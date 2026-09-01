/* Routine 1 for the multiword fp16 scheme of Fasi, Higham, Lopez, Mary &
 * Mikaitis (MIMS EPrint 2022.3, GAMM 2022).
 *
 *     A_i = fl_fp16( scale*A - sum_{k<i} A_k )
 *
 * Structurally identical to the BF16 split, with two differences that matter:
 *
 *  - fp16 keeps 11 significand bits to bf16's 8, so two words already cover
 *    22 bits and p=2 suffices against fp32 accumulation (u_low^2 = 4*u_high).
 *
 *  - fp16's exponent range is narrow (smallest normal 6.104e-5). The residual
 *    after the first word is ~2^-12 of the input, so for |a| below about
 *    0.25 the second word is already subnormal. `scale` (a power of two, so
 *    exact) lifts the operand into a range where the trailing words stay
 *    normal.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>

namespace {

constexpr int kBlock = 256;

/* fp16 limits, as FP32 values. */
constexpr float kF16Max       = 65504.0f;
constexpr float kF16MinNormal = 6.103515625e-05f;   /* 2^-14 */

template <int NSPLITS>
__device__ __forceinline__ void peelHalf(float& a, __half out[NSPLITS])
{
#pragma unroll
    for (int s = 0; s < NSPLITS; ++s) {
        out[s] = __float2half(a);           /* round to nearest even */
        a -= __half2float(out[s]);          /* exact when out[s] is normal */
    }
}

template <int NSPLITS>
__global__ void splitKernelV4(const float* __restrict__ A, int64_t lda,
                              int64_t rows, int64_t cols, float scale,
                              __half* __restrict__ S,
                              int64_t lds, int64_t stride)
{
    const int64_t r0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (r0 >= rows) return;
    const bool full = (r0 + 4 <= rows);

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        const float* __restrict__ acol = A + c * lda;
        float a[4];
        int n = 4;

        if (full) {
            const float4 v = *reinterpret_cast<const float4*>(acol + r0);
            a[0] = v.x; a[1] = v.y; a[2] = v.z; a[3] = v.w;
        } else {
            n = (int)(rows - r0);
            for (int t = 0; t < n; ++t) a[t] = acol[r0 + t];
            for (int t = n; t < 4; ++t) a[t] = 0.0f;
        }
#pragma unroll
        for (int t = 0; t < 4; ++t) a[t] *= scale;

        __half b[4][NSPLITS];
#pragma unroll
        for (int t = 0; t < 4; ++t) peelHalf<NSPLITS>(a[t], b[t]);

#pragma unroll
        for (int s = 0; s < NSPLITS; ++s) {
            __half* __restrict__ dst = S + s * stride + c * lds + r0;
            if (full) {
                __half packed[4] = { b[0][s], b[1][s], b[2][s], b[3][s] };
                *reinterpret_cast<float2*>(dst) =
                    *reinterpret_cast<const float2*>(packed);
            } else {
                for (int t = 0; t < n; ++t) dst[t] = b[t][s];
            }
        }
    }
}

template <int NSPLITS>
__global__ void splitKernelV1(const float* __restrict__ A, int64_t lda,
                              int64_t rows, int64_t cols, float scale,
                              __half* __restrict__ S,
                              int64_t lds, int64_t stride)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        float a = A[c * lda + r] * scale;
        __half b[NSPLITS];
        peelHalf<NSPLITS>(a, b);
#pragma unroll
        for (int s = 0; s < NSPLITS; ++s) S[s * stride + c * lds + r] = b[s];
    }
}

bool vectorisable(const float* dA, int64_t lda,
                  const __half* dS, int64_t lds, int64_t stride)
{
    return (reinterpret_cast<uintptr_t>(dA) % 16 == 0) && (lda % 4 == 0)
        && (reinterpret_cast<uintptr_t>(dS) % 8 == 0) && (lds % 4 == 0)
        && (stride % 4 == 0);
}

template <int NSPLITS>
void launch(const float* dA, int64_t lda, int64_t rows, int64_t cols,
            float scale, __half* dS, int64_t lds, int64_t stride,
            cudaStream_t stream)
{
    const unsigned gy = (unsigned)(cols < 65535 ? cols : 65535);
    if (vectorisable(dA, lda, dS, lds, stride)) {
        const int64_t chunks = (rows + 3) / 4;
        dim3 grid((unsigned)((chunks + kBlock - 1) / kBlock), gy);
        splitKernelV4<NSPLITS><<<grid, kBlock, 0, stream>>>(
            dA, lda, rows, cols, scale, dS, lds, stride);
    } else {
        dim3 grid((unsigned)((rows + kBlock - 1) / kBlock), gy);
        splitKernelV1<NSPLITS><<<grid, kBlock, 0, stream>>>(
            dA, lda, rows, cols, scale, dS, lds, stride);
    }
}

/* One cached 4-byte device scratch for the max reduction.
 *
 * Allocating and freeing this per call cost 0.5-3 ms -- more than the entire
 * reduction, and more than the GEMM it feeds for small operands. C++11
 * guarantees the initialisation is thread-safe; the buffer itself is not, so
 * concurrent calls from several host threads must be serialised (this is a
 * setup call that already synchronises, so it is not a hot path). */
static float* scaleScratch()
{
    static float* p = [](){
        float* q = nullptr;
        if (cudaMalloc(&q, sizeof(float)) != cudaSuccess) q = nullptr;
        return q;
    }();
    return p;
}

/* max|A| reduction. */
__global__ void maxAbsKernel(const float* __restrict__ A, int64_t lda,
                             int64_t rows, int64_t cols, float* out)
{
    __shared__ float sm[kBlock];
    float best = 0.0f;
    const int64_t total = rows * cols;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; i < total; i += stride) {
        const int64_t r = i % rows, c = i / rows;
        const float v = fabsf(A[c * lda + r]);
        if (v > best && isfinite(v)) best = v;
    }
    const int t = threadIdx.x;
    sm[t] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && sm[t + s] > sm[t]) sm[t] = sm[t + s];
        __syncthreads();
    }
    if (t == 0) atomicMax((unsigned int*)out, __float_as_uint(sm[0]));
}

/* Recomputes the split and classifies each word. */
template <int NSPLITS>
__global__ void checkKernel(const float* __restrict__ A, int64_t lda,
                            int64_t rows, int64_t cols, float scale,
                            const __half* __restrict__ S,
                            int64_t lds, int64_t stride,
                            unsigned long long* counts)
{
    unsigned long long over = 0, sub = 0, ok = 0;
    const int64_t total = rows * cols;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t gstride = (int64_t)gridDim.x * blockDim.x;

    for (; i < total; i += gstride) {
        const int64_t r = i % rows, c = i / rows;
        float resid = A[c * lda + r] * scale;
#pragma unroll
        for (int s = 0; s < NSPLITS; ++s) {
            const float w = __half2float(S[s * stride + c * lds + r]);
            const float aw = fabsf(w);
            if (!isfinite(w))                      ++over;
            else if (aw == 0.0f && resid != 0.0f)  ++sub;   /* flushed */
            else if (aw > 0.0f && aw < kF16MinNormal) ++sub; /* subnormal */
            else                                   ++ok;
            resid -= w;
        }
    }
    atomicAdd(&counts[0], over);
    atomicAdd(&counts[1], sub);
    atomicAdd(&counts[2], ok);
}

}  /* namespace */

extern "C" int mpemuMultiwordMacs(int p)
{
    if (p < 1 || p > MPEMU_MAX_SPLITS) return 0;
    return p * (p + 1) / 2;
}

extern "C" mpemuStatus_t mpemuSplitFP16(const float* dA, int64_t lda,
                                        int64_t rows, int64_t cols,
                                        int nsplits, float scale,
                                        __half* dS,
                                        int64_t lds, int64_t splitStride,
                                        cudaStream_t stream)
{
    if (!dA || !dS) return MPEMU_STATUS_INVALID_VALUE;
    if (rows < 0 || cols < 0) return MPEMU_STATUS_INVALID_VALUE;
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (lda < rows || lds < rows) return MPEMU_STATUS_INVALID_VALUE;
    if (splitStride < lds * cols) return MPEMU_STATUS_INVALID_VALUE;
    if (!(scale > 0.0f)) return MPEMU_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return MPEMU_STATUS_SUCCESS;

    switch (nsplits) {
        case 1: launch<1>(dA, lda, rows, cols, scale, dS, lds, splitStride, stream); break;
        case 2: launch<2>(dA, lda, rows, cols, scale, dS, lds, splitStride, stream); break;
        case 3: launch<3>(dA, lda, rows, cols, scale, dS, lds, splitStride, stream); break;
        case 4: launch<4>(dA, lda, rows, cols, scale, dS, lds, splitStride, stream); break;
        default: return MPEMU_STATUS_INVALID_VALUE;
    }
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}

extern "C" mpemuStatus_t mpemuAutoScaleFP16(const float* dA, int64_t lda,
                                            int64_t rows, int64_t cols,
                                            float* hScale, cudaStream_t stream)
{
    if (!dA || !hScale) return MPEMU_STATUS_INVALID_VALUE;
    *hScale = 1.0f;
    if (rows <= 0 || cols <= 0) return MPEMU_STATUS_SUCCESS;

    float* dMax = scaleScratch();
    if (!dMax) return MPEMU_STATUS_CUDA_ERROR;
    if (cudaMemsetAsync(dMax, 0, sizeof(float), stream) != cudaSuccess)
        return MPEMU_STATUS_CUDA_ERROR;
    maxAbsKernel<<<1024, kBlock, 0, stream>>>(dA, lda, rows, cols, dMax);

    float hMax = 0.0f;
    cudaMemcpyAsync(&hMax, dMax, sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return MPEMU_STATUS_CUDA_ERROR;

    if (!(hMax > 0.0f)) return MPEMU_STATUS_SUCCESS;   /* all zero */

    /* Target the top of fp16's normal range with a safety margin, so the
     * first word never overflows and the trailing words get the largest
     * possible exponent budget before going subnormal. 2^12 = 4096 leaves
     * 4 binades of headroom under 65504. */
    const float target = 4096.0f;
    int e = 0;
    frexpf(target / hMax, &e);          /* target/hMax ~ 2^e */
    *hScale = ldexpf(1.0f, e - 1);      /* largest power of two <= target/hMax */
    if (!(*hScale > 0.0f) || !isfinite(*hScale)) *hScale = 1.0f;
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuCheckSplitFP16(const float* dA, int64_t lda,
                                             int64_t rows, int64_t cols,
                                             int nsplits, float scale,
                                             const __half* dS,
                                             int64_t lds, int64_t splitStride,
                                             unsigned long long* dCounts,
                                             cudaStream_t stream)
{
    if (!dA || !dS || !dCounts) return MPEMU_STATUS_INVALID_VALUE;
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (rows <= 0 || cols <= 0) return MPEMU_STATUS_SUCCESS;

    switch (nsplits) {
        case 1: checkKernel<1><<<1024,kBlock,0,stream>>>(dA,lda,rows,cols,scale,dS,lds,splitStride,dCounts); break;
        case 2: checkKernel<2><<<1024,kBlock,0,stream>>>(dA,lda,rows,cols,scale,dS,lds,splitStride,dCounts); break;
        case 3: checkKernel<3><<<1024,kBlock,0,stream>>>(dA,lda,rows,cols,scale,dS,lds,splitStride,dCounts); break;
        case 4: checkKernel<4><<<1024,kBlock,0,stream>>>(dA,lda,rows,cols,scale,dS,lds,splitStride,dCounts); break;
        default: return MPEMU_STATUS_INVALID_VALUE;
    }
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}
