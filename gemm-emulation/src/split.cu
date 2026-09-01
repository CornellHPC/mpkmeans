/* Routine 1: split an FP32 device matrix into nsplits BF16 planes.
 *
 * b[0] = BF16(a);  b[s] = BF16(a - sum_{t<s} b[t])
 *
 * Each residual subtraction is exact in FP32 (the subtrahend is a
 * round-to-nearest 8-bit-significand approximation of the minuend, so the
 * difference is representable), which is what makes the 3-way split exact.
 *
 * The kernel is purely bandwidth bound: it reads 4*rows*cols bytes and writes
 * 2*nsplits*rows*cols. Each thread handles 4 consecutive rows of one column so
 * loads are 16-byte and stores are 8-byte per plane.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>

namespace {

constexpr int kBlock = 256;

/* Peel NSPLITS BF16 components off `a`, leaving the residual in `a`. */
template <int NSPLITS>
__device__ __forceinline__ void peel(float& a, __nv_bfloat16 out[NSPLITS])
{
#pragma unroll
    for (int s = 0; s < NSPLITS; ++s) {
        out[s] = __float2bfloat16(a);          /* round to nearest even */
        a -= __bfloat162float(out[s]);         /* exact */
    }
}

/* Vectorised path: 4 rows per thread, 16B loads / 8B stores. */
template <int NSPLITS>
__global__ void splitKernelV4(const float* __restrict__ A, int64_t lda,
                              int64_t rows, int64_t cols,
                              __nv_bfloat16* __restrict__ S,
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

        __nv_bfloat16 b[4][NSPLITS];
#pragma unroll
        for (int t = 0; t < 4; ++t) peel<NSPLITS>(a[t], b[t]);

#pragma unroll
        for (int s = 0; s < NSPLITS; ++s) {
            __nv_bfloat16* __restrict__ dst = S + s * stride + c * lds + r0;
            if (full) {
                __nv_bfloat16 packed[4] = { b[0][s], b[1][s], b[2][s], b[3][s] };
                *reinterpret_cast<float2*>(dst) =
                    *reinterpret_cast<const float2*>(packed);
            } else {
                for (int t = 0; t < n; ++t) dst[t] = b[t][s];
            }
        }
    }
}

/* Scalar fallback for unaligned leading dimensions or base pointers. */
template <int NSPLITS>
__global__ void splitKernelV1(const float* __restrict__ A, int64_t lda,
                              int64_t rows, int64_t cols,
                              __nv_bfloat16* __restrict__ S,
                              int64_t lds, int64_t stride)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    for (int64_t c = blockIdx.y; c < cols; c += gridDim.y) {
        float a = A[c * lda + r];
        __nv_bfloat16 b[NSPLITS];
        peel<NSPLITS>(a, b);
#pragma unroll
        for (int s = 0; s < NSPLITS; ++s) S[s * stride + c * lds + r] = b[s];
    }
}

bool vectorisable(const float* dA, int64_t lda,
                  const __nv_bfloat16* dS, int64_t lds, int64_t stride)
{
    return (reinterpret_cast<uintptr_t>(dA) % 16 == 0) && (lda % 4 == 0)
        && (reinterpret_cast<uintptr_t>(dS) % 8 == 0) && (lds % 4 == 0)
        && (stride % 4 == 0);
}

template <int NSPLITS>
void launch(const float* dA, int64_t lda, int64_t rows, int64_t cols,
            __nv_bfloat16* dS, int64_t lds, int64_t stride,
            cudaStream_t stream)
{
    const unsigned gy = (unsigned)(cols < 65535 ? cols : 65535);
    if (vectorisable(dA, lda, dS, lds, stride)) {
        const int64_t chunks = (rows + 3) / 4;
        dim3 grid((unsigned)((chunks + kBlock - 1) / kBlock), gy);
        splitKernelV4<NSPLITS><<<grid, kBlock, 0, stream>>>(
            dA, lda, rows, cols, dS, lds, stride);
    } else {
        dim3 grid((unsigned)((rows + kBlock - 1) / kBlock), gy);
        splitKernelV1<NSPLITS><<<grid, kBlock, 0, stream>>>(
            dA, lda, rows, cols, dS, lds, stride);
    }
}

}  /* namespace */

extern "C" mpemuStatus_t mpemuSplitBF16(const float* dA, int64_t lda,
                                        int64_t rows, int64_t cols,
                                        int nsplits,
                                        __nv_bfloat16* dS,
                                        int64_t lds, int64_t splitStride,
                                        cudaStream_t stream)
{
    if (!dA || !dS) return MPEMU_STATUS_INVALID_VALUE;
    if (rows < 0 || cols < 0) return MPEMU_STATUS_INVALID_VALUE;
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (lda < rows || lds < rows) return MPEMU_STATUS_INVALID_VALUE;
    if (splitStride < lds * cols) return MPEMU_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return MPEMU_STATUS_SUCCESS;

    switch (nsplits) {
        case 1: launch<1>(dA, lda, rows, cols, dS, lds, splitStride, stream); break;
        case 2: launch<2>(dA, lda, rows, cols, dS, lds, splitStride, stream); break;
        case 3: launch<3>(dA, lda, rows, cols, dS, lds, splitStride, stream); break;
        case 4: launch<4>(dA, lda, rows, cols, dS, lds, splitStride, stream); break;
        default: return MPEMU_STATUS_INVALID_VALUE;
    }
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}
