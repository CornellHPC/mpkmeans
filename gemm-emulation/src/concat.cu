/* Concatenated-k variant: one GEMM per bin, no temporary, no reduction.
 *
 * See the header for why sliding the right operand's base by
 * (nsplits-1-b)*k pairs A's block i with B's block b-i for every i at once.
 */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

#include <cstdint>

namespace {

constexpr int kBlock = 256;

/* Stacked REVERSED right-operand split: word j of element (r,c) goes to row
 * (nsplits-1-j)*k + r of a (nsplits*k) x n column-major buffer. */
template <typename T, int NSPLITS, bool IS_HALF>
__global__ void splitStacked(const float* __restrict__ B, int64_t ldb,
                             int64_t k, int64_t n, float scale,
                             T* __restrict__ S, int64_t lds)
{
    const int64_t r = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= k) return;

    for (int64_t c = blockIdx.y; c < n; c += gridDim.y) {
        float a = B[c * ldb + r] * scale;
        T* __restrict__ col = S + c * lds;
#pragma unroll
        for (int j = 0; j < NSPLITS; ++j) {
            T w;
            if constexpr (IS_HALF) { w = __float2half(a);     a -= __half2float(w); }
            else                   { w = __float2bfloat16(a); a -= __bfloat162float(w); }
            col[(int64_t)(NSPLITS - 1 - j) * k + r] = w;
        }
    }
}

template <typename T, bool IS_HALF>
mpemuStatus_t launchStacked(const float* dB, int64_t ldb, int64_t k, int64_t n,
                            int nsplits, float scale, T* dS, int64_t lds,
                            cudaStream_t stream)
{
    if (!dB || !dS) return MPEMU_STATUS_INVALID_VALUE;
    if (k < 0 || n < 0) return MPEMU_STATUS_INVALID_VALUE;
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (ldb < k) return MPEMU_STATUS_INVALID_VALUE;
    if (lds < (int64_t)nsplits * k) return MPEMU_STATUS_INVALID_VALUE;
    if (!(scale > 0.0f)) return MPEMU_STATUS_INVALID_VALUE;
    if (k == 0 || n == 0) return MPEMU_STATUS_SUCCESS;

    dim3 grid((unsigned)((k + kBlock - 1) / kBlock),
              (unsigned)(n < 65535 ? n : 65535));
    switch (nsplits) {
        case 1: splitStacked<T,1,IS_HALF><<<grid,kBlock,0,stream>>>(dB,ldb,k,n,scale,dS,lds); break;
        case 2: splitStacked<T,2,IS_HALF><<<grid,kBlock,0,stream>>>(dB,ldb,k,n,scale,dS,lds); break;
        case 3: splitStacked<T,3,IS_HALF><<<grid,kBlock,0,stream>>>(dB,ldb,k,n,scale,dS,lds); break;
        case 4: splitStacked<T,4,IS_HALF><<<grid,kBlock,0,stream>>>(dB,ldb,k,n,scale,dS,lds); break;
        default: return MPEMU_STATUS_INVALID_VALUE;
    }
    return (cudaGetLastError() == cudaSuccess) ? MPEMU_STATUS_SUCCESS
                                               : MPEMU_STATUS_CUDA_ERROR;
}

/* One GEMM per bin. Bin b uses inner dimension (b+1)*k, A from column 0 and B
 * from row block (nsplits-1-b)*k. */
mpemuStatus_t concatGemm(cublasHandle_t handle,
                         int64_t m, int64_t n, int64_t k,
                         float alpha, cudaDataType inType,
                         const void* dSA, int64_t lda,
                         const void* dSB, int64_t ldsb, size_t elemBytes,
                         int nsplits, int nbins,
                         float beta, float* dC, int64_t ldc)
{
    if (!handle || !dC || !dSA || !dSB) return MPEMU_STATUS_INVALID_VALUE;
    if (nsplits < 1 || nbins < 1 || nbins > nsplits) return MPEMU_STATUS_INVALID_VALUE;

    for (int b = 0; b < nbins; ++b) {
        const int64_t kk = (int64_t)(b + 1) * k;
        const char* bBase = (const char*)dSB
                          + (size_t)(nsplits - 1 - b) * k * elemBytes;
        const float bt = (b == 0) ? beta : 1.0f;
        const cublasStatus_t st = cublasGemmEx_64(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, kk,
            &alpha,
            dSA,   inType, lda,
            bBase, inType, ldsb,
            &bt,
            dC, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) return MPEMU_STATUS_CUBLAS_ERROR;
    }
    return MPEMU_STATUS_SUCCESS;
}

}  /* namespace */

extern "C" int64_t mpemuStackedLd(int64_t k, int nsplits)
{
    if (nsplits < 1) return 0;
    return ((int64_t)nsplits * k + 7) / 8 * 8;
}

extern "C" size_t mpemuStackedBytes(int64_t k, int64_t n, int nsplits, int elemBytes)
{
    if (nsplits < 1 || elemBytes < 1) return 0;
    return (size_t)mpemuStackedLd(k, nsplits) * (size_t)n * (size_t)elemBytes;
}

extern "C" mpemuStatus_t mpemuSplitBF16Stacked(const float* dB, int64_t ldb,
                                               int64_t k, int64_t n, int nsplits,
                                               __nv_bfloat16* dS, int64_t lds,
                                               cudaStream_t stream)
{
    return launchStacked<__nv_bfloat16,false>(dB, ldb, k, n, nsplits, 1.0f, dS, lds, stream);
}

extern "C" mpemuStatus_t mpemuSplitFP16Stacked(const float* dB, int64_t ldb,
                                               int64_t k, int64_t n, int nsplits,
                                               float scale,
                                               __half* dS, int64_t lds,
                                               cudaStream_t stream)
{
    return launchStacked<__half,true>(dB, ldb, k, n, nsplits, scale, dS, lds, stream);
}

extern "C" mpemuStatus_t mpemuGemmEmulatedConcat(cublasHandle_t handle,
                                                 int64_t m, int64_t n, int64_t k,
                                                 float alpha,
                                                 const __nv_bfloat16* dSA, int64_t lda,
                                                 const __nv_bfloat16* dSB, int64_t ldsb,
                                                 int nsplits, int nbins,
                                                 float beta, float* dC, int64_t ldc)
{
    return concatGemm(handle, m, n, k, alpha, CUDA_R_16BF,
                      dSA, lda, dSB, ldsb, sizeof(__nv_bfloat16),
                      nsplits, nbins, beta, dC, ldc);
}

extern "C" mpemuStatus_t mpemuGemmMultiwordConcat(cublasHandle_t handle,
                                                  int64_t m, int64_t n, int64_t k,
                                                  float alpha, float scaleA, float scaleB,
                                                  const __half* dSA, int64_t lda,
                                                  const __half* dSB, int64_t ldsb,
                                                  int nsplits, int nbins,
                                                  float beta, float* dC, int64_t ldc)
{
    if (!(scaleA > 0.0f) || !(scaleB > 0.0f)) return MPEMU_STATUS_INVALID_VALUE;
    return concatGemm(handle, m, n, k, alpha / (scaleA * scaleB), CUDA_R_16F,
                      dSA, lda, dSB, ldsb, sizeof(__half),
                      nsplits, nbins, beta, dC, ldc);
}
