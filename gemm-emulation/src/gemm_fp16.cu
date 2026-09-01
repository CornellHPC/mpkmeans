/* Routine 2 for the multiword fp16 scheme, plus its driver. */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

extern "C" mpemuStatus_t mpemuGemmFP16(cublasHandle_t handle,
                                       cublasOperation_t opA, cublasOperation_t opB,
                                       int64_t m, int64_t n, int64_t k,
                                       float alpha,
                                       const __half* dAi, int64_t lda,
                                       const __half* dBj, int64_t ldb,
                                       float beta,
                                       float* dC, int64_t ldc)
{
    if (!handle || !dC) return MPEMU_STATUS_INVALID_VALUE;
    if (k > 0 && (!dAi || !dBj)) return MPEMU_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return MPEMU_STATUS_INVALID_VALUE;

    const cublasStatus_t st = cublasGemmEx_64(
        handle, opA, opB, m, n, k,
        &alpha,
        dAi, CUDA_R_16F, lda,
        dBj, CUDA_R_16F, ldb,
        &beta,
        dC, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    return (st == CUBLAS_STATUS_SUCCESS) ? MPEMU_STATUS_SUCCESS
                                         : MPEMU_STATUS_CUBLAS_ERROR;
}

extern "C" mpemuStatus_t mpemuGemmMultiwordRange(cublasHandle_t handle,
                                                 cublasOperation_t opA, cublasOperation_t opB,
                                                 int64_t m, int64_t n, int64_t k,
                                                 float alpha,
                                                 const __half* dSA, int64_t lda, int64_t strideA,
                                                 float scaleA,
                                                 const __half* dSB, int64_t ldb, int64_t strideB,
                                                 float scaleB,
                                                 int nsplits, int macBegin, int macEnd,
                                                 float beta,
                                                 float* dC, int64_t ldc)
{
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (!(scaleA > 0.0f) || !(scaleB > 0.0f)) return MPEMU_STATUS_INVALID_VALUE;

    const int total = mpemuTermCount(nsplits);
    if (macEnd == MPEMU_ALL_MACS) macEnd = total;
    if (macBegin < 0 || macEnd > total || macBegin > macEnd)
        return MPEMU_STATUS_INVALID_VALUE;

    /* The operands were split after multiplying by scaleA / scaleB, so the
     * cross products carry a factor scaleA*scaleB. Both are powers of two in
     * normal use, making this division exact -- and identical on every
     * refinement call, so partial results stay consistent. */
    const float a = alpha / (scaleA * scaleB);

    if (macBegin == macEnd) {
        if (beta == 1.0f) return MPEMU_STATUS_SUCCESS;
        return mpemuGemmFP16(handle, opA, opB, m, n, 0, a,
                             dSA, lda, dSB, ldb, beta, dC, ldc);
    }

    mpemuTerm_t terms[MPEMU_MAX_SPLITS * MPEMU_MAX_SPLITS];
    const int nterms = mpemuTermSchedule(nsplits, macEnd, terms);
    if (nterms < macEnd) return MPEMU_STATUS_INVALID_VALUE;

    for (int t = macBegin; t < macEnd; ++t) {
        const mpemuStatus_t st = mpemuGemmFP16(
            handle, opA, opB, m, n, k, a,
            mpemuSplitPlaneFP16Const(dSA, strideA, terms[t].i), lda,
            mpemuSplitPlaneFP16Const(dSB, strideB, terms[t].j), ldb,
            (t == macBegin) ? beta : 1.0f,
            dC, ldc);
        if (st != MPEMU_STATUS_SUCCESS) return st;
    }
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuGemmMultiword(cublasHandle_t handle,
                                            cublasOperation_t opA, cublasOperation_t opB,
                                            int64_t m, int64_t n, int64_t k,
                                            float alpha,
                                            const __half* dSA, int64_t lda, int64_t strideA,
                                            float scaleA,
                                            const __half* dSB, int64_t ldb, int64_t strideB,
                                            float scaleB,
                                            int nsplits, int nmacs,
                                            float beta,
                                            float* dC, int64_t ldc)
{
    return mpemuGemmMultiwordRange(handle, opA, opB, m, n, k, alpha,
                                   dSA, lda, strideA, scaleA,
                                   dSB, ldb, strideB, scaleB,
                                   nsplits, 0, nmacs, beta, dC, ldc);
}
