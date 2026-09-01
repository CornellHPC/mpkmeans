/* Routine 2 (one BF16 cross product), the cross-product ordering, and a
 * convenience driver that chains the two. */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

extern "C" mpemuStatus_t mpemuGemmBF16(cublasHandle_t handle,
                                       cublasOperation_t opA, cublasOperation_t opB,
                                       int64_t m, int64_t n, int64_t k,
                                       float alpha,
                                       const __nv_bfloat16* dAi, int64_t lda,
                                       const __nv_bfloat16* dBj, int64_t ldb,
                                       float beta,
                                       float* dC, int64_t ldc)
{
    if (!handle || !dC) return MPEMU_STATUS_INVALID_VALUE;
    if (k > 0 && (!dAi || !dBj)) return MPEMU_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return MPEMU_STATUS_INVALID_VALUE;

    const cublasStatus_t st = cublasGemmEx_64(
        handle, opA, opB, m, n, k,
        &alpha,
        dAi, CUDA_R_16BF, lda,
        dBj, CUDA_R_16BF, ldb,
        &beta,
        dC, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    return (st == CUBLAS_STATUS_SUCCESS) ? MPEMU_STATUS_SUCCESS
                                         : MPEMU_STATUS_CUBLAS_ERROR;
}

extern "C" int mpemuTermCount(int nsplits)
{
    if (nsplits < 1 || nsplits > MPEMU_MAX_TERM_SPLITS) return 0;
    return nsplits * nsplits;
}

/* Ordered by ascending bin (i+j), then ascending i. Term (i,j) contributes on
 * the order of 2^(-8*(i+j)) relative to (0,0), so this emits the largest
 * contributions first and any prefix of the list is the best available
 * approximation for that many multiply-accumulates. */
extern "C" int mpemuTermSchedule(int nsplits, int maxTerms, mpemuTerm_t* terms)
{
    if (nsplits < 1 || nsplits > MPEMU_MAX_TERM_SPLITS || maxTerms <= 0 || !terms)
        return 0;

    int written = 0;
    for (int bin = 0; bin <= 2 * (nsplits - 1) && written < maxTerms; ++bin) {
        for (int i = 0; i <= bin && written < maxTerms; ++i) {
            const int j = bin - i;
            if (i < nsplits && j < nsplits) {
                terms[written].i = i;
                terms[written].j = j;
                ++written;
            }
        }
    }
    return written;
}

extern "C" mpemuStatus_t mpemuGemmEmulatedRange(cublasHandle_t handle,
                                                cublasOperation_t opA, cublasOperation_t opB,
                                                int64_t m, int64_t n, int64_t k,
                                                float alpha,
                                                const __nv_bfloat16* dSA, int64_t lda, int64_t strideA,
                                                const __nv_bfloat16* dSB, int64_t ldb, int64_t strideB,
                                                int nsplits, int macBegin, int macEnd,
                                                float beta,
                                                float* dC, int64_t ldc)
{
    if (nsplits < 1 || nsplits > MPEMU_MAX_SPLITS) return MPEMU_STATUS_INVALID_VALUE;

    const int total = mpemuTermCount(nsplits);
    if (macEnd == MPEMU_ALL_MACS) macEnd = total;
    if (macBegin < 0 || macEnd > total || macBegin > macEnd)
        return MPEMU_STATUS_INVALID_VALUE;

    if (macBegin == macEnd) {
        /* Empty range: still honour beta (BLAS k=0 semantics). */
        if (beta == 1.0f) return MPEMU_STATUS_SUCCESS;
        return mpemuGemmBF16(handle, opA, opB, m, n, 0, alpha,
                             dSA, lda, dSB, ldb, beta, dC, ldc);
    }

    /* Generate the schedule up to macEnd and walk the requested window; the
     * ordering is deterministic, so positions mean the same thing across
     * calls and a refinement continues exactly where the last one stopped. */
    mpemuTerm_t terms[MPEMU_MAX_SPLITS * MPEMU_MAX_SPLITS];
    const int nterms = mpemuTermSchedule(nsplits, macEnd, terms);
    if (nterms < macEnd) return MPEMU_STATUS_INVALID_VALUE;

    for (int t = macBegin; t < macEnd; ++t) {
        const mpemuStatus_t st = mpemuGemmBF16(
            handle, opA, opB, m, n, k, alpha,
            mpemuSplitPlaneConst(dSA, strideA, terms[t].i), lda,
            mpemuSplitPlaneConst(dSB, strideB, terms[t].j), ldb,
            (t == macBegin) ? beta : 1.0f,
            dC, ldc);
        if (st != MPEMU_STATUS_SUCCESS) return st;
    }
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuGemmEmulated(cublasHandle_t handle,
                                           cublasOperation_t opA, cublasOperation_t opB,
                                           int64_t m, int64_t n, int64_t k,
                                           float alpha,
                                           const __nv_bfloat16* dSA, int64_t lda, int64_t strideA,
                                           const __nv_bfloat16* dSB, int64_t ldb, int64_t strideB,
                                           int nsplits, int nmacs,
                                           float beta,
                                           float* dC, int64_t ldc)
{
    return mpemuGemmEmulatedRange(handle, opA, opB, m, n, k, alpha,
                                  dSA, lda, strideA, dSB, ldb, strideB,
                                  nsplits, 0, nmacs, beta, dC, ldc);
}
