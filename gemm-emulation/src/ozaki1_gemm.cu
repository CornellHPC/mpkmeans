/* Ozaki scheme I: the INT8 tensor-core GEMM (routine 2) and the drivers that
 * walk the slice-product schedule. */

#include "mpemu/mpemu.h"
#include "mpemu_internal.h"

extern "C" mpemuStatus_t mpemuGemmInt8(cublasHandle_t handle,
                                       cublasOperation_t opA, cublasOperation_t opB,
                                       int64_t m, int64_t n, int64_t k,
                                       int alpha,
                                       const signed char* dAi, int64_t lda,
                                       const signed char* dBj, int64_t ldb,
                                       int beta,
                                       int* dC, int64_t ldc)
{
    if (!handle || !dC) return MPEMU_STATUS_INVALID_VALUE;
    if (k > 0 && (!dAi || !dBj)) return MPEMU_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return MPEMU_STATUS_INVALID_VALUE;

    const cublasStatus_t st = cublasGemmEx_64(
        handle, opA, opB, m, n, k,
        &alpha,
        dAi, CUDA_R_8I, lda,
        dBj, CUDA_R_8I, ldb,
        &beta,
        dC, CUDA_R_32I, ldc,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

    return (st == CUBLAS_STATUS_SUCCESS) ? MPEMU_STATUS_SUCCESS
                                         : MPEMU_STATUS_CUBLAS_ERROR;
}

extern "C" mpemuStatus_t mpemuGemmOzaki1Range(cublasHandle_t handle,
                                              int64_t m, int64_t n, int64_t k,
                                              double alpha,
                                              const signed char* dSA, int64_t ldsa, int64_t strideA,
                                              const int* dExpA,
                                              const signed char* dSB, int64_t ldsb, int64_t strideB,
                                              const int* dExpB,
                                              int nslices, int alphaBits,
                                              int macBegin, int macEnd,
                                              double beta,
                                              double* dC, int64_t ldc,
                                              int* dCt, int64_t ldct,
                                              cudaStream_t stream)
{
    if (!handle || !dC || !dCt) return MPEMU_STATUS_INVALID_VALUE;
    if (nslices < 1 || nslices > MPEMU_MAX_TERM_SPLITS) return MPEMU_STATUS_INVALID_VALUE;
    if (alphaBits < 1 || alphaBits > 7) return MPEMU_STATUS_INVALID_VALUE;

    const int total = mpemuOz1Macs(nslices);
    if (macEnd == MPEMU_ALL_MACS) macEnd = total;
    if (macBegin < 0 || macEnd > total || macBegin > macEnd)
        return MPEMU_STATUS_INVALID_VALUE;

    if (macBegin == macEnd) {
        if (beta == 1.0) return MPEMU_STATUS_SUCCESS;
        /* honour beta with no contribution */
        return mpemuOz1Accumulate(dCt, ldct, m, n, 0, alphaBits,
                                  dExpA, dExpB, 0.0, beta, dC, ldc, stream);
    }

    /* One slice product is bounded by k*(2^alpha-1)^2; that fixes how many
     * may share an INT32 buffer before it can overflow. Products in the same
     * bin (i+j) carry the same 2^-((i+j+2)*alpha) scale, so grouping them
     * saves a pass over C -- but the group must respect this cap. */
    const int64_t d = ((int64_t)1 << alphaBits) - 1;
    const int64_t perGemm = (k > 0) ? k * d * d : 1;
    if (perGemm > 2147483647LL) return MPEMU_STATUS_INVALID_VALUE;  /* k too large */
    int maxAcc = (int)(2147483647LL / (perGemm > 0 ? perGemm : 1));
    if (maxAcc < 1) maxAcc = 1;

    /* Ascending-bin order: the first mpemuOz1Macs(s) entries are exactly the
     * paper's {(i,j) : i+j <= s+1}, so a prefix is always a valid scheme. */
    /* room for the full s(s+1)/2 schedule at the largest supported s */
    mpemuTerm_t terms[MPEMU_MAX_TERM_SPLITS * (MPEMU_MAX_TERM_SPLITS + 1) / 2];
    if (macEnd > (int)(sizeof terms / sizeof terms[0])) return MPEMU_STATUS_INVALID_VALUE;
    const int nterms = mpemuTermSchedule(nslices, macEnd, terms);
    if (nterms < macEnd) return MPEMU_STATUS_INVALID_VALUE;

    double curBeta = beta;
    int t = macBegin;
    while (t < macEnd) {
        const int bin = terms[t].i + terms[t].j;
        /* consecutive terms of this bin, capped by the overflow budget */
        int g = 0;
        while (t + g < macEnd && (terms[t + g].i + terms[t + g].j) == bin && g < maxAcc)
            ++g;

        for (int u = 0; u < g; ++u) {
            const mpemuTerm_t& tm = terms[t + u];
            const mpemuStatus_t st = mpemuGemmInt8(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, 1,
                mpemuOz1PlaneConst(dSA, strideA, tm.i), ldsa,
                mpemuOz1PlaneConst(dSB, strideB, tm.j), ldsb,
                (u == 0) ? 0 : 1,
                dCt, ldct);
            if (st != MPEMU_STATUS_SUCCESS) return st;
        }

        const mpemuStatus_t st = mpemuOz1Accumulate(dCt, ldct, m, n, bin, alphaBits,
                                                    dExpA, dExpB, alpha, curBeta,
                                                    dC, ldc, stream);
        if (st != MPEMU_STATUS_SUCCESS) return st;
        curBeta = 1.0;
        t += g;
    }
    return MPEMU_STATUS_SUCCESS;
}

extern "C" mpemuStatus_t mpemuGemmOzaki1(cublasHandle_t handle,
                                         int64_t m, int64_t n, int64_t k,
                                         double alpha,
                                         const signed char* dSA, int64_t ldsa, int64_t strideA,
                                         const int* dExpA,
                                         const signed char* dSB, int64_t ldsb, int64_t strideB,
                                         const int* dExpB,
                                         int nslices, int alphaBits,
                                         double beta,
                                         double* dC, int64_t ldc,
                                         int* dCt, int64_t ldct,
                                         cudaStream_t stream)
{
    return mpemuGemmOzaki1Range(handle, m, n, k, alpha,
                                dSA, ldsa, strideA, dExpA,
                                dSB, ldsb, strideB, dExpB,
                                nslices, alphaBits,
                                0, mpemuOz1Macs(nslices), beta,
                                dC, ldc, dCt, ldct, stream);
}
