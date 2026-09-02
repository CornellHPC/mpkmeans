#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include "mpkmeans/mpkmeans.h"

#define MPK_CUDA(call)                                                        \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(e_));                                  \
            return MPK_ERR_CUDA;                                              \
        }                                                                     \
    } while (0)

#define MPK_BLAS(call)                                                        \
    do {                                                                      \
        cublasStatus_t s_ = (call);                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "%s:%d cuBLAS: %d\n", __FILE__, __LINE__, (int)s_);\
            return MPK_ERR_CUBLAS;                                            \
        }                                                                     \
    } while (0)

static inline int mpk_ceil_div(int a, int b) { return (a + b - 1) / b; }

#ifdef MPK_STATS
/* Exclusion statistics are accumulated into this many banks: funnelling one
 * atomic per block onto a single global address costs more than the kernel.
 * Must be a power of two. */
#define MPK_STAT_BANKS 64
#endif

/* ------------------------------------------------------------- kernels ---
 *
 * The mixed precision iteration needs exactly three kernels of its own:
 * k_argmin (row argmin of the low precision distances, plus the condition (6)
 * reference entry), k_condition (the exclusion test), and k_update (the
 * fallback high precision refinement).  The rest is cuBLAS, cuSPARSE, CUB,
 * and the
 * shared bookkeeping below. */

/* h[i] = (half)f[i] */
void mpkLaunchToHalf(const float* src, __half* dst, long long n,
                     cudaStream_t s);

/* out[j] = ||C(j,:)||^2, FP32.  One block per centroid. */
void mpkLaunchRowNorms(const float* dC, int k, int d, float* out,
                       cudaStream_t s);

/* ARGMIN + EXCLUSION COUNT.  One warp per row.  With
 * Dt(i,j) = cnorm2[j] - 2*G(i,j) it produces
 *   jbest[i]  argmin column of Dt(i,:) (smallest index on ties)
 *   dbest[i]  Dt(i, jbest[i])
 *   gbest[i]  G (i, jbest[i])                    (= fl16(p_i^T c_jbest))
 *   gexact[i] fl32(<p_i, c_jbest[i]>), the high precision reference entry,
 *             only when use_cond6 is set -- condition (3) needs no high
 *             precision quantity at all, so none is computed
 *   row_nnz[i] how many columns survived the exclusion test
 *
 * The row of G is scanned twice, once for the argmin and once for the test,
 * but only the first scan reaches past L1: a row is k*4 bytes and is still
 * resident for the second.  Running the test as a separate kernel meant
 * streaming all of G from HBM a second time and repeating the per-row setup.
 *
 * gexact is a warp-strided fma dot in registers.  Materialising
 * prod(i,:) = P(i,:)*C(jbest[i],:) and reducing it with a cublasSgemv instead
 * is an n x d write plus an n x d read to produce n numbers; it cost
 * 175 us/iteration at n=200k, d=128 against 35 us for the fused dot.
 *
 * The tests, with
 *
 *     factor = 2*eps/(1-eps),   eps  = FP16-input GEMM relative error bound
 *     gfac   = 2*gs /(1-gs ),   gs   = FP32 dot-product relative error bound
 *     slack  = small multiple of the FP32 unit roundoff
 *
 * are, for every kk != jbest[i],
 *
 *   (3)  dbest[i] + slack*(|dbest[i]|+|Dt|) <= Dt - factor*(G(i,kk) + gbest[i])
 *   (6)  dup  [i] + slack*(|dup  [i]|+|Dt|) <= Dt - factor* G(i,kk)
 *
 * where
 *
 *   dexact[i] = cnorm2[jbest[i]] - 2*gexact[i]
 *   dup   [i] = dexact[i] + gfac*gexact[i]   >=  the exact D(i, jbest[i]).
 *
 * The slack term absorbs the FP32 rounding incurred while forming the norms
 * and Dt itself, which the paper's error model omits; it is ~4 orders of
 * magnitude below `factor` and never drives the result.
 *
 * include_best folds jbest[i] into the candidate count of every row that has
 * at least one survivor.  It is used when condition (6) is disabled: with no
 * reference entry computed, the incumbent's own high precision distance is
 * instead produced by the update step -- but only for rows (3) did not clear.
 *
 * stat_banks is 3*MPK_STAT_BANKS device counters (cond3, cond6, both) and is
 * only touched in MPK_STATS builds; pass NULL otherwise. */
void mpkLaunchArgminCount(const float* G, const float* cnorm2, const float* dP,
                          const float* dC, int n, int d, int k,
                          float factor, float gfac, float slack,
                          int use_cond3, int use_cond6, int cascade,
                          int* jbest, float* dbest, float* gbest, float* gexact,
                          unsigned long long* bestpack, int include_best,
                          int* list, int cap, unsigned int* count,
                          unsigned long long* ref_count,
                          long long* stat_banks,
                          cudaStream_t s);

/* CONDITION KERNEL, second half: with row_nnz scanned into row_ptr, re-apply
 * the same predicate and write the CSR survivor pattern (col) plus the row
 * index of each entry (rowidx, which the update kernel needs).  Touches only
 * the rows that have survivors. */

/* UPDATE KERNEL.  One warp per
 * surviving entry e:  val[e] = fl32(<P(rowidx[e],:), C(col[e],:)>). */
void mpkLaunchUpdateFlat(const float* dP, const float* dC, const float* cnorm2,
                         const int* list, int nnz, int k, int d,
                         unsigned long long* bestpack, cudaStream_t s);

/* Final assignment: start from (dexact[i], jbest[i]) with
 * dexact[i] = cnorm2[jbest[i]] - 2*gexact[i], and beat it with the refined
 * survivors, Dsurv = cnorm2[col[e]] - 2*val[e].  Ties go to the smaller
 * column index.  With gexact NULL the incumbent is instead expected to appear
 * in the row itself (see include_best), and an empty row means jbest[i] was
 * proved optimal. */
void mpkLaunchUnpack(const unsigned long long* bestpack, int n, int* assign,
                     cudaStream_t s);

/* Row-wise argmin of the dense FP32 D(i,j) = cnorm2[j] - 2*G32(i,j). */
void mpkLaunchRowArgmin32(const float* G32, const float* cnorm2, int n, int k,
                          int* assign, cudaStream_t s);

/* Count positions where a[i] != b[i]. */
void mpkLaunchCountDiff(const int* a, const int* b, int n, long long* out,
                        cudaStream_t s);

/* Centroid recomputation.  sums is k x d FP64, counts is k int. */
void mpkLaunchZero(double* sums, int* counts, int k, int d, cudaStream_t s);
void mpkLaunchAccumulate(const float* dP, const int* assign, int n, int d,
                         double* sums, int* counts, cudaStream_t s);
/* dC(j,:) = sums(j,:)/counts[j]; empty clusters keep their old centroid.
 * moved2 (k floats) receives ||c_new - c_old||^2 per centroid. */
void mpkLaunchFinalizeCentroids(const double* sums, const int* counts,
                                int k, int d, float* dC, float* moved2,
                                cudaStream_t s);

/* inertia = sum_i ||p_i - c_assign(i)||^2, FP64. */
void mpkLaunchInertia(const float* dP, const float* dC, const int* assign,
                      int n, int d, double* out, cudaStream_t s);

#ifdef MPK_STATS
/* Scores a mixed iteration against the ordinary FP32 implementation.  G32 and
 * `ref` come from a cublasSgemm on the same centroids followed by
 * k_row_argmin_32 -- that is, from the reference algorithm itself, not from a
 * separate higher precision oracle.  Accumulates:
 *   n_excluded_best += the FP32 label was neither the incumbent nor a
 *                      survivor, i.e. a condition of Theorem 1 failed,
 *   n_label_diff    += the FP32 label was reachable but not the one picked,
 *   excess          += D32(i, assign[i]) - D32(i, ref[i]).
 * Counters accumulate; the caller zeroes them. */
void mpkLaunchVerifyRef(const float* G, const float* G32, const float* cnorm2,
                        const int* jbest, const float* dbest, const float* gbest,
                        const float* gexact, const int* assign, const int* ref,
                        int n, int k, float factor, float gfac, float slack,
                        int use_cond3, int use_cond6,
                        long long* n_excluded_best, long long* n_label_diff,
                        double* excess, cudaStream_t s);
#endif
