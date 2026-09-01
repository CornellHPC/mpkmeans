/* mpkmeans.h -- Mixed precision K-Means (Lloyd's) with cluster exclusion bounds.
 *
 * Implements the "Mixed Precision K-Means" scheme: the O(ndk) -2*P*C^T GEMM is
 * evaluated with FP16 inputs, and the cluster-exclusion conditions (3) and (6)
 * of Theorem 1 decide which entries of the distance matrix must be recomputed
 * in FP32.
 *
 *   (3) is free: both sides come out of the low precision GEMM.
 *   (6) needs one high precision distance per row, computed in the same pass
 *       as the row argmin.
 *
 * The surviving entries are refined either with cusparseSDDMM (when there are
 * many of them) or with a warp-per-entry FP32 inner product (when there are
 * few); see mpkParams::sddmm_min_nnz.
 *
 * Distances drop the ||p_i||^2 term:
 *
 *     D(i,j) = ||c_j||^2 - 2 p_i^T c_j
 *
 * which shifts every entry of row i by the same constant and therefore leaves
 * both the row-wise argmin and all six conditions of Theorem 1 unchanged.
 * ||p_i||^2 is added back only when reporting inertia.
 */
#ifndef MPKMEANS_H
#define MPKMEANS_H

#include <cublas_v2.h>
#include <cusparse.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MPK_OK = 0,
    MPK_ERR_CUDA = 1,
    MPK_ERR_CUBLAS = 2,
    MPK_ERR_CUSPARSE = 3,
    MPK_ERR_INVALID = 4,
    MPK_ERR_ALLOC = 5
} mpkStatus;

/* Pass this as sddmm_min_nnz to pick the update path automatically.
 *
 * The crossover between cusparseSDDMM and the warp kernel is set by survivors
 * PER ROW, not by the total, and not by density -- measured on an A100:
 *
 *     n        k     crossover nnz   density   per row
 *     200000   256   ~2.0M            3.9%      10.0
 *     200000    64   ~2.5M           19.5%      12.5
 *     400000   256   ~3.9M            3.8%       9.7
 *
 * SDDMM tiles a row and reuses the loaded row of P across that row's entries,
 * so the reuse it wins is exactly the entries-per-row.  The crossover also
 * drifts with d (<4.7 per row at d=32, ~10 at d=128, >14.5 at d=512), which
 * MPK_SDDMM_PER_ROW models as 10*sqrt(d/128).
 *
 * Note this regime is far from where the scheme is normally used: with
 * separated clusters the survivor count is ~0.1-0.5 per row, orders of
 * magnitude below the crossover, so the warp kernel is essentially always the
 * right choice.  The threshold only matters for heavily overlapping data. */
#define MPK_SDDMM_AUTO (-1)

typedef struct {
    int   max_iter;       /* default 100                                      */
    float tol;            /* stop when max centroid movement^2 < tol; def. 0   */
    int   use_cond3;      /* default 1                                        */
    int   use_cond6;      /* default 1                                        */
    int   sddmm_min_nnz;  /* survivors needed before cusparseSDDMM is used
                           * instead of the warp kernel.  MPK_SDDMM_AUTO (the
                           * default) derives it from n and d; 0 forces SDDMM,
                           * a huge value forces the warp kernel.              */
    int   verify;         /* per-iteration FP64 oracle check (MPK_STATS only)  */
    int   verbose;        /* per-iteration log to stdout                       */
} mpkParams;

#define MPK_MAX_HIST 256

typedef struct {
    int    iters;
    double inertia;           /* sum_i ||p_i - c_assign(i)||^2, FP64 accumulate */

    /* --- high precision distance accounting (the headline metric) ---------
     * Plain Lloyd's evaluates all n*k entries of D in high precision.  The
     * mixed scheme evaluates n of them (one reference entry per row, what
     * condition (6) needs) plus one per entry that survived
     * the filter.  Everything else is decided from the FP16 GEMM alone.
     * Summed over iterations. */
    long long hp_baseline;    /* n*k per iteration                             */
    long long hp_reference;   /* n    per iteration -- the (6) reference entry  */
    long long hp_update;      /* survivors refined by SDDMM or the fallback    */

    /* --- attribution of the exclusions, over the n*(k-1) tested pairs ------
     * Only filled when the library was built with MPK_STATS.                 */
    long long tested;         /* n*(k-1) per iteration                         */
    long long excl_cond3;     /* condition (3) held                            */
    long long excl_cond6;     /* condition (6) held                            */
    long long excl_both;      /* both held                                     */

    /* --- per-iteration survivor counts, [0, min(iters, MPK_MAX_HIST)) ------ */
    int       n_hist;
    long long hist_survivors[MPK_MAX_HIST];

    /* MPK_STATS + params.verify: FP64 oracle, accumulated over all iterations */
    long long verify_excluded_best; /* filter dropped the true argmin: a real
                                     * violation of Theorem 1; must be 0       */
    long long verify_label_diff;    /* label != FP64 argmin (includes ties)     */
    double    verify_excess;        /* sum of D64(assigned) - D64(best)         */

    int stats_built;          /* 1 if built with MPK_STATS                     */
    int iters_sddmm;          /* iterations that took the cusparseSDDMM path   */
    int iters_fallback;       /* iterations that took the warp fallback        */

    /* Timings.  t_dist_ms is the headline: everything except the centroid
     * update, which is identical in both algorithms and deliberately left
     * unoptimised. */
    double t_total_ms;
    double t_dist_ms;
    double t_prep_ms;         /* C -> FP16, ||c_j||^2                          */
    double t_gemm_lo_ms;      /* FP16 P*C^T (FP32 SGEMM for the reference)     */
    double t_argmin_ms;       /* row argmin, plus the (6) reference entry      */
    double t_filter_ms;       /* conditions (3)/(6) + CSR build                */
    double t_sddmm_setup_ms;  /* SDDMM descriptors + bufferSize + preprocess   */
    double t_hp_update_ms;    /* SDDMM or fallback on the survivors            */
    double t_assign_ms;       /* final argmin                                  */
    double t_update_ms;       /* centroid recomputation (excluded from t_dist) */
} mpkStats;

void mpkParamsInit(mpkParams* p);

/* Relative error bound eps such that
 *     |fl(p^T c) - p^T c| <= eps * (p^T c)      for p, c >= 0
 * for the FP16-operand, FP32-accumulate GEMM (CUBLAS_COMPUTE_32F) with inner
 * dimension d.  Returns a negative value if the model degenerates (eps >= 1),
 * which does not happen for any d reachable by this GEMM. */
double mpkEpsilon(int d);

/* Shift P in place by M = max_ij |P(i,j)| so that every entry is
 * non-negative, as required by conditions (3) and (6).  Distances are
 * translation invariant, so the clustering is unchanged.  Returns M. */
mpkStatus mpkShiftNonNegative(float* dP, int n, int d, float* out_shift);

/* Mixed precision Lloyd's.
 *
 *   dP        n x d, row major, device, already shifted non-negative
 *   dC        k x d, row major, device, in/out (initial centroids -> final)
 *   dAssign   n,     device, out
 *
 * dP is not modified. */
mpkStatus mpkMeansMixed(cublasHandle_t blas, cusparseHandle_t sparse,
                        const float* dP, int n, int d, int k,
                        float* dC, int* dAssign,
                        const mpkParams* params, mpkStats* stats);

/* Reference Lloyd's, everything in FP32 (TF32 disabled). */
mpkStatus mpkMeansFP32(cublasHandle_t blas,
                       const float* dP, int n, int d, int k,
                       float* dC, int* dAssign,
                       const mpkParams* params, mpkStats* stats);

/* ---------------------------------------------------------------- data --- */

/* Isotropic Gaussian blobs.  Centers are drawn uniformly from
 * [-center_box, center_box]^d, points from N(center, std^2 I).
 * hP is n*d floats (row major), hLabels is n ints; either may be NULL. */
void mpkMakeBlobs(int n, int d, int k, float std, float center_box,
                  unsigned seed, float* hP, int* hLabels);

/* Pick k distinct points of dP (uniformly at random) as initial centroids. */
mpkStatus mpkInitRandomPoints(const float* dP, int n, int d, int k,
                              unsigned seed, float* dC);

#ifdef __cplusplus
}
#endif

#endif /* MPKMEANS_H */
