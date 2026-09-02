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
 * The surviving entries are enumerated as a flat list of i*k + j indices and
 * refined one warp per entry in FP32.  There is no CSR and no prefix scan.
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

typedef struct {
    int   max_iter;       /* default 100                                      */
    float tol;            /* stop when max centroid movement^2 < tol; def. 0   */
    int   use_cond3;      /* default 1                                        */
    int   use_cond6;      /* default 1                                        */
    int   cascade;        /* with both conditions on, apply (3) first and only
                           * compute the (6) reference entry for the rows (3)
                           * did not clear outright.  (3) needs no high
                           * precision quantity, so a row it clears never reads
                           * P -- and that FP32 read of P is what condition (6)
                           * actually costs.  Exclusions are identical to the
                           * unconditional (3)+(6); only the work differs.
                           * Ignored unless both conditions are on.  default 0 */
    int   verify;         /* per-iteration FP64 oracle check (MPK_STATS only)  */
    int   verbose;        /* per-iteration log to stdout                       */
} mpkParams;

#define MPK_MAX_HIST 256

typedef struct {
    int    iters;
    double inertia;           /* sum_i ||p_i - c_assign(i)||^2, FP64 accumulate */

    /* --- high precision distance accounting (the headline metric) ---------
     * Plain Lloyd's evaluates all n*k entries of D in high precision.  The
     * mixed scheme evaluates one reference entry per row that needs one (what
     * condition (6) needs before it can exclude anything: every row, or under
     * the cascade only the rows (3) did not clear) plus one per entry that
     * survived the filter.  Everything else is decided from the FP16 GEMM
     * alone.  Summed over iterations. */
    long long hp_baseline;    /* n*k per iteration                             */
    long long hp_reference;   /* reference entries actually evaluated: n per
                               * iteration for (6), fewer under the cascade    */
    long long hp_update;      /* survivors refined in FP32                     */

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

    /* Timings.  t_dist_ms is the headline: everything except the centroid
     * update, which is identical in both algorithms and deliberately left
     * unoptimised. */
    double t_total_ms;
    double t_dist_ms;
    double t_prep_ms;         /* C -> FP16, ||c_j||^2                          */
    double t_gemm_lo_ms;      /* FP16 P*C^T (FP32 SGEMM for the reference)     */
    double t_argmin_ms;       /* row argmin, plus the (6) reference entry      */
    double t_hp_update_ms;    /* FP32 inner products on the survivors          */
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
mpkStatus mpkMeansMixed(cublasHandle_t blas,
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
