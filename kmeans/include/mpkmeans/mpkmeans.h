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

/* Accumulate precision of the low precision GEMM (mpkMeansMixed only; every
 * other driver is fixed FP32).  Operands are FP16 either way -- this is only
 * the accumulator cuBLAS uses while summing the d products of a row.
 *
 *   MPK_ACCUM_FP32  CUBLAS_COMPUTE_32F, G stored as float.  The default.
 *   MPK_ACCUM_FP16  CUBLAS_COMPUTE_16F, G stored as __half -- half the
 *                   distance matrix, at a worse error bound (see mpkEpsilon):
 *                   the d-term sum now accumulates in FP16, not FP32. */
typedef enum {
    MPK_ACCUM_FP32 = 0,
    MPK_ACCUM_FP16 = 1
} mpkAccum;

typedef struct {
    int   max_iter;       /* default 100                                      */
    float tol;            /* convergence tolerance on the centroids: stop once
                           * max_j ||c_j - c_j_prev||_2 < tol.  0 (the default)
                           * disables it, leaving label stability -- no point
                           * changed cluster -- as the only stopping rule.
                           * Both rules are checked; whichever fires first
                           * stops the run.                                    */
    int   use_cond3;      /* default 1                                        */
    int   use_cond6;      /* default 1.  With use_cond3 also 0, mpkMeansMixed
                           * runs no exclusion test and no FP32 refinement at
                           * all: the low precision argmin is trusted outright.
                           * This has no correctness guarantee -- it is the
                           * naive scheme Theorem 1 exists to make safe -- and
                           * is meant for comparison, not for a run that needs
                           * a right answer.                                   */
    int   cascade;        /* with both conditions on, apply (3) first and only
                           * compute the (6) reference entry for the rows (3)
                           * did not clear outright.  (3) needs no high
                           * precision quantity, so a row it clears never reads
                           * P -- and that FP32 read of P is what condition (6)
                           * actually costs.  Exclusions are identical to the
                           * unconditional (3)+(6); only the work differs.
                           * Ignored unless both conditions are on.  default 0 */
    int   accum;          /* mpkAccum, mpkMeansMixed only.  default
                           * MPK_ACCUM_FP32                                    */
    int   mw_macs;        /* multiword only (mpkMeansMultiword): how many of
                           * the three cross products the refinement cascade
                           * may reach.  1 never refines (plain fp16), 3 is the
                           * paper's double-fp16 cut.  A stage is only paid for
                           * if the previous exclusion test left something
                           * unsettled, so this is a ceiling, not a count.
                           * default 3                                         */
    float mw_gather_frac; /* multiword only: refine narrowly -- gather the rows
                           * that still hold a survivor and run the remaining
                           * cross products only that wide -- while at most
                           * this fraction of the rows is undecided.  Above it,
                           * refine the whole matrix densely instead, since the
                           * gather stops paying for itself.  <= 0 (the default)
                           * picks the crossover from k, which is what it
                           * actually tracks: measured ~0.20 at k=32, ~0.33 at
                           * k=128, ~0.70 at k=256                              */
    int   cuvs_batch;     /* cuVS baseline only (mpkMeansCuvs): rows per 1NN
                           * tile.  0 (the default) means n, one untiled pass,
                           * which is the shape our own schemes run in; cuVS's
                           * own default is 32768, and passing that measures
                           * the library as it ships instead.               */
    float rt_theta;       /* baseline only (mpkMeansBaselineRT): the safety
                           * factor of arXiv:2407.12208 (4.13).  Must be > 2.
                           * <= 0 selects the default, 2.5, which reproduces
                           * their Algorithm 4.1's literal threshold at its
                           * stated rho = 5 -- the paper states that constant
                           * two inconsistent ways, and the reasoning for the
                           * choice is in the src/baseline.cu file comment.    */
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
    long long hp_update;      /* survivors refined in FP32.  For the multiword
                               * driver the refinement is a dense GEMM, so this
                               * counts n*k for every iteration that needed a
                               * second cross product -- the honest number of
                               * entries recomputed, not the survivor count    */

    /* --- multiword only (mpkMeansMultiword) -------------------------------
     * The headline cost here is cross products issued, not entries evaluated:
     * every stage is a whole tensor-core GEMM, and the exclusion test decides
     * whether the next one is needed at all.  mw_products / iters is the
     * number to compare -- 1.0 means the bound settled every row at plain
     * fp16 cost, 3.0 means it always needed full double-fp16. */
    long long mw_products;    /* cross products issued, summed over the run    */
    long long mw_refine_rows; /* rows the refining products were actually run
                               * over: n per iteration when the refinement went
                               * dense, the undecided count when it went narrow.
                               * / (iters*n) is the effective refine width      */

    /* --- attribution of the exclusions, over the tested pairs --------------
     * Only filled when the library was built with MPK_STATS.                 */
    long long tested;         /* pairs the scheme's test was applied to, per
                               * iteration.  n*(k-1) for the exclusion schemes,
                               * whose incumbent is exempt from being tested
                               * against itself; n*k for mpkMeansBaselineRT,
                               * whose per-entry reliability test has no
                               * incumbent and no exemption.                   */
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
 * for the FP16-operand GEMM with inner dimension d, accumulating in the
 * precision named by `accum` (an mpkAccum).  Returns a negative value if the
 * model degenerates (eps >= 1) -- for MPK_ACCUM_FP32 this does not happen for
 * any d reachable by this GEMM; for MPK_ACCUM_FP16 it can, at far smaller d,
 * since the d-term sum is accumulating in FP16 rather than FP32. */
double mpkEpsilon(int d, int accum);

/* Shift P in place by M = max_ij |P(i,j)| so that every entry is
 * non-negative, as required by conditions (3) and (6).  Distances are
 * translation invariant, so the clustering is unchanged.  Returns M. */
mpkStatus mpkShiftNonNegative(float* dP, int n, int d, float* out_shift);

/* Z-score normalize P in place: each of the d features is centred on its own
 * mean over the n points and divided by its own standard deviation, so that
 * every feature contributes on the same scale.  This is the preprocessing step
 * of arXiv:2407.12208 (Algorithm 5.1 line 1, and Section 6).
 *
 * Unlike mpkShiftNonNegative this is NOT distance preserving -- it rescales the
 * space, so inertia is on a different footing before and after.  It is offered
 * as a benchmark preprocessing step rather than being folded into any of the
 * mpkMeans* routines, so that every scheme provably sees the same input.
 *
 * A feature with zero (or numerically negligible) variance is left mean centred
 * rather than divided by zero, which is the common case for the all-zero
 * columns of a sparse LIBSVM dataset. */
mpkStatus mpkStandardize(float* dP, int n, int d);

/* Add `delta` to every entry of a k x d centroid block. */
mpkStatus mpkShiftCentroids(float* dC, int k, int d, float delta);

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
/* Baseline from arXiv:2407.12208, "Computing k-means in mixed precision":
 * a per-entry reliability test on the expanded distance formula, with the
 * failures recomputed by the direct formula in FP32.  Implemented separately
 * in src/baseline.cu.  hp_update counts the corrected entries. */
mpkStatus mpkMeansBaselineRT(cublasHandle_t blas, const float* dP, int n, int d,
                             int k, float* dC, int* dAssign,
                             const mpkParams* params, mpkStats* stats);

/* Whether the cuVS baseline was compiled in (-DMPK_CUVS=ON and a cuVS install
 * found).  mpkMeansCuvs returns MPK_ERR_INVALID when this is 0. */
int mpkHaveCuvs(void);

/* Baseline: cuVS (RAPIDS) k-means, FP32 throughout, for scale -- it says
 * whether mpkMeansFP32, which everything here is normalised against, is a fair
 * FP32 reference or a slow one.
 *
 * Same argument shape as the other drivers, and deliberately fed the same way:
 * dC must hold the shared initial centroids on entry (cuVS is told to use them
 * rather than running its own k-means++), and dP should be the UNSHIFTED
 * points, since cuVS neither needs nor benefits from the (3)/(6) shift.
 * `blas` is unused -- cuVS carries its own handles.
 *
 * stats->t_total_ms is the cuvsKMeansFit call and is the only timing directly
 * comparable to another implementation; stats->t_dist_ms is one assignment pass
 * timed separately and scaled by the iteration count.  The full account of what
 * is and is not held equal is in the src/cuvs_baseline.cu file comment; read it
 * before quoting any number from this. */
mpkStatus mpkMeansCuvs(cublasHandle_t blas, const float* dP, int n, int d,
                       int k, float* dC, int* dAssign,
                       const mpkParams* params, mpkStats* stats);

mpkStatus mpkMeansFP32(cublasHandle_t blas,
                       const float* dP, int n, int d, int k,
                       float* dC, int* dAssign,
                       const mpkParams* params, mpkStats* stats);

/* Multiword (double-fp16) Lloyd's, implemented in src/multiword.cu.
 *
 * The same conditions (3)/(6) as mpkMeansMixed, over a distance matrix built
 * one fp16 cross product at a time:
 *
 *     stage 1:  G  = C_1^T P_1
 *     stage 2:  G += C_1^T P_2
 *     stage 3:  G += C_2^T P_1     (double-fp16, the paper's p=2 cut)
 *
 * Each stage is one tensor-core GEMM over the whole matrix; the exclusion test
 * runs again after each, and a stage is only issued if the previous test left
 * something unsettled.  So the refinement is another GEMM rather than a
 * warp-per-entry FP32 dot, and the test's job is "do I need the next cross
 * product" rather than "which entries do I recompute".
 *
 * dP must be shifted non-negative, as for mpkMeansMixed -- the bound below is
 * componentwise against |A||B|, which is AB only for non-negative operands.
 * P is split once for the whole run; only C is re-split, per iteration.
 * Requires a build with -DMPK_MULTIWORD=ON; returns MPK_ERR_INVALID otherwise.
 *
 * References
 *   M. Fasi, N. J. Higham, F. Lopez, T. Mary, M. Mikaitis, "Matrix
 *   Multiplication in Multiword Arithmetic: Error Analysis and Application to
 *   GPU Tensor Cores", MIMS EPrint 2022.3 (GAMM 2022). */
mpkStatus mpkMeansMultiword(cublasHandle_t blas,
                            const float* dP, int n, int d, int k,
                            float* dC, int* dAssign,
                            const mpkParams* params, mpkStats* stats);

/* Relative error bound of the multiword product after `macs` of the three
 * cross products, for inner dimension d -- the per-stage analogue of
 * mpkEpsilon, and what mpkMeansMultiword's exclusion test uses at each stage.
 * From the bounds of the GAMM 2022 presentation (slides 7-9):
 *
 *     |E| <= ( (p+1) u_lo^p + gamma^hi_{n+p^2-1} ) |A||B|
 *
 * evaluated at the cut each stage sits on: macs=1 is the p=1 case (and equals
 * mpkEpsilon(d, MPK_ACCUM_FP32)), macs=3 is the p=2 double-fp16 bound, and
 * macs=2 sits between them, still O(u_lo) because one cross product of the
 * pair is still missing.  Negative if the model degenerates. */
double mpkMultiwordEpsilon(int d, int macs);

/* ---------------------------------------------------------------- data --- */

/* Load a LIBSVM/SVMlight sparse text file as a dense row-major n x d matrix.
 * n is the number of non-empty lines and d the largest feature index present;
 * both are outputs, since only the file knows them.  *out_P (n*d floats) and
 * *out_labels (n ints, the label column renumbered 0..nclasses-1 in order of
 * first appearance) are malloc'd and belong to the caller.  out_labels and
 * out_nclasses may be NULL.  The label count is NOT assumed to be k. */
mpkStatus mpkLoadLibsvm(const char* path, int* out_n, int* out_d,
                        float** out_P, int** out_labels, int* out_nclasses);

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
