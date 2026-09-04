/* Baseline: cuVS (RAPIDS) k-means, FP32 throughout.
 *
 * The other schemes in this repo are all ours -- the same Lloyd loop, the same
 * centroid update, differing only in how the distance matrix is formed and
 * which entries are trusted.  This one is an outside implementation, and it is
 * here to answer a question none of those can: is the FP32 baseline everything
 * is normalised against actually a competitive FP32 k-means, or is it a
 * strawman?  A speedup over our own mpkMeansFP32 means nothing if a tuned
 * library beats mpkMeansFP32 by more.
 *
 * Making that comparison mean something takes some care, since two k-means
 * implementations can differ in a dozen ways that have nothing to do with
 * arithmetic.  What is held fixed here:
 *
 *   initial centroids  cuvsKMeansInitMethod::Array, so cuVS starts from the
 *                      caller's dC -- the same dC0 every other scheme starts
 *                      from -- rather than running its own k-means++ or random
 *                      draw.  (The bench maps dC0 out of the shifted frame
 *                      first, exactly as it does for the arXiv baseline.)
 *   data               the unshifted P.  Conditions (3)/(6) need P >= 0 and the
 *                      bench shifts for them; cuVS neither needs nor benefits
 *                      from that, and the shift costs it accuracy in the same
 *                      expanded-formula way it costs the arXiv baseline, so it
 *                      gets the frame it is meant to run in.
 *   distance           L2Expanded -- the expanded formula our GEMM path uses,
 *                      not the direct one.  cuVS supports both; picking the
 *                      other would be comparing algorithms, not implementations.
 *   restarts           n_init = 1.  cuVS defaults to that too, but it decides
 *                      the whole comparison, so it is set explicitly.
 *   tiling             batch_samples = n by default, i.e. one untiled pass over
 *                      all n rows, which is what our schemes do.  cuVS ships a
 *                      default of 32768 rows per batch to cap the footprint of
 *                      the n x k distance block; params.cuvs_batch overrides
 *                      this to measure that default instead.
 *   inertia            recomputed here by mpkLaunchInertia, in FP64, from the
 *                      final centroids and labels -- the same function, on the
 *                      same data, that every other scheme's inertia comes from.
 *                      cuVS returns its own FP32 inertia from fit and predict;
 *                      using it would compare two summation orders, not two
 *                      clusterings.
 *
 * What could NOT be held fixed, and matters when reading the numbers:
 *
 *   stopping rule      ours halts when no label changed (and optionally when
 *                      ||C - C_prev||_F < tol).  cuVS halts when
 *                      cost/prior_cost > 1 - tol OR sum_j ||c_j - c_j_prev||^2
 *                      < tol, and rejects tol <= 0, so it cannot be made to run
 *                      a fixed iteration count.  Both rules fire at true
 *                      convergence, so with tol driven to ~0 (the default here)
 *                      the two stop within an iteration or so of each other --
 *                      but not identically.  Compare per iteration, not per run.
 *   label convention   our loop ends on an ASSIGNMENT: after M iterations it
 *                      holds centroids C_M and the labels assign(C_{M-1}), the
 *                      last update having landed after the last assignment.
 *                      cuvsKMeansFit ends on an UPDATE and hands back only
 *                      C_M, so the labels here come from a predict against
 *                      C_M -- self-consistent, and half a step ahead of ours.
 *                      At convergence C_M == C_{M-1} and the two coincide,
 *                      which is the regime the benchmark reports.  On a run
 *                      truncated short of convergence they do not: with -i 1
 *                      this row's clustering matches what fp32 reaches at
 *                      -i 2.  (That is also the cleanest proof that cuVS is
 *                      starting from the shared dC0 rather than its own
 *                      k-means++ -- the two trajectories line up exactly, one
 *                      assignment apart.)
 *   loop boundaries    cuvsKMeansFit is one call: assignment, centroid update,
 *                      convergence test and its own bookkeeping are inside it
 *                      and cannot be separated.  t_total_ms is that call and is
 *                      the only number directly comparable to another
 *                      implementation.  t_dist_ms is a single assignment pass,
 *                      measured separately via cuvsKMeansPredict and multiplied
 *                      by the iteration count, so that the per-iteration
 *                      distance-step column has something in it; that pass also
 *                      computes an inertia our own loop would not, so it reads
 *                      slightly high.
 *   empty clusters     cuVS reinitialises them; our finalize leaves them where
 *                      they are.  With random-point init on separated data
 *                      neither fires, but on a degenerate dataset they diverge.
 */
#include "mpk_internal.cuh"

#include <stdio.h>
#include <string.h>

#ifdef MPK_CUVS
#include <cuvs/core/c_api.h>
#include <cuvs/cluster/kmeans.h>
#include <dlpack/dlpack.h>
#endif

extern "C" int mpkHaveCuvs(void) {
#ifdef MPK_CUVS
    return 1;
#else
    return 0;
#endif
}

#ifndef MPK_CUVS

extern "C" mpkStatus mpkMeansCuvs(cublasHandle_t, const float*, int, int, int,
                                  float*, int*, const mpkParams*,
                                  mpkStats* stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
    return MPK_ERR_INVALID;   /* built without -DMPK_CUVS=ON */
}

#else

/* A DLPack view of memory we already own.  cuVS only reads the descriptor, so
 * there is nothing to manage and no deleter to run. */
static DLManagedTensor mpk_dl(void* p, int ndim, int64_t* shape,
                              DLDataTypeCode code, uint8_t bits) {
    DLManagedTensor t;
    memset(&t, 0, sizeof(t));
    t.dl_tensor.data               = p;
    t.dl_tensor.device.device_type = kDLCUDA;
    t.dl_tensor.device.device_id   = 0;
    t.dl_tensor.ndim               = ndim;
    t.dl_tensor.dtype.code         = (uint8_t)code;
    t.dl_tensor.dtype.bits         = bits;
    t.dl_tensor.dtype.lanes        = 1;
    t.dl_tensor.shape              = shape;
    t.dl_tensor.strides            = NULL;   /* row major, compact */
    t.dl_tensor.byte_offset        = 0;
    return t;
}

#define CUVS_CHK(call)                                                        \
    do {                                                                      \
        if ((call) != CUVS_SUCCESS) {                                         \
            fprintf(stderr, "%s:%d cuvs: %s\n", __FILE__, __LINE__,           \
                    cuvsGetLastErrorText());                                  \
            st = MPK_ERR_CUDA; goto done;                                     \
        }                                                                     \
    } while (0)

extern "C" mpkStatus mpkMeansCuvs(cublasHandle_t blas, const float* dP, int n,
                                  int d, int k, float* dC, int* dAssign,
                                  const mpkParams* params, mpkStats* stats) {
    (void)blas;   /* cuVS carries its own handles; the argument keeps the
                   * driver signature uniform with the others */
    if (n <= 0 || d <= 0 || k <= 0 || k > n) return MPK_ERR_INVALID;
    mpkParams P;
    if (params) P = *params; else mpkParamsInit(&P);
    memset(stats, 0, sizeof(*stats));
#ifdef MPK_STATS
    stats->stats_built = 1;
#endif

    mpkStatus st = MPK_OK;
    cuvsResources_t   res = 0;
    cuvsKMeansParams_t pr = NULL;
    double*     dInertia  = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL, e2 = NULL, e3 = NULL;
    cudaStream_t s = NULL;

    int64_t shX[2], shC[2], shL[1];
    DLManagedTensor X, C, L;
    double fit_inertia = 0.0, pred_inertia = 0.0;
    int    n_iter = 0;
    float  fit_ms = 0.f, pred_ms = 0.f;

    if (cudaMalloc((void**)&dInertia, sizeof(double)) != cudaSuccess)
        return MPK_ERR_ALLOC;

    CUVS_CHK(cuvsResourcesCreate(&res));
    CUVS_CHK(cuvsStreamGet(res, &s));
    CUVS_CHK(cuvsKMeansParamsCreate(&pr));

    pr->n_clusters = k;
    pr->init       = Array;          /* start from the caller's dC */
    pr->metric     = L2Expanded;
    pr->max_iter   = P.max_iter;
    pr->n_init     = 1;
    pr->hierarchical = false;
    /* cuVS rejects tol <= 0, so a fixed iteration count is not available.  In
     * the bench's default mode our loop stops on label stability; the nearest
     * cuVS rule is "the cost stopped falling and the centroids stopped moving",
     * which is what a tol driven to the bottom of the float range gives.  With
     * --convergence the bench sets P.tol on ||C - C_prev||_F, and cuVS's shift
     * clause is a SUM of SQUARED shifts -- the squared Frobenius norm of the
     * same block.  So tol^2 here is not merely comparable to our rule, it is
     * the same test written the other way round. */
    pr->tol = (P.tol > 0.f) ? (double)P.tol * (double)P.tol : 1e-30;
    /* one untiled pass over all n rows, matching our single n x k GEMM.
     * cuvs_batch > 0 asks for cuVS's own tiling instead (its default is 32768). */
    pr->batch_samples = (P.cuvs_batch > 0) ? P.cuvs_batch : n;

    shX[0] = n; shX[1] = d;
    shC[0] = k; shC[1] = d;
    shL[0] = n;
    X = mpk_dl((void*)dP, 2, shX, kDLFloat, 32);
    C = mpk_dl((void*)dC, 2, shC, kDLFloat, 32);
    L = mpk_dl((void*)dAssign, 1, shL, kDLInt, 32);

    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    cudaEventRecord(e0, s);
    CUVS_CHK(cuvsKMeansFit(res, pr, &X, NULL, &C, &fit_inertia, &n_iter));
    cudaEventRecord(e1, s);

    /* labels, and with them one timed assignment pass -- fit does not hand the
     * labels back, so this call has to happen anyway */
    cudaEventRecord(e2, s);
    CUVS_CHK(cuvsKMeansPredict(res, pr, &X, NULL, &C, &L, true, &pred_inertia));
    cudaEventRecord(e3, s);

    CUVS_CHK(cuvsStreamSync(res));
    cudaEventSynchronize(e3);
    cudaEventElapsedTime(&fit_ms, e0, e1);
    cudaEventElapsedTime(&pred_ms, e2, e3);

    stats->iters       = n_iter;
    stats->t_total_ms  = fit_ms;
    stats->t_dist_ms   = (double)pred_ms * (double)(n_iter > 0 ? n_iter : 1);
    /* every pair is evaluated, every one of them in FP32: no exclusion, so the
     * "eliminated" column is 0 by construction rather than by measurement */
    stats->hp_baseline = (long long)n * k * (n_iter > 0 ? n_iter : 1);
    stats->hp_update   = stats->hp_baseline;

    cudaMemset(dInertia, 0, sizeof(double));
    mpkLaunchInertia(dP, dC, dAssign, n, d, dInertia, s);
    cudaMemcpy(&stats->inertia, dInertia, sizeof(double), cudaMemcpyDeviceToHost);

done:
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (e2) cudaEventDestroy(e2);
    if (e3) cudaEventDestroy(e3);
    if (pr)  cuvsKMeansParamsDestroy(pr);
    if (res) cuvsResourcesDestroy(res);
    cudaFree(dInertia);
    return st;
}

#endif  /* MPK_CUVS */
