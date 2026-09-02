/* Mixed precision Lloyd's iteration.
 *
 * Per iteration:
 *   1. C -> FP16, ||c_j||^2 in FP32
 *   2. G = fl16(P C^T)                              (cublasGemmEx, FP16 in)
 *   3. one pass over each row of G: its argmin -> jbest, dbest, gbest; for (6)
 *      the high precision reference entry gexact[i] = fl32(<p_i, c_jbest[i]>);
 *      and the conditions (3)/(6) of Theorem 1  -> a survivor count per row
 *   4. scan the counts, then write the CSR survivor pattern
 *   5. the survivors are refined in FP32, one warp per surviving flat index
 *      of them and by a warp-per-entry inner product if there are few
 *   6. argmin over survivors and jbest             -> assignment
 *   7. centroid recomputation (FP64 accumulation, FP32 result)
 *
 * Step 7 is identical in the mixed and the FP32 reference path and is left
 * unoptimised; it is reported separately and excluded from t_dist_ms.
 */
#include "mpk_internal.cuh"

#include <cub/cub.cuh>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr double U_FP16 = 1.0 / 2048.0;      /* 2^-11 */
constexpr double U_FP32 = 1.0 / 16777216.0;  /* 2^-24 */
constexpr float  SLACK  = 8.0f * (float)U_FP32;

/* Phase timer.  start()/stop() only record events; the elapsed time is read
 * back by collect() after the caller has synchronised for another reason.
 * Synchronising inside stop() left the GPU idle between phases and inflated
 * every phase measurement well above its true kernel time. */
struct Timer {
    cudaEvent_t a, b;
    double* acc = nullptr;
    bool    live = false;
    Timer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { cudaEventRecord(a, s); }
    void stop(cudaStream_t s, double* dst) {
        cudaEventRecord(b, s);
        acc = dst;
        live = true;
    }
    void collect() {
        if (!live) return;
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        *acc += ms;
        live = false;
    }
    /* blocking variant, for the one timer that spans the whole loop */
    void stop_sync(cudaStream_t s, double* dst) {
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        *dst += ms;
    }
};

/* Grow a device buffer to at least `need` bytes. */
cudaError_t grow(void** p, size_t* cap, size_t need) {
    if (*cap >= need) return cudaSuccess;
    if (*p) cudaFree(*p);
    *p = nullptr;
    /* geometric, not proportional: in the dense-survivor regime nnz climbs by
     * more than a quarter per iteration, so a 25% margin reallocated on nearly
     * every one of them */
    size_t want = *cap ? *cap : 256;
    while (want < need) want *= 2;
    cudaError_t e = cudaMalloc(p, want);
    *cap = (e == cudaSuccess) ? want : 0;
    return e;
}

struct MaxAbsOp {
    __host__ __device__ float operator()(float a, float b) const {
        return fmaxf(fabsf(a), fabsf(b));
    }
};

}  /* namespace */

extern "C" void mpkParamsInit(mpkParams* p) {
    p->max_iter      = 100;
    p->tol           = 0.f;
    p->use_cond3     = 1;
    p->use_cond6     = 1;
    p->cascade       = 0;
    p->accum         = MPK_ACCUM_FP32;
    p->rt_theta      = 0.f;
    p->verify        = 0;
    p->verbose       = 0;
}

/* |fl(p^T c) - p^T c| <= eps * p^T c for non-negative p, c, for the FP16
 * operand GEMM accumulating in the precision named by `accum`.
 *
 * FP16 operands rounded from FP32 contribute (1+u16)^2 per term regardless of
 * accum -- operand precision does not change here.  The length-d summation
 * contributes gamma_d = d*u/(1-d*u), where u is the accumulator's own unit
 * roundoff: u32 for CUBLAS_COMPUTE_32F, u16 for CUBLAS_COMPUTE_16F.  Since
 * every term is non-negative the per-term factors combine into a single
 * relative bound on the sum. */
extern "C" double mpkEpsilon(int d, int accum) {
    const double u = (accum == MPK_ACCUM_FP16) ? U_FP16 : U_FP32;
    const double g = (double)d * u;
    if (g >= 1.0) return -1.0;
    const double gamma = g / (1.0 - g);
    const double eps = (1.0 + U_FP16) * (1.0 + U_FP16) * (1.0 + gamma) - 1.0;
    return (eps >= 1.0) ? -1.0 : eps;
}

/* ------------------------------------------------------------ preprocess -- */

namespace {
__global__ void k_add_const(float* p, long long n, float c) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] += c;
}
}

/* Add a constant to every entry of a k x d centroid block.  Used to move the
 * initial centers between the shifted and unshifted frames so that every
 * configuration starts from the same points. */
extern "C" mpkStatus mpkShiftCentroids(float* dC, int k, int d, float delta) {
    const long long total = (long long)k * d;
    long long blocks = (total + 255) / 256;
    int grid = (int)(blocks > 8192 ? 8192 : (blocks < 1 ? 1 : blocks));
    k_add_const<<<grid, 256>>>(dC, total, delta);
    MPK_CUDA(cudaGetLastError());
    return MPK_OK;
}

extern "C" mpkStatus mpkShiftNonNegative(float* dP, int n, int d, float* out_shift) {
    const long long total = (long long)n * d;
    void*  tmp = nullptr;
    size_t bytes = 0;
    float* dmax = nullptr;
    MPK_CUDA(cudaMalloc(&dmax, sizeof(float)));
    cub::DeviceReduce::Reduce(tmp, bytes, dP, dmax, total, MaxAbsOp(), 0.f);
    if (cudaMalloc(&tmp, bytes) != cudaSuccess) { cudaFree(dmax); return MPK_ERR_ALLOC; }
    cub::DeviceReduce::Reduce(tmp, bytes, dP, dmax, total, MaxAbsOp(), 0.f);
    float M = 0.f;
    cudaError_t e = cudaMemcpy(&M, dmax, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tmp);
    cudaFree(dmax);
    if (e != cudaSuccess) return MPK_ERR_CUDA;

    long long blocks = (total + 255) / 256;
    int grid = (int)(blocks > 8192 ? 8192 : (blocks < 1 ? 1 : blocks));
    k_add_const<<<grid, 256>>>(dP, total, M);
    MPK_CUDA(cudaGetLastError());
    if (out_shift) *out_shift = M;
    return MPK_OK;
}

/* --------------------------------------------------------------- FP32 ---- */

extern "C" mpkStatus mpkMeansFP32(cublasHandle_t blas, const float* dP, int n,
                                  int d, int k, float* dC, int* dAssign,
                                  const mpkParams* params, mpkStats* stats) {
    if (n <= 0 || d <= 0 || k <= 0 || k > n) return MPK_ERR_INVALID;
    mpkParams P;
    if (params) P = *params; else mpkParamsInit(&P);
    memset(stats, 0, sizeof(*stats));
#ifdef MPK_STATS
    stats->stats_built = 1;
#endif

    cudaStream_t s = nullptr;
    cublasGetStream(blas, &s);
    cublasMath_t saved;
    cublasGetMathMode(blas, &saved);
    MPK_BLAS(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));

    float  *G = nullptr, *cnorm2 = nullptr, *moved2 = nullptr;
    double *sums = nullptr;
    int    *counts = nullptr, *prev = nullptr;
    double *dInertia = nullptr;
    long long* dDiff = nullptr;
    long long* h_changed = nullptr;
    float* h_moved = nullptr;
    mpkStatus st = MPK_OK;
#define ALLOC(p, bytes) if (cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
    ALLOC(G, (size_t)n * k * sizeof(float));
    ALLOC(cnorm2, (size_t)k * sizeof(float));
    ALLOC(moved2, (size_t)k * sizeof(float));
    ALLOC(sums, (size_t)k * d * sizeof(double));
    ALLOC(counts, (size_t)k * sizeof(int));
    ALLOC(prev, (size_t)n * sizeof(int));
    ALLOC(dInertia, sizeof(double));
    ALLOC(dDiff, sizeof(long long));
#undef ALLOC
    if (cudaHostAlloc((void**)&h_changed, sizeof(long long), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_moved, (size_t)k * sizeof(float),
                      cudaHostAllocDefault) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    cudaMemset(prev, 0xff, (size_t)n * sizeof(int));

    {
        Timer tt, t0, t1, t4, t5;
        tt.start(s);
        const float one = 1.f, zero = 0.f;
        for (int it = 0; it < P.max_iter; ++it) {
            t0.start(s);
            mpkLaunchRowNorms(dC, k, d, cnorm2, s);
            t0.stop(s, &stats->t_prep_ms);
            t1.start(s);
            /* G_cm(k x n) = C_cm^T * P_cm  ==  G row major (n x k) = P C^T */
            if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, &one,
                            dC, d, dP, d, &zero, G, k) != CUBLAS_STATUS_SUCCESS) {
                st = MPK_ERR_CUBLAS; goto done;
            }
            t1.stop(s, &stats->t_gemm_lo_ms);
            t4.start(s);
            mpkLaunchRowArgmin32(G, cnorm2, n, k, dAssign, s);
            t4.stop(s, &stats->t_assign_ms);
            stats->hp_baseline += (long long)n * k;
            cudaMemsetAsync(dDiff, 0, sizeof(long long), s);
            mpkLaunchCountDiff(dAssign, prev, n, dDiff, s);
            cudaMemcpyAsync(h_changed, dDiff, sizeof(long long),
                            cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            const long long changed = *h_changed;
            cudaMemcpyAsync(prev, dAssign, (size_t)n * sizeof(int),
                            cudaMemcpyDeviceToDevice, s);

            t0.collect(); t1.collect(); t4.collect(); t5.collect();

            stats->iters = it + 1;
            if (changed == 0) break;

            t5.start(s);
            mpkLaunchZero(sums, counts, k, d, s);
            mpkLaunchAccumulate(dP, dAssign, n, d, sums, counts, s);
            mpkLaunchFinalizeCentroids(sums, counts, k, d, dC, moved2, s);
            t5.stop(s, &stats->t_update_ms);

            /* convergence on the centroids: max_j ||c_j - c_j_prev||_2 < tol.
             * k_finalize leaves the squared movements in moved2, so the
             * comparison is done on the max of those and squared once, not
             * rooted k times. */
            if (P.tol > 0.f) {
                cudaMemcpyAsync(h_moved, moved2, (size_t)k * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                float mx = 0.f;
                for (int j = 0; j < k; ++j) mx = fmaxf(mx, h_moved[j]);
                if (mx < P.tol * P.tol) break;
            }
            if (P.verbose)
                fprintf(stderr, "[fp32] iter %3d  changed %lld\n", it, changed);
        }
        tt.stop_sync(s, &stats->t_total_ms);
        t5.collect();
    }
    stats->t_dist_ms = stats->t_prep_ms + stats->t_gemm_lo_ms + stats->t_assign_ms;

    cudaMemset(dInertia, 0, sizeof(double));
    mpkLaunchInertia(dP, dC, dAssign, n, d, dInertia, s);
    cudaMemcpy(&stats->inertia, dInertia, sizeof(double), cudaMemcpyDeviceToHost);

done:
    cublasSetMathMode(blas, saved);
    cudaFree(G); cudaFree(cnorm2); cudaFree(moved2); cudaFree(sums);
    cudaFree(counts); cudaFree(prev); cudaFree(dInertia); cudaFree(dDiff);
    cudaFreeHost(h_changed); cudaFreeHost(h_moved);
    return st;
}

/* -------------------------------------------------------------- mixed ---- */

extern "C" mpkStatus mpkMeansMixed(cublasHandle_t blas,
                                   const float* dP, int n, int d, int k,
                                   float* dC, int* dAssign,
                                   const mpkParams* params, mpkStats* stats) {
    if (n <= 0 || d <= 0 || k <= 0 || k > n) return MPK_ERR_INVALID;
    mpkParams P;
    if (params) P = *params; else mpkParamsInit(&P);
    memset(stats, 0, sizeof(*stats));
#ifdef MPK_STATS
    stats->stats_built = 1;
#endif
    /* use_cond3 == use_cond6 == 0 is not an error: it is the no-refinement
     * mode, which runs no exclusion test and no FP32 recomputation at all --
     * see k_argmin_count's REFINE parameter.  It has no correctness
     * guarantee; it exists for comparison against the schemes that do. */
    const int refine = (P.use_cond3 || P.use_cond6) ? 1 : 0;

    /* Condition (6) is the only one that needs a high precision quantity, so
     * it alone pays for it.  With (3) alone the incumbent's own high
     * precision distance is folded into the update step, and only for rows
     * that (3) did not clear outright. */
    const int need_ref     = (refine && P.use_cond6) ? 1 : 0;
    const int include_best = (refine && !need_ref) ? 1 : 0;
    /* the cascade only means anything when both conditions are on: with (6)
     * alone there is no free test to put in front of the reference entry */
    const int cascade_on   = (refine && P.cascade && P.use_cond3 && P.use_cond6)
                            ? 1 : 0;

    /* factor/gfac are unused in the no-refinement mode (the kernel never
     * reaches the test that reads them), so a degenerate error model there
     * is not fatal -- only a refining run needs a real bound. */
    const double eps = mpkEpsilon(d, P.accum);
    if (refine && eps < 0.0) {
        fprintf(stderr,
                "mpkMeansMixed: error model degenerate for d=%d (u_l*d >= 1).\n",
                d);
        return MPK_ERR_INVALID;
    }
    const float factor = (eps >= 0.0) ? (float)(2.0 * eps / (1.0 - eps)) : 0.f;
    /* The reference entry is a warp-strided fma dot: no product rounding, and
     * the tree reduction is shallower than a sequential sum, so gamma_d is a
     * safe overestimate of its error.  Always FP32: the reference precision
     * does not follow P.accum, which only names the GEMM's accumulator. */
    const double gs_raw = (double)d * U_FP32;
    const double gs = gs_raw / (1.0 - gs_raw);
    const float gfac = (float)(2.0 * gs / (1.0 - gs));


    cudaStream_t s = nullptr;
    cublasGetStream(blas, &s);

    __half *P16 = nullptr, *C16 = nullptr;
    /* G lives in exactly one of these, chosen by P.accum: float for a
     * CUBLAS_COMPUTE_32F GEMM, __half for CUBLAS_COMPUTE_16F -- half the
     * distance matrix, which is the entire point of the FP16 accumulator.
     * mpkLaunchArgminCount/mpkLaunchVerifyRef are templated on G's type and
     * read it at its native width; nothing here widens it back to float. */
    float  *Gf = nullptr;
    __half *Gh = nullptr;
    float  *cnorm2 = nullptr, *moved2 = nullptr;
    float  *dbest = nullptr, *gbest = nullptr, *gexact = nullptr;
    int    *jbest = nullptr, *prev = nullptr;
    unsigned long long *bestpack = nullptr;  /* per row (distance, index) key */
    unsigned int       *dNnz = nullptr;      /* survivor list length          */
    unsigned long long *dRefCnt = nullptr;  /* rows that evaluated a (6)
                                            * reference entry, over the run */
#ifdef MPK_STATS
    /* The oracle is the ordinary FP32 algorithm: a full cublasSgemm on the
     * same centroids, then the same row argmin the FP32 driver uses.  Nothing
     * higher precision is involved -- the question is whether the exclusion
     * machinery reaches the decision the unfiltered FP32 code would have. */
    float* G32   = nullptr;
    int*   dARef = nullptr;
#endif
    double *sums = nullptr, *dInertia = nullptr;
    int    *counts = nullptr;
    /* dCnt[0] changed, [1] verify: true argmin excluded, [2] verify: label
     * != FP64 argmin */
    long long *dCnt = nullptr;
    long long *dStatBanks = nullptr;
    double *dExcess = nullptr;         /* verify only */

    /* Pinned staging for the two per-iteration device-to-host reads.  A
     * pageable destination turns cudaMemcpyAsync into a synchronous staged
     * copy, which serialises the host against the whole stream. */
    int*       h_nnz = nullptr;
    long long* h_changed = nullptr;
    float* h_moved = nullptr;

    /* the one survivor buffer: a flat list of i*k + j */
    void*  list_buf = nullptr; size_t list_cap = 0, list_cap_entries = 0;
    double t_argmin_keep = 0.0;

    mpkStatus st = MPK_OK;
#define ALLOC(p, bytes) if (cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
    ALLOC(P16, (size_t)n * d * sizeof(__half));
    ALLOC(C16, (size_t)k * d * sizeof(__half));
    if (P.accum == MPK_ACCUM_FP16) { ALLOC(Gh, (size_t)n * k * sizeof(__half)); }
    else                           { ALLOC(Gf, (size_t)n * k * sizeof(float)); }
    ALLOC(cnorm2, (size_t)k * sizeof(float));
    ALLOC(moved2, (size_t)k * sizeof(float));
    ALLOC(dbest,  (size_t)n * sizeof(float));
    ALLOC(gbest,  (size_t)n * sizeof(float));
    if (need_ref) ALLOC(gexact, (size_t)n * sizeof(float));
    if (cascade_on) ALLOC(dRefCnt, sizeof(unsigned long long));
#ifdef MPK_STATS
    if (P.verify) {
        ALLOC(G32,   (size_t)n * k * sizeof(float));
        ALLOC(dARef, (size_t)n * sizeof(int));
    }
#endif
    ALLOC(jbest,    (size_t)n * sizeof(int));
    ALLOC(bestpack, (size_t)n * sizeof(unsigned long long));
    ALLOC(dNnz,     sizeof(unsigned int));
    ALLOC(prev,     (size_t)n * sizeof(int));
    ALLOC(sums,   (size_t)k * d * sizeof(double));
    ALLOC(counts, (size_t)k * sizeof(int));
    ALLOC(dInertia, sizeof(double));
    ALLOC(dCnt, 3 * sizeof(long long));
#ifdef MPK_STATS
    ALLOC(dStatBanks, 3 * MPK_STAT_BANKS * sizeof(long long));
    if (P.verify) ALLOC(dExcess, sizeof(double));
#endif
#undef ALLOC

    if (cudaHostAlloc((void**)&h_nnz, sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**)&h_changed, sizeof(long long), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_moved, (size_t)k * sizeof(float),
                      cudaHostAllocDefault) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }

    /* Size the survivor list for two entries per row up front.  cudaMalloc
     * costs milliseconds, so growing it from inside a timed iteration would
     * charge whichever configuration ran first for the allocation. */
    if (grow(&list_buf, &list_cap, (size_t)2 * n * sizeof(int)) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    list_cap_entries = list_cap / sizeof(int);

    cudaMemset(prev, 0xff, (size_t)n * sizeof(int));
    cudaMemset(dCnt, 0, 3 * sizeof(long long));
    if (dRefCnt) cudaMemset(dRefCnt, 0, sizeof(unsigned long long));
#ifdef MPK_STATS
    cudaMemset(dStatBanks, 0, 3 * MPK_STAT_BANKS * sizeof(long long));
#endif
    if (dExcess) cudaMemset(dExcess, 0, sizeof(double));

    /* P is fixed: convert once. */
    mpkLaunchToHalf(dP, P16, (long long)n * d, s);

    {
        Timer tt, t0, t1, t4, t5, t6, t7;
        tt.start(s);
        cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
        const float f_one = 1.f, f_zero = 0.f;

        for (int it = 0; it < P.max_iter; ++it) {
            /* ---- 1/2: low precision GEMM --------------------------------- */
            t0.start(s);
            mpkLaunchToHalf(dC, C16, (long long)k * d, s);
            mpkLaunchRowNorms(dC, k, d, cnorm2, s);
            t0.stop(s, &stats->t_prep_ms);

            t1.start(s);
            cublasStatus_t bs;
            if (P.accum == MPK_ACCUM_FP16) {
                const __half h_one = __float2half(1.f), h_zero = __float2half(0.f);
                bs = cublasGemmEx(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d,
                    &h_one, C16, CUDA_R_16F, d, P16, CUDA_R_16F, d,
                    &h_zero, Gh, CUDA_R_16F, k,
                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
            } else {
                bs = cublasGemmEx(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d,
                    &f_one, C16, CUDA_R_16F, d, P16, CUDA_R_16F, d,
                    &f_zero, Gf, CUDA_R_32F, k,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            }
            if (bs != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "cublasGemmEx failed: %d\n", (int)bs);
                st = MPK_ERR_CUBLAS; goto done;
            }
            t1.stop(s, &stats->t_gemm_lo_ms);

            /* ---- 3/4: row argmin, the (6) reference entry, the exclusion
             * test and the survivor list, all in one pass over the row ------
             * The survivors go into one flat append-only list of i*k + j.
             * There is no CSR and no scan; the count comes back from a device
             * counter, and if the list did not fit the buffer is grown and the
             * pass repeated, which happens once or twice per run at most. */
            /* GT deduced from whichever of Gf/Gh is live this run, so the
             * two accumulate precisions share this call site. */
            auto launch_argmin = [&](auto* Gptr) {
                mpkLaunchArgminCount(Gptr, cnorm2, dP, dC, n, d, k, factor, gfac,
                                     SLACK, P.use_cond3, P.use_cond6, cascade_on,
                                     refine, jbest, dbest, gbest, gexact, bestpack,
                                     include_best, (int*)list_buf,
                                     (int)list_cap_entries, dNnz, dRefCnt,
                                     dStatBanks, s);
            };
            int nnz = 0;
            for (;;) {
                cudaMemsetAsync(dNnz, 0, sizeof(unsigned int), s);
                t7.start(s);
                if (P.accum == MPK_ACCUM_FP16) launch_argmin(Gh);
                else                           launch_argmin(Gf);
                t7.stop(s, &stats->t_argmin_ms);
                cudaMemcpyAsync(h_nnz, dNnz, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                nnz = (int)*(unsigned int*)h_nnz;
                if ((size_t)nnz <= list_cap_entries) break;
                /* the overflowing pass is discarded, and not charged for */
                t7.collect();
                stats->t_argmin_ms = t_argmin_keep;
                if (dRefCnt) cudaMemset(dRefCnt, 0, sizeof(unsigned long long));
#ifdef MPK_STATS
                cudaMemset(dStatBanks, 0, 3 * MPK_STAT_BANKS * sizeof(long long));
#endif
                if (grow(&list_buf, &list_cap, (size_t)nnz * sizeof(int))
                    != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
                list_cap_entries = list_cap / sizeof(int);
            }
            t7.collect();
            t_argmin_keep = stats->t_argmin_ms;

            /* ---- 5: the high precision entries --------------------------- */
            if (nnz > 0) {
                t4.start(s);
                mpkLaunchUpdateFlat(dP, dC, cnorm2, (const int*)list_buf, nnz,
                                    k, d, bestpack, s);
                t4.stop(s, &stats->t_hp_update_ms);
            }

            /* ---- 6: the label is the low half of the packed best --------- */
            t5.start(s);
            mpkLaunchUnpack(bestpack, n, dAssign, s);
            t5.stop(s, &stats->t_assign_ms);

            stats->tested       += (long long)n * (k - 1);
            stats->hp_baseline  += (long long)n * k;
            stats->hp_update    += nnz;

#ifdef MPK_STATS
            /* ---- oracle check: the plain FP32 iteration, same centroids ---
             * Untimed, and deliberately the real thing -- cublasSgemm under
             * PEDANTIC math, exactly as mpkMeansFP32 runs it, so `dARef` is
             * the label the reference implementation would have produced from
             * this same dC. */
            if (P.verify) {
                cublasMath_t vm;
                cublasGetMathMode(blas, &vm);
                cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
                const float v_one = 1.f, v_zero = 0.f;
                if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, &v_one,
                                dC, d, dP, d, &v_zero, G32, k)
                    != CUBLAS_STATUS_SUCCESS) { st = MPK_ERR_CUBLAS; goto done; }
                cublasSetMathMode(blas, vm);
                mpkLaunchRowArgmin32(G32, cnorm2, n, k, dARef, s);
                /* refine == 0 ran no exclusion test, so "reachable" there
                 * means the trivial (false, false) predicate, not condition
                 * (6) alone -- gating both flags on refine keeps that. */
                auto launch_verify = [&](auto* Gptr) {
                    mpkLaunchVerifyRef(Gptr, G32, cnorm2, jbest, dbest, gbest,
                                       gexact, dAssign, dARef, n, k, factor, gfac,
                                       SLACK, P.use_cond3 && refine,
                                       P.use_cond6 && refine,
                                       dCnt + 1, dCnt + 2, dExcess, s);
                };
                if (P.accum == MPK_ACCUM_FP16) launch_verify(Gh);
                else                           launch_verify(Gf);
            }
#endif

            /* ---- convergence --------------------------------------------- */
            cudaMemsetAsync(dCnt, 0, sizeof(long long), s);
            mpkLaunchCountDiff(dAssign, prev, n, dCnt, s);
            cudaMemcpyAsync(h_changed, dCnt, sizeof(long long),
                            cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            const long long changed = *h_changed;
            cudaMemcpyAsync(prev, dAssign, (size_t)n * sizeof(int),
                            cudaMemcpyDeviceToDevice, s);

            t0.collect(); t1.collect(); t4.collect(); t5.collect();

            stats->iters = it + 1;
            if (P.verbose)
                fprintf(stderr,
                        "[mixed] iter %3d  changed %8lld  survivors %9d / %lld"
                        "  (%.4f%%)\n",
                        it, changed, nnz, (long long)n * (k - 1),
                        100.0 * nnz / ((double)n * (k - 1)));
            if (changed == 0) break;

            /* ---- 7: centroid update (excluded from t_dist_ms) ------------
             * Identical code in every scheme, and timed in every scheme, so
             * that the column reads as the constant it is rather than as a
             * blank next to the FP32 run's number. */
            t6.start(s);
            mpkLaunchZero(sums, counts, k, d, s);
            mpkLaunchAccumulate(dP, dAssign, n, d, sums, counts, s);
            mpkLaunchFinalizeCentroids(sums, counts, k, d, dC, moved2, s);
            /* stop_sync, not stop+collect: the update is the last thing in the
             * iteration, so nothing downstream would have forced the event to
             * complete and cudaEventElapsedTime would read back 0. */
            t6.stop_sync(s, &stats->t_update_ms);

            /* convergence on the centroids: max_j ||c_j - c_j_prev||_2 < tol.
             * k_finalize leaves the squared movements in moved2, so the
             * comparison is done on the max of those and squared once, not
             * rooted k times. */
            if (P.tol > 0.f) {
                cudaMemcpyAsync(h_moved, moved2, (size_t)k * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                float mx = 0.f;
                for (int j = 0; j < k; ++j) mx = fmaxf(mx, h_moved[j]);
                if (mx < P.tol * P.tol) break;
            }
        }
        tt.stop_sync(s, &stats->t_total_ms);
    }
    stats->t_dist_ms = stats->t_prep_ms + stats->t_gemm_lo_ms +
                       stats->t_argmin_ms + stats->t_hp_update_ms +
                       stats->t_assign_ms;

    /* the reference entries actually evaluated, accumulated on the device over
     * the whole run so the loop never has to synchronise for it */
    if (need_ref) {
        if (cascade_on) {
            unsigned long long hr = 0;
            cudaMemcpy(&hr, dRefCnt, sizeof(hr), cudaMemcpyDeviceToHost);
            stats->hp_reference = (long long)hr;
        } else {
            stats->hp_reference = (long long)n * stats->iters;
        }
    }

#ifdef MPK_STATS
    {
        long long h[3];
        cudaMemcpy(h, dCnt, 3 * sizeof(long long), cudaMemcpyDeviceToHost);
        long long banks[3 * MPK_STAT_BANKS];
        cudaMemcpy(banks, dStatBanks, sizeof(banks), cudaMemcpyDeviceToHost);
        for (int b = 0; b < MPK_STAT_BANKS; ++b) {
            stats->excl_cond3 += banks[b];
            stats->excl_cond6 += banks[MPK_STAT_BANKS + b];
            stats->excl_both  += banks[2 * MPK_STAT_BANKS + b];
        }
        stats->verify_excluded_best = h[1];
        stats->verify_label_diff    = h[2];
        if (dExcess)
            cudaMemcpy(&stats->verify_excess, dExcess, sizeof(double),
                       cudaMemcpyDeviceToHost);
    }
#endif
    cudaMemset(dInertia, 0, sizeof(double));
    mpkLaunchInertia(dP, dC, dAssign, n, d, dInertia, s);
    cudaMemcpy(&stats->inertia, dInertia, sizeof(double), cudaMemcpyDeviceToHost);

done:
    cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
    cudaFree(P16); cudaFree(C16); cudaFree(Gf); cudaFree(Gh);
    cudaFree(cnorm2); cudaFree(moved2); cudaFree(dbest); cudaFree(gbest);
    cudaFree(gexact); cudaFree(dRefCnt); cudaFree(jbest);
    cudaFree(bestpack); cudaFree(dNnz);
#ifdef MPK_STATS
    cudaFree(G32); cudaFree(dARef);
#endif
    cudaFree(prev); cudaFree(sums); cudaFree(counts);
    cudaFree(dInertia); cudaFree(dCnt); cudaFree(dStatBanks);
    cudaFree(dExcess);
    cudaFreeHost(h_nnz); cudaFreeHost(h_changed); cudaFreeHost(h_moved);
    cudaFree(list_buf);
    return st;
}
