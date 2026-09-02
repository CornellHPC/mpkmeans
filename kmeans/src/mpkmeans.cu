/* Mixed precision Lloyd's iteration.
 *
 * Per iteration:
 *   1. C -> FP16, ||c_j||^2 in FP32
 *   2. G = fl16(P C^T)                              (cublasGemmEx, FP16 in)
 *   3. one pass over each row of G: its argmin -> jbest, dbest, gbest; for (6)
 *      the high precision reference entry gexact[i] = fl32(<p_i, c_jbest[i]>);
 *      and the conditions (3)/(6) of Theorem 1  -> a survivor count per row
 *   4. scan the counts, then write the CSR survivor pattern
 *   5. the survivors are refined in FP32, by cusparseSDDMM if there are many
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
    p->sddmm_min_nnz = MPK_SDDMM_AUTO;
    p->verify        = 0;
    p->verbose       = 0;
}

/* |fl(p^T c) - p^T c| <= eps * p^T c for non-negative p, c, for the FP16
 * operand / FP32 accumulate GEMM (CUBLAS_COMPUTE_32F).
 *
 * FP16 operands rounded from FP32 contribute (1+u16)^2 per term; the length-d
 * FP32 summation contributes gamma_d = d*u32/(1-d*u32).  Since every term is
 * non-negative the per-term factors combine into a single relative bound on
 * the sum. */
extern "C" double mpkEpsilon(int d) {
    const double g = (double)d * U_FP32;
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
        != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
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

            if (P.tol > 0.f) {
                float* hm = (float*)malloc((size_t)k * sizeof(float));
                cudaMemcpyAsync(hm, moved2, (size_t)k * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                float mx = 0.f;
                for (int j = 0; j < k; ++j) mx = fmaxf(mx, hm[j]);
                free(hm);
                if (mx < P.tol) break;
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
    cudaFreeHost(h_changed);
    return st;
}

/* -------------------------------------------------------------- mixed ---- */

extern "C" mpkStatus mpkMeansMixed(cublasHandle_t blas, cusparseHandle_t sparse,
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
    if (!P.use_cond3 && !P.use_cond6) {
        fprintf(stderr, "mpkMeansMixed: at least one condition must be on\n");
        return MPK_ERR_INVALID;
    }

    /* Condition (6) is the only one that needs a high precision quantity, so
     * it alone pays for it.  With (3) alone the incumbent's own high
     * precision distance is folded into the update step, and only for rows
     * that (3) did not clear outright. */
    const int need_ref     = P.use_cond6 ? 1 : 0;
    const int include_best = need_ref ? 0 : 1;
    /* the cascade only means anything when both conditions are on: with (6)
     * alone there is no free test to put in front of the reference entry */
    const int cascade_on   = (P.cascade && P.use_cond3 && P.use_cond6) ? 1 : 0;

    const double eps = mpkEpsilon(d);
    if (eps < 0.0) {
        fprintf(stderr,
                "mpkMeansMixed: error model degenerate for d=%d (u_l*d >= 1).\n",
                d);
        return MPK_ERR_INVALID;
    }
    const float factor = (float)(2.0 * eps / (1.0 - eps));
    /* The reference entry is a warp-strided fma dot: no product rounding, and
     * the tree reduction is shallower than a sequential sum, so gamma_d is a
     * safe overestimate of its error. */
    const double gs_raw = (double)d * U_FP32;
    const double gs = gs_raw / (1.0 - gs_raw);
    const float gfac = (float)(2.0 * gs / (1.0 - gs));

    /* Resolve the update-path crossover.  It tracks survivors per row rather
     * than the total (SDDMM's win is the reuse of a row of P across that row's
     * entries), and drifts with d; see MPK_SDDMM_AUTO.  The fit is from three
     * (n,k) points and three d points, so it is a coarse guide, but near the
     * crossover the two paths are within ~10% of each other by definition. */
    long long sddmm_min = P.sddmm_min_nnz;
    if (P.sddmm_min_nnz < 0) {
        double per_row = 10.0 * sqrt((double)d / 128.0);
        if (per_row <  4.0) per_row =  4.0;
        if (per_row > 24.0) per_row = 24.0;
        sddmm_min = (long long)(per_row * (double)n);
    }

    cudaStream_t s = nullptr;
    cublasGetStream(blas, &s);
    cusparseSetStream(sparse, s);

    __half *P16 = nullptr, *C16 = nullptr;
    float  *G = nullptr, *cnorm2 = nullptr, *moved2 = nullptr;
    float  *dbest = nullptr, *gbest = nullptr, *gexact = nullptr;
    int    *jbest = nullptr, *row_nnz = nullptr, *row_ptr = nullptr, *prev = nullptr;
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

    void*  scan_tmp = nullptr; size_t scan_cap = 0, scan_bytes = 0;
    void*  col_buf  = nullptr; size_t col_cap = 0;
    void*  row_buf  = nullptr; size_t row_cap = 0;
    void*  val_buf  = nullptr; size_t val_cap = 0;
    void*  sddmm_buf = nullptr; size_t sddmm_cap = 0;

    mpkStatus st = MPK_OK;
#define ALLOC(p, bytes) if (cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
    ALLOC(P16, (size_t)n * d * sizeof(__half));
    ALLOC(C16, (size_t)k * d * sizeof(__half));
    ALLOC(G,   (size_t)n * k * sizeof(float));
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
    ALLOC(jbest,   (size_t)n * sizeof(int));
    ALLOC(row_nnz, (size_t)n * sizeof(int));
    ALLOC(row_ptr, (size_t)(n + 1) * sizeof(int));
    ALLOC(prev,    (size_t)n * sizeof(int));
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
            != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }

    cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, row_nnz, row_ptr + 1, n, s);
    if (grow(&scan_tmp, &scan_cap, scan_bytes) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }

    /* Size the survivor buffers for one entry per row up front.  cudaMalloc
     * costs milliseconds and grow() would otherwise call it from inside a
     * timed iteration -- the first iteration always has the most survivors,
     * so whichever configuration ran first was charged for the allocation. */
    if (grow(&col_buf, &col_cap, (size_t)n * sizeof(int))   != cudaSuccess ||
        grow(&row_buf, &row_cap, (size_t)n * sizeof(int))   != cudaSuccess ||
        grow(&val_buf, &val_cap, (size_t)n * sizeof(float)) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }

    cudaMemset(prev, 0xff, (size_t)n * sizeof(int));
    cudaMemset(dCnt, 0, 3 * sizeof(long long));
    if (dRefCnt) cudaMemset(dRefCnt, 0, sizeof(unsigned long long));
#ifdef MPK_STATS
    cudaMemset(dStatBanks, 0, 3 * MPK_STAT_BANKS * sizeof(long long));
#endif
    if (dExcess) cudaMemset(dExcess, 0, sizeof(double));
    cudaMemset(row_ptr, 0, sizeof(int));

    /* P is fixed: convert once. */
    mpkLaunchToHalf(dP, P16, (long long)n * d, s);

    {
        Timer tt, t0, t1, t3, t4, t5, t6, t7, t8, t9;
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
            const cublasStatus_t bs = cublasGemmEx(
                blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d,
                &f_one, C16, CUDA_R_16F, d, P16, CUDA_R_16F, d,
                &f_zero, G, CUDA_R_32F, k,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (bs != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "cublasGemmEx failed: %d\n", (int)bs);
                st = MPK_ERR_CUBLAS; goto done;
            }
            t1.stop(s, &stats->t_gemm_lo_ms);

            /* ---- 3: row argmin, the (6) reference entry and the exclusion
             * test, all in one pass over the row of G ---------------------- */
            t7.start(s);
            mpkLaunchArgminCount(G, cnorm2, dP, dC, n, d, k, factor, gfac, SLACK,
                                 P.use_cond3, P.use_cond6, cascade_on,
                                 jbest, dbest, gbest,
                                 gexact, row_nnz, include_best, dRefCnt,
                                 dStatBanks, s);
            t7.stop(s, &stats->t_argmin_ms);

            /* ---- 4: scan the counts into row_ptr and write the pattern --- */
            t3.start(s);
            cub::DeviceScan::InclusiveSum(scan_tmp, scan_bytes, row_nnz,
                                          row_ptr + 1, n, s);
            cudaMemcpyAsync(h_nnz, row_ptr + n, sizeof(int),
                            cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            const int nnz = *h_nnz;

            t3.stop(s, &stats->t_filter_ms);

            /* The survivor buffers are resized here, between the two halves of
             * the filter phase and outside both of them.  cudaFree/cudaMalloc
             * of tens of megabytes is milliseconds and it synchronises; timed,
             * it swamps the kernel it is standing next to and does so
             * unevenly across configurations. */
            if (nnz > 0) {
                if (grow(&col_buf, &col_cap, (size_t)nnz * sizeof(int)) != cudaSuccess ||
                    grow(&row_buf, &row_cap, (size_t)nnz * sizeof(int)) != cudaSuccess ||
                    grow(&val_buf, &val_cap, (size_t)nnz * sizeof(float)) != cudaSuccess) {
                    st = MPK_ERR_ALLOC; goto done;
                }
                t8.start(s);
                mpkLaunchConditionFill(G, cnorm2, jbest, dbest, gbest, gexact, n, k,
                                       factor, gfac, SLACK, P.use_cond3,
                                       P.use_cond6, row_ptr, (int*)col_buf,
                                       (int*)row_buf, include_best, s);
                t8.stop(s, &stats->t_filter_ms);
            }

            stats->tested       += (long long)n * (k - 1);
            stats->hp_baseline  += (long long)n * k;
            /* hp_reference accumulates in dRefCnt on the device and is read
             * once after the loop: a per-iteration read would cost a host
             * synchronisation for a number nothing in the loop consumes */
            stats->hp_update    += nnz;
            if (stats->n_hist < MPK_MAX_HIST)
                stats->hist_survivors[stats->n_hist++] = nnz;

            /* ---- 5: high precision refinement of the survivors ----------- */
            if (nnz > 0 && (long long)nnz >= sddmm_min) {
                stats->iters_sddmm++;
                t6.start(s);
                cusparseConstDnMatDescr_t mA = nullptr, mB = nullptr;
                cusparseSpMatDescr_t      mC = nullptr;
                /* A: P, n x d, row major.  B: C^T, d x k -- the k x d row major
                 * buffer dC read as column major with ld = d is exactly C^T. */
                MPK_SPARSE(cusparseCreateConstDnMat(&mA, n, d, d, dP,
                                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));
                MPK_SPARSE(cusparseCreateConstDnMat(&mB, d, k, d, dC,
                                                    CUDA_R_32F, CUSPARSE_ORDER_COL));
                MPK_SPARSE(cusparseCreateCsr(&mC, n, k, nnz, row_ptr, col_buf,
                                             val_buf, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
                size_t need = 0;
                MPK_SPARSE(cusparseSDDMM_bufferSize(
                    sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &f_one, mA, mB, &f_zero, mC,
                    CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &need));
                t6.stop(s, &stats->t_sddmm_setup_ms);
                if (grow(&sddmm_buf, &sddmm_cap, need ? need : 1) != cudaSuccess) {
                    st = MPK_ERR_ALLOC; goto done;
                }
                t9.start(s);
                MPK_SPARSE(cusparseSDDMM_preprocess(
                    sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &f_one, mA, mB, &f_zero, mC,
                    CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, sddmm_buf));
                t9.stop(s, &stats->t_sddmm_setup_ms);
                t4.start(s);
                MPK_SPARSE(cusparseSDDMM(
                    sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, &f_one, mA, mB, &f_zero, mC,
                    CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, sddmm_buf));
                cusparseDestroyDnMat(mA);
                cusparseDestroyDnMat(mB);
                cusparseDestroySpMat(mC);
                t4.stop(s, &stats->t_hp_update_ms);
            } else if (nnz > 0) {
                stats->iters_fallback++;
                t4.start(s);
                mpkLaunchUpdate(dP, dC, (const int*)col_buf, (const int*)row_buf,
                                nnz, d, (float*)val_buf, s);
                t4.stop(s, &stats->t_hp_update_ms);
            }

            /* ---- 6: final assignment ------------------------------------- */
            t5.start(s);
            mpkLaunchFinalAssign(row_ptr, (const int*)col_buf,
                                 (const float*)val_buf, cnorm2, jbest, gexact,
                                 n, dAssign, s);
            t5.stop(s, &stats->t_assign_ms);

#ifdef MPK_STATS
            /* ---- oracle check: the plain FP32 iteration, same centroids ---
             * Untimed, and deliberately the real thing -- cublasSgemm under
             * PEDANTIC math, exactly as mpkMeansFP32 runs it, so `dARef` is
             * the label the reference implementation would have produced from
             * this same dC.  Anything the bounds excluded that the reference
             * then picks is a genuine violation. */
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
                mpkLaunchVerifyRef(G32, cnorm2, row_ptr, (const int*)col_buf,
                                   jbest, dAssign, dARef, n, k,
                                   dCnt + 1, dCnt + 2, dExcess, s);
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

            t0.collect(); t1.collect(); t7.collect();
            t3.collect(); t8.collect(); t6.collect(); t9.collect();
            t4.collect(); t5.collect();

            stats->iters = it + 1;
            if (P.verbose)
                fprintf(stderr,
                        "[mixed] iter %3d  changed %8lld  survivors %9d / %lld"
                        "  (%.4f%%)\n",
                        it, changed, nnz, (long long)n * (k - 1),
                        100.0 * nnz / ((double)n * (k - 1)));
            if (changed == 0) break;

            /* ---- 7: centroid update (excluded from t_dist_ms) ------------ */
            mpkLaunchZero(sums, counts, k, d, s);
            mpkLaunchAccumulate(dP, dAssign, n, d, sums, counts, s);
            mpkLaunchFinalizeCentroids(sums, counts, k, d, dC, moved2, s);

            if (P.tol > 0.f) {
                float* hm = (float*)malloc((size_t)k * sizeof(float));
                cudaMemcpyAsync(hm, moved2, (size_t)k * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                float mx = 0.f;
                for (int j = 0; j < k; ++j) mx = fmaxf(mx, hm[j]);
                free(hm);
                if (mx < P.tol) break;
            }
        }
        tt.stop_sync(s, &stats->t_total_ms);
    }
    stats->t_dist_ms = stats->t_prep_ms + stats->t_gemm_lo_ms +
                       stats->t_argmin_ms + stats->t_filter_ms +
                       stats->t_sddmm_setup_ms + stats->t_hp_update_ms +
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
    cudaFree(P16); cudaFree(C16); cudaFree(G);
    cudaFree(cnorm2); cudaFree(moved2); cudaFree(dbest); cudaFree(gbest);
    cudaFree(gexact); cudaFree(dRefCnt); cudaFree(jbest); cudaFree(row_nnz);
#ifdef MPK_STATS
    cudaFree(G32); cudaFree(dARef);
#endif
    cudaFree(row_ptr); cudaFree(prev); cudaFree(sums); cudaFree(counts);
    cudaFree(dInertia); cudaFree(dCnt); cudaFree(dStatBanks);
    cudaFree(dExcess);
    cudaFreeHost(h_nnz); cudaFreeHost(h_changed);
    cudaFree(scan_tmp); cudaFree(col_buf); cudaFree(row_buf); cudaFree(val_buf);
    cudaFree(sddmm_buf);
    return st;
}
