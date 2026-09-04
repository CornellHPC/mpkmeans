/* Baseline: "Computing k-means in mixed precision" (arXiv:2407.12208).
 *
 * A DIFFERENT strategy from the Theorem 1 exclusion bounds in mpkmeans.cu, and
 * kept in its own file for that reason.  Where our scheme asks "can this
 * centroid be ruled out as the argmin", this one asks, entry by entry, "is the
 * expanded distance formula numerically trustworthy here":
 *
 *     expanded:  D(i,j) = ||p_i||^2 + ||c_j||^2 - 2 p_i^T c_j      (cheap, GEMM)
 *     direct:    D(i,j) = || p_i - c_j ||^2                        (no cancellation)
 *
 * The expanded form is what makes the GEMM usable, but it cancels: when the two
 * points are close, D is a small difference of large numbers.  So each entry is
 * computed in low precision from the GEMM, tested, and only the entries that
 * fail the test are recomputed with the direct formula in FP32.
 *
 * The test is the paper's (4.13), the input-quantization form of the fallback
 * rule (4.7) -- Section 4.3, with the error floor of Corollary 4.3:
 *
 *     E_l(p,c) = 2 * gamma_l(d+2) * ( ||p~||^2 + ||c~||^2 ),
 *     gamma_l(m) = m*u_l / (1 - m*u_l),        u_l = FP16 unit roundoff
 *
 *     accept the expanded value  iff  D_lo(i,j) > theta * E_l(p,c),
 *     otherwise recompute by the direct formula in FP32.
 *
 * theta plays the role of their safety factor (their tilde-rho), required > 2.
 * Its default here is 2.5, and that number needs justifying, because the paper
 * is internally inconsistent about it:
 *
 *   - Section 4.3's analysis defines E^l(x~,y~) := 2 * gamma^l_{r+2} *
 *     (||x~||^2 + ||y~||^2) and states the test as d^l_r,exp > rho~ * E^l,
 *     which is (4.13).  `efac` below is that definition, factor of 2 included.
 *   - Algorithm 4.1, the boxed pseudocode for the mixed precision distance
 *     computation that the k-means loop (Algorithm 5.1) actually calls, writes
 *     the same test at its line 7 as E <- fl_l( rho * gamma^l_{r+2} *
 *     (dxx + dyy) ) -- with no factor of 2.
 *   - The running text of Section 6.1 says "the safety factors in the fallback
 *     rules (4.7) and (4.18) are set to rho = rho~ = 5".
 *
 * So for the same named rho = 5 the two readings differ by exactly a factor of
 * 2 in the threshold.  Read against (4.13), which is what this file implements,
 * the prose would put theta at 5.  We default to 2.5 instead, which reproduces
 * Algorithm 4.1's literal threshold at its own stated rho = 5: the boxed
 * pseudocode is the executable specification, and so the reading another
 * reimplementation of this paper is most likely to have transcribed.  Both
 * readings are defensible; this is a choice, and `--theta` overrides it.
 *
 * By (4.14) theta controls the relative error of what is accepted: entries that
 * pass are good to 1/(theta-1) and to 2/(theta-2) against the unquantized
 * distance.  At the default theta = 2.5 those are 2/3 and 4 -- loose bounds,
 * and much looser than the 1/4 and 2/3 the prose reading's theta = 5 would give.
 * The bounds are worst case; what actually changes is the fallback rate, which
 * is what the benchmark measures.  Note the direction: a smaller theta is a
 * lower threshold, so MORE entries pass the test and FEWER fall back to the
 * direct formula.  This is their FP16->FP32 variant, one of the two they report
 * as the best trade-off.
 *
 * One deliberate deviation: the norms are taken from the FP32 P and C rather
 * than from their FP16 images.  Their (4.12) bounds the difference by a factor
 * (1+u_l)^2, which is far below the safety factor.
 *
 * Unlike the exclusion scheme, this test is per entry and independent of the
 * argmin, so it cannot use "this centroid is not the nearest" to skip work.
 */
#include "mpk_internal.cuh"

#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define RT_WARP 32
#define RT_WPB  8
#define RT_NTHR 256
#define RT_FULL 0xffffffffu

__device__ __forceinline__ float rt_warp_sum(float x) {
    for (int o = RT_WARP / 2; o > 0; o >>= 1) x += __shfl_down_sync(RT_FULL, x, o);
    return x;
}
__device__ __forceinline__ unsigned int rt_ford(float f) {
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ unsigned long long rt_pack(float dist, int j) {
    return ((unsigned long long)rt_ford(dist) << 32) | (unsigned int)j;
}

/* One warp per row.  Take the argmin over the entries the test accepts, and
 * append the entries it rejects to a flat list for correction. */
__global__ void k_rt_scan(const float* __restrict__ G,
                          const float* __restrict__ pnorm2,
                          const float* __restrict__ cnorm2,
                          int n, int k, float thresh,
                          unsigned long long* __restrict__ bestpack,
                          int* __restrict__ list, int cap,
                          unsigned int* __restrict__ count) {
    const int lane = threadIdx.x & (RT_WARP - 1);
    const int step = gridDim.x * RT_WPB;

    for (int i = blockIdx.x * RT_WPB + (threadIdx.x >> 5); i < n; i += step) {
        const float* g  = G + (size_t)i * k;
        const float  pn = pnorm2[i];

        unsigned long long mine = rt_pack(INFINITY, k);
        int cnt = 0;
        for (int c0 = 0; c0 < k; c0 += RT_WARP) {
            const int j = c0 + lane;
            bool bad = false;
            if (j < k) {
                const float s  = pn + cnorm2[j];
                const float dl = fmaf(-2.f, g[j], s);   /* expanded, low precision */
                if (dl > thresh * s) {                  /* (4.13) holds: keep it */
                    const unsigned long long p = rt_pack(dl, j);
                    if (p < mine) mine = p;
                } else {
                    bad = true;                         /* cancellation: correct it */
                }
            }
            cnt += __popc(__ballot_sync(RT_FULL, bad));
        }
        for (int o = RT_WARP / 2; o > 0; o >>= 1) {
            const unsigned long long o2 = __shfl_down_sync(RT_FULL, mine, o);
            if (o2 < mine) mine = o2;
        }
        /* the shuffle must be executed by the whole warp, so broadcast first
         * and only then let lane 0 do the store */
        mine = __shfl_sync(RT_FULL, mine, 0);
        if (lane == 0) bestpack[i] = mine;

        if (!cnt) continue;
        unsigned int pos = 0;
        if (lane == 0) pos = atomicAdd(count, (unsigned int)cnt);
        pos = __shfl_sync(RT_FULL, pos, 0);
        for (int c0 = 0; c0 < k; c0 += RT_WARP) {
            const int j = c0 + lane;
            bool bad = false;
            if (j < k) {
                const float s  = pn + cnorm2[j];
                bad = !(fmaf(-2.f, g[j], s) > thresh * s);
            }
            const unsigned int m = __ballot_sync(RT_FULL, bad);
            if (bad) {
                const unsigned int slot = pos + __popc(m & ((1u << lane) - 1u));
                if (slot < (unsigned int)cap) list[slot] = i * k + j;
            }
            pos += __popc(m);
        }
    }
}

/* One warp per flagged entry: the direct formula in FP32, straight into the
 * row's packed best. */
__global__ void k_rt_correct(const float* __restrict__ P,
                             const float* __restrict__ C,
                             const int* __restrict__ list, int nnz,
                             int k, int d,
                             unsigned long long* __restrict__ bestpack) {
    const int lane = threadIdx.x & (RT_WARP - 1);
    const int step = gridDim.x * RT_WPB;
    for (int e = blockIdx.x * RT_WPB + (threadIdx.x >> 5); e < nnz; e += step) {
        const int idx = list[e];
        const int i   = idx / k;
        const int j   = idx - i * k;
        const float* p = P + (size_t)i * d;
        const float* c = C + (size_t)j * d;
        float acc = 0.f;
        for (int t = lane; t < d; t += RT_WARP) {
            const float v = p[t] - c[t];
            acc = fmaf(v, v, acc);
        }
        const float dd = rt_warp_sum(acc);
        if (lane == 0) atomicMin(&bestpack[i], rt_pack(dd, j));
    }
}

__global__ void k_rt_unpack(const unsigned long long* __restrict__ bestpack,
                            int n, int* __restrict__ assign) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) assign[i] = (int)(bestpack[i] & 0xffffffffull);
}

/* local copy: the driver's Timer lives in mpkmeans.cu, and this file is meant
 * to stand on its own */
struct RTTimer {
    cudaEvent_t a, b; double* acc = nullptr; bool live = false;
    RTTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~RTTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { cudaEventRecord(a, s); }
    void stop(cudaStream_t s, double* dst) { cudaEventRecord(b, s); acc = dst; live = true; }
    void collect() { if (!live) return; float ms = 0.f;
                     cudaEventElapsedTime(&ms, a, b); *acc += ms; live = false; }
    /* drop the pending interval without accumulating it, for a pass whose
     * result is being thrown away.  Zeroing *acc instead would also wipe every
     * previous iteration's contribution, since *acc is a run total. */
    void discard() { live = false; }
    void stop_sync(cudaStream_t s, double* dst) {
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b); *dst += ms;
    }
};

static inline int rt_grid(int n, int cap) {
    const int full = (n + RT_WPB - 1) / RT_WPB;
    return full < cap ? (full < 1 ? 1 : full) : cap;
}

static cudaError_t rt_grow(void** p, size_t* cap, size_t need) {
    if (*cap >= need) return cudaSuccess;
    if (*p) cudaFree(*p);
    *p = nullptr;
    size_t want = *cap ? *cap : 256;
    while (want < need) want *= 2;
    cudaError_t e = cudaMalloc(p, want);
    *cap = (e == cudaSuccess) ? want : 0;
    return e;
}

extern "C" mpkStatus mpkMeansBaselineRT(cublasHandle_t blas, const float* dP,
                                        int n, int d, int k, float* dC,
                                        int* dAssign, const mpkParams* params,
                                        mpkStats* stats) {
    if (n <= 0 || d <= 0 || k <= 0 || k > n) return MPK_ERR_INVALID;
    mpkParams P;
    if (params) P = *params; else mpkParamsInit(&P);
    memset(stats, 0, sizeof(*stats));
#ifdef MPK_STATS
    stats->stats_built = 1;
#endif
    /* the paper's safety factor: > 2 required.  2.5 by default -- see the
     * file comment above for why that, and not the 5 their prose names. */
    float theta = P.rt_theta > 0.f ? P.rt_theta : 2.5f;
    /* the computable error floor E_l, less the (||p||^2 + ||c||^2) factor */
    const double u_l   = 4.8828125e-4;             /* FP16 unit roundoff, 2^-11 */
    const double md    = (double)d + 2.0;
    const double gam_l = md * u_l / (1.0 - md * u_l);
    const float  efac  = (float)(2.0 * gam_l);
    if (md * u_l >= 1.0) return MPK_ERR_INVALID;   /* analysis needs (d+2)u < 1 */
    const float thresh = theta * efac;

    cudaStream_t s = nullptr;
    cublasGetStream(blas, &s);
    cublasMath_t saved;
    cublasGetMathMode(blas, &saved);

    __half *P16 = nullptr, *C16 = nullptr;
    float  *G = nullptr, *cnorm2 = nullptr, *pnorm2 = nullptr, *moved2 = nullptr;
    int    *prev = nullptr, *counts = nullptr;
    double *sums = nullptr, *dInertia = nullptr;
    unsigned long long* bestpack = nullptr;
    unsigned int* dNnz = nullptr;
    long long* dDiff = nullptr;
    long long* h_changed = nullptr;
    float* h_moved = nullptr;
    unsigned int* h_nnz = nullptr;
    void* list_buf = nullptr; size_t list_cap = 0, list_cap_entries = 0;

    mpkStatus st = MPK_OK;
#define ALLOC(p, bytes) if (cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
    ALLOC(P16, (size_t)n * d * sizeof(__half));
    ALLOC(C16, (size_t)k * d * sizeof(__half));
    ALLOC(G,   (size_t)n * k * sizeof(float));
    ALLOC(cnorm2, (size_t)k * sizeof(float));
    ALLOC(pnorm2, (size_t)n * sizeof(float));
    ALLOC(moved2, (size_t)k * sizeof(float));
    ALLOC(bestpack, (size_t)n * sizeof(unsigned long long));
    ALLOC(dNnz, sizeof(unsigned int));
    ALLOC(prev, (size_t)n * sizeof(int));
    ALLOC(sums, (size_t)k * d * sizeof(double));
    ALLOC(counts, (size_t)k * sizeof(int));
    ALLOC(dInertia, sizeof(double));
    ALLOC(dDiff, sizeof(long long));
#undef ALLOC
    if (cudaHostAlloc((void**)&h_changed, sizeof(long long), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_nnz, sizeof(unsigned int), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_moved, (size_t)k * sizeof(float),
                      cudaHostAllocDefault) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    if (rt_grow(&list_buf, &list_cap, (size_t)2 * n * sizeof(int)) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    list_cap_entries = list_cap / sizeof(int);

    cudaMemset(prev, 0xff, (size_t)n * sizeof(int));
    /* P -> FP16 is the one-time setup both schemes pay, and both leave it
     * untimed.  ||p||^2 is NOT: this scheme needs it and the exclusion scheme
     * does not (it drops ||p||^2 by translation invariance), so leaving it
     * outside the timers would hand the baseline a free extra pass over P.
     * It is charged to prep, once, and amortised over the run like everything
     * else there. */
    mpkLaunchToHalf(dP, P16, (long long)n * d, s);

    {
        RTTimer tt, t0, t1, t4, t5, t6, t7;
        tt.start(s);
        t0.start(s);
        mpkLaunchRowNorms(dP, n, d, pnorm2, s);   /* P is fixed: once */
        t0.stop_sync(s, &stats->t_prep_ms);
        cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
        const float f_one = 1.f, f_zero = 0.f;

        for (int it = 0; it < P.max_iter; ++it) {
            t0.start(s);
            mpkLaunchToHalf(dC, C16, (long long)k * d, s);
            mpkLaunchRowNorms(dC, k, d, cnorm2, s);
            t0.stop(s, &stats->t_prep_ms);

            t1.start(s);
            if (cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d,
                             &f_one, C16, CUDA_R_16F, d, P16, CUDA_R_16F, d,
                             &f_zero, G, CUDA_R_32F, k,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                != CUBLAS_STATUS_SUCCESS) { st = MPK_ERR_CUBLAS; goto done; }
            t1.stop(s, &stats->t_gemm_lo_ms);

            int nnz = 0;
            for (;;) {
                cudaMemsetAsync(dNnz, 0, sizeof(unsigned int), s);
                t7.start(s);
                k_rt_scan<<<rt_grid(n, 8192), RT_NTHR, 0, s>>>(
                    G, pnorm2, cnorm2, n, k, thresh, bestpack,
                    (int*)list_buf, (int)list_cap_entries, dNnz);
                t7.stop(s, &stats->t_argmin_ms);
                cudaMemcpyAsync(h_nnz, dNnz, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                nnz = (int)*h_nnz;
                if ((size_t)nnz <= list_cap_entries) break;
                /* the overflowing scan is discarded, and not charged for */
                t7.discard();
                if (rt_grow(&list_buf, &list_cap, (size_t)nnz * sizeof(int))
                    != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
                list_cap_entries = list_cap / sizeof(int);
            }
            t7.collect();

            if (nnz > 0) {
                t4.start(s);
                k_rt_correct<<<rt_grid(nnz, 8192), RT_NTHR, 0, s>>>(
                    dP, dC, (const int*)list_buf, nnz, k, d, bestpack);
                t4.stop(s, &stats->t_hp_update_ms);
            }
            t5.start(s);
            k_rt_unpack<<<(n + RT_NTHR - 1) / RT_NTHR, RT_NTHR, 0, s>>>(
                bestpack, n, dAssign);
            t5.stop(s, &stats->t_assign_ms);

            /* n*k, not n*(k-1): that exemption belongs to the exclusion
             * scheme, where each row's incumbent is never tested against
             * itself.  This baseline has no incumbent -- k_rt_scan applies the
             * reliability test to every entry of every row. */
            stats->tested      += (long long)n * k;
            stats->hp_baseline += (long long)n * k;
            stats->hp_update   += nnz;

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
            /* No early exit on label stability.  Without --convergence the loop
             * runs exactly max_iter iterations, which is what makes a fixed
             * iteration count mean the same thing here and in the mp-kmeans
             * driver; with it, the only stopping rule is the Frobenius test
             * below -- the rule that package and cuVS also use.  A
             * label-stability break on top of that fired at a completely
             * different point for the low precision schemes than for FP32
             * (coarse distances stop flipping marginal points early), so runs
             * of different lengths were being compared as if equal.
             *
             * `changed` is still computed: it is what --verbose prints, and
             * keeping the memcpy and sync leaves the loop's synchronisation
             * structure, and hence its timings, unchanged. */

            t6.start(s);
            mpkLaunchZero(sums, counts, k, d, s);
            mpkLaunchAccumulate(dP, dAssign, n, d, sums, counts, s);
            mpkLaunchFinalizeCentroids(sums, counts, k, d, dC, moved2, s);
            /* stop_sync, not stop+collect: the update is the last thing in the
             * iteration, so nothing downstream would have forced the event to
             * complete and cudaEventElapsedTime would read back 0. */
            t6.stop_sync(s, &stats->t_update_ms);

            /* same stopping rule as the exclusion scheme, so a convergence run
             * compares the same number of iterations of the same recurrence */
            if (P.tol > 0.f) {
                cudaMemcpyAsync(h_moved, moved2, (size_t)k * sizeof(float),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                double fro2 = 0.0;
                for (int j = 0; j < k; ++j) fro2 += (double)h_moved[j];
                if (fro2 < (double)P.tol * (double)P.tol) break;
            }
        }
        tt.stop_sync(s, &stats->t_total_ms);
    }
    stats->t_dist_ms = stats->t_prep_ms + stats->t_gemm_lo_ms +
                       stats->t_argmin_ms + stats->t_hp_update_ms +
                       stats->t_assign_ms;

    cudaMemset(dInertia, 0, sizeof(double));
    mpkLaunchInertia(dP, dC, dAssign, n, d, dInertia, s);
    cudaMemcpy(&stats->inertia, dInertia, sizeof(double), cudaMemcpyDeviceToHost);

done:
    cublasSetMathMode(blas, saved);
    cudaFree(P16); cudaFree(C16); cudaFree(G); cudaFree(cnorm2);
    cudaFree(pnorm2); cudaFree(moved2); cudaFree(bestpack); cudaFree(dNnz);
    cudaFree(prev); cudaFree(sums); cudaFree(counts); cudaFree(dInertia);
    cudaFree(dDiff); cudaFree(list_buf);
    if (h_changed) cudaFreeHost(h_changed);
    if (h_nnz) cudaFreeHost(h_nnz);
    if (h_moved) cudaFreeHost(h_moved);
    return st;
}
