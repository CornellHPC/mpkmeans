/* Multiword (double-fp16) Lloyd's iteration.
 *
 * A THIRD strategy, next to the Theorem 1 exclusion bounds in mpkmeans.cu and
 * the per-entry reliability test in baseline.cu.  The exclusion machinery is
 * exactly the one in mpkmeans.cu -- the same conditions (3)/(6), the same
 * kernel -- but what sits underneath it is different: instead of one FP16 GEMM
 * plus a per-entry FP32 fallback, the distance matrix is built out of the
 * multiword decomposition of
 *
 *   M. Fasi, N. J. Higham, F. Lopez, T. Mary, M. Mikaitis, "Matrix
 *   Multiplication in Multiword Arithmetic: Error Analysis and Application to
 *   GPU Tensor Cores", MIMS EPrint 2022.3 (GAMM 2022).
 *
 * P and C are each split into two fp16 words,
 *
 *     A_1 = fl16(A),   A_2 = fl16(A - A_1),
 *
 * and the product is assembled one cross product at a time:
 *
 *     stage 1:  G  = C_1^T P_1
 *     stage 2:  G += C_1^T P_2
 *     stage 3:  G += C_2^T P_1          (= double-fp16, the paper's p=2 cut)
 *
 * Each stage is one tensor-core GEMM over the WHOLE matrix, and between
 * stages the exclusion test runs again on the sharper G.  A stage is only
 * paid for if the previous test left something unsettled: when no entry
 * survives, every row's argmin is already proved and the remaining cross
 * products are never issued.
 *
 * That is the whole point of the scheme.  In mpkmeans.cu the refinement is a
 * warp-per-entry FP32 dot -- memory bound, and it loses badly when survivors
 * are dense.  Here the refinement is another tensor-core GEMM, so the dense
 * case costs 3 GEMMs instead of 1 rather than millions of scattered dots, and
 * the exclusion test's job shifts from "which entries do I recompute" to "do
 * I need the next cross product at all".
 *
 * P is split ONCE for the whole run: it never changes.  Only C is re-split,
 * once per iteration.
 */
#include "mpk_internal.cuh"

#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------- epsilon --- */
/* The bounds of the GAMM 2022 presentation, slides 7-9.  With the split
 *
 *     A = sum_i A_i + dA,   |dA| <= u_lo^p |A|
 *
 * and every product with i + j > p + 1 dropped, the paper's error bound is
 *
 *     |E| <= ( (p+1) u_lo^p + gamma^hi_{n + p^2 - 1} ) |A||B|              (slide 8)
 *
 * which for fp16 (u_lo = 2^-11) against fp32 accumulation (u_hi = 2^-24) gives
 * the table on slide 9: p=1 is 2*2^-11 + n*2^-24, p=2 ("double-fp16") is
 * n*2^-24.
 *
 * We need it at every stage, not just at the paper's cut, because the
 * exclusion test runs after each cross product.  Since P and C are both
 * non-negative here (mpkShiftNonNegative), |A||B| = AB and the componentwise
 * bound is a relative one, which is exactly the shape conditions (3)/(6)
 * want -- see mpkEpsilon in mpkmeans.cu, whose value this reduces to at
 * macs = 1.
 *
 * Stage by stage, writing A for C and B for P:
 *
 *   macs = 1   A_1 B_1                 dropped A_1B_2 + A_2B_1 ~ 2 u_lo |A||B|
 *   macs = 2   + A_1 B_2               dropped A_2 B ~ u_lo |A||B|
 *   macs = 3   + A_2 B_1               dropped A_2B_2 ~ u_lo^2, the p=2 bound
 */
extern "C" double mpkMultiwordEpsilon(int d, int macs) {
    const double u_lo = 1.0 / 2048.0;        /* fp16, 2^-11  */
    const double u_hi = 1.0 / 16777216.0;    /* fp32, 2^-24  */
    if (macs < 1 || macs > 3) return -1.0;

    /* gamma^hi_m: the accumulation term.  m grows by one per extra cross
     * product, matching the paper's n + p^2 - 1 at the p = 1 and p = 2 cuts. */
    const int    m     = d + (macs == 1 ? 0 : macs == 2 ? 1 : 3);
    const double g_raw = (double)m * u_hi;
    if (g_raw >= 1.0) return -1.0;
    const double gamma = g_raw / (1.0 - g_raw);

    /* the dropped cross products, as a relative factor on |A||B| */
    double dropped;
    if      (macs == 1) dropped = 2.0 * u_lo + u_lo * u_lo;   /* (1+u_lo)^2 - 1 */
    else if (macs == 2) dropped = u_lo + u_lo * u_lo;
    else                dropped = 3.0 * u_lo * u_lo;          /* the p=2 bound  */

    const double eps = (1.0 + dropped) * (1.0 + gamma) - 1.0;
    return (eps >= 1.0) ? -1.0 : eps;
}

#ifndef MPK_MULTIWORD

/* Built without the mpemu multiword library.  Keep the symbol so callers
 * link, and say so rather than silently doing something else. */
extern "C" mpkStatus mpkMeansMultiword(cublasHandle_t, const float*, int, int,
                                       int, float*, int*, const mpkParams*,
                                       mpkStats* stats) {
    if (stats) memset(stats, 0, sizeof(*stats));
    fprintf(stderr, "mpkMeansMultiword: built without mpemu "
                    "(configure with -DMPK_MULTIWORD=ON)\n");
    return MPK_ERR_INVALID;
}

#else  /* MPK_MULTIWORD */

#include "mpemu/mpemu.h"

#define MW_WARP 32
#define MW_WPB  8
#define MW_NTHR 256

namespace {

/* Order preserving float -> uint, so a packed (distance, index) key compares
 * as an unsigned integer.  Local copies of kernels.cu's mpk_ford/mpk_pack;
 * this file is meant to stand on its own. */
__device__ __forceinline__ unsigned int mw_ford(float f) {
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ unsigned long long mw_pack(float dist, int j) {
    return ((unsigned long long)mw_ford(dist) << 32) | (unsigned int)j;
}

/* Which rows still have a survivor.  The exclusion test hands back a flat list
 * of i*k + j, and the refinement wants the distinct rows out of it: those are
 * the only points whose distances are still undecided, and so the only ones
 * the remaining cross products have to be computed for.
 *
 * Two passes rather than a sort: mark, then compact.  Every write in the mark
 * pass stores the same value, so the races between entries of one row are
 * benign. */
__global__ void k_mw_mark_rows(const int* __restrict__ list, int nnz, int k,
                               unsigned char* __restrict__ rowflag) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (; e < nnz; e += stride) rowflag[list[e] / k] = 1;
}

/* rowlist[t] = the t-th unsettled row; rowmap[i] = t for those rows.  Order is
 * whatever the atomics produce -- nothing downstream depends on it, only on
 * rowmap and rowlist agreeing. */
__global__ void k_mw_compact_rows(const unsigned char* __restrict__ rowflag,
                                  int n, int* __restrict__ rowlist,
                                  int* __restrict__ rowmap,
                                  unsigned int* __restrict__ count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        if (!rowflag[i]) continue;
        const unsigned int t = atomicAdd(count, 1u);
        rowlist[t] = i;
        rowmap[i]  = (int)t;
    }
}

/* Pull the unsettled rows' fp16 words out of the split P planes.  P is seen
 * column-major d x n, so a "row of P" is a column here: d contiguous halves.
 * Both planes are gathered in one launch. */
__global__ void k_mw_gather_cols(const __half* __restrict__ src, int64_t lds,
                                 int64_t stride_src, int d,
                                 const int* __restrict__ rowlist, int m,
                                 __half* __restrict__ dst, int64_t stride_dst) {
    const int t = blockIdx.x;                      /* one block per row */
    if (t >= m) return;
    const int i = rowlist[t];
    for (int p = 0; p < 2; ++p) {
        const __half* s = src + (int64_t)p * stride_src + (int64_t)i * lds;
        __half*       o = dst + (int64_t)p * stride_dst + (int64_t)t * lds;
        for (int e = threadIdx.x; e < d; e += blockDim.x) o[e] = s[e];
    }
}

/* The refinement, such as it is: the survivors' distances are already sitting
 * in G, so there is nothing to compute -- one thread per surviving flat index
 * reads its entry and reduces it into that row's packed best.
 *
 * This is where the multiword scheme differs most from mpkmeans.cu: there the
 * equivalent kernel is a warp-per-entry FP32 dot over d elements; here the
 * arithmetic already happened, inside a tensor-core GEMM, and all that is left
 * is a gather.
 *
 * G2 is the narrow path's extra cross products, k x m, holding only the
 * gathered rows: the value for row i is then G(i,j) + G2(rowmap[i], j).  Pass
 * G2 = NULL when the refinement was dense and G already carries everything. */
__global__ void k_mw_gather(const float* __restrict__ G,
                            const float* __restrict__ G2,
                            const int* __restrict__ rowmap,
                            const float* __restrict__ cnorm2,
                            const int* __restrict__ list, int nnz, int k,
                            unsigned long long* __restrict__ bestpack) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (; e < nnz; e += stride) {
        const int idx = list[e];
        const int i   = idx / k;
        const int j   = idx - i * k;
        float g = G[idx];
        if (G2) g += G2[(size_t)rowmap[i] * k + j];
        atomicMin(&bestpack[i], mw_pack(fmaf(-2.f, g, cnorm2[j]), j));
    }
}

__global__ void k_mw_unpack(const unsigned long long* __restrict__ bestpack,
                            int n, int* __restrict__ assign) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) assign[i] = (int)(bestpack[i] & 0xffffffffull);
}

/* local copy: the driver's Timer lives in mpkmeans.cu, and this file is meant
 * to stand on its own */
struct MWTimer {
    cudaEvent_t a, b; double* acc = nullptr; bool live = false;
    MWTimer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~MWTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(cudaStream_t s) { cudaEventRecord(a, s); }
    void stop(cudaStream_t s, double* dst) { cudaEventRecord(b, s); acc = dst; live = true; }
    void collect() { if (!live) return; float ms = 0.f;
                     cudaEventElapsedTime(&ms, a, b); *acc += ms; live = false; }
    /* drop the pending interval without accumulating it (see RTTimer) */
    void discard() { live = false; }
    void stop_sync(cudaStream_t s, double* dst) {
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b); *dst += ms;
    }
};

cudaError_t mw_grow(void** p, size_t* cap, size_t need) {
    if (*cap >= need) return cudaSuccess;
    if (*p) cudaFree(*p);
    *p = nullptr;
    size_t want = *cap ? *cap : 256;
    while (want < need) want *= 2;
    cudaError_t e = cudaMalloc(p, want);
    *cap = (e == cudaSuccess) ? want : 0;
    return e;
}

}  /* namespace */

extern "C" mpkStatus mpkMeansMultiword(cublasHandle_t blas, const float* dP,
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
    if (!P.use_cond3 && !P.use_cond6) {
        fprintf(stderr, "mpkMeansMultiword: at least one condition must be on\n");
        return MPK_ERR_INVALID;
    }
    /* how far the cascade may go: 1 means never refine (plain fp16), 3 is the
     * paper's double-fp16 cut.  Anything else is out of range for a 2-word
     * split. */
    const int max_macs = (P.mw_macs >= 1 && P.mw_macs <= 3) ? P.mw_macs : 3;

    const int need_ref     = P.use_cond6 ? 1 : 0;
    const int include_best = need_ref ? 0 : 1;
    const int cascade_on   = (P.cascade && P.use_cond3 && P.use_cond6) ? 1 : 0;

    /* Condition (6)'s reference entry is an FP32 dot straight from dP/dC, as
     * in mpkmeans.cu -- it is n values, not n*k, and it is tighter than any
     * partial multiword value we could read out of G, so it stays exact.
     * gfac therefore keeps the FP32 dot-product bound; only `factor`, which
     * scales the entries of G, moves with the stage. */
    const double U_HI   = 1.0 / 16777216.0;
    const float  SLACK  = 8.0f * (float)U_HI;
    const double gs_raw = (double)d * U_HI;
    const double gs     = gs_raw / (1.0 - gs_raw);
    const float  gfac   = (float)(2.0 * gs / (1.0 - gs));

    /* factor per stage, from mpkMultiwordEpsilon above */
    float factor[4] = {0.f, 0.f, 0.f, 0.f};
    for (int m = 1; m <= 3; ++m) {
        const double eps = mpkMultiwordEpsilon(d, m);
        if (eps < 0.0) {
            fprintf(stderr, "mpkMeansMultiword: error model degenerate for "
                            "d=%d at %d cross product(s)\n", d, m);
            return MPK_ERR_INVALID;
        }
        factor[m] = (float)(2.0 * eps / (1.0 - eps));
    }

    /* When to gather rather than refine densely.  Measured crossover on an
     * A100 at n=200k, d=128: narrow starts winning below ~20% of the rows at
     * k=32, ~33% at k=128 and ~70% at k=256.  It moves with k because the
     * refining GEMM's cost scales with the output width k while the gather's
     * does not -- it copies m*d fp16 words whatever k is -- so the wider the
     * output, the more of the row range it is worth narrowing.  k/384 fits
     * those points; the clamps keep it sane outside the measured range.
     * A positive params.mw_gather_frac overrides all of this. */
    double gather_frac = (double)k / 384.0;
    if (gather_frac < 0.15) gather_frac = 0.15;
    if (gather_frac > 0.75) gather_frac = 0.75;
    if (P.mw_gather_frac > 0.f) gather_frac = (double)P.mw_gather_frac;

    cudaStream_t s = nullptr;
    cublasGetStream(blas, &s);

    /* --- split buffers.  Both matrices are seen column-major with d rows:
     * dP is row-major n x d, i.e. column-major d x n with lda = d, and dC is
     * row-major k x d, i.e. column-major d x k with lda = d.  So the two
     * splits share a leading dimension. --- */
    const int64_t lds     = mpemuSplitLd(d);
    const int64_t strideP = mpemuSplitStride(lds, n);
    const int64_t strideC = mpemuSplitStride(lds, k);

    __half *dSP = nullptr, *dSC = nullptr;
    float  *G = nullptr, *cnorm2 = nullptr, *moved2 = nullptr;
    float  *dbest = nullptr, *gbest = nullptr, *gexact = nullptr;
    int    *jbest = nullptr, *prev = nullptr, *counts = nullptr;
    double *sums = nullptr, *dInertia = nullptr;
    unsigned long long *bestpack = nullptr, *dRefCnt = nullptr;
    unsigned int *dNnz = nullptr;
    long long *dDiff = nullptr, *dStatBanks = nullptr;
    long long *h_changed = nullptr;
    unsigned int *h_nnz = nullptr;
    float *h_moved = nullptr;
    void  *list_buf = nullptr; size_t list_cap = 0, list_cap_entries = 0;
    /* the narrow-refinement scratch: which rows are still undecided, their
     * gathered fp16 words, and the extra cross products for just those rows */
    unsigned char *rowflag = nullptr;
    int   *rowlist = nullptr, *rowmap = nullptr;
    unsigned int  *dRowCnt = nullptr;
    __half *dSPsub = nullptr; size_t sub_cap = 0;
    float  *G2buf = nullptr;  size_t g2_cap = 0;
    float  scaleP = 1.f, scaleC = 1.f;

    mpkStatus st = MPK_OK;
#define ALLOC(p, bytes) if (cudaMalloc((void**)&(p), (bytes)) != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
    ALLOC(dSP, mpemuSplitBytes(d, n, 2));
    ALLOC(dSC, mpemuSplitBytes(d, k, 2));
    ALLOC(G,      (size_t)n * k * sizeof(float));
    ALLOC(cnorm2, (size_t)k * sizeof(float));
    ALLOC(moved2, (size_t)k * sizeof(float));
    ALLOC(dbest,  (size_t)n * sizeof(float));
    ALLOC(gbest,  (size_t)n * sizeof(float));
    if (need_ref)   ALLOC(gexact, (size_t)n * sizeof(float));
    if (cascade_on) ALLOC(dRefCnt, sizeof(unsigned long long));
    ALLOC(jbest,    (size_t)n * sizeof(int));
    ALLOC(bestpack, (size_t)n * sizeof(unsigned long long));
    ALLOC(dNnz,     sizeof(unsigned int));
    ALLOC(prev,     (size_t)n * sizeof(int));
    ALLOC(sums,     (size_t)k * d * sizeof(double));
    ALLOC(counts,   (size_t)k * sizeof(int));
    ALLOC(dInertia, sizeof(double));
    ALLOC(dDiff,    sizeof(long long));
    ALLOC(rowflag,  (size_t)n);
    ALLOC(rowlist,  (size_t)n * sizeof(int));
    ALLOC(rowmap,   (size_t)n * sizeof(int));
    ALLOC(dRowCnt,  sizeof(unsigned int));
#ifdef MPK_STATS
    ALLOC(dStatBanks, 3 * MPK_STAT_BANKS * sizeof(long long));
#endif
#undef ALLOC
    if (cudaHostAlloc((void**)&h_changed, sizeof(long long), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_nnz, sizeof(unsigned int), cudaHostAllocDefault)
            != cudaSuccess ||
        cudaHostAlloc((void**)&h_moved, (size_t)k * sizeof(float),
                      cudaHostAllocDefault) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    if (mw_grow(&list_buf, &list_cap, (size_t)2 * n * sizeof(int)) != cudaSuccess) {
        st = MPK_ERR_ALLOC; goto done;
    }
    list_cap_entries = list_cap / sizeof(int);

    cudaMemset(prev, 0xff, (size_t)n * sizeof(int));
#ifdef MPK_STATS
    cudaMemset(dStatBanks, 0, 3 * MPK_STAT_BANKS * sizeof(long long));
#endif

    /* P is fixed: scale and split it once, for the whole run. */
    if (mpemuAutoScaleFP16(dP, d, d, n, &scaleP, s) != MPEMU_STATUS_SUCCESS ||
        mpemuSplitFP16(dP, d, d, n, 2, scaleP, dSP, lds, strideP, s)
            != MPEMU_STATUS_SUCCESS) {
        st = MPK_ERR_CUDA; goto done;
    }

    {
        MWTimer tt, t0, t1, t2, t4, t5, t6, t7;
        tt.start(s);
        cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);

        for (int it = 0; it < P.max_iter; ++it) {
            /* ---- 1: C -> two fp16 words, plus its norms ------------------
             * The only per-iteration split: the centroids moved, the points
             * did not. */
            t0.start(s);
            mpkLaunchRowNorms(dC, k, d, cnorm2, s);
            t0.stop(s, &stats->t_prep_ms);
            if (mpemuAutoScaleFP16(dC, d, d, k, &scaleC, s) != MPEMU_STATUS_SUCCESS ||
                mpemuSplitFP16(dC, d, d, k, 2, scaleC, dSC, lds, strideC, s)
                    != MPEMU_STATUS_SUCCESS) {
                st = MPK_ERR_CUDA; goto done;
            }

            /* ---- 2: the leading cross product, then one test ------------
             * G_cm(k x n) = C_cm^T * P_cm  ==  G row major (n x k) = P C^T,
             * the same orientation every other driver uses.
             *
             * The cascade is 1 -> 3, not 1 -> 2 -> 3.  The middle point is
             * dominated: adding only one of the two bin-1 products takes the
             * bound from 2*u_lo to u_lo, a factor of two, which excludes
             * almost nothing extra (measured: 1% fewer survivors on separated
             * blobs) while costing a whole GEMM and a whole test pass.  The
             * two cut points worth having are exactly the paper's p=1 and
             * p=2. */
            t1.start(s);
            if (mpemuGemmMultiwordRange(
                    blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, 1.0f,
                    dSC, lds, strideC, scaleC,
                    dSP, lds, strideP, scaleP,
                    /*nsplits=*/2, /*macBegin=*/0, /*macEnd=*/1, /*beta=*/0.0f,
                    G, k) != MPEMU_STATUS_SUCCESS) {
                st = MPK_ERR_CUBLAS; goto done;
            }
            t1.stop(s, &stats->t_gemm_lo_ms);
            stats->mw_products += 1;

            int nnz = 0;
            for (;;) {
                cudaMemsetAsync(dNnz, 0, sizeof(unsigned int), s);
                t7.start(s);
                mpkLaunchArgminCount<float>(
                    G, cnorm2, dP, dC, n, d, k, factor[1], gfac, SLACK,
                    P.use_cond3, P.use_cond6, cascade_on, /*refine=*/1,
                    jbest, dbest, gbest, gexact, bestpack, include_best,
                    (int*)list_buf, (int)list_cap_entries, dNnz, dRefCnt,
                    dStatBanks, s);
                t7.stop(s, &stats->t_argmin_ms);
                cudaMemcpyAsync(h_nnz, dNnz, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                nnz = (int)*h_nnz;
                if ((size_t)nnz <= list_cap_entries) break;
                /* the overflowing scan is discarded, and not charged for */
                t7.discard();
                if (mw_grow(&list_buf, &list_cap, (size_t)nnz * sizeof(int))
                    != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
                list_cap_entries = list_cap / sizeof(int);
            }
            t7.collect();
            t1.collect();
            stats->tested      += (long long)n * (k - 1);
            stats->hp_baseline += (long long)n * k;

            /* ---- 3: the remaining cross products, for whoever needs them --
             * Only the rows that still hold a survivor are undecided, so only
             * they need the rest of the product.  Whether that is worth
             * gathering for depends on how many there are: a narrow GEMM over
             * m rows beats a dense one over n whenever m is a small enough
             * fraction of n, and loses below that because the gather itself
             * costs a pass over P's words and the GEMM gets too skinny to run
             * well.  P.mw_gather_frac is that crossover. */
            int m_rows = 0;
            const float* G2 = nullptr;
            if (nnz > 0 && max_macs > 1) {
                cudaMemsetAsync(dRowCnt, 0, sizeof(unsigned int), s);
                cudaMemsetAsync(rowflag, 0, (size_t)n, s);
                t2.start(s);
                {
                    const int gm = mpk_ceil_div(nnz, MW_NTHR);
                    k_mw_mark_rows<<<gm > 8192 ? 8192 : (gm < 1 ? 1 : gm),
                                     MW_NTHR, 0, s>>>(
                        (const int*)list_buf, nnz, k, rowflag);
                    const int gc = mpk_ceil_div(n, MW_NTHR);
                    k_mw_compact_rows<<<gc > 8192 ? 8192 : (gc < 1 ? 1 : gc),
                                        MW_NTHR, 0, s>>>(
                        rowflag, n, rowlist, rowmap, dRowCnt);
                }
                cudaMemcpyAsync(h_nnz, dRowCnt, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                m_rows = (int)*h_nnz;
                t2.stop(s, &stats->t_prep_ms);

                const int narrow = (double)m_rows <= gather_frac * (double)n;
                if (narrow && m_rows > 0) {
                    /* gather those rows' words, then a GEMM only that wide */
                    if (mw_grow((void**)&dSPsub, &sub_cap,
                                mpemuSplitBytes(d, m_rows, 2)) != cudaSuccess ||
                        mw_grow((void**)&G2buf, &g2_cap,
                                (size_t)m_rows * k * sizeof(float))
                            != cudaSuccess) { st = MPK_ERR_ALLOC; goto done; }
                    const int64_t strideSub = mpemuSplitStride(lds, m_rows);
                    t2.start(s);
                    k_mw_gather_cols<<<m_rows, MW_NTHR, 0, s>>>(
                        dSP, lds, strideP, d, rowlist, m_rows,
                        dSPsub, strideSub);
                    t2.stop(s, &stats->t_prep_ms);

                    t1.start(s);
                    if (mpemuGemmMultiwordRange(
                            blas, CUBLAS_OP_T, CUBLAS_OP_N, k, m_rows, d, 1.0f,
                            dSC, lds, strideC, scaleC,
                            dSPsub, lds, strideSub, scaleP,
                            /*nsplits=*/2, /*macBegin=*/1, /*macEnd=*/max_macs,
                            /*beta=*/0.0f, G2buf, k) != MPEMU_STATUS_SUCCESS) {
                        st = MPK_ERR_CUBLAS; goto done;
                    }
                    t1.stop_sync(s, &stats->t_gemm_lo_ms);
                    G2 = G2buf;
                    stats->mw_products    += max_macs - 1;
                    stats->mw_refine_rows += m_rows;
                    stats->hp_update      += (long long)m_rows * k;
                } else {
                    /* too many rows left to be worth gathering: refine the
                     * whole matrix in place, as the dense scheme does */
                    t1.start(s);
                    if (mpemuGemmMultiwordRange(
                            blas, CUBLAS_OP_T, CUBLAS_OP_N, k, n, d, 1.0f,
                            dSC, lds, strideC, scaleC,
                            dSP, lds, strideP, scaleP,
                            /*nsplits=*/2, /*macBegin=*/1, /*macEnd=*/max_macs,
                            /*beta=*/1.0f, G, k) != MPEMU_STATUS_SUCCESS) {
                        st = MPK_ERR_CUBLAS; goto done;
                    }
                    t1.stop_sync(s, &stats->t_gemm_lo_ms);
                    stats->mw_products    += max_macs - 1;
                    stats->mw_refine_rows += n;
                    stats->hp_update      += (long long)n * k;
                }
            }
            if (P.verbose)
                fprintf(stderr, "[mw]   survivors %9d / %-10lld (%.4f%%)  "
                                "rows %7d / %-8d %s\n", nnz,
                        (long long)n * (k - 1),
                        100.0 * nnz / ((double)n * (k - 1)), m_rows, n,
                        G2 ? "narrow" : (nnz ? "dense" : "settled"));

            /* ---- the survivors' distances are already in G: gather them --- */
            if (nnz > 0) {
                t4.start(s);
                const int grid = mpk_ceil_div(nnz, MW_NTHR) > 8192
                               ? 8192 : mpk_ceil_div(nnz, MW_NTHR);
                k_mw_gather<<<grid < 1 ? 1 : grid, MW_NTHR, 0, s>>>(
                    G, G2, rowmap, cnorm2, (const int*)list_buf, nnz, k,
                    bestpack);
                t4.stop(s, &stats->t_hp_update_ms);
            }

            t5.start(s);
            k_mw_unpack<<<mpk_ceil_div(n, MW_NTHR), MW_NTHR, 0, s>>>(
                bestpack, n, dAssign);
            t5.stop(s, &stats->t_assign_ms);

            /* ---- convergence --------------------------------------------- */
            cudaMemsetAsync(dDiff, 0, sizeof(long long), s);
            mpkLaunchCountDiff(dAssign, prev, n, dDiff, s);
            cudaMemcpyAsync(h_changed, dDiff, sizeof(long long),
                            cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            const long long changed = *h_changed;
            cudaMemcpyAsync(prev, dAssign, (size_t)n * sizeof(int),
                            cudaMemcpyDeviceToDevice, s);

            t0.collect(); t2.collect(); t4.collect(); t5.collect();
            stats->iters = it + 1;
            if (P.verbose)
                fprintf(stderr, "[mw] iter %3d  changed %8lld\n", it, changed);
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

            /* ---- centroid update (excluded from t_dist_ms) ---------------- */
            t6.start(s);
            mpkLaunchZero(sums, counts, k, d, s);
            mpkLaunchAccumulate(dP, dAssign, n, d, sums, counts, s);
            mpkLaunchFinalizeCentroids(sums, counts, k, d, dC, moved2, s);
            t6.stop_sync(s, &stats->t_update_ms);

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
        long long banks[3 * MPK_STAT_BANKS];
        cudaMemcpy(banks, dStatBanks, sizeof(banks), cudaMemcpyDeviceToHost);
        for (int b = 0; b < MPK_STAT_BANKS; ++b) {
            stats->excl_cond3 += banks[b];
            stats->excl_cond6 += banks[MPK_STAT_BANKS + b];
            stats->excl_both  += banks[2 * MPK_STAT_BANKS + b];
        }
    }
#endif
    cudaMemset(dInertia, 0, sizeof(double));
    mpkLaunchInertia(dP, dC, dAssign, n, d, dInertia, s);
    cudaMemcpy(&stats->inertia, dInertia, sizeof(double), cudaMemcpyDeviceToHost);

done:
    cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH);
    cudaFree(dSP); cudaFree(dSC); cudaFree(G); cudaFree(cnorm2);
    cudaFree(moved2); cudaFree(dbest); cudaFree(gbest); cudaFree(gexact);
    cudaFree(dRefCnt); cudaFree(jbest); cudaFree(bestpack); cudaFree(dNnz);
    cudaFree(prev); cudaFree(sums); cudaFree(counts); cudaFree(dInertia);
    cudaFree(dDiff); cudaFree(dStatBanks); cudaFree(list_buf);
    cudaFree(rowflag); cudaFree(rowlist); cudaFree(rowmap); cudaFree(dRowCnt);
    cudaFree(dSPsub); cudaFree(G2buf);
    if (h_changed) cudaFreeHost(h_changed);
    if (h_nnz)     cudaFreeHost(h_nnz);
    if (h_moved)   cudaFreeHost(h_moved);
    return st;
}

#endif /* MPK_MULTIWORD */
