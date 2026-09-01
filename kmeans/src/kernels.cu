#include "mpk_internal.cuh"
#include <cub/cub.cuh>
#include <climits>

namespace {

constexpr int WARP  = 32;
constexpr int WPB   = 8;                 /* warps per block */
constexpr int NTHR  = WARP * WPB;        /* 256 */
constexpr unsigned FULL = 0xffffffffu;

/* ---------------------------------------------------- warp primitives ---- */

struct ArgMin {
    float v;
    int   j;
};

__device__ __forceinline__ ArgMin amin(ArgMin a, ArgMin b) {
    /* smaller index wins ties, so the reduction is order independent */
    if (b.v < a.v || (b.v == a.v && b.j < a.j)) return b;
    return a;
}

__device__ __forceinline__ ArgMin warp_argmin(ArgMin x) {
    for (int off = WARP / 2; off > 0; off >>= 1) {
        ArgMin o;
        o.v = __shfl_down_sync(FULL, x.v, off);
        o.j = __shfl_down_sync(FULL, x.j, off);
        x = amin(x, o);
    }
    return x;  /* valid in lane 0 */
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int off = WARP / 2; off > 0; off >>= 1) x += __shfl_down_sync(FULL, x, off);
    return x;
}

__device__ __forceinline__ unsigned long long warp_sum(unsigned long long x) {
    for (int off = WARP / 2; off > 0; off >>= 1) x += __shfl_down_sync(FULL, x, off);
    return x;
}

__device__ __forceinline__ double warp_sum(double x) {
    for (int off = WARP / 2; off > 0; off >>= 1) x += __shfl_down_sync(FULL, x, off);
    return x;
}

__device__ __forceinline__ float block_sum(float mine, float* sh) {
    const int t = threadIdx.x;
    sh[t] = mine;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (t < off) sh[t] += sh[t + off];
        __syncthreads();
    }
    return sh[0];
}

/* --------------------------------------------------------------- casts --- */

__global__ void k_to_half(const float* __restrict__ src, __half* __restrict__ dst,
                          long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = __float2half(src[i]);
}

/* --------------------------------------------------------------- norms --- */

__global__ void k_row_norms(const float* __restrict__ C, int d,
                            float* __restrict__ out) {
    __shared__ float sh[NTHR];
    const int j = blockIdx.x;
    const float* c = C + (size_t)j * d;
    float acc = 0.f;
    for (int t = threadIdx.x; t < d; t += blockDim.x) acc = fmaf(c[t], c[t], acc);
    float s = block_sum(acc, sh);
    if (threadIdx.x == 0) out[j] = s;
}

/* --------------------------------------------- argmin + exclusion test ---- */

/* Per-row constants of the two tests, hoisted out of the column loop.
 *
 *   (3)  dbest + slack*(|dbest| + |dt|) <= dt - factor*(g_j + gbest)
 *   (6)  dup   + slack*(|dup|   + |dt|) <= dt - factor* g_j
 *
 * becomes, with lhs = x + slack*|x| precomputed,
 *
 *        lhs + slack*|dt| <= dt - factor*(...)
 *
 * The reassociation perturbs only the slack term, itself a deliberate
 * overestimate ~8*u32, so the net change is O(u32^2); the FP64 oracle checks
 * the result either way. */
struct RowTest {
    float lhs3;     /* dbest + slack*|dbest|               */
    float lhs6;     /* dup   + slack*|dup|                 */
    float gbest;    /* fl16 inner product at the incumbent */
    int   jb;
};

/* True when column j survives, i.e. neither enabled condition can prove it
 * loses to the incumbent.
 *
 * C3/C6 are template parameters, not arguments: as runtime flags the compiler
 * kept both tests live in every instantiation, and the (3)+(6) configuration
 * cost as much as (3) and (6) separately added together. */
template <bool C3, bool C6>
__device__ __forceinline__ bool survives(float gj, float cn, const RowTest& t,
                                         float factor, float slack) {
    const float dt  = fmaf(-2.f, gj, cn);
    const float adt = fabsf(dt);
    const float gk  = fmaxf(gj, 0.f);        /* the bound assumes g >= 0 */
    if (C3) {
        const float rhs = dt - factor * (gk + t.gbest);
        if (fmaf(slack, adt, t.lhs3) <= rhs) return false;
    }
    if (C6) {
        const float rhs = dt - factor * gk;
        if (fmaf(slack, adt, t.lhs6) <= rhs) return false;
    }
    return true;
}

/* Build the per-row test from what the argmin produced. */
__device__ __forceinline__ RowTest row_test(int jb, float db, float gb, float ge,
                                            float cn_jb, float gfac, float slack,
                                            bool c3, bool c6) {
    RowTest t;
    t.jb    = jb;
    t.gbest = gb;
    t.lhs3  = 0.f;
    t.lhs6  = 0.f;
    if (c3) t.lhs3 = fmaf(slack, fabsf(db), db);
    if (c6) {
        /* dexact = cnorm2[jb] - 2*gexact, inflated into an upper bound on the
         * exact D(i, jb) before it is compared */
        const float du = fmaf(gfac, ge, fmaf(-2.f, ge, cn_jb));
        t.lhs6 = fmaf(slack, fabsf(du), du);
    }
    return t;
}

/* ARGMIN + COUNT.  One warp per row, grid stride.
 *
 * Two scans of the same row of G: the first takes the argmin, the second
 * applies the exclusion test and counts the survivors.  Only the first touches
 * memory below L1 -- a row is k*4 bytes, so it is still resident when the
 * second scan runs.  Splitting these into two kernels meant streaming all of G
 * from HBM twice and paying the per-row setup twice.
 *
 * The row reduction is cub::WarpReduce with cub::ArgMin, whose tie break (on
 * equal values keep the smaller key) is the column ordering we want.  It is
 * measurably better than the equivalent hand-rolled shuffle tree: the argmin
 * phase went from 3.22 to 2.70 ms over 20 iterations at n=200k, d=128, k=64.
 *
 * IPT > 0 additionally holds the row in registers, warp striped: lane L keeps
 * columns L, L+32, ...  The second scan then needs no loads at all, and cnorm2
 * is loop invariant across rows so it is hoisted out too.  That pays only while
 * IPT is small: it is worth ~13% at k <= 64 and loses from k = 128 up, where
 * the register pressure costs more than the L1 traffic it saves.  IPT == 0 is
 * the streaming fallback used there.
 *
 * REF also computes the one high precision quantity condition (6) needs,
 * gexact[i] = fl32(<p_i, c_jbest[i]>), as a warp-strided fma dot in registers.
 * Materialising the elementwise product and reducing it with a cublasSgemv
 * instead is an n x d write plus an n x d read to produce n numbers; it cost
 * 175 us/iteration at n=200k, d=128 against 35 us for the fused dot. */
template <int IPT, bool REF, bool C3, bool C6>
__global__ void k_argmin_count(const float* __restrict__ G,
                               const float* __restrict__ cnorm2,
                               const float* __restrict__ P,
                               const float* __restrict__ C,
                               int n, int d, int k,
                               float factor, float gfac, float slack,
                               int* __restrict__ jbest,
                               float* __restrict__ dbest,
                               float* __restrict__ gbest,
                               float* __restrict__ gexact,
                               int* __restrict__ row_nnz, int include_best,
                               unsigned long long* __restrict__ banks) {
    using KV         = cub::KeyValuePair<int, float>;
    using WarpReduce = cub::WarpReduce<KV>;
    __shared__ typename WarpReduce::TempStorage red[WPB];
#ifdef MPK_STATS
    __shared__ int sh3[WPB], sh6[WPB], shB[WPB];
    int n3 = 0, n6 = 0, nb = 0;
#endif
    const int lane = threadIdx.x & (WARP - 1);
    const int w    = threadIdx.x >> 5;
    const int step = gridDim.x * WPB;

    /* loop invariant: this lane's slice of cnorm2 */
    constexpr int NREG = IPT > 0 ? IPT : 1;
    float cn[NREG];
    if constexpr (IPT > 0) {
#pragma unroll
        for (int t = 0; t < IPT; ++t) {
            const int j = lane + t * WARP;
            cn[t] = (j < k) ? cnorm2[j] : INFINITY;
        }
    }

    /* every lane of a warp shares i, so the warp iterates as a whole and the
     * full-mask intrinsics below stay well defined */
    for (int i = blockIdx.x * WPB + w; i < n; i += step) {
        const float* g = G + (size_t)i * k;

        /* ---- 1. row argmin of Dt(i,:) = cnorm2[:] - 2*G(i,:) ------------- */
        float gv[NREG];
        KV mine{k, INFINITY};
        if constexpr (IPT > 0) {
#pragma unroll
            for (int t = 0; t < IPT; ++t) {
                const int j = lane + t * WARP;
                gv[t] = (j < k) ? g[j] : 0.f;
                if (j < k)
                    mine = cub::ArgMin()(mine, KV{j, fmaf(-2.f, gv[t], cn[t])});
            }
        } else {
            for (int j = lane; j < k; j += WARP)
                mine = cub::ArgMin()(mine, KV{j, fmaf(-2.f, g[j], cnorm2[j])});
        }
        const KV r = WarpReduce(red[w]).Reduce(mine, cub::ArgMin());

        const int   jb = __shfl_sync(FULL, r.key, 0);
        const float db = __shfl_sync(FULL, r.value, 0);
        const float gb = g[jb];          /* uniform address, hits L1 */

        /* ---- 2. the reference entry, when condition (6) is on ------------ */
        float ge = 0.f;
        if (REF) {
            const float* p = P + (size_t)i * d;
            const float* c = C + (size_t)jb * d;
            float acc = 0.f;
            for (int t = lane; t < d; t += WARP) acc = fmaf(p[t], c[t], acc);
            ge = warp_sum(acc);
            ge = __shfl_sync(FULL, ge, 0);
        }
        if (lane == 0) {
            jbest[i] = jb;
            dbest[i] = db;
            gbest[i] = gb;
            if (REF) gexact[i] = ge;
        }

        /* ---- 3. the exclusion test over the rest of the row -------------- */
        /* cnorm2[jb] is a uniform address into a k-element array: L1 */
        const RowTest t0 = row_test(jb, db, gb, ge, C6 ? cnorm2[jb] : 0.f,
                                    gfac, slack, C3, C6);
        int cnt = 0;
        if constexpr (IPT > 0) {
            /* gv and cn are already in registers: no memory traffic here */
#pragma unroll
            for (int t = 0; t < IPT; ++t) {
                const int j = lane + t * WARP;
                const bool live = (j < k && j != jb);
                const bool keep = live &&
                                  survives<C3, C6>(gv[t], cn[t], t0, factor, slack);
                cnt += __popc(__ballot_sync(FULL, keep));
#ifdef MPK_STATS
                const bool h3 = C3 && live &&
                    !survives<true, false>(gv[t], cn[t], t0, factor, slack);
                const bool h6 = C6 && live &&
                    !survives<false, true>(gv[t], cn[t], t0, factor, slack);
                n3 += __popc(__ballot_sync(FULL, h3));
                n6 += __popc(__ballot_sync(FULL, h6));
                nb += __popc(__ballot_sync(FULL, h3 && h6));
#endif
            }
        } else {
            for (int c0 = 0; c0 < k; c0 += WARP) {
                const int j = c0 + lane;
                bool keep = false;
                if (j < k && j != jb)
                    keep = survives<C3, C6>(g[j], cnorm2[j], t0, factor, slack);
                cnt += __popc(__ballot_sync(FULL, keep));
#ifdef MPK_STATS
                /* attribution needs each condition on its own, so it costs an
                 * extra evaluation; only compiled in for measurement runs */
                bool h3 = false, h6 = false;
                if (j < k && j != jb) {
                    h3 = C3 && !survives<true, false>(g[j], cnorm2[j], t0,
                                                      factor, slack);
                    h6 = C6 && !survives<false, true>(g[j], cnorm2[j], t0,
                                                      factor, slack);
                }
                n3 += __popc(__ballot_sync(FULL, h3));
                n6 += __popc(__ballot_sync(FULL, h6));
                nb += __popc(__ballot_sync(FULL, h3 && h6));
#endif
            }
        }
        /* a row the conditions cleared completely needs no high precision work
         * at all -- that is the point of the test */
        if (lane == 0) row_nnz[i] = (cnt && include_best) ? cnt + 1 : cnt;
    }

#ifdef MPK_STATS
    if (lane == 0) { sh3[w] = n3; sh6[w] = n6; shB[w] = nb; }
    __syncthreads();
    if (threadIdx.x == 0 && banks) {
        int a = 0, b = 0, cc = 0;
        for (int t = 0; t < WPB; ++t) { a += sh3[t]; b += sh6[t]; cc += shB[t]; }
        const int bank = blockIdx.x & (MPK_STAT_BANKS - 1);
        if (a)  atomicAdd(&banks[bank], (unsigned long long)a);
        if (b)  atomicAdd(&banks[MPK_STAT_BANKS + bank], (unsigned long long)b);
        if (cc) atomicAdd(&banks[2 * MPK_STAT_BANKS + bank],
                          (unsigned long long)cc);
    }
#else
    (void)banks;
#endif
}

__global__ void k_row_argmin_32(const float* __restrict__ G,
                                const float* __restrict__ cnorm2,
                                int n, int k, int* __restrict__ assign) {
    const int lane = threadIdx.x & (WARP - 1);
    const int i    = blockIdx.x * WPB + (threadIdx.x >> 5);
    if (i >= n) return;
    const float* g = G + (size_t)i * k;

    ArgMin mine = {INFINITY, k};
    for (int j = lane; j < k; j += WARP)
        mine = amin(mine, ArgMin{fmaf(-2.f, g[j], cnorm2[j]), j});
    ArgMin r = warp_argmin(mine);
    if (lane == 0) assign[i] = r.j;
}

/* ------------------------------------------------- CSR fill (condition) --- */

/* The second half of the exclusion test: with row_nnz scanned into row_ptr,
 * re-apply the predicate and write the survivor pattern.  Only the rows that
 * actually have survivors are touched, which is a small fraction of them. */
template <bool C3, bool C6>
__global__ void k_condition_fill(const float* __restrict__ G,
                                 const float* __restrict__ cnorm2,
                                 const int* __restrict__ jbest,
                                 const float* __restrict__ dbest,
                                 const float* __restrict__ gbest,
                                 const float* __restrict__ gexact,
                                 int n, int k,
                                 float factor, float gfac, float slack,
                                 const int* __restrict__ row_ptr,
                                 int* __restrict__ col,
                                 int* __restrict__ rowidx, int include_best) {
    const int lane = threadIdx.x & (WARP - 1);
    const int step = gridDim.x * WPB;

    for (int i = blockIdx.x * WPB + (threadIdx.x >> 5); i < n; i += step) {
        int base = row_ptr[i];
        if (base == row_ptr[i + 1]) continue;      /* nothing survived here */

        const int   jb = jbest[i];
        const RowTest t = row_test(jb, dbest[i], gbest[i],
                                   C6 ? gexact[i] : 0.f, cnorm2[jb],
                                   gfac, slack, C3, C6);

        if (include_best) {
            if (lane == 0) {
                col[base]    = jb;   /* (6) produced no value for the incumbent,
                                      * so it is refined with the survivors */
                rowidx[base] = i;
            }
            base += 1;
        }

        const float* g = G + (size_t)i * k;
        for (int c0 = 0; c0 < k; c0 += WARP) {
            const int j = c0 + lane;
            bool keep = false;
            if (j < k && j != jb)
                keep = survives<C3, C6>(g[j], cnorm2[j], t, factor, slack);
            const unsigned mk = __ballot_sync(FULL, keep);
            if (keep) {
                const int e = base + __popc(mk & ((1u << lane) - 1u));
                col[e]    = j;
                rowidx[e] = i;
            }
            base += __popc(mk);
        }
    }
}

/* ------------------------------------------------------ update kernel ---- */

/* One warp per surviving entry: the FP32 inner product cusparseSDDMM would
 * have computed.  Cheaper than SDDMM when the survivor count is small, since
 * SDDMM pays for descriptors, a bufferSize query and a preprocess pass every
 * iteration (the pattern changes every iteration). */
__global__ void k_update(const float* __restrict__ P, const float* __restrict__ C,
                         const int* __restrict__ col,
                         const int* __restrict__ rowidx,
                         int nnz, int d, float* __restrict__ val) {
    const int lane = threadIdx.x & (WARP - 1);
    const int e    = blockIdx.x * WPB + (threadIdx.x >> 5);
    if (e >= nnz) return;
    const float* p = P + (size_t)rowidx[e] * d;
    const float* c = C + (size_t)col[e]    * d;
    float acc = 0.f;
    for (int t = lane; t < d; t += WARP) acc = fmaf(p[t], c[t], acc);
    const float s = warp_sum(acc);
    if (lane == 0) val[e] = s;
}

/* ------------------------------------------------------ final argmin ----- */

/* One THREAD per row, not one warp: survivor rows hold a handful of entries
 * and the overwhelming majority hold none, so a warp per row left 31 of 32
 * lanes idle and needed 32x the blocks. */
__global__ void k_final_assign(const int* __restrict__ row_ptr,
                               const int* __restrict__ col,
                               const float* __restrict__ val,
                               const float* __restrict__ cnorm2,
                               const int* __restrict__ jbest,
                               const float* __restrict__ gexact,
                               int n, int* __restrict__ assign) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int jb = jbest[i];
    const int b = row_ptr[i], e = row_ptr[i + 1];
    if (b == e) { assign[i] = jb; return; }   /* proved optimal outright */

    /* With (6) on, the incumbent's high precision distance came from the
     * argmin kernel's dot; with (3) alone it sits in the row itself. */
    ArgMin mine = gexact ? ArgMin{fmaf(-2.f, gexact[i], cnorm2[jb]), jb}
                         : ArgMin{INFINITY, jb};
    for (int t = b; t < e; ++t) {
        const int j = col[t];
        mine = amin(mine, ArgMin{fmaf(-2.f, val[t], cnorm2[j]), j});
    }
    assign[i] = mine.j;
}

/* -------------------------------------------------------- verification --- */

#ifdef MPK_STATS
/* Recompute the whole row of ||p_i - c_j||^2 in FP64 and check
 *   (a) the true argmin survived the filter, and
 *   (b) the label the mixed pipeline produced is that argmin.
 * (a) failing means a condition of Theorem 1 was violated; (b) can also fail on
 * an exact tie or an FP32 tie flip, which is why they are counted apart. */
__global__ void k_verify64(const float* __restrict__ P, const float* __restrict__ C,
                           const int* __restrict__ row_ptr,
                           const int* __restrict__ col,
                           const int* __restrict__ jbest,
                           const int* __restrict__ assign,
                           int n, int d, int k,
                           unsigned long long* __restrict__ n_excluded_best,
                           unsigned long long* __restrict__ n_label_diff,
                           double* __restrict__ excess) {
    const int lane = threadIdx.x & (WARP - 1);
    const int i    = blockIdx.x * WPB + (threadIdx.x >> 5);
    if (i >= n) return;
    const float* p = P + (size_t)i * d;

    double dmin = INFINITY, dpick = 0.0;
    int    jmin = 0;
    const int a = assign[i];

    for (int j = 0; j < k; ++j) {
        const float* c = C + (size_t)j * d;
        double acc = 0.0;
        for (int t = lane; t < d; t += WARP) {
            const double dv = (double)p[t] - (double)c[t];
            acc = fma(dv, dv, acc);
        }
        acc = warp_sum(acc);
        acc = __shfl_sync(FULL, acc, 0);
        if (acc < dmin) { dmin = acc; jmin = j; }
        if (j == a) dpick = acc;
    }
    if (lane) return;

    bool survived = (jmin == jbest[i]);
    if (!survived) {
        for (int t = row_ptr[i]; t < row_ptr[i + 1]; ++t)
            if (col[t] == jmin) { survived = true; break; }
    }
    if (!survived) atomicAdd(n_excluded_best, 1ull);
    if (a != jmin) {
        atomicAdd(n_label_diff, 1ull);
        atomicAdd(excess, dpick - dmin);
    }
}
#endif /* MPK_STATS */

/* ---------------------------------------------------------- bookkeeping -- */

__global__ void k_count_diff(const int* __restrict__ a, const int* __restrict__ b,
                             int n, unsigned long long* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    unsigned long long c = 0;
    for (; i < n; i += stride) c += (a[i] != b[i]);
    c = warp_sum(c);
    if ((threadIdx.x & (WARP - 1)) == 0 && c) atomicAdd(out, c);
}

__global__ void k_zero(double* sums, int* counts, int k, int d) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long t = i; t < (long long)k * d; t += stride) sums[t] = 0.0;
    for (long long t = i; t < k; t += stride) counts[t] = 0;
}

__global__ void k_accumulate(const float* __restrict__ P,
                             const int* __restrict__ assign,
                             long long n, int d,
                             double* __restrict__ sums,
                             int* __restrict__ counts) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long t = idx; t < n * d; t += stride) {
        const long long i = t / d;
        const int f = (int)(t - i * d);
        atomicAdd(&sums[(size_t)assign[i] * d + f], (double)P[t]);
    }
    for (long long i = idx; i < n; i += stride) atomicAdd(&counts[assign[i]], 1);
}

__global__ void k_finalize(const double* __restrict__ sums,
                           const int* __restrict__ counts,
                           int k, int d, float* __restrict__ C,
                           float* __restrict__ moved2) {
    __shared__ float sh[NTHR];
    const int j = blockIdx.x;
    const int cnt = counts[j];
    float acc = 0.f;
    for (int t = threadIdx.x; t < d; t += blockDim.x) {
        const size_t o = (size_t)j * d + t;
        const float old = C[o];
        const float nv = (cnt > 0) ? (float)(sums[o] / (double)cnt) : old;
        C[o] = nv;
        const float dv = nv - old;
        acc = fmaf(dv, dv, acc);
    }
    float s = block_sum(acc, sh);
    if (threadIdx.x == 0) moved2[j] = s;
}

__global__ void k_inertia(const float* __restrict__ P, const float* __restrict__ C,
                          const int* __restrict__ assign, int n, int d,
                          double* __restrict__ out) {
    const int lane = threadIdx.x & (WARP - 1);
    const int i    = blockIdx.x * WPB + (threadIdx.x >> 5);
    double acc = 0.0;
    if (i < n) {
        const float* p = P + (size_t)i * d;
        const float* c = C + (size_t)assign[i] * d;
        for (int t = lane; t < d; t += WARP) {
            const double dv = (double)p[t] - (double)c[t];
            acc = fma(dv, dv, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0 && acc != 0.0) atomicAdd(out, acc);
}

}  /* anonymous namespace */

/* ------------------------------------------------------------ launchers -- */

static inline int grid1d(long long items) {
    long long b = (items + NTHR - 1) / NTHR;
    return (int)(b > 65535LL * 16 ? 65535LL * 16 : (b < 1 ? 1 : b));
}

void mpkLaunchToHalf(const float* src, __half* dst, long long n, cudaStream_t s) {
    k_to_half<<<grid1d(n), NTHR, 0, s>>>(src, dst, n);
}

void mpkLaunchRowNorms(const float* dC, int k, int d, float* out, cudaStream_t s) {
    k_row_norms<<<k, NTHR, 0, s>>>(dC, d, out);
}

/* Enough blocks to fill the device several times over, but not one per row:
 * at n = 200k the row-per-warp grids were 25k blocks, and for the kernels that
 * skip most rows the scheduling cost dominated the work. */
static inline int rowgrid(int n, int cap) {
    const int full = mpk_ceil_div(n, WPB);
    return full < cap ? (full < 1 ? 1 : full) : cap;
}

/* C3/C6 are compile time, so the configuration is dispatched here.  REF (the
 * condition (6) reference entry) is needed exactly when (6) is enabled. */
void mpkLaunchArgminCount(const float* G, const float* cnorm2, const float* dP,
                          const float* dC, int n, int d, int k,
                          float factor, float gfac, float slack,
                          int use_cond3, int use_cond6,
                          int* jbest, float* dbest, float* gbest, float* gexact,
                          int* row_nnz, int include_best, long long* stat_banks,
                          cudaStream_t s) {
    const int grid = rowgrid(n, 8192);
    unsigned long long* b = (unsigned long long*)stat_banks;
#define MPK_ARGMIN_LAUNCH(IPT, REF, C3, C6)                                   \
    k_argmin_count<IPT, REF, C3, C6><<<grid, NTHR, 0, s>>>(                   \
        G, cnorm2, dP, dC, n, d, k, factor, gfac, slack,                      \
        jbest, dbest, gbest, gexact, row_nnz, include_best, b)
#define MPK_ARGMIN_BY_COND(IPT)                                               \
    do {                                                                      \
        if (use_cond3 && use_cond6) MPK_ARGMIN_LAUNCH(IPT, true, true, true);  \
        else if (use_cond3)         MPK_ARGMIN_LAUNCH(IPT, false, true, false);\
        else                        MPK_ARGMIN_LAUNCH(IPT, true, false, true); \
    } while (0)
    /* Register residency pays only for a couple of columns per lane; past
     * k = 64 it measured slower than streaming the row.  IPT must cover the
     * whole row, so the cutoff is on ceil(k/32). */
    const int need = mpk_ceil_div(k, WARP);
    if      (need <= 1) MPK_ARGMIN_BY_COND(1);
    else if (need <= 2) MPK_ARGMIN_BY_COND(2);
    else                MPK_ARGMIN_BY_COND(0);
#undef MPK_ARGMIN_BY_COND
#undef MPK_ARGMIN_LAUNCH
}

void mpkLaunchRowArgmin32(const float* G32, const float* cnorm2, int n, int k,
                          int* assign, cudaStream_t s) {
    k_row_argmin_32<<<mpk_ceil_div(n, WPB), NTHR, 0, s>>>(G32, cnorm2, n, k, assign);
}

void mpkLaunchConditionFill(const float* G, const float* cnorm2, const int* jbest,
                            const float* dbest, const float* gbest,
                            const float* gexact, int n, int k, float factor,
                            float gfac, float slack, int use_cond3, int use_cond6,
                            const int* row_ptr, int* col, int* rowidx,
                            int include_best, cudaStream_t s) {
    /* touches only the rows that have survivors, so it wants a small grid */
    const int grid = rowgrid(n, 2048);
    if (use_cond3 && use_cond6)
        k_condition_fill<true, true><<<grid, NTHR, 0, s>>>(
            G, cnorm2, jbest, dbest, gbest, gexact, n, k, factor, gfac, slack,
            row_ptr, col, rowidx, include_best);
    else if (use_cond3)
        k_condition_fill<true, false><<<grid, NTHR, 0, s>>>(
            G, cnorm2, jbest, dbest, gbest, gexact, n, k, factor, gfac, slack,
            row_ptr, col, rowidx, include_best);
    else
        k_condition_fill<false, true><<<grid, NTHR, 0, s>>>(
            G, cnorm2, jbest, dbest, gbest, gexact, n, k, factor, gfac, slack,
            row_ptr, col, rowidx, include_best);
}

void mpkLaunchUpdate(const float* dP, const float* dC, const int* col,
                     const int* rowidx, int nnz, int d, float* val,
                     cudaStream_t s) {
    k_update<<<mpk_ceil_div(nnz, WPB), NTHR, 0, s>>>(
        dP, dC, col, rowidx, nnz, d, val);
}

void mpkLaunchFinalAssign(const int* row_ptr, const int* col, const float* val,
                          const float* cnorm2, const int* jbest,
                          const float* gexact, int n, int* assign, cudaStream_t s) {
    k_final_assign<<<mpk_ceil_div(n, NTHR), NTHR, 0, s>>>(
        row_ptr, col, val, cnorm2, jbest, gexact, n, assign);
}

void mpkLaunchCountDiff(const int* a, const int* b, int n, long long* out,
                        cudaStream_t s) {
    k_count_diff<<<mpk_ceil_div(n, NTHR), NTHR, 0, s>>>(
        a, b, n, (unsigned long long*)out);
}

void mpkLaunchZero(double* sums, int* counts, int k, int d, cudaStream_t s) {
    k_zero<<<grid1d((long long)k * d) , NTHR, 0, s>>>(sums, counts, k, d);
}

void mpkLaunchAccumulate(const float* dP, const int* assign, int n, int d,
                         double* sums, int* counts, cudaStream_t s) {
    long long total = (long long)n * d;
    long long blocks = (total + NTHR - 1) / NTHR;
    int grid = (int)(blocks > 8192 ? 8192 : (blocks < 1 ? 1 : blocks));
    k_accumulate<<<grid, NTHR, 0, s>>>(dP, assign, n, d, sums, counts);
}

void mpkLaunchFinalizeCentroids(const double* sums, const int* counts, int k,
                                int d, float* dC, float* moved2, cudaStream_t s) {
    k_finalize<<<k, NTHR, 0, s>>>(sums, counts, k, d, dC, moved2);
}

void mpkLaunchInertia(const float* dP, const float* dC, const int* assign, int n,
                      int d, double* out, cudaStream_t s) {
    k_inertia<<<mpk_ceil_div(n, WPB), NTHR, 0, s>>>(dP, dC, assign, n, d, out);
}

#ifdef MPK_STATS
void mpkLaunchVerify64(const float* dP, const float* dC, const int* row_ptr,
                       const int* col, const int* jbest, const int* assign,
                       int n, int d, int k, long long* n_excluded_best,
                       long long* n_label_diff, double* excess, cudaStream_t s) {
    k_verify64<<<mpk_ceil_div(n, WPB), NTHR, 0, s>>>(
        dP, dC, row_ptr, col, jbest, assign, n, d, k,
        (unsigned long long*)n_excluded_best, (unsigned long long*)n_label_diff,
        excess);
}
#endif
