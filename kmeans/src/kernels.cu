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
/* A row's running best as one 64 bit key, so the update kernel can reduce into
 * it with a single atomicMin and no per-row structure of any kind.
 *
 * The float goes in the high 32 bits under the standard order preserving map
 * (flip the sign bit for non-negatives, invert everything for negatives), so
 * unsigned comparison of the key matches float comparison of the distance --
 * distances here can be negative, since ||p||^2 is dropped.  The centroid
 * index sits in the low bits, so an exact tie is broken towards the smaller
 * index, which is what amin() does. */
__device__ __forceinline__ unsigned int mpk_ford(float f) {
    const unsigned int b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ unsigned long long mpk_pack(float dist, int j) {
    return ((unsigned long long)mpk_ford(dist) << 32) | (unsigned int)j;
}

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
 * 175 us/iteration at n=200k, d=128 against 35 us for the fused dot.
 *
 * GT is the storage type of G: float for an FP32-accumulate GEMM, __half for
 * an FP16-accumulate one.  The point of the FP16-accumulate path is a distance
 * matrix half the size, so G is read at its native width and widened to float
 * only in the register that holds it -- g_load() below -- never by a separate
 * conversion pass over the whole matrix.
 *
 * REFINE is the other axis: false means no exclusion test and no FP32
 * refinement at all -- the row argmin of the low precision G is trusted
 * outright and packed as the final answer.  That is a structurally different
 * kernel body (it returns right after step 1), not C3=C6=false, which still
 * runs the exclusion test and would flag every column a survivor. */
__device__ __forceinline__ float g_load(const float* g, int j) { return g[j]; }
__device__ __forceinline__ float g_load(const __half* g, int j) {
    return __half2float(g[j]);
}

template <typename GT, int IPT, bool REF, bool C3, bool C6, bool CASCADE, bool REFINE>
__global__ void k_argmin_count(const GT* __restrict__ G,
                               const float* __restrict__ cnorm2,
                               const float* __restrict__ P,
                               const float* __restrict__ C,
                               int n, int d, int k,
                               float factor, float gfac, float slack,
                               int* __restrict__ jbest,
                               float* __restrict__ dbest,
                               float* __restrict__ gbest,
                               float* __restrict__ gexact,
                               unsigned long long* __restrict__ bestpack,
                               int include_best,
                               int* __restrict__ list, int cap,
                               unsigned int* __restrict__ count,
                               unsigned long long* __restrict__ ref_count,
                               unsigned long long* __restrict__ banks) {
    using KV         = cub::KeyValuePair<int, float>;
    using WarpReduce = cub::WarpReduce<KV>;
    using SumReduce  = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage red[WPB];
    /* separate storage from red[]: both are shuffle based and so empty on this
     * arch, and keeping them apart avoids needing a __syncwarp between the two
     * reductions of a row */
    __shared__ typename SumReduce::TempStorage sred[WPB];
    /* how many rows of this block evaluated a reference entry.  Only the
     * cascade has to count: for (6) and (3)+(6) it is every row. */
    __shared__ int shref[WPB];
    int nref = 0;
    (void)nref; (void)shref;
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
        const GT* g = G + (size_t)i * k;

        /* ---- 1. row argmin of Dt(i,:) = cnorm2[:] - 2*G(i,:) ------------- */
        float gv[NREG];
        KV mine{k, INFINITY};
        if constexpr (IPT > 0) {
#pragma unroll
            for (int t = 0; t < IPT; ++t) {
                const int j = lane + t * WARP;
                gv[t] = (j < k) ? g_load(g, j) : 0.f;
                if (j < k)
                    mine = cub::ArgMin()(mine, KV{j, fmaf(-2.f, gv[t], cn[t])});
            }
        } else {
            for (int j = lane; j < k; j += WARP)
                mine = cub::ArgMin()(mine, KV{j, fmaf(-2.f, g_load(g, j), cnorm2[j])});
        }
        const KV r = WarpReduce(red[w]).Reduce(mine, cub::ArgMin());

        const int   jb = __shfl_sync(FULL, r.key, 0);
        const float db = __shfl_sync(FULL, r.value, 0);
        const float gb = g_load(g, jb);  /* uniform address, hits L1 */

        /* ---- 0. no refinement at all: the low precision argmin is the
         * final answer, packed straight from (db, jb).  Nothing past this
         * point runs -- no cascade, no reference entry, no exclusion test,
         * no survivor list -- so an unsafe/no-refinement config costs exactly
         * the row argmin and nothing else. */
        if constexpr (!REFINE) {
            if (lane == 0) {
                jbest[i]    = jb;
                dbest[i]    = db;
                gbest[i]    = gb;
                bestpack[i] = mpk_pack(db, jb);
            }
            continue;
        }

        /* ---- 2. cascade: condition (3) first, because it is free ---------
         * (3) needs no high precision quantity, so a row it clears completely
         * can be settled without ever touching P.  That read of P in FP32 is
         * what condition (6) actually costs -- it is the same number of bytes
         * the low precision GEMM moves in total -- so skipping it on the rows
         * (3) already handled is where the saving is.  The surviving rows then
         * go on to pay for their reference entry exactly as before, and the
         * set of exclusions is unchanged: a pair (3) rules out is ruled out by
         * (3) or (6) too. */
        bool want_ref = REF;
        if constexpr (CASCADE) {
            const RowTest t3 = row_test(jb, db, gb, 0.f, 0.f, gfac, slack,
                                        true, false);
            int c3 = 0;
            if constexpr (IPT > 0) {
#pragma unroll
                for (int t = 0; t < IPT; ++t) {
                    const int j    = lane + t * WARP;
                    const bool live = (j < k && j != jb);
                    const bool keep = live && survives<true, false>(
                                                  gv[t], cn[t], t3, factor, slack);
                    c3 += __popc(__ballot_sync(FULL, keep));
#ifdef MPK_STATS
                    n3 += __popc(__ballot_sync(FULL, live && !keep));
#endif
                }
            } else {
                for (int c0 = 0; c0 < k; c0 += WARP) {
                    const int j     = c0 + lane;
                    const bool live = (j < k && j != jb);
                    const bool keep = live && survives<true, false>(
                                                  g_load(g, j), cnorm2[j], t3, factor, slack);
                    c3 += __popc(__ballot_sync(FULL, keep));
#ifdef MPK_STATS
                    n3 += __popc(__ballot_sync(FULL, live && !keep));
#endif
                }
            }
            /* c3 comes from a ballot, so it is already warp uniform and the
             * reduction and the branch below stay convergent */
            want_ref = (c3 != 0);
        }

        /* ---- 3. the reference entry, when condition (6) needs one --------- */
        float ge = 0.f;
        if (REF && want_ref) {
            const float* p = P + (size_t)i * d;
            const float* c = C + (size_t)jb * d;
            float acc = 0.f;
            for (int t = lane; t < d; t += WARP) acc = fmaf(p[t], c[t], acc);
            ge = SumReduce(sred[w]).Sum(acc);
            ge = __shfl_sync(FULL, ge, 0);   /* CUB leaves the total in lane 0 */
        }
        /* only the cascade needs this counted: everywhere else every row
         * evaluates a reference entry, so the total is n by construction */
        if constexpr (CASCADE) { if (want_ref) ++nref; }
        if (lane == 0) {
            jbest[i] = jb;
            dbest[i] = db;
            gbest[i] = gb;
            /* A cascade row that (3) cleared has no reference entry.  Writing 0
             * rather than leaving it undefined keeps the filter's (6) term
             * finite; it can only read "(6) excluded nothing here", which is
             * exactly true, because (6) was never evaluated on that row. */
            if (REF) gexact[i] = want_ref ? ge : 0.f;
            /* Seed the row's best.  With (6) the incumbent's own FP32 distance
             * is already known, so it competes from the start; with (3) there
             * is none, so the incumbent is instead appended to the candidate
             * list by the filter and INF lets it win. */
            bestpack[i] = (REF && want_ref)
                        ? mpk_pack(fmaf(-2.f, ge, cnorm2[jb]), jb)
                        : mpk_pack(INFINITY, jb);
        }

        /* ---- 4. the exclusion test, emitting surviving flat indices ------
         * This stays in the argmin kernel because the row is already here:
         * with IPT > 0 it sits in registers and the test costs no traffic at
         * all, and even when streamed it is hot in L1.  Splitting it into its
         * own kernel measured slower for exactly one reason -- that kernel has
         * to read the n x k row of G a second time. */
        const RowTest t0 = row_test(jb, db, gb, ge, C6 ? cnorm2[jb] : 0.f,
                                    gfac, slack, C3, C6);
        constexpr int NM = IPT > 0 ? IPT : 1;
        unsigned int mm[NM];
        int cnt = 0;
        if (CASCADE && !want_ref) {
#pragma unroll
            for (int t = 0; t < NM; ++t) mm[t] = 0u;   /* (3) emptied the row */
        } else if constexpr (IPT > 0) {
#pragma unroll
            for (int t = 0; t < IPT; ++t) {
                const int j = lane + t * WARP;
                const bool live = (j < k && j != jb);
                const bool keep = live &&
                                  survives<C3, C6>(gv[t], cn[t], t0, factor, slack);
                mm[t] = __ballot_sync(FULL, keep);
                cnt += __popc(mm[t]);
#ifdef MPK_STATS
                const bool h3 = C3 && live &&
                    !survives<true, false>(gv[t], cn[t], t0, factor, slack);
                const bool h6 = C6 && live &&
                    !survives<false, true>(gv[t], cn[t], t0, factor, slack);
                if constexpr (!CASCADE) n3 += __popc(__ballot_sync(FULL, h3));
                n6 += __popc(__ballot_sync(FULL, h6));
                nb += __popc(__ballot_sync(FULL, h3 && h6));
#endif
            }
        } else {
            for (int c0 = 0; c0 < k; c0 += WARP) {
                const int j = c0 + lane;
                const bool live = (j < k && j != jb);
                const bool keep = live &&
                    survives<C3, C6>(g_load(g, j), cnorm2[j], t0, factor, slack);
                cnt += __popc(__ballot_sync(FULL, keep));
#ifdef MPK_STATS
                bool h3 = false, h6 = false;
                if (live) {
                    h3 = C3 && !survives<true, false>(g_load(g, j), cnorm2[j], t0,
                                                      factor, slack);
                    h6 = C6 && !survives<false, true>(g_load(g, j), cnorm2[j], t0,
                                                      factor, slack);
                }
                if constexpr (!CASCADE) n3 += __popc(__ballot_sync(FULL, h3));
                n6 += __popc(__ballot_sync(FULL, h6));
                nb += __popc(__ballot_sync(FULL, h3 && h6));
#endif
            }
        }

        /* with (3) alone the incumbent has no reference entry of its own, so a
         * row that kept anything has to price it alongside the survivors */
        if (include_best && cnt) ++cnt;
        if (!cnt) continue;

        unsigned int pos;
        if (lane == 0) pos = atomicAdd(count, (unsigned int)cnt);
        pos = __shfl_sync(FULL, pos, 0);

        if constexpr (IPT > 0) {
#pragma unroll
            for (int t = 0; t < IPT; ++t) {
                if (mm[t] & (1u << lane)) {
                    const unsigned int slot =
                        pos + __popc(mm[t] & ((1u << lane) - 1u));
                    if (slot < (unsigned int)cap)
                        list[slot] = i * k + (lane + t * WARP);
                }
                pos += __popc(mm[t]);
            }
        } else {
            /* second look at the row, straight out of L1 */
            for (int c0 = 0; c0 < k; c0 += WARP) {
                const int j = c0 + lane;
                const bool keep = (j < k && j != jb) &&
                    survives<C3, C6>(g_load(g, j), cnorm2[j], t0, factor, slack);
                const unsigned int m = __ballot_sync(FULL, keep);
                if (keep) {
                    const unsigned int slot =
                        pos + __popc(m & ((1u << lane) - 1u));
                    if (slot < (unsigned int)cap) list[slot] = i * k + j;
                }
                pos += __popc(m);
            }
        }
        if (include_best && lane == 0 && pos < (unsigned int)cap)
            list[pos] = i * k + jb;
    }

    if constexpr (CASCADE) if (ref_count) {
        if (lane == 0) shref[w] = nref;
        __syncthreads();
        if (threadIdx.x == 0) {
            int t = 0;
            for (int q = 0; q < WPB; ++q) t += shref[q];
            if (t) atomicAdd(ref_count, (unsigned long long)t);
        }
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

/* ------------------------------------------------------ the condition ----- */

/* Enumerate the flat indices i*k + j that survive the exclusion test.
 *
 * There is no CSR, no scan and no per-row structure: survivors go into one
 * flat append-only list.  The only thing that needs care is the append itself.
 * One atomicAdd per warp per row-chunk would put hundreds of thousands of
 * serialised atomics on a single counter, so slots are reserved once per block
 * per row-batch: each warp records its own chunk masks, the block sums the
 * per-warp counts, one thread claims the whole block's range, and the warps
 * then write into it with an intra-warp prefix from the same masks.  The masks
 * live in dynamic shared memory (ceil(k/32) words per warp), so the row is
 * tested once and read once. */
/* ------------------------------------------------ high precision update --- */

/* One warp per surviving flat index: recover (i, j), take the FP32 inner
 * product, and reduce it straight into that row's packed best.  No values are
 * stored and there is no separate selection pass. */
__global__ void k_update_flat(const float* __restrict__ P,
                              const float* __restrict__ C,
                              const float* __restrict__ cnorm2,
                              const int* __restrict__ list, int nnz,
                              int k, int d,
                              unsigned long long* __restrict__ bestpack) {
    const int lane = threadIdx.x & (WARP - 1);
    const int step = gridDim.x * WPB;
    for (int e = blockIdx.x * WPB + (threadIdx.x >> 5); e < nnz; e += step) {
        const int idx = list[e];
        const int i   = idx / k;
        const int j   = idx - i * k;
        const float* p = P + (size_t)i * d;
        const float* c = C + (size_t)j * d;
        float acc = 0.f;
        for (int t = lane; t < d; t += WARP) acc = fmaf(p[t], c[t], acc);
        const float dot = warp_sum(acc);
        if (lane == 0)
            atomicMin(&bestpack[i], mpk_pack(fmaf(-2.f, dot, cnorm2[j]), j));
    }
}

/* The label is just the low half of the packed best. */
__global__ void k_unpack(const unsigned long long* __restrict__ bestpack,
                         int n, int* __restrict__ assign) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) assign[i] = (int)(bestpack[i] & 0xffffffffull);
}

/* -------------------------------------------------------- verification --- */

#ifdef MPK_STATS
/* Oracle check.  The oracle itself is the ordinary FP32 implementation --
 * cublasSgemm plus k_row_argmin_32 on the same centroids -- so `ref` here is
 * the label the unfiltered FP32 algorithm assigns to row i.  This kernel only
 * scores the mixed run against it:
 *
 *   n_excluded_best  the FP32 answer was not even reachable: it was neither
 *                    the incumbent nor in the survivor list, so a condition of
 *                    Theorem 1 does not hold.  Must be 0.
 *   n_label_diff     it was reachable but the mixed run did not pick it.
 *   excess           what that cost, in the FP32 distances themselves.
 */
/* Reachability is now checked by re-evaluating the predicate rather than by
 * searching a survivor list: the flat list holds exactly the pairs for which
 * survives<C3,C6>() is true, so asking the test again about (i, r) answers
 * "was r in the list" exactly, without the list being sorted or indexed. */
template <typename GT, bool C3, bool C6>
__global__ void k_verify_ref(const GT* __restrict__ G,
                             const float* __restrict__ G32,
                             const float* __restrict__ cnorm2,
                             const int* __restrict__ jbest,
                             const float* __restrict__ dbest,
                             const float* __restrict__ gbest,
                             const float* __restrict__ gexact,
                             const int* __restrict__ assign,
                             const int* __restrict__ ref,
                             int n, int k, float factor, float gfac, float slack,
                             unsigned long long* __restrict__ n_excluded_best,
                             unsigned long long* __restrict__ n_label_diff,
                             double* __restrict__ excess) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int r  = ref[i];
    const int a  = assign[i];
    const int jb = jbest[i];

    bool reachable = (r == jb);
    if (!reachable) {
        const RowTest t0 = row_test(jb, dbest[i], gbest[i],
                                    C6 ? gexact[i] : 0.f,
                                    C6 ? cnorm2[jb] : 0.f,
                                    gfac, slack, C3, C6);
        reachable = survives<C3, C6>(g_load(G + (size_t)i * k, r), cnorm2[r], t0,
                                     factor, slack);
    }
    if (!reachable) atomicAdd(n_excluded_best, 1ull);

    if (a != r) {
        atomicAdd(n_label_diff, 1ull);
        const float* g = G32 + (size_t)i * k;
        atomicAdd(excess, (double)fmaf(-2.f, g[a], cnorm2[a])
                        - (double)fmaf(-2.f, g[r], cnorm2[r]));
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
 * condition (6) reference entry) is needed exactly when (6) is enabled.
 * GT is the storage type of G (see k_argmin_count).  refine == 0 selects the
 * no-refinement short circuit and ignores use_cond3/use_cond6/cascade. */
template <typename GT>
void mpkLaunchArgminCount(const GT* G, const float* cnorm2, const float* dP,
                          const float* dC, int n, int d, int k,
                          float factor, float gfac, float slack,
                          int use_cond3, int use_cond6, int cascade, int refine,
                          int* jbest, float* dbest, float* gbest, float* gexact,
                          unsigned long long* bestpack, int include_best,
                          int* list, int cap, unsigned int* count,
                          unsigned long long* ref_count,
                          long long* stat_banks, cudaStream_t s) {
    const int grid = rowgrid(n, 8192);
    unsigned long long* b = (unsigned long long*)stat_banks;
#define MPK_ARGMIN_LAUNCH(IPT, REF, C3, C6, CAS, RFN)                         \
    k_argmin_count<GT, IPT, REF, C3, C6, CAS, RFN><<<grid, NTHR, 0, s>>>(      \
        G, cnorm2, dP, dC, n, d, k, factor, gfac, slack,                      \
        jbest, dbest, gbest, gexact, bestpack, include_best, list, cap,       \
        count, ref_count, b)
#define MPK_ARGMIN_BY_COND(IPT)                                               \
    do {                                                                      \
        if (!refine) {                                                        \
            MPK_ARGMIN_LAUNCH(IPT, false, false, false, false, false);        \
        } else if (use_cond3 && use_cond6) {                                  \
            if (cascade) MPK_ARGMIN_LAUNCH(IPT, true, true, true, true, true); \
            else         MPK_ARGMIN_LAUNCH(IPT, true, true, true, false, true);\
        }                                                                     \
        else if (use_cond3) MPK_ARGMIN_LAUNCH(IPT, false, true, false, false, true);\
        else                MPK_ARGMIN_LAUNCH(IPT, true, false, true, false, true); \
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
template void mpkLaunchArgminCount<float>(const float*, const float*, const float*,
    const float*, int, int, int, float, float, float, int, int, int, int,
    int*, float*, float*, float*, unsigned long long*, int, int*, int,
    unsigned int*, unsigned long long*, long long*, cudaStream_t);
template void mpkLaunchArgminCount<__half>(const __half*, const float*, const float*,
    const float*, int, int, int, float, float, float, int, int, int, int,
    int*, float*, float*, float*, unsigned long long*, int, int*, int,
    unsigned int*, unsigned long long*, long long*, cudaStream_t);

void mpkLaunchRowArgmin32(const float* G32, const float* cnorm2, int n, int k,
                          int* assign, cudaStream_t s) {
    k_row_argmin_32<<<mpk_ceil_div(n, WPB), NTHR, 0, s>>>(G32, cnorm2, n, k, assign);
}

void mpkLaunchUpdateFlat(const float* dP, const float* dC, const float* cnorm2,
                         const int* list, int nnz, int k, int d,
                         unsigned long long* bestpack, cudaStream_t s) {
    const int grid = rowgrid(nnz, 8192);
    k_update_flat<<<grid, NTHR, 0, s>>>(dP, dC, cnorm2, list, nnz, k, d, bestpack);
}

void mpkLaunchUnpack(const unsigned long long* bestpack, int n, int* assign,
                     cudaStream_t s) {
    k_unpack<<<mpk_ceil_div(n, NTHR), NTHR, 0, s>>>(bestpack, n, assign);
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
/* GT matches whatever mpkLaunchArgminCount ran with -- G here is the mixed
 * run's own distance matrix, not the always-FP32 oracle G32. */
template <typename GT>
void mpkLaunchVerifyRef(const GT* G, const float* G32, const float* cnorm2,
                        const int* jbest, const float* dbest, const float* gbest,
                        const float* gexact, const int* assign, const int* ref,
                        int n, int k, float factor, float gfac, float slack,
                        int use_cond3, int use_cond6,
                        long long* n_excluded_best, long long* n_label_diff,
                        double* excess, cudaStream_t s) {
    const int grid = mpk_ceil_div(n, NTHR);
    unsigned long long* eb = (unsigned long long*)n_excluded_best;
    unsigned long long* ld = (unsigned long long*)n_label_diff;
#define MPK_VERIFY_LAUNCH(C3, C6)                                             \
    k_verify_ref<GT, C3, C6><<<grid, NTHR, 0, s>>>(                           \
        G, G32, cnorm2, jbest, dbest, gbest, gexact, assign, ref, n, k,       \
        factor, gfac, slack, eb, ld, excess)
    /* refine==false runs (see k_argmin_count) still call this with
     * use_cond3 == use_cond6 == 0, which is a real request here: it means
     * "no exclusion test", not "condition (6) alone" -- survives<false,false>
     * is trivially true, so n_excluded_best stays vacuously 0 while
     * n_label_diff/excess still score the low precision argmin for real. */
    if (use_cond3 && use_cond6) MPK_VERIFY_LAUNCH(true, true);
    else if (use_cond3)         MPK_VERIFY_LAUNCH(true, false);
    else if (use_cond6)         MPK_VERIFY_LAUNCH(false, true);
    else                        MPK_VERIFY_LAUNCH(false, false);
#undef MPK_VERIFY_LAUNCH
}
template void mpkLaunchVerifyRef<float>(const float*, const float*, const float*,
    const int*, const float*, const float*, const float*, const int*, const int*,
    int, int, float, float, float, int, int,
    long long*, long long*, double*, cudaStream_t);
template void mpkLaunchVerifyRef<__half>(const __half*, const float*, const float*,
    const int*, const float*, const float*, const float*, const int*, const int*,
    int, int, float, float, float, int, int,
    long long*, long long*, double*, cudaStream_t);
#endif
