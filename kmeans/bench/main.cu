/* Driver: gaussian blobs -> mixed precision K-Means vs FP32 K-Means.
 *
 * Conditions (3) and (6) are implemented as separate configurations and run
 * side by side, because they cost different things:
 *
 *   (3)      needs no high precision quantity to be evaluated, so none is
 *            computed.  A row it clears completely costs nothing in high
 *            precision.  A row it does not clear pays for its own incumbent.
 *   (6)      needs one high precision distance per row -- the incumbent's --
 *            whether or not it ends up excluding anything.
 *   (3)+(6)  pays for the reference entry and excludes with whichever fires.
 */
#include "mpkmeans/mpkmeans.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHK(call)                                                             \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d CUDA: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(e_));                                  \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static void usage(const char* p) {
    printf("usage: %s [options]\n"
           "  -n <int>       points                     (default 200000)\n"
           "  -d <int>       features                   (default 64)\n"
           "  -k <int>       clusters                   (default 32)\n"
           "  -s <float>     blob standard deviation    (default 1.0)\n"
           "  -b <float>     center box half width      (default 10.0)\n"
           "  -i <int>       max iterations             (default 50)\n"
           "  -e <int>       rng seed                   (default 1)\n"
           "  -r <int>       timed repeats, last wins   (default 1)\n"
           "  --only <c>     run one config only: 3, 6 or 36\n"
           "  --sddmm-min <n> survivors needed to use cusparseSDDMM\n"
           "                 (default: auto, from n and d)\n"
           "  --force-sddmm  always use cusparseSDDMM\n"
           "  --force-warp   always use the warp fallback update kernel\n"
           "  --no-verify    skip the per-iteration FP64 oracle check\n"
           "                 (the oracle needs a -DMPK_STATS=ON build; it is\n"
           "                  on by default there and absent otherwise)\n"
           "  --csv          one machine readable line per config\n"
           "  --csv-header   print the CSV header and exit\n"
           "  -v             verbose per-iteration log\n", p);
}

/* Fraction of points whose (mixed, fp32) cluster labels agree after matching
 * the two labelings greedily by contingency-table mass. */
static double label_agreement(const std::vector<int>& a, const std::vector<int>& b,
                              int k) {
    std::vector<long long> cont((size_t)k * k, 0);
    for (size_t i = 0; i < a.size(); ++i) cont[(size_t)a[i] * k + b[i]]++;
    std::vector<char> used_a(k, 0), used_b(k, 0);
    long long matched = 0;
    for (int step = 0; step < k; ++step) {
        long long best = -1; int bi = -1, bj = -1;
        for (int i = 0; i < k; ++i) {
            if (used_a[i]) continue;
            for (int j = 0; j < k; ++j) {
                if (used_b[j]) continue;
                if (cont[(size_t)i * k + j] > best) {
                    best = cont[(size_t)i * k + j]; bi = i; bj = j;
                }
            }
        }
        if (bi < 0) break;
        used_a[bi] = used_b[bj] = 1;
        matched += best;
    }
    return (double)matched / (double)a.size();
}

struct Config {
    const char* name;
    int cond3, cond6;
};

static const Config kConfigs[3] = {
    {"(3)",     1, 0},
    {"(6)",     0, 1},
    {"(3)+(6)", 1, 1},
};

int main(int argc, char** argv) {
    int   n = 200000, d = 64, k = 32, max_iter = 50, seed = 1, repeats = 1;
    float blob_std = 1.0f, box = 10.0f;
    int   verify = 1, verbose = 0, csv = 0, only = -1;
    int   sddmm_min = MPK_SDDMM_AUTO;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto next = [&](void) -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if (!strcmp(a, "-n")) n = atoi(next());
        else if (!strcmp(a, "-d")) d = atoi(next());
        else if (!strcmp(a, "-k")) k = atoi(next());
        else if (!strcmp(a, "-s")) blob_std = (float)atof(next());
        else if (!strcmp(a, "-b")) box = (float)atof(next());
        else if (!strcmp(a, "-i")) max_iter = atoi(next());
        else if (!strcmp(a, "-e")) seed = atoi(next());
        else if (!strcmp(a, "-r")) repeats = atoi(next());
        else if (!strcmp(a, "--no-verify")) verify = 0;
        else if (!strcmp(a, "--sddmm-min")) sddmm_min = atoi(next());
        else if (!strcmp(a, "--force-sddmm")) sddmm_min = 0;
        else if (!strcmp(a, "--force-warp")) sddmm_min = INT32_MAX;
        else if (!strcmp(a, "--only")) {
            const char* v = next();
            only = !strcmp(v, "3") ? 0 : !strcmp(v, "6") ? 1 :
                   !strcmp(v, "36") ? 2 : -1;
            if (only < 0) { usage(argv[0]); return 1; }
        }
        else if (!strcmp(a, "-v")) verbose = 1;
        else if (!strcmp(a, "--csv")) csv = 1;
        else if (!strcmp(a, "--csv-header")) {
            printf("n,d,k,std,seed,cond,iters,eps,"
                   "hp_baseline,hp_reference,hp_update,hp_total,pct_eliminated,"
                   "pct_reference,pct_update,pct_cond3,pct_cond6,pct_cond3_only,"
                   "pct_cond6_only,violations,label_diff,inertia,inertia_fp32,"
                   "rel_inertia,ms_dist,ms_dist_fp32,speedup,ms_prep,ms_gemm,"
                   "ms_argmin,ms_filter,ms_setup,ms_hpupdate,"
                   "ms_assign,iters_sddmm,iters_warp\n");
            return 0;
        }
        else { usage(argv[0]); return 1; }
    }
    if (n <= 0 || d <= 0 || k <= 0 || k > n) { usage(argv[0]); return 1; }

    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    const double eps = mpkEpsilon(d);
    if (eps < 0) {
        fprintf(stderr, "error model degenerate for d=%d in this mode\n", d);
        return 1;
    }
    if (!csv) {
        printf("device   : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        printf("problem  : n=%d d=%d k=%d  blobs(std=%.3g, box=%.3g, seed=%d)\n",
               n, d, k, blob_std, box, seed);
        printf("gemm     : FP16 operands, FP32 accumulate\n");
        printf("eps      : %.6e   ->  factor 2*eps/(1-eps) = %.6e\n",
               eps, 2.0 * eps / (1.0 - eps));
        if (sddmm_min == MPK_SDDMM_AUTO)
            printf("update   : SDDMM/warp crossover auto (~%.0f survivors per "
                   "row for d=%d)\n",
                   fmin(24.0, fmax(4.0, 10.0 * sqrt((double)d / 128.0))), d);
        else
            printf("update   : cusparseSDDMM when survivors >= %d, else warp "
                   "kernel\n", sddmm_min);
    }

    /* ------------------------------------------------------------ data --- */
    std::vector<float> hP((size_t)n * d);
    std::vector<int>   hTruth(n);
    mpkMakeBlobs(n, d, k, blob_std, box, (unsigned)seed, hP.data(), hTruth.data());

    float *dP = nullptr, *dCmix = nullptr, *dCref = nullptr, *dC0 = nullptr;
    int   *dAmix = nullptr, *dAref = nullptr;
    CHK(cudaMalloc(&dP,    (size_t)n * d * sizeof(float)));
    CHK(cudaMalloc(&dCmix, (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dCref, (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dC0,   (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dAmix, (size_t)n * sizeof(int)));
    CHK(cudaMalloc(&dAref, (size_t)n * sizeof(int)));
    CHK(cudaMemcpy(dP, hP.data(), (size_t)n * d * sizeof(float),
                   cudaMemcpyHostToDevice));

    /* preprocessing required by conditions (3) and (6) */
    float shift = 0.f;
    if (mpkShiftNonNegative(dP, n, d, &shift) != MPK_OK) return 1;
    if (!csv) printf("shift    : P += %.6g (all entries non-negative)\n", shift);

    if (mpkInitRandomPoints(dP, n, d, k, (unsigned)seed, dC0) != MPK_OK) return 1;

    cublasHandle_t blas;   cublasCreate(&blas);
    cusparseHandle_t sparse; cusparseCreate(&sparse);

    mpkParams par; mpkParamsInit(&par);
    par.max_iter      = max_iter;
    par.sddmm_min_nnz = sddmm_min;
    par.verbose       = verbose;

    mpkStats smix[3], sref;
    memset(smix, 0, sizeof(smix));
    memset(&sref, 0, sizeof(sref));

    /* warm up cuBLAS/cuSPARSE so the first timed run is not charged for it */
    {
        mpkParams w = par; w.max_iter = 1; w.verify = 0; w.verbose = 0;
        mpkStats junk;
        CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansMixed(blas, sparse, dP, n, d, k, dCmix, dAmix, &w, &junk);
        CHK(cudaMemcpy(dCref, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansFP32(blas, dP, n, d, k, dCref, dAref, &w, &junk);
    }

    for (int r = 0; r < repeats; ++r) {
        CHK(cudaMemcpy(dCref, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        par.verify = 0;
        if (mpkMeansFP32(blas, dP, n, d, k, dCref, dAref, &par, &sref)
            != MPK_OK) { fprintf(stderr, "fp32 failed\n"); return 1; }
    }

    std::vector<int> aref(n);
    CHK(cudaMemcpy(aref.data(), dAref, (size_t)n * sizeof(int),
                   cudaMemcpyDeviceToHost));

    int ok = 1;
    std::vector<std::vector<int>> amix(3, std::vector<int>(n));
    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        par.use_cond3 = kConfigs[c].cond3;
        par.use_cond6 = kConfigs[c].cond6;
        for (int r = 0; r < repeats; ++r) {
            CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                           cudaMemcpyDeviceToDevice));
            par.verify = (r == repeats - 1) ? verify : 0;
            if (mpkMeansMixed(blas, sparse, dP, n, d, k, dCmix, dAmix, &par,
                              &smix[c]) != MPK_OK) {
                fprintf(stderr, "mixed %s failed\n", kConfigs[c].name);
                return 1;
            }
        }
        CHK(cudaMemcpy(amix[c].data(), dAmix, (size_t)n * sizeof(int),
                       cudaMemcpyDeviceToHost));
    }

    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        const double base = (double)(S.hp_baseline ? S.hp_baseline : 1);
        const long long hp = S.hp_reference + S.hp_update;
        const double tb = (double)(S.tested ? S.tested : 1);
        const double rel = fabs(S.inertia - sref.inertia) /
                           fmax(sref.inertia, 1e-300);
        const int cok = (!verify || !S.stats_built || S.verify_excluded_best == 0)
                        && rel < 1e-4;
        ok = ok && cok;

        if (csv) {
            printf("%d,%d,%d,%g,%d,%s,%d,%.6e,"
                   "%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,"
                   "%.6f,%.6f,%.6f,%.6f,%lld,%lld,"
                   "%.9e,%.9e,%.3e,%.3f,%.3f,%.4f,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d\n",
                   n, d, k, blob_std, seed, kConfigs[c].name, S.iters, eps,
                   S.hp_baseline, S.hp_reference, S.hp_update, hp,
                   100.0 * (1.0 - (double)hp / base),
                   100.0 * (double)S.hp_reference / base,
                   100.0 * (double)S.hp_update / base,
                   100.0 * (double)S.excl_cond3 / tb,
                   100.0 * (double)S.excl_cond6 / tb,
                   100.0 * (double)(S.excl_cond3 - S.excl_both) / tb,
                   100.0 * (double)(S.excl_cond6 - S.excl_both) / tb,
                   S.verify_excluded_best, S.verify_label_diff,
                   S.inertia, sref.inertia, rel,
                   S.t_dist_ms, sref.t_dist_ms,
                   sref.t_dist_ms / fmax(S.t_dist_ms, 1e-9),
                   S.t_prep_ms, S.t_gemm_lo_ms, S.t_argmin_ms,
                   S.t_filter_ms, S.t_sddmm_setup_ms, S.t_hp_update_ms,
                   S.t_assign_ms, S.iters_sddmm, S.iters_fallback);
        }
    }
    if (csv) {
        cusparseDestroy(sparse); cublasDestroy(blas);
        cudaFree(dP); cudaFree(dCmix); cudaFree(dCref); cudaFree(dC0);
        cudaFree(dAmix); cudaFree(dAref);
        return ok ? 0 : 2;
    }

    /* ------------------------------------------- high precision work ----- */
    printf("\nhigh precision distance evaluations PER ITERATION\n");
    printf("  plain Lloyd's evaluates every entry of D: n*k = %lld per iteration\n",
           (long long)n * k);
    printf("  %-8s %6s %14s %14s %14s %11s\n",
           "cond", "iters", "reference", "survivors", "total", "ELIMINATED");
    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        const double it = (double)(S.iters ? S.iters : 1);
        const double base = (double)n * k;
        const double ref = (double)S.hp_reference / it;
        const double sur = (double)S.hp_update / it;
        printf("  %-8s %6d %14.0f %14.0f %14.0f %10.4f%%\n",
               kConfigs[c].name, S.iters, ref, sur, ref + sur,
               100.0 * (1.0 - (ref + sur) / base));
    }
    printf("  reference = the incumbent's own FP32 distance, which condition (6)\n"
           "  needs before it can exclude anything: n per iteration.  Condition\n"
           "  (3) needs no such quantity, so it computes none; the incumbent is\n"
           "  instead counted among the survivors, and only for the rows (3) did\n"
           "  not clear outright.\n");

    {
        int it_min = 0, it_max = 0;
        for (int c = 0; c < 3; ++c) {
            if (only >= 0 && only != c) continue;
            const int v = smix[c].iters;
            if (!it_min || v < it_min) it_min = v;
            if (v > it_max) it_max = v;
        }
        if (it_max != it_min) {
            printf("\n  NOTE: the configurations converged in different numbers of\n"
                   "  iterations (%d..%d).  Per-iteration numbers are comparable;\n"
                   "  run totals are not.  Totals over the whole run:\n", it_min, it_max);
            for (int c = 0; c < 3; ++c) {
                if (only >= 0 && only != c) continue;
                const mpkStats& S = smix[c];
                printf("    %-8s %d iters:  %lld of %lld\n", kConfigs[c].name,
                       S.iters, S.hp_reference + S.hp_update, S.hp_baseline);
            }
        }
    }

    if (smix[0].stats_built) {
        printf("\nexclusion attribution, over the n*(k-1) tested pairs\n");
        printf("  %-8s %10s %10s %10s %10s %10s\n",
               "cond", "(3) held", "(6) held", "both", "(3) only", "(6) only");
        for (int c = 0; c < 3; ++c) {
            if (only >= 0 && only != c) continue;
            const mpkStats& S = smix[c];
            const double tb = (double)(S.tested ? S.tested : 1);
            printf("  %-8s %9.4f%% %9.4f%% %9.4f%% %9.4f%% %9.4f%%\n",
                   kConfigs[c].name,
                   100.0 * (double)S.excl_cond3 / tb,
                   100.0 * (double)S.excl_cond6 / tb,
                   100.0 * (double)S.excl_both / tb,
                   100.0 * (double)(S.excl_cond3 - S.excl_both) / tb,
                   100.0 * (double)(S.excl_cond6 - S.excl_both) / tb);
        }
    } else {
        printf("\nexclusion attribution: not collected "
               "(rebuild with -DMPK_STATS=ON)\n");
    }

    /* ------------------------------------------------------- timing ------ */
    printf("\ntiming, ms over the whole run, centroid update excluded\n");
    printf("  %-8s %5s %7s %8s %8s %8s %7s %7s %9s %8s\n",
           "cond", "iters", "prep", "gemm", "argmin", "filter",
           "setup", "update", "assign", "TOTAL");
    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        printf("  %-8s %5d %7.2f %8.2f %8.2f %8.2f %7.2f %7.2f %9.2f %8.2f\n",
               kConfigs[c].name, S.iters, S.t_prep_ms, S.t_gemm_lo_ms,
               S.t_argmin_ms, S.t_filter_ms, S.t_sddmm_setup_ms,
               S.t_hp_update_ms, S.t_assign_ms, S.t_dist_ms);
    }
    printf("  %-8s %5d %7.2f %8.2f %8s %8s %7s %7s %9.2f %8.2f\n",
           "fp32", sref.iters, sref.t_prep_ms, sref.t_gemm_lo_ms, "-", "-",
           "-", "-", sref.t_assign_ms, sref.t_dist_ms);

    printf("\n  %-8s %10s %10s %8s   %s\n", "cond", "ms/iter", "vs fp32",
           "update", "(centroid update, excluded above)");
    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        const double per = S.t_dist_ms / fmax(S.iters, 1);
        const double rper = sref.t_dist_ms / fmax(sref.iters, 1);
        printf("  %-8s %10.3f %9.3fx %8.2f   sddmm iters %d, warp iters %d\n",
               kConfigs[c].name, per, rper / fmax(per, 1e-9), S.t_update_ms,
               S.iters_sddmm, S.iters_fallback);
    }
    printf("  %-8s %10.3f %9s %8.2f\n", "fp32",
           sref.t_dist_ms / fmax(sref.iters, 1), "1.000x", sref.t_update_ms);

    /* ---------------------------------------------------- verification --- */
    /* Two separate claims, and they are not the same thing:
     *
     *   (a) the filter never discards the true nearest centroid.  That is what
     *       Theorem 1 asserts, and it must hold exactly -- any nonzero count
     *       is a broken bound.
     *   (b) the label finally emitted is the true nearest centroid.  This is
     *       (a) plus the final argmin picking the right one among what
     *       survived, and it can differ on exact ties and on FP32 tie flips
     *       without the bound being wrong at all.  `excess` is what those
     *       cost: sum over disagreeing rows of D(assigned) - D(nearest). */
    printf("\nverification");
    if (verify && smix[0].stats_built) {
        printf("  -- FP64 oracle, recomputed for every row of every iteration\n");
        printf("  %-8s %20s %22s %12s\n", "cond", "bound violations",
               "label != true argmin", "excess");
        for (int c = 0; c < 3; ++c) {
            if (only >= 0 && only != c) continue;
            const mpkStats& S = smix[c];
            const long long rows = (long long)n * S.iters;
            printf("  %-8s %9lld / %-8lld %9lld / %-8lld %12.4e\n",
                   kConfigs[c].name, S.verify_excluded_best, rows,
                   S.verify_label_diff, rows, S.verify_excess);
        }
        printf("  bound violations must be 0: the true nearest centroid was\n"
               "  neither the incumbent nor among the survivors, so a condition\n"
               "  of Theorem 1 does not hold.  label != true argmin is not a\n"
               "  violation; excess is the total distance those rows gave up.\n");
    } else if (verify) {
        printf("  -- SKIPPED: this binary was built without MPK_STATS, so the\n"
               "  FP64 oracle is compiled out.  Rebuild with -DMPK_STATS=ON.\n");
    } else {
        printf("  -- SKIPPED: --no-verify\n");
    }

    printf("\nclustering\n");
    for (int c = 0; c < 3; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        long long identical = 0;
        for (int i = 0; i < n; ++i) identical += (amix[c][i] == aref[i]);
        const double rel = fabs(S.inertia - sref.inertia) /
                           fmax(sref.inertia, 1e-300);
        printf("  %-8s inertia rel diff %.3e   labels == fp32 %.3f%%   "
               "truth %.4f\n", kConfigs[c].name, rel,
               100.0 * (double)identical / n,
               label_agreement(amix[c], hTruth, k));
    }
    printf("  %-8s %54s truth %.4f\n", "fp32", "",
           label_agreement(aref, hTruth, k));

    /* The theorem guarantees the true nearest centroid is never filtered out.
     * It does not guarantee bit-identical labels to an FP32 run: once an FP32
     * tie flips, the two trajectories separate.  So gate on the violation
     * counter and on the clustering quality, not on label equality. */
    printf("\n%s\n", ok ? "PASS" : "CHECK FAILED");

    cusparseDestroy(sparse);
    cublasDestroy(blas);
    cudaFree(dP); cudaFree(dCmix); cudaFree(dCref); cudaFree(dC0);
    cudaFree(dAmix); cudaFree(dAref);
    return ok ? 0 : 2;
}
