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
           "  --maxiters <int>  same as -i\n"
           "  -e <int>       rng seed                   (default 1)\n"
           "  -r <int>       timed repeats, last wins   (default 1)\n"
           "  --dataset <f>  LIBSVM file instead of blobs; n and d come from\n"
           "                 the file and -n/-d are ignored\n"
           "  --zscore       z-score normalize every feature of P before any\n"
           "                 scheme sees it, as arXiv:2407.12208 does (their\n"
           "                 Algorithm 5.1 line 1).  Off by default.  Changes\n"
           "                 the SSE, since it rescales the space; a constant\n"
           "                 feature is left mean centred, not divided by 0\n"
           "  --convergence  iterate to convergence rather than a fixed count:\n"
           "                 stop at max_j ||c_j - c_j_prev||_2 < tol, or at\n"
           "                 --maxiters, whichever comes first\n"
           "  --tol <float>  the tolerance above         (default 1e-8)\n"
           "  --theta <f>    rt-base safety factor of (4.13), must be > 2\n"
           "                 (default 2.5; the paper names this constant two\n"
           "                  inconsistent ways -- see the file comment at the\n"
           "                  top of src/baseline.cu for which was picked)\n"
           "  --accum <a>    low precision GEMM accumulator: fp32 or fp16\n"
           "                 (default fp32; mpkMeansMixed configs only --\n"
           "                  rt-base and fp32 are unaffected)\n"
           "  --gather-frac <f>  multiword only: refine narrowly while at most\n"
           "                 this fraction of rows is undecided, else densely\n"
           "                 (default 0.35)\n"
           "  --macs <int>   multiword only: how many of the three fp16 cross\n"
           "                 products the refinement cascade may reach, 1..3\n"
           "                 (default 3 = double-fp16; a stage is only issued\n"
           "                  if the previous exclusion test left survivors)\n"
           "  --only <c>     run one config only: 3, 6, 36, c (cascade),\n"
           "                 rt (arXiv:2407.12208 baseline), raw (no exclusion\n"
           "                 test, no FP32 refinement -- the low precision\n"
           "                 argmin trusted outright, no correctness guarantee),\n"
           "                 mw (multiword double-fp16)\n"
           "  --no-verify    skip the per-iteration FP64 oracle check\n"
           "                 (the oracle needs a -DMPK_STATS=ON build; it is\n"
           "                  on by default there and absent otherwise)\n"
           "  --csv          one machine readable line per config\n"
           "  --csv-header   print the CSV header and exit\n"
           "  -v             verbose per-iteration log\n", p);
}

/* Fraction of points whose (mixed, fp32) cluster labels agree after matching
 * the two labelings greedily by contingency-table mass. */
/* ka and kb need not be equal: on a real dataset the label column has however
 * many classes it has, which is rarely the k we cluster into.  The greedy
 * matching then runs for min(ka, kb) steps and the rest counts as disagreement,
 * so the number is only comparable across configurations, never an accuracy. */
static double label_agreement(const std::vector<int>& a, const std::vector<int>& b,
                              int ka, int kb) {
    std::vector<long long> cont((size_t)ka * kb, 0);
    for (size_t i = 0; i < a.size(); ++i) cont[(size_t)a[i] * kb + b[i]]++;
    std::vector<char> used_a(ka, 0), used_b(kb, 0);
    long long matched = 0;
    const int steps = ka < kb ? ka : kb;
    for (int step = 0; step < steps; ++step) {
        long long best = -1; int bi = -1, bj = -1;
        for (int i = 0; i < ka; ++i) {
            if (used_a[i]) continue;
            for (int j = 0; j < kb; ++j) {
                if (used_b[j]) continue;
                if (cont[(size_t)i * kb + j] > best) {
                    best = cont[(size_t)i * kb + j]; bi = i; bj = j;
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
    int cond3, cond6, cascade;
    int driver;   /* 0 mpkMeansMixed, 1 mpkMeansBaselineRT, 2 mpkMeansMultiword */
};

/* (3)->(6) is the cascade: the same exclusions as (3)+(6), but the reference
 * entry -- and with it the FP32 read of P, which is what (6) costs -- is
 * evaluated only for the rows (3) did not clear outright. */
/* rt-base is the arXiv:2407.12208 reliability-test scheme, a different
 * strategy entirely (per-entry cancellation test, not cluster exclusion), run
 * here as a baseline. */
/* raw asks mpkMeansMixed for cond3 = cond6 = 0, which used to be an error and
 * is now the no-refinement mode: the low precision argmin is trusted outright,
 * with no exclusion test and no FP32 recomputation at all.  It has no
 * correctness guarantee -- it is the naive scheme Theorem 1 exists to make
 * safe -- and is here so that guarantee has something to be measured against. */
/* mw is the multiword (double-fp16) scheme: the same conditions (3)+(6), but
 * over a distance matrix assembled one fp16 cross product at a time, with the
 * test re-run between products to decide whether the next one is needed.  Its
 * refinement is another tensor-core GEMM rather than a warp-per-entry FP32
 * dot, so the number to watch for it is cross products per iteration, not
 * entries refined. */
#define MPK_NCFG 7
static const Config kConfigs[MPK_NCFG] = {
    {"(3)",     1, 0, 0, 0},
    {"(6)",     0, 1, 0, 0},
    {"(3)+(6)", 1, 1, 0, 0},
    {"(3)->(6)",1, 1, 1, 0},
    {"rt-base", 0, 0, 0, 1},
    {"raw",     0, 0, 0, 0},
    {"mw",      1, 1, 0, 2},
};

int main(int argc, char** argv) {
    int   n = 200000, d = 64, k = 32, max_iter = 50, seed = 1, repeats = 1;
    float blob_std = 1.0f, box = 10.0f;
    int   verify = 1, verbose = 0, csv = 0, only = -1;
    const char* dataset = nullptr;
    int   converge = 0;
    float tol = 1e-8f;
    float rt_theta = 0.f;      /* 0 -> the paper's 5 */
    int   accum = MPK_ACCUM_FP32;
    int   mw_macs = 3;          /* multiword: ceiling on the cross products */
    float mw_gather_frac = 0.f;     /* multiword: 0 = derive from k */
    int   zscore = 0;               /* --zscore: standardize P up front */

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
        else if (!strcmp(a, "-i") || !strcmp(a, "--maxiters"))
            max_iter = atoi(next());
        else if (!strcmp(a, "--dataset")) dataset = next();
        else if (!strcmp(a, "--convergence")) converge = 1;
        else if (!strcmp(a, "--zscore")) zscore = 1;
        else if (!strcmp(a, "--tol")) tol = (float)atof(next());
        else if (!strcmp(a, "--theta")) rt_theta = (float)atof(next());
        else if (!strcmp(a, "--macs")) mw_macs = atoi(next());
        else if (!strcmp(a, "--gather-frac")) mw_gather_frac = (float)atof(next());
        else if (!strcmp(a, "--accum")) {
            const char* v = next();
            if      (!strcmp(v, "fp32")) accum = MPK_ACCUM_FP32;
            else if (!strcmp(v, "fp16")) accum = MPK_ACCUM_FP16;
            else { usage(argv[0]); return 1; }
        }
        else if (!strcmp(a, "-e")) seed = atoi(next());
        else if (!strcmp(a, "-r")) repeats = atoi(next());
        else if (!strcmp(a, "--no-verify")) verify = 0;
        else if (!strcmp(a, "--only")) {
            const char* v = next();
            only = !strcmp(v, "3") ? 0 : !strcmp(v, "6") ? 1 :
                   !strcmp(v, "36") ? 2 : !strcmp(v, "c") ? 3 :
                   !strcmp(v, "rt") ? 4 : !strcmp(v, "raw") ? 5 :
                   !strcmp(v, "mw") ? 6 : -1;
            if (only < 0) { usage(argv[0]); return 1; }
        }
        else if (!strcmp(a, "-v")) verbose = 1;
        else if (!strcmp(a, "--csv")) csv = 1;
        else if (!strcmp(a, "--csv-header")) {
            printf("n,d,k,std,seed,zscore,cond,iters,eps,"
                   "hp_baseline,hp_reference,hp_update,hp_total,pct_eliminated,"
                   "pct_reference,pct_update,pct_cond3,pct_cond6,pct_cond3_only,"
                   "pct_cond6_only,violations,label_diff,inertia,inertia_fp32,"
                   "rel_inertia,ms_dist,ms_dist_fp32,speedup,ms_prep,ms_gemm,"
                   "ms_argmin,ms_hpupdate,ms_assign\n");
            return 0;
        }
        else { usage(argv[0]); return 1; }
    }
    /* A dataset defines n and d -- only the file knows them -- so it is read
     * before anything that depends on the shape, and -n/-d are ignored. */
    float* hP = nullptr;
    std::vector<int> hTruth;
    int n_truth_classes = k;
    if (dataset) {
        int fn = 0, fd = 0, nc = 0; int* lab = nullptr;
        if (mpkLoadLibsvm(dataset, &fn, &fd, &hP, &lab, &nc) != MPK_OK) return 1;
        n = fn; d = fd; n_truth_classes = nc;
        hTruth.assign(lab, lab + n);
        free(lab);
    }
    if (n <= 0 || d <= 0 || k <= 0 || k > n) { usage(argv[0]); return 1; }

    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));
    const double eps = mpkEpsilon(d, accum);
    if (eps < 0) {
        fprintf(stderr, "error model degenerate for d=%d, accum=%s\n", d,
                accum == MPK_ACCUM_FP16 ? "fp16" : "fp32");
        return 1;
    }
    if (!csv) {
        printf("device   : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        if (dataset)
            printf("problem  : n=%d d=%d k=%d  %s (%d classes in the label "
                   "column)\n", n, d, k, dataset, n_truth_classes);
        else
            printf("problem  : n=%d d=%d k=%d  blobs(std=%.3g, box=%.3g, "
                   "seed=%d)\n", n, d, k, blob_std, box, seed);
        if (converge)
            printf("stopping : max_j ||c_j - c_j_prev||_2 < %.3g, or %d "
                   "iterations\n", tol, max_iter);
        else
            printf("stopping : label stability, or %d iterations\n", max_iter);
        printf("gemm     : FP16 operands, %s accumulate (rt-base is always FP32)\n",
               accum == MPK_ACCUM_FP16 ? "FP16" : "FP32");
        printf("eps      : %.6e   ->  factor 2*eps/(1-eps) = %.6e\n",
               eps, 2.0 * eps / (1.0 - eps));
        printf("update   : flat survivor list, one warp per entry, FP32\n");
    }

    /* ------------------------------------------------------------ data --- */
    if (!dataset) {
        hP = (float*)malloc((size_t)n * d * sizeof(float));
        if (!hP) { fprintf(stderr, "out of host memory\n"); return 1; }
        hTruth.resize(n);
        mpkMakeBlobs(n, d, k, blob_std, box, (unsigned)seed, hP, hTruth.data());
    }

    float *dP = nullptr, *dCmix = nullptr, *dCref = nullptr, *dC0 = nullptr;
    int   *dAmix = nullptr, *dAref = nullptr;
    CHK(cudaMalloc(&dP,    (size_t)n * d * sizeof(float)));
    CHK(cudaMalloc(&dCmix, (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dCref, (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dC0,   (size_t)k * d * sizeof(float)));
    CHK(cudaMalloc(&dAmix, (size_t)n * sizeof(int)));
    CHK(cudaMalloc(&dAref, (size_t)n * sizeof(int)));
    CHK(cudaMemcpy(dP, hP, (size_t)n * d * sizeof(float),
                   cudaMemcpyHostToDevice));
    free(hP); hP = nullptr;

    /* Z-scoring, when asked for, happens here and nowhere else: before the
     * unshifted copy below and before mpkShiftNonNegative, so that every
     * scheme -- the shifted one the exclusion conditions run on and the
     * unshifted one the baseline runs on -- sees the same normalized data.
     * Doing it after the copy would silently give the two frames different
     * datasets and invalidate every comparison in this file.
     *
     * It matters most to the baseline: its reliability test measures against
     * ||p||^2 + ||c||^2, so unlike the exclusion bounds (which drop ||p||^2 by
     * translation invariance) it is sensitive to both where the data sits and
     * how it is scaled.  This is arXiv:2407.12208's own preprocessing step. */
    if (zscore) {
        if (mpkStandardize(dP, n, d) != MPK_OK) return 1;
        if (!csv) printf("zscore   : every feature centred on its mean and "
                         "scaled to unit variance\n");
    }

    /* Conditions (3) and (6) need a non-negative P; the arXiv:2407.12208
     * baseline does not, and the shift actively hurts it -- it inflates
     * ||p||^2 and so the error floor E_l, which is exactly what its
     * reliability test measures against.  Adding a scalar to every entry is a
     * rigid translation, so it leaves all distances and the clustering
     * untouched and only changes the conditioning: keeping an unshifted copy
     * and giving it to the baseline compares each method on the data it is
     * meant to run on. */
    float* dPraw = nullptr;
    CHK(cudaMalloc((void**)&dPraw, (size_t)n * d * sizeof(float)));
    CHK(cudaMemcpy(dPraw, dP, (size_t)n * d * sizeof(float),
                   cudaMemcpyDeviceToDevice));

    float shift = 0.f;
    if (mpkShiftNonNegative(dP, n, d, &shift) != MPK_OK) return 1;
    if (!csv) printf("shift    : P += %.6g for (3)/(6); rt-base runs unshifted\n",
                     shift);

    if (mpkInitRandomPoints(dP, n, d, k, (unsigned)seed, dC0) != MPK_OK) return 1;

    cublasHandle_t blas;   cublasCreate(&blas);

    mpkParams par; mpkParamsInit(&par);
    par.max_iter      = max_iter;
    par.tol           = converge ? tol : 0.f;
    par.rt_theta      = rt_theta;
    par.accum         = accum;
    par.mw_macs       = mw_macs;
    par.mw_gather_frac= mw_gather_frac;
    par.verbose       = verbose;

    mpkStats smix[MPK_NCFG], sref;
    memset(smix, 0, sizeof(smix));
    memset(&sref, 0, sizeof(sref));

    /* warm up cuBLAS/cuSPARSE so the first timed run is not charged for it */
    {
        mpkParams w = par; w.max_iter = 1; w.verify = 0; w.verbose = 0;
        mpkStats junk;
        CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansMixed(blas, dP, n, d, k, dCmix, dAmix, &w, &junk);
        CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansBaselineRT(blas, dPraw, n, d, k, dCmix, dAmix, &w, &junk);
        CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansMultiword(blas, dP, n, d, k, dCmix, dAmix, &w, &junk);
        CHK(cudaMemcpy(dCref, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        mpkMeansFP32(blas, dP, n, d, k, dCref, dAref, &w, &junk);
    }

    for (int r = 0; r < repeats; ++r) {
        CHK(cudaMemcpy(dCref, dC0, (size_t)k * d * sizeof(float),
                       cudaMemcpyDeviceToDevice));
        par.verify = 0;   /* the FP32 run IS the oracle; nothing to check it against */
        if (mpkMeansFP32(blas, dP, n, d, k, dCref, dAref, &par, &sref)
            != MPK_OK) { fprintf(stderr, "fp32 failed\n"); return 1; }
    }

    std::vector<int> aref(n);
    CHK(cudaMemcpy(aref.data(), dAref, (size_t)n * sizeof(int),
                   cudaMemcpyDeviceToHost));

    int ok = 1;
    std::vector<std::vector<int>> amix(MPK_NCFG, std::vector<int>(n));
    for (int c = 0; c < MPK_NCFG; ++c) {
        if (only >= 0 && only != c) continue;
        par.use_cond3 = kConfigs[c].cond3;
        par.use_cond6 = kConfigs[c].cond6;
        par.cascade   = kConfigs[c].cascade;
        for (int r = 0; r < repeats; ++r) {
            CHK(cudaMemcpy(dCmix, dC0, (size_t)k * d * sizeof(float),
                           cudaMemcpyDeviceToDevice));
            par.verify = (r == repeats - 1) ? verify : 0;
            /* dC0 was picked from the shifted P; undo it for rt-base so both
             * start from the same points in their own frame.  The multiword
             * driver stays in the shifted frame: its bound, like (3)/(6)'s,
             * is componentwise against |A||B| and needs non-negative P. */
            if (kConfigs[c].driver == 1)
                mpkShiftCentroids(dCmix, k, d, -shift);
            const mpkStatus rc =
                kConfigs[c].driver == 1
                    ? mpkMeansBaselineRT(blas, dPraw, n, d, k, dCmix, dAmix,
                                         &par, &smix[c])
              : kConfigs[c].driver == 2
                    ? mpkMeansMultiword(blas, dP, n, d, k, dCmix, dAmix, &par,
                                        &smix[c])
                    : mpkMeansMixed(blas, dP, n, d, k, dCmix, dAmix, &par,
                                    &smix[c]);
            if (rc != MPK_OK) {
                fprintf(stderr, "mixed %s failed\n", kConfigs[c].name);
                return 1;
            }
        }
        CHK(cudaMemcpy(amix[c].data(), dAmix, (size_t)n * sizeof(int),
                       cudaMemcpyDeviceToHost));
    }

    for (int c = 0; c < MPK_NCFG; ++c) {
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
            printf("%d,%d,%d,%g,%d,%d,%s,%d,%.6e,"
                   "%lld,%lld,%lld,%lld,%.6f,%.6f,%.6f,"
                   "%.6f,%.6f,%.6f,%.6f,%lld,%lld,"
                   "%.9e,%.9e,%.3e,%.3f,%.3f,%.4f,"
                   "%.3f,%.3f,%.3f,%.3f,%.3f\n",
                   n, d, k, blob_std, seed, zscore, kConfigs[c].name,
                   S.iters, eps,
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
                   S.t_hp_update_ms, S.t_assign_ms);
        }
    }
    if (csv) {
        cublasDestroy(blas);
        cudaFree(dP); cudaFree(dPraw); cudaFree(dCmix); cudaFree(dCref); cudaFree(dC0);
        cudaFree(dAmix); cudaFree(dAref);
        return ok ? 0 : 2;
    }

    /* ------------------------------------------- high precision work ----- */
    printf("\nhigh precision distance evaluations PER ITERATION\n");
    printf("  plain Lloyd's evaluates every entry of D: n*k = %lld per iteration\n",
           (long long)n * k);
    printf("  %-8s %6s %14s %14s %14s %11s\n",
           "cond", "iters", "reference", "survivors", "total", "ELIMINATED");
    for (int c = 0; c < MPK_NCFG; ++c) {
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
           "  needs before it can exclude anything: n per iteration for (6) and\n"
           "  (3)+(6).  Condition (3) needs no such quantity, so it computes\n"
           "  none; the incumbent is instead counted among the survivors, and\n"
           "  only for the rows (3) did not clear outright.  (3)->(6) is the\n"
           "  cascade: (3) is applied first and a reference entry is evaluated\n"
           "  only for the rows (3) left behind, so its reference count falls\n"
           "  well below n while its survivor count matches (3)+(6) exactly.\n");

    {
        int it_min = 0, it_max = 0;
        for (int c = 0; c < MPK_NCFG; ++c) {
            if (only >= 0 && only != c) continue;
            const int v = smix[c].iters;
            if (!it_min || v < it_min) it_min = v;
            if (v > it_max) it_max = v;
        }
        if (it_max != it_min) {
            printf("\n  NOTE: the configurations converged in different numbers of\n"
                   "  iterations (%d..%d).  Per-iteration numbers are comparable;\n"
                   "  run totals are not.  Totals over the whole run:\n", it_min, it_max);
            for (int c = 0; c < MPK_NCFG; ++c) {
                if (only >= 0 && only != c) continue;
                const mpkStats& S = smix[c];
                printf("    %-8s %d iters:  %lld of %lld\n", kConfigs[c].name,
                       S.iters, S.hp_reference + S.hp_update, S.hp_baseline);
            }
        }
    }

    /* with --only, results land in smix[only]; smix[0] may never have run */
    const int sref_idx = only >= 0 ? only : 0;

    if (smix[sref_idx].stats_built) {
        printf("\nexclusion attribution, over the pairs each scheme tested\n");
        printf("  for (3)->(6), (6) is only ever evaluated on the rows (3) did\n"
               "  not clear, so its share is over that subset, not over all n\n");
        printf("  the denominator is n*(k-1) for the exclusion schemes, whose\n"
               "  incumbent is exempt, and n*k for rt-base, which tests every\n"
               "  entry; rt-base excludes nothing, so its row is 0 either way\n");
        printf("  %-8s %10s %10s %10s %10s %10s\n",
               "cond", "(3) held", "(6) held", "both", "(3) only", "(6) only");
        for (int c = 0; c < MPK_NCFG; ++c) {
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
    printf("  %-8s %5s %7s %8s %10s %9s %9s %9s\n",
           "cond", "iters", "prep", "gemm", "argmin+cond",
           "update", "assign", "TOTAL");
    for (int c = 0; c < MPK_NCFG; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        printf("  %-8s %5d %7.2f %8.2f %10.2f %9.2f %9.2f %9.2f\n",
               kConfigs[c].name, S.iters, S.t_prep_ms, S.t_gemm_lo_ms,
               S.t_argmin_ms, S.t_hp_update_ms, S.t_assign_ms, S.t_dist_ms);
    }
    printf("  %-8s %5d %7.2f %8.2f %10s %9s %9.2f %9.2f\n",
           "fp32", sref.iters, sref.t_prep_ms, sref.t_gemm_lo_ms, "-", "-",
           sref.t_assign_ms, sref.t_dist_ms);

    printf("\n  %-8s %10s %10s %8s   %s\n", "cond", "ms/iter", "vs fp32",
           "update", "(centroid update, excluded above)");
    for (int c = 0; c < MPK_NCFG; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        const double per = S.t_dist_ms / fmax(S.iters, 1);
        const double rper = sref.t_dist_ms / fmax(sref.iters, 1);
        printf("  %-8s %10.3f %9.3fx %8.2f\n",
               kConfigs[c].name, per, rper / fmax(per, 1e-9), S.t_update_ms);
    }
    printf("  %-8s %10.3f %9s %8.2f\n", "fp32",
           sref.t_dist_ms / fmax(sref.iters, 1), "1.000x", sref.t_update_ms);

    /* The FP32 refinement is the same kernel for every scheme here -- one warp
     * per flagged entry, WPB=8, 256 threads, the same grid cap -- differing
     * only in whether it forms p.c or ||p-c||^2.  So its cost is a count, not
     * a rate, and the schemes are separated by how many entries they flag, not
     * by how fast they refine one.  ns/entry says whether that holds: if two
     * schemes disagree there, something other than the count is at work. */
    printf("\n  refinement cost, normalised -- same kernel for every scheme\n");
    printf("  %-8s %14s %12s %12s\n", "cond", "entries/iter", "ms/iter",
           "ns/entry");
    for (int c = 0; c < MPK_NCFG; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        const double it = fmax(S.iters, 1);
        printf("  %-8s %14.1f %12.4f %12.2f\n", kConfigs[c].name,
               (double)S.hp_update / it, S.t_hp_update_ms / it,
               S.hp_update ? S.t_hp_update_ms * 1e6 / (double)S.hp_update : 0.0);
    }

    /* The multiword scheme does not refine entries at all -- it refines the
     * whole matrix, one fp16 cross product at a time, and the exclusion test
     * decides whether the next product is issued.  So its cost is a product
     * count, and the row above (a gather out of an already-refined G) says
     * nothing about it. */
    {
        int any_mw = 0;
        for (int c = 0; c < MPK_NCFG; ++c)
            if ((only < 0 || only == c) && kConfigs[c].driver == 2) any_mw = 1;
        if (any_mw) {
            printf("\n  multiword cost -- cross products, not entries\n");
            printf("  %-8s %16s %14s %14s %14s\n", "cond", "products/iter",
                   "ms gemm/iter", "eps at cut", "refine width");
            for (int c = 0; c < MPK_NCFG; ++c) {
                if (only >= 0 && only != c) continue;
                if (kConfigs[c].driver != 2) continue;
                const mpkStats& S = smix[c];
                const double it = fmax(S.iters, 1);
                const double per = (double)S.mw_products / it;
                /* the bound actually reached, at the average product count */
                const int cut = (int)(per + 0.5) < 1 ? 1
                              : ((int)(per + 0.5) > 3 ? 3 : (int)(per + 0.5));
                printf("  %-8s %16.2f %14.4f %14.3e %13.1f%%\n",
                       kConfigs[c].name, per, S.t_gemm_lo_ms / it,
                       mpkMultiwordEpsilon(d, cut),
                       100.0 * (double)S.mw_refine_rows / (it * (double)n));
            }
            printf("  1.00 means the bound settled every row from the leading\n"
                   "  product alone; 3.00 means it always needed full\n"
                   "  double-fp16.  eps is mpkMultiwordEpsilon at that cut --\n"
                   "  the accuracy the answer actually carries.\n");
        }
    }

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
    if (verify && smix[sref_idx].stats_built) {
        printf("  -- against a full FP32 sgemm iteration on the same centroids,\n"
               "     every row of every iteration\n");
        printf("  %-8s %20s %22s %12s\n", "cond", "bound violations",
               "label != fp32 label", "excess");
        for (int c = 0; c < MPK_NCFG; ++c) {
            if (only >= 0 && only != c) continue;
            const mpkStats& S = smix[c];
            const long long rows = (long long)n * S.iters;
            printf("  %-8s %9lld / %-8lld %9lld / %-8lld %12.4e\n",
                   kConfigs[c].name, S.verify_excluded_best, rows,
                   S.verify_label_diff, rows, S.verify_excess);
        }
        printf("  the oracle is the ordinary FP32 implementation -- the same\n"
               "  cublasSgemm and row argmin mpkMeansFP32 runs -- evaluated on\n"
               "  the mixed run's own centroids each iteration.\n"
               "  bound violations must be 0: the FP32 answer was neither the\n"
               "  incumbent nor among the survivors, so a condition of Theorem 1\n"
               "  does not hold.  label != fp32 label counts rows where the FP32\n"
               "  answer was reachable but not chosen; excess is what that cost\n"
               "  in the FP32 distances themselves.\n"
               "  rt-base has no per-iteration hookup into this oracle at all --\n"
               "  a different strategy entirely -- so its row always reads\n"
               "  0/rows: an absence of measurement, not a passing check.\n"
               "  raw ran no exclusion test, so \"reachable\" is trivially every\n"
               "  row: its bound violations column is 0 by construction, for the\n"
               "  same reason, but label != fp32 label and excess are real --\n"
               "  they score how often trusting the low precision argmin outright\n"
               "  got the wrong answer, and by how much.\n"
               "  The clustering table below is the other way to see this, for\n"
               "  every scheme including rt-base.\n");
    } else if (verify) {
        printf("  -- SKIPPED: this binary was built without MPK_STATS, so the\n"
               "  FP64 oracle is compiled out.  Rebuild with -DMPK_STATS=ON.\n");
    } else {
        printf("  -- SKIPPED: --no-verify\n");
    }

    printf("\nclustering\n");
    for (int c = 0; c < MPK_NCFG; ++c) {
        if (only >= 0 && only != c) continue;
        const mpkStats& S = smix[c];
        long long identical = 0;
        for (int i = 0; i < n; ++i) identical += (amix[c][i] == aref[i]);
        const double rel = fabs(S.inertia - sref.inertia) /
                           fmax(sref.inertia, 1e-300);
        printf("  %-8s inertia rel diff %.3e   labels == fp32 %.3f%%   "
               "truth %.4f\n", kConfigs[c].name, rel,
               100.0 * (double)identical / n,
               label_agreement(amix[c], hTruth, k, n_truth_classes));
    }
    printf("  %-8s %54s truth %.4f\n", "fp32", "",
           label_agreement(aref, hTruth, k, n_truth_classes));

    /* The theorem guarantees the true nearest centroid is never filtered out.
     * It does not guarantee bit-identical labels to an FP32 run: once an FP32
     * tie flips, the two trajectories separate.  So gate on the violation
     * counter and on the clustering quality, not on label equality. */
    printf("\n%s\n", ok ? "PASS" : "CHECK FAILED");


    cublasDestroy(blas);
    cudaFree(dP); cudaFree(dPraw); cudaFree(dCmix); cudaFree(dCref); cudaFree(dC0);
    cudaFree(dAmix); cudaFree(dAref);
    return ok ? 0 : 2;
}
