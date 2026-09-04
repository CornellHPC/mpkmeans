#!/usr/bin/env python3
"""rt baseline driver: the AUTHORS' implementation of arXiv:2407.12208.

`mp-kmeans` on PyPI (github.com/inEXASCALE/mp-kmeans) is Carson, Chen and Liu's
own code for the paper that this repo's C++ `rt-base` reimplements.  Running it
directly removes the reimplementation from the comparison: whatever it reports
is what the method does, not what we understood it to do.

Kernel `fp16_fp32` -- their FP16 GEMM with the FP32 direct-formula fallback,
which is the variant the paper reports as the best trade-off and the one our
rt-base was written against.

What is held equal to the C++ benchmark, and how:

  subsampling   NOT shared.  This draws with numpy's Generator and the bench
                with std::mt19937, so the same -e picks different rows.  When
                --subsample is in play, use --dump-subsample here and hand the
                bench that file with --bin, or the two cluster different data.
  data          --bin and --libsvm read exactly the files mpkmeans_bench reads,
                so the matrix clustered is the same one, bit for bit.  --blobs
                mirrors mpkMakeBlobs' construction (centers uniform in
                [-b,b]^d, balanced clusters, N(0,std) noise) but NOT its RNG:
                libstdc++ and numpy draw normals differently, so synthetic runs
                are distributionally matched, not identical.  For an exact
                synthetic comparison, generate once and hand both sides the
                same --bin file.
  initial       k distinct rows drawn without replacement, seeded by --seed.
  centroids     --dump-centroids writes them, and mpkmeans_bench --init-centroids
                reads that file, so both sides can be pinned to one choice.
                init_method is 'random', never 'kmeans++': the comparison is of
                distance kernels, and k-means++ would change the trajectory.
  normalization normalize=None is forced.  The package z-scores by default,
                which would silently apply a second, different normalization on
                top of --zscore and put the supplied centroids in the wrong
                space.  --zscore here is the same per-feature standardization
                mpkStandardize does, applied before anything else.
  inertia       recomputed in FP64 from the final centers and labels, the same
                quantity mpkmeans_bench reports -- not their inertia_, which is
                a mixed-precision sum in whatever space the model was fitted.

What cannot be held equal:

  stopping      both stop on ||C - C_prev||_F < tol over the whole k x d
                block.  The C++ side was moved onto that rule to match this
                package (and cuVS, whose shift clause is the same quantity),
                so at equal --tol the two run the same recurrence to the same
                point.  One residual difference: our loop also breaks when no
                label changed, which this package does not test -- but stable
                labels give identical centroids and hence ||C - C_prev||_F = 0,
                so the two fire on the same iteration.
  loop shape    their fit ends on a centroid update and then re-assigns, so the
                labels are half a Lloyd step ahead of ours, exactly as cuVS's
                are.  At convergence this does not matter.
  fallback rate their kernel does the reliability test and the fallback inside
                one CUDA call and does not report how many entries fell back.
                The hp_* columns are therefore not measured, and are written the
                way the cuvs row writes them: every pair counted as evaluated,
                0% eliminated.  Do not read that as "it eliminated nothing".

REQUIRES torch 2.12.x.  The wheel ships prebuilt CUDA extensions with no torch
pin, and on torch 2.14 every kernel fails `TORCH_CHECK(P.is_contiguous())` on
tensors that are demonstrably contiguous -- an ABI mismatch that looks like a
bad argument.  See eval/README.md for the environment.
"""
import argparse, csv, os, sys, time

import numpy as np

CSV_COLUMNS = [
    "dataset", "n", "d", "k", "std", "box", "seed", "zscore", "accum", "cond",
    "iters", "eps", "hp_baseline", "hp_reference", "hp_update", "hp_total",
    "pct_eliminated", "pct_reference", "pct_update", "pct_cond3", "pct_cond6",
    "pct_cond3_only", "pct_cond6_only", "violations", "label_diff", "inertia",
    "inertia_fp32", "rel_inertia", "ms_dist", "ms_dist_fp32", "speedup",
    "ms_prep", "ms_gemm", "ms_argmin", "ms_hpupdate", "ms_assign",
    "ms_total", "ms_total_fp32", "iters_fp32",
]


# ------------------------------------------------------------------ data ---
def load_bin(path, d):
    b = os.path.getsize(path)
    if b % (d * 4):
        sys.exit(f"{path}: {b} bytes is not a whole number of {d}-dim float32 "
                 f"rows -- wrong --dim, or truncated")
    return np.fromfile(path, dtype=np.float32).reshape(-1, d)


def load_libsvm(path):
    """Densify to the largest feature index, as mpkLoadLibsvm does."""
    rows, maxidx = 0, 0
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            rows += 1
            for tok in line.split()[1:]:
                i, _, v = tok.partition(":")
                if v and i.isdigit():
                    maxidx = max(maxidx, int(i))
    if not rows or not maxidx:
        sys.exit(f"{path}: no usable rows")
    X = np.zeros((rows, maxidx), dtype=np.float32)
    with open(path) as f:
        i = 0
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for tok in line.split()[1:]:
                idx, _, v = tok.partition(":")
                if v and idx.isdigit():
                    j = int(idx)
                    if 1 <= j <= maxidx:
                        X[i, j - 1] = float(v)
            i += 1
    return X


def make_blobs(n, d, k, std, box, seed):
    """mpkMakeBlobs' construction, numpy's RNG -- see the module docstring."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-box, box, size=(k, d)).astype(np.float32)
    lab = (np.arange(n, dtype=np.int64) * k) // n          # balanced clusters
    X = centers[lab] + rng.normal(0.0, std, size=(n, d)).astype(np.float32)
    return X.astype(np.float32), lab


def zscore(X):
    """Per-feature standardization, matching mpkStandardize: FP64 moments, and
    a zero-variance feature is left mean-centred rather than divided by zero."""
    mu = X.mean(axis=0, dtype=np.float64)
    sd = X.astype(np.float64).std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((X - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------- run ---
def fit_once(X_t, C0_t, kernel, kappa, max_iter, tol, seed):
    """One mp-kmeans fit, wall-clock timed with the device synchronised."""
    import torch
    from mp_kmeans.clustering import KMeansPlusPlus

    model = KMeansPlusPlus(
        n_clusters=C0_t.shape[0],
        kernel=kernel,
        kappa=kappa,
        max_iter=max_iter,
        tol=tol,
        normalize=None,          # we normalize, not the model -- see docstring
        random_state=seed,
        init_method="random",    # never kmeans++
        verbose=False,
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.fit(X_t, C0_t.clone())
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    return model, ms


def inertia_fp64(X_t, centers_t, labels_t):
    """Sum of squared distances to the assigned centre, in FP64 -- the same
    quantity mpkLaunchInertia computes for every other scheme."""
    import torch
    X = X_t.double()
    C = centers_t.double()[labels_t]
    return float(((X - C) ** 2).sum().item())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bin", help="headerless row-major float32 matrix")
    src.add_argument("--libsvm", help="LIBSVM text file")
    src.add_argument("--blobs", action="store_true", help="synthetic gaussians")
    ap.add_argument("-d", "--dim", type=int, default=64,
                    help="dimension; required with --bin, features with --blobs")
    ap.add_argument("-n", type=int, default=200000, help="--blobs only")
    ap.add_argument("-k", type=int, required=True, help="clusters")
    ap.add_argument("-s", "--std", type=float, default=1.0, help="--blobs only")
    ap.add_argument("-b", "--box", type=float, default=10.0, help="--blobs only")
    ap.add_argument("-e", "--seed", type=int, default=1)
    ap.add_argument("--maxiters", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--kappa", type=float, default=5.0,
                    help="their safety factor, the paper's rho (default 5)")
    ap.add_argument("--kernel", default="fp16_fp32")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--dump-subsample",
                    help="write the subsampled matrix here as raw float32.  "
                         "REQUIRED for a matched comparison when --subsample "
                         "is used: this driver draws with numpy's Generator "
                         "and mpkmeans_bench with std::mt19937, so the same "
                         "-e gives DIFFERENT subsets.  Point the bench at this "
                         "file with --bin instead of letting it subsample.")
    ap.add_argument("--centroids", help="read initial centroids (k x d float32)")
    ap.add_argument("--dump-centroids",
                    help="write the initial centroids, for mpkmeans_bench "
                         "--init-centroids")
    ap.add_argument("--reference", choices=["fp32", "none"], default="none",
                    help="also fit with their uniform-fp32 kernel from the same "
                         "centroids, filling the *_fp32 columns.  Off by "
                         "default: those columns would then hold THIS package's "
                         "fp32, not mpkMeansFP32, which is a different "
                         "reference from every other row's and doubles the "
                         "runtime for a number the plots do not use.")
    ap.add_argument("--csv", help="append a row here (header written if new)")
    ap.add_argument("--csv-header", action="store_true",
                    help="print the CSV header and exit")
    args = ap.parse_args()

    if args.csv_header:
        print(",".join(CSV_COLUMNS))
        return

    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible to torch")

    # ---- data -------------------------------------------------------------
    truth = None
    if args.bin:
        X = load_bin(args.bin, args.dim); name = os.path.basename(args.bin)
    elif args.libsvm:
        X = load_libsvm(args.libsvm); name = os.path.basename(args.libsvm)
    else:
        X, truth = make_blobs(args.n, args.dim, args.k, args.std, args.box,
                              args.seed)
        name = "blobs"
    n, d = X.shape

    if 0 < args.subsample < n:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n, size=args.subsample, replace=False))
        X = np.ascontiguousarray(X[idx])
        n = X.shape[0]
        if args.dump_subsample:
            X.astype(np.float32).tofile(args.dump_subsample)
            print(f"wrote {args.dump_subsample} ({n} x {X.shape[1]} float32) "
                  f"-- give this to mpkmeans_bench --bin, not --subsample")

    if args.zscore:
        X = zscore(X)

    if args.k >= n:
        sys.exit(f"k={args.k} >= n={n}")

    # ---- initial centroids ------------------------------------------------
    if args.centroids:
        C0 = load_bin(args.centroids, d)
        if C0.shape[0] != args.k:
            sys.exit(f"{args.centroids}: {C0.shape[0]} centroids, expected {args.k}")
    else:
        rng = np.random.default_rng(args.seed)
        C0 = np.ascontiguousarray(X[rng.choice(n, size=args.k, replace=False)])
    if args.dump_centroids:
        C0.astype(np.float32).tofile(args.dump_centroids)
        print(f"wrote {args.dump_centroids} ({args.k} x {d} float32)")

    X_t = torch.from_numpy(np.ascontiguousarray(X)).cuda().contiguous()
    C0_t = torch.from_numpy(np.ascontiguousarray(C0)).cuda().contiguous()

    print(f"problem  : n={n} d={d} k={args.k}  {name}"
          f"{'  (z-scored)' if args.zscore else ''}")
    print(f"kernel   : {args.kernel}  kappa={args.kappa}  init=random  "
          f"normalize=None")
    print(f"stopping : ||C_new - C_old||_F < {args.tol:g}, or {args.maxiters} "
          f"iterations")

    # a first CUDA call pays for context setup and kernel load; keep that out
    # of the timed fit
    fit_once(X_t[: min(n, 4096)].contiguous(), C0_t, args.kernel, args.kappa,
             2, args.tol, args.seed)

    model, ms = fit_once(X_t, C0_t, args.kernel, args.kappa, args.maxiters,
                         args.tol, args.seed)
    inert = inertia_fp64(X_t, model.cluster_centers_, model.labels_)
    print(f"{args.kernel:12s} iters={model.n_iter_:4d}  {ms:9.2f} ms  "
          f"inertia={inert:.9e}")

    ref_ms = ref_iters = 0.0
    ref_inert = float("nan")
    label_diff = 0
    if args.reference == "fp32":
        rmodel, ref_ms = fit_once(X_t, C0_t, "fp32", args.kappa, args.maxiters,
                                  args.tol, args.seed)
        ref_inert = inertia_fp64(X_t, rmodel.cluster_centers_, rmodel.labels_)
        ref_iters = rmodel.n_iter_
        label_diff = int((model.labels_ != rmodel.labels_).sum().item())
        print(f"{'fp32':12s} iters={rmodel.n_iter_:4d}  {ref_ms:9.2f} ms  "
              f"inertia={ref_inert:.9e}")
        rel = abs(inert - ref_inert) / max(abs(ref_inert), 1e-300)
        print(f"rel inertia vs their own fp32: {rel:.3e}   "
              f"labels differing: {label_diff} ({100.0*label_diff/n:.3f}%)")
    else:
        rel = float("nan")

    if truth is not None:
        print(f"(synthetic: {args.k} planted clusters; blobs are "
              f"distributionally matched to mpkMakeBlobs, not identical)")

    # ---- CSV --------------------------------------------------------------
    if args.csv:
        pairs = float(n) * args.k * max(model.n_iter_, 1)
        row = {c: 0 for c in CSV_COLUMNS}
        row.update(
            dataset=name, n=n, d=d, k=args.k, std=args.std, box=args.box,
            seed=args.seed, zscore=int(args.zscore), accum=args.kernel,
            cond="rt-mp", iters=model.n_iter_, eps=0,
            # their kernel does the test and the fallback inside one call and
            # reports neither, so this is "not measured", not "none eliminated"
            hp_baseline=int(pairs), hp_reference=0, hp_update=int(pairs),
            hp_total=int(pairs), pct_eliminated=0.0,
            violations=0, label_diff=label_diff,
            inertia=f"{inert:.9e}",
            inertia_fp32=f"{ref_inert:.9e}", rel_inertia=f"{rel:.3e}",
            # the fit is one library call: no stage is separable, so ms_dist is
            # left 0 and the row appears only in the end-to-end plots
            ms_dist=0.0, ms_dist_fp32=0.0, speedup=0.0,
            ms_total=f"{ms:.3f}", ms_total_fp32=f"{ref_ms:.3f}",
            iters_fp32=int(ref_iters),
        )
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"appended a row to {args.csv}")


if __name__ == "__main__":
    main()
