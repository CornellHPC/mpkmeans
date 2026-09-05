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
                with std::mt19937, so the same -e picks different rows.  Nor
                are --blobs, which no two RNGs reproduce.  Use --dump-data and
                hand the bench that file with --bin -- run_one.sh wires it up.
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
                The chosen kernel is then scored against this package's own
                fp32 kernel run on the same data from the same centroids
                through the same loop, so the relative inertia difference
                isolates the distance kernel and nothing else.

What cannot be held equal:

  stopping      --convergence on both sides stops on ||C - C_prev||_F < tol
                over the whole k x d block; without it both run exactly
                --maxiters iterations and never exit early.  The C++ side was moved onto that rule to match this
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

# The paper's safety factor rho.  5 is the value it prescribes and the measured
# knee at d = 200 -- the smallest value driving the kernel's argmin errors to
# zero.  The knee falls as d rises, roughly as 1/(d*u_l), reaching ~1 by d=768,
# which is why the evaluation runs both 1 and 5 rather than one of them.  See
# eval/README.md; kappa_kernel.py re-derives it.
KAPPA_DEFAULT = 5.0

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


def resolve_bin_dim(path, explicit_dim):
    """A --bin file's dimension is a property of the file, not something to
    retype on the command line -- and a wrong guess does not fail, it just
    reshapes the matrix into garbage as long as the byte count still divides.

    Two sources know the real value without being asked: a --dump-data
    sidecar (<path>.meta, written "n d") is exact; a manifest.tsv next to a
    scripts/fetch_superkmeans_datasets.sh dataset (matched by the
    data_<id>.bin filename) is authoritative for that id.  --dim is now only
    a fallback for a file neither source recognises, and if it disagrees with
    one that does, that is treated as a mistake (a flag copied from a
    different invocation), not a request to override -- the file wins."""
    found, src = None, None

    meta_path = path + ".meta"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _n_meta, d_meta = (int(x) for x in f.read().split())
        found, src = d_meta, meta_path

    if found is None:
        base = os.path.basename(path)
        if base.startswith("data_") and base.endswith(".bin") \
                and not base.endswith("_test.bin"):
            dset_id = base[len("data_"):-len(".bin")]
            man_path = os.path.join(os.path.dirname(path), os.pardir,
                                     "manifest.tsv")
            if os.path.exists(man_path):
                with open(man_path, newline="") as f:
                    for row in csv.DictReader(f, delimiter="\t"):
                        if row.get("id") == dset_id:
                            found, src = int(row["d"]), man_path
                            break

    if found is not None and explicit_dim is not None and explicit_dim != found:
        sys.exit(f"{path}: --dim {explicit_dim} disagrees with {src} "
                  f"(d={found}) -- fix whichever one is wrong, this refuses "
                  f"to silently pick one")

    if found is not None:
        return found
    if explicit_dim is not None:
        return explicit_dim
    sys.exit(f"{path}: cannot determine dimension -- no .meta sidecar, no "
             f"manifest.tsv entry for it, and no --dim given")


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


def iter_stats(model):
    """Per-iteration wall times from the model, in ms.

    The package times each Lloyd iteration itself, and its loop ends with a
    .item() on the centroid shift, so each of those times is bounded by a
    device sync and is usable.  Reported because the total divided by the
    iteration count is NOT the per-iteration cost until the allocator is warm:
    on a cold process iteration 0 costs 208 ms against a 16 ms steady state.
    Printing the spread makes a warm-up that did not take visible at once,
    rather than silently inflating an average."""
    it = [t * 1e3 for t in model.timing_.get("iterations", [])]
    if not it:
        return None
    body = it[1:] or it
    body = sorted(body)
    med = body[len(body) // 2]
    return {"first": it[0], "median": med, "min": min(it), "max": max(it),
            "n": len(it)}


def profile_fit(X_t, C0_t, kernel, kappa, iters, seed):
    """Split one fit into distance / centroid-update / everything else.

    The package exposes no hooks, so this wraps the CUDA entry points on the
    mp_kmeans.euclidean_cuda module for the duration of one fit.  That measures
    the calls the Lloyd loop actually makes, at the shapes it actually uses --
    unlike timing the kernel standalone, which measures a call the loop never
    makes and, done badly, invents a number (see the cuvs t_dist_ms note in
    src/cuvs_baseline.cu for how that goes wrong).

    Each wrapper synchronises, so this fit is SLOWER than the timed one and its
    total is not comparable to ms_total.  Only the split is taken from it, and
    it runs for a few iterations because the loop is in steady state.
    """
    import torch
    from mp_kmeans import euclidean_cuda as E

    acc = {"dist": 0.0, "update": 0.0}
    originals = {}

    def wrap(name, bucket):
        fn = getattr(E, name)
        originals[name] = fn

        def timed(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            acc[bucket] += (time.perf_counter() - t0) * 1e3
            return out
        setattr(E, name, timed)

    for name in dir(E):
        if name.startswith("pairwise_euclidean"):
            wrap(name, "dist")
        elif name.startswith("update_centers"):
            wrap(name, "update")
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model, _ = fit_once(X_t, C0_t, kernel, kappa, iters, 1e-45, seed)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
    finally:
        for name, fn in originals.items():
            setattr(E, name, fn)

    it = max(model.n_iter_, 1)
    # the loop makes one extra distance call after it ends, for the final
    # labels, so the per-iteration figure divides by iterations + 1
    return {"dist": acc["dist"] / (it + 1), "update": acc["update"] / it,
            "other": (wall - acc["dist"] - acc["update"]) / it,
            "iters": it, "wall": wall}


def inertia_fp64(X_t, centers_t, labels_t, chunk=200_000):
    """Sum of squared distances to the assigned centre, in FP64 -- the same
    quantity mpkLaunchInertia computes for every other scheme.

    Chunked over rows: X_t.double() and centers_t.double()[labels_t] each
    materialize a full n x d FP64 array, so at n=999000, d=1536 (openai) the
    two together are ~24.6 GB on top of whatever the fit itself still holds --
    measured to OOM a 40 GB card outright.  Chunking bounds the extra memory
    by chunk x d instead of n x d, independent of dataset size."""
    C64 = centers_t.double()
    total = 0.0
    n = X_t.shape[0]
    for i in range(0, n, chunk):
        Xc = X_t[i:i + chunk].double()
        Cc = C64[labels_t[i:i + chunk]]
        total += ((Xc - Cc) ** 2).sum().item()
    return float(total)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bin", help="headerless row-major float32 matrix")
    src.add_argument("--libsvm", help="LIBSVM text file")
    src.add_argument("--blobs", action="store_true", help="synthetic gaussians")
    ap.add_argument("-d", "--dim", type=int, default=None,
                    help="--blobs: feature count (default 64).  --bin: only "
                    "needed as a fallback -- a .meta sidecar or a "
                    "manifest.tsv entry is read first, see resolve_bin_dim")
    ap.add_argument("-n", type=int, default=200000, help="--blobs only")
    ap.add_argument("-k", type=int, required=True, help="clusters")
    ap.add_argument("-s", "--std", type=float, default=1.0, help="--blobs only")
    ap.add_argument("-b", "--box", type=float, default=10.0, help="--blobs only")
    ap.add_argument("-e", "--seed", type=int, default=1)
    ap.add_argument("--maxiters", type=int, default=400)
    ap.add_argument("--convergence", action="store_true",
                    help="stop at ||C - C_prev||_F < --tol.  WITHOUT it the fit "
                         "runs exactly --maxiters iterations, matching "
                         "mpkmeans_bench without --convergence.  (The package "
                         "rejects tol <= 0, so 'fixed count' is implemented as "
                         "a subnormal tol: it can then only fire on an exactly "
                         "zero centroid shift, i.e. on a fit that has stopped "
                         "moving at all and whose further iterations would be "
                         "no-ops anyway.)")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--kappa", type=float, default=KAPPA_DEFAULT,
                    help=f"the paper's safety factor rho (default "
                         f"{KAPPA_DEFAULT:g}).  The evaluation runs 1 and 5: "
                         f"5 is the paper's value and the knee at d=200, and "
                         f"the knee falls to ~1 by d=768.")
    ap.add_argument("--kernel", default="fp16_fp32")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--dump-data", "--dump-subsample", dest="dump_data",
                    help="write the matrix this run will cluster -- after any "
                         "--subsample, before any --zscore -- as raw float32, "
                         "plus a <file>.meta holding 'n d'.  Hand that file to "
                         "mpkmeans_bench with --bin and the two cluster exactly "
                         "the same numbers.  Needed whenever the source is not "
                         "already a shared .bin: --subsample draws with numpy's "
                         "Generator here and std::mt19937 there, and --blobs "
                         "cannot be reproduced across the two at all.  "
                         "run_one.sh does this for you.")
    ap.add_argument("--centroids", help="read initial centroids (k x d float32)")
    ap.add_argument("--dump-centroids",
                    help="write the initial centroids, for mpkmeans_bench "
                         "--init-centroids")
    ap.add_argument("--no-profile", dest="profile", action="store_false",
                    help="skip the extra instrumented fit that splits the time "
                         "into distance / centroid update / other.  That fit "
                         "runs a few iterations with a device sync around every "
                         "kernel, so it costs a little and is not part of the "
                         "reported ms_total.")
    ap.add_argument("--reference", choices=["fp32", "none"], default="fp32",
                    help="fit the same data from the same centroids with this "
                         "package's fp32 kernel, and report the chosen "
                         "kernel's inertia relative to it.  That is the "
                         "controlled comparison -- same loop, same data, same "
                         "start, only the distance kernel differs.  --reference "
                         "none skips it and halves the runtime, leaving "
                         "rel_inertia unset.")
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
        bin_dim = resolve_bin_dim(args.bin, args.dim)
        X = load_bin(args.bin, bin_dim); name = os.path.basename(args.bin)
    elif args.libsvm:
        X = load_libsvm(args.libsvm); name = os.path.basename(args.libsvm)
    else:
        X, truth = make_blobs(args.n, args.dim or 64, args.k, args.std,
                              args.box, args.seed)
        name = "blobs"
    n, d = X.shape

    if 0 < args.subsample < n:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n, size=args.subsample, replace=False))
        X = np.ascontiguousarray(X[idx])
        n = X.shape[0]

    # Before --zscore, so the other side applies its own and lands in the same
    # space; the centroids below are dumped after it, which is the frame
    # mpkmeans_bench --init-centroids expects.
    if args.dump_data:
        X.astype(np.float32).tofile(args.dump_data)
        with open(args.dump_data + ".meta", "w") as f:
            f.write(f"{n} {X.shape[1]}\n")
        print(f"wrote {args.dump_data} ({n} x {X.shape[1]} float32) "
              f"and {os.path.basename(args.dump_data)}.meta")

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
    print(f"kernel   : {args.kernel}  kappa={args.kappa:g}  init=random  "
          f"normalize=None")
    if args.convergence:
        print(f"stopping : ||C - C_prev||_F < {args.tol:g}, or {args.maxiters} "
              f"iterations")
    else:
        print(f"stopping : none -- exactly {args.maxiters} iterations")

    # A first fit pays for CUDA context setup, kernel load, and torch's
    # caching allocator reserving the n x k distance block and the workspaces.
    # Measured on 150k x 200, k=64: fit 0 takes 569 ms and fits 1-5 take 327,
    # so ~240 ms of that is one-off.  The warm-up is therefore at FULL SIZE --
    # a small slice warms the context but not the allocator, and leaves most of
    # the one-off cost inside the timed run.
    fit_once(X_t, C0_t, args.kernel, args.kappa, 2, 1e-45, args.seed)

    # mirror mpkmeans_bench: --convergence enables the Frobenius test, and
    # without it the fit runs the full --maxiters
    eff_tol = args.tol if args.convergence else 1e-45
    model, ms = fit_once(X_t, C0_t, args.kernel, args.kappa, args.maxiters,
                         eff_tol, args.seed)
    inert = inertia_fp64(X_t, model.cluster_centers_, model.labels_)
    print(f"{args.kernel:12s} iters={model.n_iter_:4d}  {ms:9.2f} ms  "
          f"inertia={inert:.9e}")
    st = iter_stats(model)
    if st:
        print(f"  per iteration: median {st['median']:.2f} ms  "
              f"(min {st['min']:.2f}, max {st['max']:.2f}, "
              f"first {st['first']:.2f})")
        if st["first"] > 1.5 * st["median"]:
            print(f"  WARNING: iteration 0 cost {st['first']/st['median']:.1f}x "
                  f"the median -- the full-size warm-up did not take, and "
                  f"{ms:.0f} ms includes one-off allocator cost")

    if args.profile:
        pr = profile_fit(X_t, C0_t, args.kernel, args.kappa,
                         min(args.maxiters, 8), args.seed)
        tot = pr["dist"] + pr["update"] + pr["other"]
        print(f"  where it goes, per iteration (separate instrumented fit):")
        print(f"    {'distance (' + args.kernel + ')':24s} {pr['dist']:8.3f} ms"
              f"   {100*pr['dist']/tot:5.1f}%")
        print(f"    {'centroid update':24s} {pr['update']:8.3f} ms"
              f"   {100*pr['update']/tot:5.1f}%")
        print(f"    {'argmin, shift, python':24s} {pr['other']:8.3f} ms"
              f"   {100*pr['other']/tot:5.1f}%")
        # scaled to the timed fit's iteration count, for the ms_dist column
        dist_total = pr["dist"] * (model.n_iter_ + 1)

    dist_total = 0.0 if not args.profile else dist_total
    ref_ms = ref_iters = 0.0
    ref_inert = float("nan")
    label_diff = 0
    rel = float("nan")
    if args.kernel in ("fp32", "fp32_uniform") and args.reference == "fp32":
        # the chosen kernel IS the reference; refitting it would only measure
        # this package's own run-to-run noise
        print("rel inertia vs fp32: 0 by construction (--kernel is fp32)")
        ref_inert, ref_ms, ref_iters, rel = inert, ms, model.n_iter_, 0.0
    elif args.reference == "fp32":
        rmodel, ref_ms = fit_once(X_t, C0_t, "fp32", args.kappa, args.maxiters,
                                  eff_tol, args.seed)
        ref_inert = inertia_fp64(X_t, rmodel.cluster_centers_, rmodel.labels_)
        ref_iters = rmodel.n_iter_
        label_diff = int((model.labels_ != rmodel.labels_).sum().item())
        print(f"{'fp32':12s} iters={rmodel.n_iter_:4d}  {ref_ms:9.2f} ms  "
              f"inertia={ref_inert:.9e}")
        rel = abs(inert - ref_inert) / max(abs(ref_inert), 1e-300)
        signed = (inert - ref_inert) / max(abs(ref_inert), 1e-300)
        print(f"\n  {args.kernel} vs fp32 (same package, same data, same "
              f"centroids, same loop):")
        print(f"    rel inertia diff : {rel:.3e}  ({signed:+.3e} signed -- "
              f"{'worse' if signed > 0 else 'better'} than fp32)")
        print(f"    labels differing : {label_diff} of {n} "
              f"({100.0*label_diff/n:.3f}%)")

    if truth is not None and not args.dump_data:
        print(f"(synthetic: {args.k} planted clusters.  These blobs match "
              f"mpkMakeBlobs' construction but not its RNG, so they are NOT "
              f"the points mpkmeans_bench would generate.  Use --dump-data, or "
              f"run_one.sh, to give it these ones.)")

    # ---- CSV --------------------------------------------------------------
    if args.csv:
        pairs = float(n) * args.k * max(model.n_iter_, 1)
        row = {c: 0 for c in CSV_COLUMNS}
        row.update(
            dataset=name, n=n, d=d, k=args.k, std=args.std, box=args.box,
            seed=args.seed, zscore=int(args.zscore), accum=args.kernel,
            cond=f"rt-mp-k{args.kappa:g}", iters=model.n_iter_, eps=0,
            # their kernel does the test and the fallback inside one call and
            # reports neither, so this is "not measured", not "none eliminated"
            hp_baseline=int(pairs), hp_reference=0, hp_update=int(pairs),
            hp_total=int(pairs), pct_eliminated=0.0,
            violations=0, label_diff=label_diff,
            inertia=f"{inert:.9e}",
            inertia_fp32=f"{ref_inert:.9e}", rel_inertia=f"{rel:.3e}",
            # ms_dist is the distance step only -- the pairwise kernel the
            # Lloyd loop calls, measured inside that loop by profile_fit and
            # scaled to this run's iteration count.  It is a sum of measured
            # stage times, like the benchmark's, but taken from a separately
            # instrumented fit whose syncs perturb it slightly; 0 means
            # --no-profile, i.e. not measured.
            ms_dist=f"{dist_total:.3f}", ms_dist_fp32=0.0, speedup=0.0,
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
