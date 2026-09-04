# Evaluation workflow

Three scripts, run in order. Everything else here is machinery they call.

```sh
./1_prep_datasets.sh                 # download + prepare, ~2 h, ~120 GB
./2_gen_jobs.sh                      # write runs/jobs/*.sbatch and runs/run_all.sh
cd runs && ./run_all.sh              # submit (or ./run_all.sh --local to run in series)
cd .. && ./3_plot.sh                 # runs/plots/*.png and runs/plots/summary.csv
```

`2_gen_jobs.sh --dry-run` counts the matrix without writing anything, and
`--only synth|libsvm|vector` restricts it to one family. `run_all.sh <pattern>`
submits the subset whose job name contains `<pattern>`.

## What gets run

## The rt baseline is the authors' own code

`rt-base` in these runs is **not** our C++ reimplementation. Every bench
invocation is issued with `--skip rt`, and the arXiv:2407.12208 baseline comes
instead from `mp-kmeans` on PyPI — Carson, Chen and Liu's own package for that
paper — driven by `rt_baseline_mp.py` with kernel `fp16_fp32`,
`init_method='random'` (never k-means++) and `normalize=None`. Its rows carry
`cond=rt-mp` and land in `results/<job>.rt.csv`.

It is emitted once per (dataset, k) under the `--accum fp32` arm only: their
kernel is `fp16_fp32` whatever our `--accum` says, so running it under both
would just duplicate the row.

**Environment.** The wheel ships prebuilt CUDA extensions for **cpython-3.12
only**, with no torch pin, and on torch 2.14 every kernel fails
`TORCH_CHECK(P.is_contiguous())` on tensors that are demonstrably contiguous —
an ABI mismatch that presents as a bad argument. What works:

```sh
mamba create -p $SCRATCH/envs/mpk -c conda-forge python=3.12 pip
$SCRATCH/envs/mpk/bin/pip install mp-kmeans
$SCRATCH/envs/mpk/bin/pip install --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.12.1+cu130"
```

`2_gen_jobs.sh` bakes `$SCRATCH/envs/mpk/bin/python` into the jobs; override
with `RT_PYTHON=... ./2_gen_jobs.sh`. A missing interpreter makes the rt rows
be skipped with a message, not the whole sweep fail.

**Reading its numbers.** `rt-mp`'s `inertia_fp32`, `ms_total_fp32` and
`iters_fp32` come from that package's *own* fp32 fit, not from `mpkMeansFP32`,
so a "speedup vs fp32" built from them measures a different thing than every
other row. The plotter therefore keeps `rt-mp` out of the ratio plots and shows
it in the absolute `ms/iteration` plots, where no reference is implied. Its
`hp_*` columns are not measured either: the reliability test and the fallback
happen inside one CUDA call that reports neither count.

Stopping rules now agree. The C++ side was moved onto
`||C - C_prev||_F < tol` — Frobenius over the whole k x d centroid block —
because that is the rule this package uses, and it is also exactly cuVS's
centroid-shift clause, so all three schemes stop on the same quantity. Our loop
additionally breaks when no label changed, which this package does not test,
but stable labels give identical centroids and hence a zero Frobenius shift, so
the two fire on the same iteration.

One thing it still does not share with the C++ bench: its labels come from a
re-assignment after the final update, so they sit half a Lloyd step ahead of
ours — exactly as cuVS's do.

### Why the rt-mp row is ~20x slower, and what it is not

Measured on yandex 150000 x 200, k=64, 20 iterations (A100-SXM4):

| | ms / iteration |
| --- | --- |
| centroid update (`update_centers_fp32_with_reinit`) | **11.25** |
| distance (`pairwise_euclidean_fp16_fp32`, kappa=5) | 4.49 |
| `argmin(dim=1)` | 0.11 |
| centroid-shift norm + `.item()` | 0.04 |
| **total** | **15.9** |

`rt_baseline_mp.py` reports that split itself, per run: it wraps the package's
CUDA entry points for one extra short fit and measures the calls the Lloyd loop
actually makes. (`--no-profile` skips it.) The distance figure lands in the
CSV's `ms_dist`, so `run_one.sh` can put it beside the benchmark's own
distance-step column -- on yandex 150k x 200, k=64 that reads 4.92 ms/iter for
`fp16_fp32` against 0.30 for our FP32 and 0.33-0.52 for the mixed schemes.
| *for scale:* their `pairwise_euclidean_single` | *0.51* |
| *for scale:* a plain `X@C.T` fp16 GEMM, same shape | *0.079* |

**It is not warm-up.** Repeated identical fits in one process: 569, 327, 328,
327, 327, 327 ms. About 240 ms is one-off (context, kernel load, allocator
reserving the n x k block) and the rest is steady state. `rt_baseline_mp.py`
warms up at full size for that reason — a small slice warms the context but not
the allocator.

**It is not the Python loop.** The per-iteration `.item()` host sync costs
0.035 ms, and `argmin` 0.11.

**71% of it is the centroid update** -- recomputing each centroid as the mean
of the points assigned to it -- which has nothing to do with precision. It
costs 7.5-7.8 ms per 100k rows, flat in k between 32 and 256, so it is linear
in n and roughly an order of magnitude slower per row than ours.

**The rest is the fallback, not the FP16 GEMM.** Sweeping kappa at fixed data:
0 -> 0.48 ms, 0.5 -> 0.50, **5 -> 4.43**, 50 -> 3.71, 1e6 -> 3.71. At kappa=0
the mixed kernel matches their own fp32 (0.51); at the paper's kappa=5 it is 9x
that. The reliability test's FP32 direct-formula fallback is the entire
expense — the kernel costs 57x the FP16 GEMM it is built around. (kappa=5 being
slower than kappa=50 is consistent with warp divergence: partial fallback is
worse than total fallback.)

This is the same shape of result as our own reimplementation of the method: it
falls back on ~97% of entries, so it is nearly exact and pays accordingly.

### What kappa buys

kappa is the paper's safety factor rho: the reliability test accepts the
expanded FP16 distance when it exceeds `kappa * E_l`, and otherwise recomputes
that entry with the direct formula in FP32. Scoring the kernel alone against an
FP64 ground truth on converged centroids -- clustering dynamics removed, so only
kernel accuracy is left:

**yandex, n=150k d=200 k=64**

| kappa | max rel err | mean rel err | argmin wrong | ms |
| --- | --- | --- | --- | --- |
| 0 | 5.60e-03 | 7.75e-05 | 70 (0.047%) | 0.48 |
| 1 | 2.25e-03 | 7.68e-05 | 67 | 0.49 |
| 2 | 7.66e-04 | 7.50e-05 | 64 | 0.49 |
| 3 | 5.21e-04 | 6.95e-05 | 39 | 0.85 |
| **5** | 1.78e-04 | **6.88e-07** | **0** | **4.66** |
| 10+ | 1.17e-06 | 1.64e-07 | 0 | 3.7 |
| *their fp32* | *9.58e-06* | *1.57e-07* | *0* | *0.51* |

Monotone, and the paper's kappa = 5 is exactly the knee: the smallest value that
drives argmin errors to zero and the mean error down two orders of magnitude.
It costs 9x on the distance step to get there.

**The knee moves with d, and fast.** The error floor carries gamma_l(d+2) with
u_l = 2^-11, so the threshold grows with the dimension and the kappa at which
everything falls back scales roughly as 1/(d*u_l):

| dataset | d | predicted knee | observed | cost at the knee |
| --- | --- | --- | --- | --- |
| yandex | 200 | ~4.6 | 5 | 0.48 -> 4.66 ms (9.7x) |
| arxiv | 768 | ~0.8 | 1 | 2.48 -> 76.8 ms (31x) |
| wiki | 3072 | ~0.3 | 0.5 | 4.46 -> 78.2 ms (17.5x) |

Past d = 2046, `(d+2)*u_l >= 1` and the model degenerates outright -- the same
limit that made our own C++ rt-base decline those problems. At wiki's d = 3072
the test can never accept anything, so at every kappa the method computes the
FP16 GEMM *and* the full FP32 direct formula, for 78 ms against 4.8 ms for
their plain fp32 kernel. The mixed-precision path is pure overhead there.

One genuine win: at full fallback the kernel is **more accurate than their own
fp32 kernel** (arxiv: 2.4e-06 vs 2.6e-05 max error, and their fp32 gets one
argmin wrong where the fallback gets none). The fallback uses the direct
formula, which does not cancel; `pairwise_euclidean_single` uses the expanded
one, which does.

**But the clustering barely notices, and a single-seed sweep cannot even see
it.** Over full 20-iteration fits from one shared set of centroids the trend is
real but tiny: inertia falls slightly from kappa 0 to 3 and, from kappa 5 on,
matches the fp32 run to ten digits -- which is what the kernel table predicts,
since argmin errors reach zero at 5. The trouble is the size of it:

| | spread in final inertia |
| --- | --- |
| vary kappa 0 -> 1e6, same starting centroids | 7.5e-05 |
| fix kappa = 5, vary starting centroids (8 seeds) | **3.6e-03** |

Choosing a different 64 starting points moves the answer **48x more** than
sweeping kappa over its whole range, so one seed reads a quantity far below its
own noise floor -- which is why the first sweep here showed sign flips and no
order. To measure kappa end to end, average over ~20 initialisations; to
measure the kernel, do not cluster at all (`kappa_kernel.py`).

The substantive point survives either way. At kappa = 0 only 0.047% of points
get the wrong nearest centroid per iteration and Lloyd's absorbs that; label
disagreement against fp32 falls only from 0.51% to 0.33% across the entire
kappa range. The reliability test does exactly what it claims at the kernel
level, and the outer loop is robust enough that you are paying 9x to 31x for
accuracy the clustering mostly does not need.

### run_one.sh — one problem, both implementations, identical inputs

```sh
./run_one.sh --out /tmp/cmp --bin $DATA/data_yandex.bin -d 200 -k 64 \
             --subsample 150000 --zscore --maxiters 400 --convergence
```

It runs the driver first so it can materialise the exact matrix it clusters
(after `--subsample`, before `--zscore`) and the centroids it draws, then points
`mpkmeans_bench` at both with `--bin` and `--init-centroids`. Neither side gets
to draw its own, so the two cluster the same numbers from the same start — which
is the only way `--blobs` can be compared at all, since no two RNGs reproduce
each other's points.

Common flags (`-k -n -d -s -b -e --maxiters --convergence --tol --zscore
--subsample`) go to both; `--kappa` and `--kernel` go to the driver, and
`--bench-extra` / `--rt-extra` pass anything else through verbatim. It prints a
per-iteration comparison and leaves `bench.csv` and `rt.csv` in `--out`.

To pin both sides to one set of initial centroids by hand:

```sh
rt_baseline_mp.py ... --dump-centroids c0.bin
mpkmeans_bench    ... --init-centroids c0.bin     # read in the unshifted frame
```

Without that the two draw their own seeded samples. `--bin` and `--libsvm`
already read byte-identical data; `--blobs` mirrors `mpkMakeBlobs`'
construction but not its RNG, so synthetic runs are distributionally matched,
not identical.

## What gets run

Fixed for every invocation, per `instructions/eval.md`: `--maxiters 400
--convergence`, both accumulators (`--accum fp32|fp16`), and both normalization
states (with and without `--zscore`). Every invocation runs all eight schemes
— (3), (6), (3)+(6), (3)->(6), raw, mw, rt-base, cuvs — and emits one CSV row
each.

| family | source | axes |
| --- | --- | --- |
| synthetic | generated in-process | separation b ∈ {10, 5, 2, 0}, n ∈ {200k, 1M}, d and k over 32…1024 |
| LIBSVM | csie.ntu.edu.tw | k over 32…1024; d is the data's |
| vector indexing | SuperKMeans Drive corpora | k over 32…1024; d is the data's |

`--sweep 1d` (the default) sweeps d at a fixed k and k at a fixed d — 11 points
per (box, n), 544 invocations overall. `--sweep grid` does all 36 (d, k) pairs
instead, 1344 invocations. The instruction reads either way; 1d matches the
existing `scripts/sweep.sh` convention and is what fits in a sitting.

## Granularity

`--granularity config` (default) writes one jobscript per (dataset, separation,
n, accumulator, normalization) with the d/k sweep inside it: 64 jobs, each
appending to one CSV. `--granularity run` splits to one script per single
invocation if you would rather parallelise harder — at several hundred jobs'
worth of scheduler overhead.

## Feasibility — read `runs/infeasible.tsv`

The benchmark holds P on the device twice (shifted for (3)/(6), unshifted for
rt-base and cuvs) and the multiword driver splits it into two fp16 planes, so
the footprint is roughly `12·n·d + 8·n·k + 44·n` bytes. The generator computes
that for every point and, on real data, sets a `--subsample` that fits rather
than emitting a job that will die. Synthetic points over budget are skipped
outright: shrinking one would quietly answer a different question than the
sweep asked.

Two consequences worth knowing before reading any result:

- **news20 cannot be run at all.** It is 19,996 × 1,355,191, and `mpkLoadLibsvm`
  densifies to the largest feature index — 108 GB, past its own 60 GB ceiling,
  on the host, at load time, before subsampling could help. Row count is not
  the problem; the index space is.
- **Subsampled datasets do not share their subsample.** `mpkmeans_bench` draws
  with `std::mt19937` and `rt_baseline_mp.py` with numpy's `Generator`, so for
  mnist8m (always) and arxiv (at k=1024) the `rt-mp` row clusters a different
  subset of the same file than the other schemes. Same size, same
  distribution, different rows. To make them identical, run the driver once
  with `--dump-data` and point both sides at that file with `--bin`, which is
  what `run_one.sh` does automatically. The generator prints which datasets
  this affects.
- **mnist8m is always subsampled**, to between 1.9M and 3.5M of its 8.1M rows
  depending on k. The subsample is recorded in the CSV's `n` column, so the
  plots show what was actually clustered.

## Exit codes

`mpkmeans_bench` exits 2 when its own correctness check does not pass, with the
rows still written. On real data that is usually `raw` — the config with no
exclusion test and no guarantee, which exists precisely to be the thing the
bounds are measured against. The jobscripts count those separately from real
failures and keep going. `runs/plots/summary.csv` carries `max_violations` per
scheme; a nonzero entry there is a genuine correctness bug, and the plotting
script prints a loud warning for it.

## Files

    1_prep_datasets.sh   download + prepare (LIBSVM, and the SuperKMeans
                         corpora via ../scripts/fetch_superkmeans_datasets.sh)
    rt_baseline_mp.py    the rt baseline, via the paper authors' own package
    2_gen_jobs.sh        write the jobscripts and run_all.sh
    3_plot.sh            venv bootstrap + plot_results.py
    plot_results.py      the plotting itself
    lib/datasets.sh      dataset table and memory model, shared by 1 and 2
    runs/                generated: jobs/, results/, logs/, plots/, manifests

Everything is normalised per iteration. The schemes stop on their own
convergence criteria and so run different numbers of iterations on the same
problem; the CSV carries both `iters` and `iters_fp32` so that division can be
done honestly. `ms_dist` is the distance step, which is what the exclusion
conditions change; `ms_total` is the whole fit and is the only column the cuvs
row can be compared on, since cuVS is one opaque library call.
