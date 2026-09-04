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
| `update_centers_fp32_with_reinit` | **11.25** |
| `pairwise_euclidean_fp16_fp32` (kappa=5) | 4.49 |
| `argmin(dim=1)` | 0.11 |
| centroid-shift norm + `.item()` | 0.04 |
| **total** | **15.9** |
| *for scale:* their `pairwise_euclidean_single` | *0.51* |
| *for scale:* a plain `X@C.T` fp16 GEMM, same shape | *0.079* |

**It is not warm-up.** Repeated identical fits in one process: 569, 327, 328,
327, 327, 327 ms. About 240 ms is one-off (context, kernel load, allocator
reserving the n x k block) and the rest is steady state. `rt_baseline_mp.py`
warms up at full size for that reason — a small slice warms the context but not
the allocator.

**It is not the Python loop.** The per-iteration `.item()` host sync costs
0.035 ms, and `argmin` 0.11.

**71% of it is the centroid update**, which has nothing to do with precision.
It costs 7.5-7.8 ms per 100k rows, flat in k between 32 and 256, so it is
linear in n and roughly an order of magnitude slower per row than our own
M-step.

**The rest is the fallback, not the FP16 GEMM.** Sweeping kappa at fixed data:
0 -> 0.48 ms, 0.5 -> 0.50, **5 -> 4.43**, 50 -> 3.71, 1e6 -> 3.71. At kappa=0
the mixed kernel matches their own fp32 (0.51); at the paper's kappa=5 it is 9x
that. The reliability test's FP32 direct-formula fallback is the entire
expense — the kernel costs 57x the FP16 GEMM it is built around. (kappa=5 being
slower than kappa=50 is consistent with warp divergence: partial fallback is
worse than total fallback.)

This is the same shape of result as our own reimplementation of the method: it
falls back on ~97% of entries, so it is nearly exact and pays accordingly.

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
