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
