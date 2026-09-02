# scripts/

Two of these are the ones you run after a change. The rest are diagnostics you
reach for only when a number moves and you want to know why.

Everything needs a GPU, so everything goes through `srun`. Get an allocation
once and reuse it:

```
salloc -A m1266_g -C gpu -q interactive -t 60 -N 1 --no-shell
squeue -u $USER -o "%.10i %.9T %.9N %.10L"      # note the JOBID and TIME_LEFT
srun --jobid <JOBID> --ntasks=1 --gpus=1 ./scripts/check.sh
```

`jrun.sh` wraps that last line: `JOBID=<id> ./scripts/jrun.sh ./scripts/check.sh`.

Allocations expire (they have run out mid-task more than once), so check
`TIME_LEFT` before starting a long sweep and batch runs into one `srun`.

## The two builds

| | configure with | has | use for |
|---|---|---|---|
| `build/` | `-DCMAKE_BUILD_TYPE=Release` | nothing extra | **timing** |
| `build-stats/` | `... -DMPK_STATS=ON` | attribution counters + the oracle | **correctness**, elimination rates |

The stats build is slower on purpose. Never quote a timing from it, and never
quote an elimination rate or a violation count from the other one.

```
cmake -S . -B build       -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-stats -DCMAKE_BUILD_TYPE=Release -DMPK_STATS=ON
cmake --build build -j && cmake --build build-stats -j
```

---

## (a) Correctness — `check.sh`

```
srun --jobid <JOBID> --ntasks=1 --gpus=1 ./scripts/check.sh
```

Seven shapes x four configurations. Each run checks **every row of every
iteration** against the ordinary FP32 implementation — a full `cublasSgemm`
under `CUBLAS_PEDANTIC_MATH` plus the same row argmin `mpkMeansFP32` uses,
evaluated on the mixed run's own centroids. Exits non-zero on any failure, so
it works as a gate.

Two columns, and they mean different things:

- **`violations` must be 0.** The FP32 answer was neither the incumbent nor in
  the survivor list, so a condition of Theorem 1 does not hold. Any non-zero
  value means the bound or the filter is wrong. This is the real test.
- **`labeldiff` need not be 0.** The FP32 answer *was* reachable, but a
  near-tie ordered differently, because the survivor inner product and cuBLAS
  sum in different orders and disagree in the last bits. A few per 10^5 is
  normal and it varies with the update path. Watch it for sudden jumps, not for
  being non-zero.

The shapes are chosen to hit different code paths, so don't trim the list
casually:

| shape | why |
|---|---|
| `-n 20000 -d 32 -k 16` | `k<=32`: the `NM=1` register path |
| `-n 200000 -d 128 -k 64` | the reference point, `NM=2` |
| `-n 200000 -d 128 -k 256` | `k>64`: the `IPT=0` streaming path, which re-reads the row from L1 |
| `-n 50000 -d 512 -k 32 -b 3` | large `d`, long inner products |
| `-n 100000 -d 64 -k 128 -b 2` | overlapping clusters, many survivors |
| `-n 200000 -d 128 -k 64 -b 0` | single blob: exercises survivor-list growth |
| `-n 200000 -d 128 -k 64 -e 7` | a different random draw |

If you change the kernels, run this before you look at any timing.

## (b) Performance — `bench.sh`

```
srun --jobid <JOBID> --ntasks=1 --gpus=1 ./scripts/bench.sh
```

ms/iteration of the distance step (centroid update excluded), as the **minimum
of 5 interleaved samples**. Interleaving is not optional: run order moves
results by ~10% at `k=256`, which is enough to invert the ranking between two
configurations. A single sample once told me the cascade beat `(3)`; repeated
properly, it does not.

It reports two regimes, because a change can easily help one and hurt the
other:

- **`box=10`, separated blobs.** Survivors are ~0.3 per row, so nearly every
  row is settled by the bound alone and the **argmin dominates**.
- **`box=0`, one single Gaussian blob.** Survivors run to millions, so the
  **update kernel and the survivor list dominate**. Everything loses to FP32
  here — the number to watch is whether your change moved it, not whether it
  clears 1.000x.

Points start at `d=64, k=64` because that is the worst case found so far:
condition (6)'s fixed `O(nd)` cost is amortised over an `O(ndk)` GEMM, so the
margin shrinks with both `d` and `k`. At `d=64, k=64` separated, the best
configuration sits at ~0.97x — it actually loses. If you only measure
`d=128, k=256` you will think everything is fine.

Knobs: `REPS`, `N`, `I`, `PTS` (as `d:k` pairs), `BOXES`.

```
REPS=3 PTS="64:64 128:64" BOXES=10 ./scripts/bench.sh    # quick check
```

---

## Real data and convergence runs

Both gates above use synthetic blobs, which is what you want while changing
kernels — the shape is a knob. For a run on real data, or for iterating to
convergence instead of a fixed count, the bench takes these directly:

| flag | meaning |
|---|---|
| `--dataset <file>` | LIBSVM/SVMlight sparse text, densified to `n x d`. `n` is the number of non-empty lines and `d` the largest feature index, both read from the file; `-n` and `-d` are ignored. The label column becomes the ground truth for the `truth` column, renumbered in order of first appearance, and its class count need not equal `k`. |
| `--maxiters <int>` | same as `-i`. |
| `--convergence` | stop at `max_j ||c_j - c_j_prev||_2 < tol` or at `--maxiters`, whichever comes first, instead of at label stability. |
| `--tol <float>` | that tolerance, default `1e-8`. |
| `--theta <float>` | the `rt-base` safety factor of (4.13), default 5 (the paper's value). Must be `> 2`. |

```
./build-stats/mpkmeans_bench --dataset data/foo.libsvm -k 64 \
    --convergence --maxiters 300
```

Two things to know before reading the output of a convergence run:

- At `1e-8` in FP32 the tolerance is essentially "the centroids did not move",
  so it usually fires on the same iteration as label stability. It separates
  from it only at looser tolerances — `1e-1` on the single-blob case stops at
  15 iterations where label stability takes 269.
- Configurations may stop at *different* iterations, so compare `ms/iter`, not
  the run totals. The `iters` column is printed for exactly this reason.

## Reading the timing tables

Three tables come out of every run, and they answer different questions.

- **`timing`** — totals over the run, per phase. Only comparable between
  configurations that ran the same number of iterations.
- **`ms/iter` / `vs fp32`** — the headline. `update` beside it is the centroid
  recomputation, which is identical code in every scheme and excluded from
  `ms/iter`; it should read roughly the same for all of them.
- **`refinement cost, normalised`** — `entries/iter`, `ms/iter`, `ns/entry` for
  the FP32 refinement. Every scheme here refines with the *same* kernel (one
  warp per flagged entry, `WPB=8`, 256 threads, the same grid cap), differing
  only in whether it forms `p.c` or `||p-c||^2`. So a scheme's refinement cost
  is a **count**, not a rate, and `ns/entry` is the check on that: if two
  schemes disagree there, something other than the count is at work and the
  comparison is not apples to apples. In practice they agree to within a
  factor of two, and `rt-base` sits at the cheap end because its flagged
  entries come in dense runs of consecutive `j` within a row.

  The exception is a nearly empty kernel: at `-b 0`, `rt-base` flags ~117
  entries per iteration and `ns/entry` reads ~80, which is launch overhead
  divided by almost nothing, not a per-entry cost.

---

## Diagnostics

Reach for these when a gate number moves and you want the cause. None of them
are gates; none of them exit non-zero.

| script | question it answers |
|---|---|
| `sweep_d.sh` | How do the GEMM and the argmin scale with `d` at fixed `k`? This is what established that condition (6)'s extra cost is exactly an `n*d*4` FP32 read of `P` running at ~1.5 TB/s, and that (3) is flat in `d` because it never touches `P`. |
| `sweep_cascade.sh` | `(3)`, `(6)`, `(3)+(6)`, `(3)->(6)` against `d` at `k=64`. Shows the cascade's argmin is nearly flat in `d` where unconditional `(3)+(6)` grows linearly. |
| `sweep_cascade_k.sh` | The same four against `k` at `d=128`, min over interleaved repeats. Use it when you touch the cascade. |
| `sweep_hard.sh` | How elimination degrades as clusters stop being separated (`-b` from 10 down to 0). Shows (3) collapsing far faster than (6), and the cascade degenerating into plain `(3)+(6)` once (3) stops clearing rows. Env: `N D K I`. |
| `sweep_blob.sh` | Single blob, two axes: `std` (which changes nothing — the non-negativity shift is proportional to it, so the problem is scale invariant) and `d` (which is what actually makes it hard). |
| `sweep.sh` | The broad CSV sweep over `k`, `d`, separation, `n` and seed, for `RESULTS.md`. Writes `results/sweep.csv`. Slow. Uses `build-stats`, so its timings are not comparable to `bench.sh`. |
| `jrun.sh` | `JOBID=<id> ./scripts/jrun.sh <cmd>` — runs a command on an existing allocation. |

## Reading the bench output directly

`mpkmeans_bench` on its own prints, in order: the elimination table (high
precision evaluations per iteration, the headline metric), the exclusion
attribution, the phase timings, ms/iter vs FP32, the oracle check, and the
clustering comparison. `--only 3|6|36|c` runs one configuration, `--no-verify`
skips the oracle, `--csv` emits one machine-readable line per configuration
(`--csv-header` for the field names).

`-b` is the knob that controls difficulty, not `-s`. See `sweep_blob.sh`.
