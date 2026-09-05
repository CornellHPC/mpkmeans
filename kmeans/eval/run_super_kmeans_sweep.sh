#!/usr/bin/env bash
# One-off sweep: the authors' mp-kmeans package (rt_baseline_mp.py) over every
# SuperKMeans vector-indexing dataset that scripts/fetch_superkmeans_datasets.sh
# can produce, at a k sweep and both kappa values.
#
# This is NOT part of the three-script eval/ pipeline (2_gen_jobs.sh
# deliberately restricts the vector family to openai/arxiv/wiki, per
# instructions/eval.md) -- it is a standalone run over all nine datasets in
# the manifest, requested separately.  d is never passed on the command line:
# rt_baseline_mp.py now reads it from manifest.tsv next to the data (see
# resolve_bin_dim there), so there is nothing here to get out of sync with the
# files it downloaded.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib/datasets.sh"   # mem_bytes, max_n_for

DATA="${DATA:-/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets}"
RT_PY="$HERE/rt_baseline_mp.py"
RT_PYTHON="${RT_PYTHON:-/pscratch/sd/j/jbellav/envs/mpk/bin/python}"
OUT="${OUT:-$REPO/eval/runs/super_kmeans_sweep}"
CSV="$OUT/results.csv"
LOG="$OUT/log.txt"
# For an apples-to-apples run through mpkmeans_bench afterward: the exact
# (post-subsample) matrix each cell clustered, plus the exact initial
# centroids each k drew -- --bin + --init-centroids on the bench side then
# starts from the same data, same draw.  Centroids depend on (dataset,k) but
# not kappa (drawn before the kernel choice matters), so one dump per k
# covers both kappa runs; data depends only on the dataset, since every
# dataset here is subsampled by the fixed mp-kmeans n cap below, not by the
# k-dependent memory budget -- verified per-dataset at dump time, not assumed.
DUMP_DIR="${DUMP_DIR:-$DATA/superkmeans-subsampled}"

GPU_MEM_GB="${GPU_MEM_GB:-40}"
HEADROOM=85            # same margin 2_gen_jobs.sh plans mpkmeans_bench with;
                        # the python driver holds fewer copies than the C++
                        # bench this budget was calibrated for, so it is
                        # conservative here, not tight
KS=(32 64 128 256 512 1024)
KAPPAS=(1 5)
MAXITERS=50   # fixed count, no --convergence: both sides run exactly this
              # many iterations, so runtime is bounded and comparable across
              # datasets/k instead of tracking real data's slow convergence
SEED=1
KERNEL=fp16_fp32

# mp-kmeans' own CUDA kernels (not our GPU-memory budget above) break above
# roughly n=1,048,560 -- measured by bisection on arxiv: 1,048,560 rows always
# ran, 1,048,570 and up always failed with `torch.AcceleratorError: CUDA
# error: invalid argument` inside distances_float.argmin(dim=1), regardless
# of k or d.  That is a hard cap in the package under test, not something the
# eval memory model would ever catch (34 GB comfortably fits far more rows at
# small k), so it is enforced separately here.  1e6 is a round number safely
# under the measured boundary.
MP_KMEANS_MAX_N=1000000

[[ -x "$RT_PYTHON" ]] || { echo "no python at $RT_PYTHON (set RT_PYTHON)" >&2; exit 1; }
[[ -s "$DATA/manifest.tsv" ]] || { echo "no manifest at $DATA/manifest.tsv -- run scripts/fetch_superkmeans_datasets.sh first" >&2; exit 1; }

mkdir -p "$OUT" "$DUMP_DIR"
# RESUME=1 (default): keep any existing CSV and skip (dataset,k,kappa) rows
# already in it, rather than starting over -- a run can span several walltime
# allocations.  RESUME=0 starts clean.
RESUME="${RESUME:-1}"
if [[ "$RESUME" == 0 ]]; then
    : > "$LOG"
    rm -f "$CSV"
else
    : >> "$LOG"
fi

BUDGET=$(( GPU_MEM_GB * 1000000000 * HEADROOM / 100 ))
COMMON=(--maxiters "$MAXITERS" -e "$SEED" --kernel "$KERNEL")
# Under an existing allocation, run this through srun (set SRUN="srun --jobid=
# ... -n1 -G1"); left empty, the driver runs directly on whatever node the
# shell is already on.
SRUN=${MPK_SRUN-}

echo "python    : $RT_PYTHON"
echo "data      : $DATA"
echo "out       : $OUT"
echo "dumps     : $DUMP_DIR/<id>/data.bin + centroids_k<k>.bin"
echo "budget    : $(( BUDGET / 1000000000 )) GB of a ${GPU_MEM_GB} GB card (${HEADROOM}%)"
echo "k sweep   : ${KS[*]}"
echo "kappa     : ${KAPPAS[*]}"
echo

n_run=0; n_skip=0
# Read the manifest on fd 3, not stdin: srun below inherits stdin from this
# shell, and a plain `while read ... done < manifest.tsv` hands srun (and
# transitively rt_baseline_mp.py) the *same* fd -- the first srun call reads
# straight through the rest of the manifest and this loop silently ends after
# one dataset.  Measured: exactly that happened, 12/12 cells for the first
# row (arxiv) and nothing else, no error, exit 0.
while IFS=$'\t' read -r -u 3 id hdf5 n d test_n normalized train_bytes test_bytes; do
    [[ "$id" == "id" ]] && continue
    path="$DATA/data/data_${id}.bin"
    if [[ ! -s "$path" ]]; then
        echo "SKIP $id: $path not present" | tee -a "$LOG"
        n_skip=$(( n_skip + 1 ))
        continue
    fi
    dsdir="$DUMP_DIR/$id"
    mkdir -p "$dsdir"
    dumped_data_sub=""   # subsample size the currently-dumped data.bin has
    for k in "${KS[@]}"; do
        if [[ $k -ge $n ]]; then
            echo "SKIP $id k=$k: k >= n=$n" | tee -a "$LOG"
            n_skip=$(( n_skip + 1 ))
            continue
        fi
        need=$(mem_bytes "$n" "$d" "$k")
        sub=$n
        reason=""
        if [[ $need -gt $BUDGET ]]; then
            mem_sub=$(max_n_for "$d" "$k" "$BUDGET")
            if [[ $mem_sub -lt $(( k * 10 )) || $mem_sub -lt 1000 ]]; then
                echo "SKIP $id k=$k: needs $(( need/1000000000 ))GB > budget, and the subsample that would fit ($mem_sub) is too small" | tee -a "$LOG"
                n_skip=$(( n_skip + 1 ))
                continue
            fi
            sub=$mem_sub
            reason="fit budget"
        fi
        if [[ $sub -gt $MP_KMEANS_MAX_N ]]; then
            sub=$MP_KMEANS_MAX_N
            reason="mp-kmeans n cap"
        fi
        extra=()
        if [[ $sub -lt $n ]]; then
            extra=(--subsample "$sub")
            echo "  ($id k=$k: subsampling $n -> $sub -- $reason)" | tee -a "$LOG"
        fi
        for kap in "${KAPPAS[@]}"; do
            fname="$(basename "$path")"
            if [[ "$RESUME" != 0 && -s "$CSV" ]] && awk -F, -v f="$fname" -v k="$k" -v c="rt-mp-k$kap" \
                    'NR>1 && $1==f && $4==k && $10==c {found=1} END{exit !found}' "$CSV"; then
                echo "SKIP $id k=$k kappa=$kap: already in $CSV" | tee -a "$LOG"
                continue
            fi
            dump_args=()
            if [[ "$kap" == "${KAPPAS[0]}" ]]; then
                # centroids depend on (dataset,k), not kappa -- dump once here,
                # reused by every other kappa at this k
                dump_args+=(--dump-centroids "$dsdir/centroids_k${k}.bin")
                if [[ "$dumped_data_sub" != "$sub" ]]; then
                    [[ -n "$dumped_data_sub" ]] && echo "  WARNING: $id subsample size changed ($dumped_data_sub -> $sub) at k=$k -- re-dumping data.bin; centroids dumped at earlier k no longer match it" | tee -a "$LOG"
                    dump_args+=(--dump-data "$dsdir/data.bin")
                    dumped_data_sub=$sub
                fi
            fi
            echo "=== $id  n=$n d=$d k=$k kappa=$kap ===" | tee -a "$LOG"
            # shellcheck disable=SC2086
            if ! $SRUN "$RT_PYTHON" "$RT_PY" --bin "$path" "${COMMON[@]}" \
                    -k "$k" --kappa "$kap" "${extra[@]}" "${dump_args[@]}" --csv "$CSV" \
                    </dev/null >>"$LOG" 2>&1; then
                echo "  FAILED (see $LOG)" | tee -a "$LOG"
            fi
            n_run=$(( n_run + 1 ))
        done
    done
done 3< "$DATA/manifest.tsv"

echo
echo "ran $n_run invocation(s), skipped $n_skip"
echo "CSV: $CSV"
echo "log: $LOG"
