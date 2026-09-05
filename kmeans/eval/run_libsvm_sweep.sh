#!/usr/bin/env bash
# One-off sweep, LIBSVM family: the authors' mp-kmeans package
# (rt_baseline_mp.py) over every LIBSVM dataset named in instructions/eval.md
# (mnist8m, covtype, Wiki10-31K, news20, real-sim, epsilon), at a k sweep and
# both kappa values.  Mirrors run_super_kmeans_sweep.sh exactly (same k
# sweep, same kappas, same fixed-50-iteration stopping rule, same GPU-memory
# and mp-kmeans-n-cap subsampling, same dump-for-mpkmeans_bench convention) --
# only the dataset source and its feasibility check differ.
#
# LIBSVM_SETS is a bash array (eval/lib/datasets.sh), not a file read on
# stdin, so this script does NOT need the fd-3 workaround
# run_super_kmeans_sweep.sh needs for manifest.tsv -- there is no stdin for
# srun to steal here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib/datasets.sh"   # mem_bytes, max_n_for, LIBSVM_SETS, LIBSVM_DENSE_LIMIT

DATA="${DATA:-/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets}"
RT_PY="$HERE/rt_baseline_mp.py"
RT_PYTHON="${RT_PYTHON:-/pscratch/sd/j/jbellav/envs/mpk/bin/python}"
OUT="${OUT:-$REPO/eval/runs/libsvm_sweep}"
CSV="$OUT/results.csv"
LOG="$OUT/log.txt"
# Same rationale as run_super_kmeans_sweep.sh: exact centroids per (dataset,k)
# and the exact densified+subsampled matrix per dataset, for a later
# mpkmeans_bench --bin ... --init-centroids ... run from the same start.
DUMP_DIR="${DUMP_DIR:-$DATA/libsvm-subsampled}"

GPU_MEM_GB="${GPU_MEM_GB:-40}"
HEADROOM=85
KS=(32 64 128 256 512 1024)
KAPPAS=(1 5)
MAXITERS=50   # fixed count, no --convergence -- see run_super_kmeans_sweep.sh
SEED=1
KERNEL=fp16_fp32

# mp-kmeans' own CUDA kernels fail above n~1,048,560 regardless of k or d --
# see run_super_kmeans_sweep.sh for how this was found.
MP_KMEANS_MAX_N=1000000

[[ -x "$RT_PYTHON" ]] || { echo "no python at $RT_PYTHON (set RT_PYTHON)" >&2; exit 1; }
[[ -d "$DATA/libsvm" ]] || { echo "no $DATA/libsvm -- run: eval/1_prep_datasets.sh --only libsvm" >&2; exit 1; }

mkdir -p "$OUT" "$DUMP_DIR"
RESUME="${RESUME:-1}"
if [[ "$RESUME" == 0 ]]; then
    : > "$LOG"
    rm -f "$CSV"
else
    : >> "$LOG"
fi

BUDGET=$(( GPU_MEM_GB * 1000000000 * HEADROOM / 100 ))
COMMON=(--maxiters "$MAXITERS" -e "$SEED" --kernel "$KERNEL")
SRUN=${MPK_SRUN-}

echo "python    : $RT_PYTHON"
echo "data      : $DATA/libsvm"
echo "out       : $OUT"
echo "dumps     : $DUMP_DIR/<id>/data.bin + centroids_k<k>.bin"
echo "budget    : $(( BUDGET / 1000000000 )) GB of a ${GPU_MEM_GB} GB card (${HEADROOM}%)"
echo "k sweep   : ${KS[*]}"
echo "kappa     : ${KAPPAS[*]}"
echo

n_run=0; n_skip=0
for row in "${LIBSVM_SETS[@]}"; do
    IFS='|' read -r id url bz name n d <<<"$row"
    path="$DATA/libsvm/$name"
    if [[ ! -s "$path" ]]; then
        echo "SKIP $id: $path not present" | tee -a "$LOG"
        n_skip=$(( n_skip + 1 ))
        continue
    fi
    dense=$(( n * d * 4 ))
    if [[ $dense -gt $LIBSVM_DENSE_LIMIT ]]; then
        # mpkLoadLibsvm (and our own load_libsvm, which mirrors it) densifies
        # the WHOLE file to its largest feature index before any subsampling
        # can help -- this is a host-memory wall hit at load time, not
        # something --subsample works around.  Same finding this repo already
        # made for the C++ bench (see eval/lib/datasets.sh): news20's 108 GB
        # dense form exceeds the 60 GB the loader is willing to build.
        echo "SKIP $id: dense $(( dense/1000000000 ))GB exceeds the ${LIBSVM_DENSE_LIMIT}-byte densify limit -- not a --subsample-able problem, the loader builds the full matrix before subsampling runs" | tee -a "$LOG"
        n_skip=$(( n_skip + 1 ))
        continue
    fi
    dsdir="$DUMP_DIR/$id"
    mkdir -p "$dsdir"
    dumped_data_sub=""
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
            fname="$name"
            if [[ "$RESUME" != 0 && -s "$CSV" ]] && awk -F, -v f="$fname" -v k="$k" -v c="rt-mp-k$kap" \
                    'NR>1 && $1==f && $4==k && $10==c {found=1} END{exit !found}' "$CSV"; then
                echo "SKIP $id k=$k kappa=$kap: already in $CSV" | tee -a "$LOG"
                continue
            fi
            dump_args=()
            if [[ "$kap" == "${KAPPAS[0]}" ]]; then
                dump_args+=(--dump-centroids "$dsdir/centroids_k${k}.bin")
                if [[ "$dumped_data_sub" != "$sub" ]]; then
                    [[ -n "$dumped_data_sub" ]] && echo "  WARNING: $id subsample size changed ($dumped_data_sub -> $sub) at k=$k -- re-dumping data.bin; centroids dumped at earlier k no longer match it" | tee -a "$LOG"
                    dump_args+=(--dump-data "$dsdir/data.bin")
                    dumped_data_sub=$sub
                fi
            fi
            echo "=== $id  n=$n d=$d k=$k kappa=$kap ===" | tee -a "$LOG"
            # shellcheck disable=SC2086
            if ! $SRUN "$RT_PYTHON" "$RT_PY" --libsvm "$path" "${COMMON[@]}" \
                    -k "$k" --kappa "$kap" "${extra[@]}" "${dump_args[@]}" --csv "$CSV" \
                    </dev/null >>"$LOG" 2>&1; then
                echo "  FAILED (see $LOG)" | tee -a "$LOG"
            fi
            n_run=$(( n_run + 1 ))
        done
    done
done

echo
echo "ran $n_run invocation(s), skipped $n_skip"
echo "CSV: $CSV"
echo "log: $LOG"
