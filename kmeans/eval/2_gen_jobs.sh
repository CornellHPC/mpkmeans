#!/usr/bin/env bash
# (2/3) Generate the SLURM jobscripts for the evaluation.
#
# One script per configuration under <out>/jobs/, plus <out>/run_all.sh which
# submits every one of them.  Each script is self-contained and can be sbatch'd
# on its own, so a single configuration can be re-run without touching the rest.
#
# A "configuration" here is one (dataset, separation, n, accumulator,
# normalization) point; the sweep over d and k runs inside it, appending to one
# CSV.  --granularity run splits that further, to one script per single
# mpkmeans_bench invocation, if you want to parallelise harder at the cost of
# several hundred jobs' worth of scheduler overhead.
#
# Fixed for every run, per instructions/eval.md:
#   --maxiters 400 --convergence      and both accumulators, both --zscore states
#
# Nothing is silently dropped.  A configuration that cannot fit in GPU memory is
# either given a --subsample that makes it fit -- recorded in the CSV and in the
# manifest -- or written to <out>/infeasible.tsv with the arithmetic that ruled
# it out.  Read that file before believing the evaluation is complete.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib/datasets.sh"

DATA=/global/homes/j/jbellav/m4646/hoon/mpkmeans-dsets
OUT="$REPO/eval/runs"
BIN="$REPO/build/mpkmeans_bench"
ACCOUNT=m1266
QOS=shared
WALLTIME=06:00:00
CONSTRAINT=gpu
GPU_MEM_GB=40
HEADROOM=85           # percent of the card we are willing to plan for
GRANULARITY=config
SWEEP=1d              # 1d: sweep d at fixed k and k at fixed d; grid: all pairs
PIVOT_D=128
PIVOT_K=128
DIMS=(32 64 128 256 512 1024)
KS=(32 64 128 256 512 1024)
NS=(200000 1000000)
BOXES=(10 5 2 0)
SEED=1
ONLY_FAMILY=""        # synth | libsvm | vector; empty means all

usage() {
    cat <<USAGE
usage: $(basename "$0") [options]

  --data DIR        prepared datasets (default $DATA)
  --out DIR         where to write jobs/, results/, logs/ (default $OUT)
  --bin PATH        mpkmeans_bench (default $BIN)
  --account A       SLURM account   (default $ACCOUNT)
  --qos Q           SLURM qos       (default $QOS)
  --time T          walltime        (default $WALLTIME)
  --gpu-mem GB      card size for the feasibility model (default $GPU_MEM_GB)
  --sweep 1d|grid   1d sweeps d at k=$PIVOT_K and k at d=$PIVOT_D;
                    grid does every (d,k) pair.  1d is $(( ${#DIMS[@]} + ${#KS[@]} - 1 )) points per
                    (box,n), grid is $(( ${#DIMS[@]} * ${#KS[@]} )).  (default $SWEEP)
  --granularity config|run
                    one script per (dataset,box,n,accum,zscore) with the sweep
                    inside, or one per single invocation (default $GRANULARITY)
  --dims "..."      override the d sweep   (default ${DIMS[*]})
  --ks "..."        override the k sweep   (default ${KS[*]})
  --ns "..."        override the n values  (default ${NS[*]})
  --boxes "..."     override the separations (default ${BOXES[*]})
  --pivot-d D / --pivot-k K   the fixed axis in a 1d sweep
  --only F          restrict to one family: synth, libsvm or vector
  --dry-run         count the jobs, write nothing

Synthetic axes: box ${BOXES[*]}, n ${NS[*]}, d ${DIMS[*]}, k ${KS[*]}.
Real datasets sweep k only -- d is whatever the data is.
USAGE
}

DRY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --bin) BIN="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_GB="$2"; shift 2 ;;
        --sweep) SWEEP="$2"; shift 2 ;;
        --granularity) GRANULARITY="$2"; shift 2 ;;
        --dims) read -r -a DIMS <<<"$2"; shift 2 ;;
        --ks) read -r -a KS <<<"$2"; shift 2 ;;
        --ns) read -r -a NS <<<"$2"; shift 2 ;;
        --boxes) read -r -a BOXES <<<"$2"; shift 2 ;;
        --pivot-d) PIVOT_D="$2"; shift 2 ;;
        --pivot-k) PIVOT_K="$2"; shift 2 ;;
        --only) ONLY_FAMILY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

BUDGET=$(( GPU_MEM_GB * 1000000000 * HEADROOM / 100 ))
JOBS="$OUT/jobs"; RESULTS="$OUT/results"; LOGS="$OUT/logs"
INFEAS="$OUT/infeasible.tsv"; MANIFEST="$OUT/manifest.tsv"

echo "bin       : $BIN"
echo "data      : $DATA"
echo "out       : $OUT"
echo "budget    : $(( BUDGET / 1000000000 )) GB of a ${GPU_MEM_GB} GB card (${HEADROOM}%)"
echo "sweep     : $SWEEP    granularity: $GRANULARITY"

if [[ $DRY -eq 0 ]]; then
    [[ -x "$BIN" ]] || { echo "no such binary: $BIN (build it first)" >&2; exit 1; }
    mkdir -p "$JOBS" "$RESULTS" "$LOGS"
    printf 'job\tfamily\tdataset\tbox\tn\tsweep\taccum\tzscore\truns\tpeak_gb\n' > "$MANIFEST"
    printf 'family\tdataset\tn\td\tk\tneed_gb\tbudget_gb\treason\n' > "$INFEAS"
fi

NJOB=0; NRUN=0; NSKIP=0
SUBSAMPLED_SETS=""

# ---------------------------------------------------------------- emit ----
# Accumulates bench argument lines into CUR_RUNS, then writes one script.
CUR_RUNS=()
CUR_RT=()
CUR_PEAK=0

emit_job() {  # name family dataset box n sweeplabel accum zscore
    local name=$1 family=$2 dset=$3 box=$4 n=$5 slabel=$6 accum=$7 z=$8
    [[ ${#CUR_RUNS[@]} -eq 0 ]] && { CUR_RT=(); return 0; }
    NJOB=$(( NJOB + 1 ))
    NRUN=$(( NRUN + ${#CUR_RUNS[@]} ))
    if [[ $DRY -eq 1 ]]; then CUR_RUNS=(); CUR_RT=(); CUR_PEAK=0; return 0; fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$family" "$dset" "$box" "$n" "$slabel" "$accum" "$z" \
        "${#CUR_RUNS[@]}" "$(( CUR_PEAK / 1000000000 ))" >> "$MANIFEST"

    {
        cat <<HDR
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH -q ${QOS}
#SBATCH -t ${WALLTIME}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -J ${name}
#SBATCH -o ${LOGS}/${name}.out
#SBATCH -e ${LOGS}/${name}.err
# Generated by eval/2_gen_jobs.sh -- edit that, not this.
#   family ${family}   dataset ${dset}   accum ${accum}   zscore ${z}
#   ${#CUR_RUNS[@]} invocations, peak device footprint ~$(( CUR_PEAK / 1000000000 )) GB
set -uo pipefail

BIN=${BIN}
OUT=${RESULTS}/${name}.csv

# Header first, then one row per config per invocation.  Written fresh each
# time so a re-run replaces its own results instead of doubling them.
"\$BIN" --csv-header > "\$OUT"

# Under sbatch the work goes through srun.  Set MPK_SRUN to something else --
# or to empty -- to run the binary directly, which is what you want inside an
# existing interactive allocation, where a nested srun would just queue behind
# itself.
SRUN=\${MPK_SRUN-srun -n1 -G1}

# mpkmeans_bench exits 2 when its own correctness check does not pass -- the
# rows are still written, and on real data the 'raw' config (no exclusion test,
# no guarantee) trips it routinely by design.  That is a result, not a failure.
# Anything else is a genuine failure and the job reports it.
RT_PY=${RT_PY}
RT_PYTHON=${RT_PYTHON}
RT_OUT=${RESULTS}/${name}.rt.csv
rm -f "\$RT_OUT"

fail=0
checks=0
run() {
    echo "+ \$*" >&2
    \$SRUN "\$BIN" --csv "\$@" >> "\$OUT"
    local rc=\$?
    case \$rc in
        0) ;;
        2) echo "CHECK NOT PASSED (rows kept): \$*" >&2; checks=\$(( checks + 1 )) ;;
        *) echo "FAILED rc=\$rc: \$*" >&2; fail=1 ;;
    esac
}

# The arXiv:2407.12208 baseline, from its authors' package.  A missing or
# broken environment must not take the whole sweep down with it, so this is
# reported and skipped rather than fatal -- see eval/README.md for the venv.
rt_ok=1
if [[ ! -x "\$RT_PYTHON" ]]; then
    echo "rt baseline SKIPPED: no python at \$RT_PYTHON" >&2
    rt_ok=0
fi
run_rt() {
    [[ \$rt_ok -eq 1 ]] || return 0
    echo "+ rt \$*" >&2
    \$SRUN "\$RT_PYTHON" "\$RT_PY" --csv "\$RT_OUT" "\$@" \
        || { echo "rt baseline FAILED: \$*" >&2; checks=\$(( checks + 1 )); }
}

HDR
        printf '%s\n' "${CUR_RUNS[@]}"
        if [[ ${#CUR_RT[@]} -gt 0 ]]; then
            printf '\n# --- rt baseline: mp-kmeans, kernel fp16_fp32 ---\n'
            printf '%s\n' "${CUR_RT[@]}"
        fi
        cat <<'FTR'

if [[ $fail -ne 0 ]]; then
    echo "one or more invocations failed outright; see the .err log" >&2
    exit 1
fi
if [[ $checks -ne 0 ]]; then
    echo "$checks invocation(s) did not pass the correctness check; rows kept."
    echo "On real data that is usually the 'raw' config, which has no guarantee."
    echo "Check rel_inertia and violations per cond in the plots' summary.csv."
fi
echo "ok"
FTR
    } > "$JOBS/$name.sbatch"
    chmod +x "$JOBS/$name.sbatch"
    CUR_RUNS=(); CUR_RT=(); CUR_PEAK=0
}

# Adds one invocation if it fits, else records why not.
# add_run family dataset n d k <extra bench args...>
add_run() {
    local family=$1 dset=$2 n=$3 d=$4 k=$5; shift 5
    local extra="$*"
    local need sub

    if [[ $k -ge $n ]]; then
        [[ $DRY -eq 0 ]] && printf '%s\t%s\t%s\t%s\t%s\t-\t-\tk >= n\n' \
            "$family" "$dset" "$n" "$d" "$k" >> "$INFEAS"
        NSKIP=$(( NSKIP + 1 )); return 0
    fi

    need=$(mem_bytes "$n" "$d" "$k")
    if [[ $need -gt $BUDGET ]]; then
        if [[ "$family" == synth ]]; then
            # synthetic sizes are ours to choose; shrinking one would silently
            # answer a different question than the sweep asked
            [[ $DRY -eq 0 ]] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tover budget\n' \
                "$family" "$dset" "$n" "$d" "$k" \
                "$(( need / 1000000000 ))" "$(( BUDGET / 1000000000 ))" >> "$INFEAS"
            NSKIP=$(( NSKIP + 1 )); return 0
        fi
        # real data: cluster as much of it as fits, and say so
        sub=$(max_n_for "$d" "$k" "$BUDGET")
        if [[ $sub -lt $(( k * 10 )) || $sub -lt 1000 ]]; then
            [[ $DRY -eq 0 ]] && printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tsubsample %s too small\n' \
                "$family" "$dset" "$n" "$d" "$k" \
                "$(( need / 1000000000 ))" "$(( BUDGET / 1000000000 ))" "$sub" >> "$INFEAS"
            NSKIP=$(( NSKIP + 1 )); return 0
        fi
        extra="$extra --subsample $sub"
        SUBSAMPLED_SETS="$SUBSAMPLED_SETS $dset"
        need=$(mem_bytes "$sub" "$d" "$k")
    fi

    [[ $need -gt $CUR_PEAK ]] && CUR_PEAK=$need
    CUR_RUNS+=("run -k $k $extra")
    # The same point through the authors' implementation.  --skip rt above
    # removed our reimplementation; this puts the real one back.  Their kernel
    # is fp16_fp32 whatever OUR --accum says, so it is emitted once, under the
    # fp32 arm, rather than duplicated across both.
    [[ "$extra" != *"--accum fp32"* ]] && return 0
    local rt_extra="$extra"
    rt_extra="${rt_extra//--dataset /--libsvm }"
    rt_extra="${rt_extra//--accum fp32/}"
    rt_extra="${rt_extra//--accum fp16/}"
    rt_extra="${rt_extra//--skip rt/}"
    [[ "$family" == synth ]] && rt_extra="--blobs $rt_extra"
    CUR_RT+=("run_rt -k $k $rt_extra")
}

# The arXiv:2407.12208 baseline comes from its authors' own package via
# eval/rt_baseline_mp.py.  Our reimplementation of it is no longer a config in
# mpkmeans_bench at all, so there is nothing to skip.
COMMON="--maxiters 400 --convergence -e $SEED"
RT_PY="${RT_PY:-$HERE/rt_baseline_mp.py}"
RT_PYTHON="${RT_PYTHON:-/pscratch/sd/j/jbellav/envs/mpk/bin/python}"

# ------------------------------------------------------------ synthetic ---
if [[ -z "$ONLY_FAMILY" || "$ONLY_FAMILY" == synth ]]; then
for box in "${BOXES[@]}"; do
  for n in "${NS[@]}"; do
    for accum in fp32 fp16; do
      for z in 0 1; do
        zf=""; [[ $z == 1 ]] && zf=" --zscore"
        nlab=$(( n / 1000 ))k
        base="synth_b${box}_n${nlab}_${accum}_z${z}"

        if [[ "$SWEEP" == grid ]]; then
            for d in "${DIMS[@]}"; do for k in "${KS[@]}"; do
                add_run synth blobs "$n" "$d" "$k" \
                    "-n $n -d $d -b $box $COMMON --accum $accum$zf"
            done; done
            emit_job "${base}_grid" synth blobs "$box" "$n" grid "$accum" "$z"
        else
            # d at the pivot k, then k at the pivot d; the pivot point itself
            # is run once, in the d sweep
            for d in "${DIMS[@]}"; do
                add_run synth blobs "$n" "$d" "$PIVOT_K" \
                    "-n $n -d $d -b $box $COMMON --accum $accum$zf"
            done
            for k in "${KS[@]}"; do
                [[ $k == "$PIVOT_K" ]] && continue
                add_run synth blobs "$n" "$PIVOT_D" "$k" \
                    "-n $n -d $PIVOT_D -b $box $COMMON --accum $accum$zf"
            done
            emit_job "${base}_sweep" synth blobs "$box" "$n" 1d "$accum" "$z"
        fi
      done
    done
  done
done

fi

# --------------------------------------------------------------- LIBSVM ---
if [[ -z "$ONLY_FAMILY" || "$ONLY_FAMILY" == libsvm ]]; then
for row in "${LIBSVM_SETS[@]}"; do
    IFS='|' read -r id url bz name n d <<<"$row"
    path="$DATA/libsvm/$name"
    dense=$(( n * d * 4 ))
    for accum in fp32 fp16; do
      for z in 0 1; do
        zf=""; [[ $z == 1 ]] && zf=" --zscore"
        if [[ $dense -gt $LIBSVM_DENSE_LIMIT ]]; then
            # mpkLoadLibsvm refuses this before any subsampling can help: the
            # densification happens on the host, at full n, at load time
            if [[ $DRY -eq 0 && "$accum" == fp32 && "$z" == 0 ]]; then
                printf '%s\t%s\t%s\t%s\t-\t%s\t-\tdense %s GB exceeds the loader'"'"'s 60 GB densify limit\n' \
                    libsvm "$id" "$n" "$d" "$(( dense / 1000000000 ))" \
                    "$(( dense / 1000000000 ))" >> "$INFEAS"
            fi
            NSKIP=$(( NSKIP + 1 )); continue
        fi
        for k in "${KS[@]}"; do
            add_run libsvm "$id" "$n" "$d" "$k" \
                "--dataset $path $COMMON --accum $accum$zf"
        done
        emit_job "libsvm_${id}_${accum}_z${z}" libsvm "$id" - "$n" ksweep "$accum" "$z"
      done
    done
done

fi

# ------------------------------------------------------- vector indexing --
if [[ -z "$ONLY_FAMILY" || "$ONLY_FAMILY" == vector ]]; then
for row in "${BIN_SETS[@]}"; do
    IFS='|' read -r id file d <<<"$row"
    path="$DATA/data/$file"
    # n comes from the file when it exists; fall back to the manifest so that
    # --dry-run works before 1_prep_datasets.sh has run
    if [[ -s "$path" ]]; then
        n=$(( $(stat -c %s "$path") / (d * 4) ))
    else
        n=$(awk -v i="$id" -F'\t' '$1==i {print $3}' "$DATA/manifest.tsv" 2>/dev/null || true)
        [[ -z "$n" ]] && { echo "  (skipping $id: not prepared and not in the manifest)"; continue; }
    fi
    for accum in fp32 fp16; do
      for z in 0 1; do
        zf=""; [[ $z == 1 ]] && zf=" --zscore"
        for k in "${KS[@]}"; do
            add_run vector "$id" "$n" "$d" "$k" \
                "--bin $path -d $d $COMMON --accum $accum$zf"
        done
        emit_job "vector_${id}_${accum}_z${z}" vector "$id" - "$n" ksweep "$accum" "$z"
      done
    done
done

fi

# -------------------------------------------------------------- run_all ---
if [[ $DRY -eq 0 ]]; then
    cat > "$OUT/run_all.sh" <<RA
#!/usr/bin/env bash
# Submit every generated job.  Generated by eval/2_gen_jobs.sh.
#
#   ./run_all.sh              submit all
#   ./run_all.sh synth        submit only jobs whose name matches 'synth'
#   ./run_all.sh --local      run them here, in series, without sbatch.  Inside
#                             an interactive allocation set MPK_SRUN= as well,
#                             so the runs do not nest srun inside srun.
set -uo pipefail
cd "\$(dirname "\$0")"

LOCAL=0
[[ "\${1:-}" == "--local" ]] && { LOCAL=1; shift; }
PAT="\${1:-}"

n=0
for j in jobs/*.sbatch; do
    [[ -n "\$PAT" && "\$j" != *"\$PAT"* ]] && continue
    n=\$(( n + 1 ))
    if [[ \$LOCAL -eq 1 ]]; then
        echo "=== \$j ==="
        bash "\$j" || echo "FAILED: \$j" >&2
    else
        sbatch "\$j"
    fi
done
echo "\$n job(s) \$([[ \$LOCAL -eq 1 ]] && echo run || echo submitted)"
RA
    chmod +x "$OUT/run_all.sh"
fi

echo
echo "jobs      : $NJOB"
echo "runs      : $NRUN"
echo "skipped   : $NSKIP  (see $INFEAS)"
if [[ -n "$SUBSAMPLED_SETS" ]]; then
    uniq_sets=$(echo "$SUBSAMPLED_SETS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
    echo
    echo "NOTE: these datasets are subsampled to fit: $uniq_sets"
    echo "  mpkmeans_bench and rt_baseline_mp.py draw subsamples with different"
    echo "  RNGs, so for these the rt-mp row clusters a DIFFERENT subset of the"
    echo "  same file than the other schemes do.  Sizes and distributions match;"
    echo "  the rows do not.  To make them identical, run the driver once with"
    echo "  --dump-subsample and point both at that file with --bin."
fi
if [[ $DRY -eq 0 ]]; then
    echo
    echo "wrote $JOBS/*.sbatch and $OUT/run_all.sh"
    if [[ $(wc -l < "$INFEAS") -gt 1 ]]; then
        echo
        echo "NOT RUNNABLE -- these configurations were asked for and cannot be run:"
        column -t -s $'\t' "$INFEAS" | sed 's/^/  /'
    fi
fi
