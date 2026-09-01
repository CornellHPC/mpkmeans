#!/usr/bin/env bash
# Locate the crossover between cusparseSDDMM and the warp kernel for the high
# precision update of the surviving entries.
#
#   JOBID=<slurm id> ./scripts/tune_sddmm.sh          # via scripts/jrun.sh
#   ./scripts/tune_sddmm.sh                           # directly
#
# The knob that matters is the survivor count PER ROW, and the way to sweep it
# is the centre box (-b): shrinking it makes the blobs overlap, so the filter
# excludes less and the survivor count climbs by orders of magnitude.  Blob
# std (-s) barely moves it, because in high d randomly placed centres are far
# apart whatever the std.
#
# Reported per iteration: nnz, nnz/row, the SDDMM path (setup + operation) and
# the warp path.  The crossover is where the last two columns cross; it sits
# near 10 survivors per row at d=128 and drifts roughly as sqrt(d).
set -u
BIN=${BIN:-./build/mpkmeans_bench}
RUN=cat
if [ -n "${JOBID:-}" ]; then RUN="./scripts/jrun.sh"; fi
invoke() { if [ -n "${JOBID:-}" ]; then ./scripts/jrun.sh "$BIN" "$@"; else "$BIN" "$@"; fi; }

printf "%5s %5s %8s %6s %12s %8s %9s %9s %8s\n" \
       "d" "k" "n" "box" "nnz/iter" "nnz/row" "sddmm_us" "warp_us" "winner"

probe() {  # d k n box
    local D=$1 K=$2 N=$3 BX=$4
    local a b nnz sd wp
    a=$(invoke --csv --no-verify --force-sddmm --only 3 -n "$N" -d "$D" -k "$K" \
            -b "$BX" -i 12 2>/dev/null \
        | awk -F, '{printf "%.0f %.2f", $11/$7, ($32+$33)*1000/$7}')
    b=$(invoke --csv --no-verify --force-warp --only 3 -n "$N" -d "$D" -k "$K" \
            -b "$BX" -i 12 2>/dev/null | awk -F, '{printf "%.2f", $33*1000/$7}')
    nnz=${a%% *}; sd=${a##* }; wp=$b
    printf "%5s %5s %8s %6s %12s %8.2f %9s %9s %8s\n" "$D" "$K" "$N" "$BX" \
        "$nnz" "$(echo "$nnz / $N" | bc -l)" "$sd" "$wp" \
        "$(awk -v s="$sd" -v w="$wp" 'BEGIN{print (s<w)?"sddmm":"warp"}')"
}

# survivors per row from ~0.5 up through the crossover and well past it
for BX in 10 3 1.5 1 0.9 0.8 0.7 0.6 0.5; do probe 128 256 200000 "$BX"; done
# the crossover is per row, so it should move with n and stay put in k
for BX in 1 0.9 0.8; do probe 128 256 400000 "$BX"; done
for BX in 1 0.7 0.5; do probe 128  64 200000 "$BX"; done
# and it drifts with d
for BX in 0.9 0.8; do probe  32 256 200000 "$BX"; done
for BX in 0.9 0.8; do probe 512 256 200000 "$BX"; done
