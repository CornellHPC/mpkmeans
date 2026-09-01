#!/usr/bin/env bash
# Sweep the parameters that drive how many high precision distance evaluations
# the exclusion conditions let us throw out.  Every run carries the FP64 oracle
# check, so `violations` in the output must stay 0 throughout.
#
# Each invocation emits three rows, one per condition configuration:
# (3) alone, (6) alone, and (3)+(6).
set -u
# Use the MPK_STATS build: the exclusion attribution and the FP64 oracle are
# compiled out of the default build.
BIN=${BIN:-./build-stats/mpkmeans_bench}
OUT=${1:-results/sweep.csv}
mkdir -p "$(dirname "$OUT")"
"$BIN" --csv-header > "$OUT"

run() { "$BIN" --csv "$@" >> "$OUT" || echo "FAILED: $*" >&2; }

# k: sets the n/(n*k) = 1/k floor from the exact reference entries
for k in 8 16 32 64 128 256; do
    run -n 100000 -d 64 -k "$k" -i 20 -s 1.0
done

# d: inner dimension, drives eps and the FP16 rounding
for d in 16 32 64 128 256 512 1024; do
    run -n 100000 -d "$d" -k 32 -i 20 -s 1.0
done

# separation: the blob std relative to a fixed center box is what actually
# decides how many entries stay ambiguous after the FP16 GEMM
for s in 0.25 0.5 1.0 2.0 4.0 8.0; do
    run -n 100000 -d 64 -k 32 -i 20 -s "$s"
done

# n: should leave the percentages flat
for n in 25000 50000 100000 200000 400000; do
    run -n "$n" -d 64 -k 32 -i 20 -s 1.0
done

# seeds, to show the numbers are not an artifact of one draw
for e in 1 2 3 4 5; do
    run -n 100000 -d 64 -k 32 -i 20 -s 1.0 -e "$e"
done

echo "wrote $OUT"
