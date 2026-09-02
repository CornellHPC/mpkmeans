#!/bin/bash
# PERFORMANCE GATE.  Run after any change you expect to affect speed, and
# compare against the numbers you recorded before it.
#
# Uses the TIMING build (./build), which compiles out the attribution counters
# and the oracle -- those cost kernel time and would mask what you changed.
#
# Two regimes, because they stress different things and a change can easily
# help one and hurt the other:
#
#   box=10  separated blobs.  Survivors are ~0.3 per row, so almost every row
#           is settled by the bound alone and the argmin dominates.
#   box=0   one single Gaussian blob, no cluster structure.  Survivors run to
#           millions, so the update kernel and the survivor list dominate.
#           Everything is slower than FP32 here; the number to watch is whether
#           your change moved it, not whether it beats 1.000x.
#
# Points start at d=64, k=64: the smallest d and k tested, and the worst case
# so far, since condition (6)'s fixed O(nd) cost is amortised over an O(ndk)
# GEMM and both shrink with d and k.
#
# ms/iteration of the distance step (centroid update excluded), reported as the
# minimum of REPS interleaved samples.  Interleaving matters: run order moves
# results by ~10% at k=256, enough to invert the ranking between two
# configurations, so never compare single samples taken back to back.
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/mpkmeans_bench}
REPS=${REPS:-5}
N=${N:-200000}; I=${I:-20}
PTS=${PTS:-"64:64 128:64 64:256 128:256"}   # d:k
BOXES=${BOXES:-"10 0"}
[ -x "$BIN" ] || { echo "no $BIN -- build the timing config first"; exit 1; }

for BOX in $BOXES; do
  case $BOX in
    10) echo "=== separated blobs (box=10): the bound does the work ===" ;;
    0)  echo "=== single blob (box=0): the survivors do the work ===" ;;
    *)  echo "=== box=$BOX ===" ;;
  esac
  printf "%-12s %-6s %9s %9s %9s %9s %9s\n" \
         "d,k" "fp32" "(3)" "(6)" "(3)+(6)" "(3)->(6)" "best"
  for PT in $PTS; do
    D=${PT%%:*}; K=${PT##*:}
    declare -A best
    for r in $(seq 1 "$REPS"); do
      for C in 3 6 36 c; do
        v=$($BIN -n $N -d $D -k $K -i $I -b $BOX -r 2 --only $C --no-verify \
            2>/dev/null |
            awk '/cond +ms\/iter/{f=1;next} f&&/^  \(/{print $2; exit}')
        [ -n "${v:-}" ] || v=999
        best[$C]=$(awk -v a="$v" -v b="${best[$C]:-999}" 'BEGIN{print (a<b)?a:b}')
      done
    done
    fp=$($BIN -n $N -d $D -k $K -i $I -b $BOX -r 2 --only 3 --no-verify \
         2>/dev/null |
         awk '/cond +ms\/iter/{f=1;next} f&&/^  fp32/{print $2; exit}')
    awk -v dk="$D,$K" -v f="$fp" -v a="${best[3]}" -v b="${best[6]}" \
        -v c="${best[36]}" -v e="${best[c]}" 'BEGIN{
          m=a; n="(3)";
          if (b<m) {m=b; n="(6)"}
          if (c<m) {m=c; n="(3)+(6)"}
          if (e<m) {m=e; n="(3)->(6)"}
          printf "%-12s %-6.3f %8.3f%s %8.3f%s %8.3f%s %8.3f%s %9s\n",
                 dk, f, a, (a==m?"*":" "), b, (b==m?"*":" "),
                 c, (c==m?"*":" "), e, (e==m?"*":" "),
                 sprintf("%.2fx %s", f/m, n);
        }'
    unset best
  done
  echo
done
echo "ms/iter of the distance step; * marks the fastest configuration."
echo "'best' is that configuration's speedup over the FP32 reference."
