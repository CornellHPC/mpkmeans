#!/bin/bash
# Interleaved repeated sampling: run order rotates so no config is always first
# and no config is always charged for the GPU warming up.
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build/mpkmeans_bench
K=${1:-256}
REPS=${2:-5}
declare -A best
for r in $(seq 1 $REPS); do
  for C in 3 6 36 c; do
    v=$($B -n 200000 -d 128 -k $K -i 20 -r 2 --only $C --no-verify 2>/dev/null |
        awk '/cond +ms\/iter/{f=1;next} f&&/^  \(/{print $2; exit}')
    cur=${best[$C]:-999}
    best[$C]=$(awk -v a="$v" -v b="$cur" 'BEGIN{print (a<b)?a:b}')
  done
done
fp=$($B -n 200000 -d 128 -k $K -i 20 -r 2 --only 3 --no-verify 2>/dev/null |
     awk '/cond +ms\/iter/{f=1;next} f&&/^  fp32/{print $2; exit}')
echo "k=$K  min of $REPS samples, ms/iter    (fp32 $fp)"
for C in 3 6 36 c; do
  awk -v c="$C" -v v="${best[$C]}" -v f="$fp" \
      'BEGIN{printf "  %-4s %7.3f  %6.3fx\n", c, v, f/v}'
done
