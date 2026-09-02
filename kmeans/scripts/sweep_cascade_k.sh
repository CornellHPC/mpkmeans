#!/bin/bash
# argmin and whole-distance-step cost against k, min over repeated interleaved
# samples: single samples at k = 256 vary by ~10% and invert the ranking.
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build/mpkmeans_bench
REPS=${1:-5}
for K in 32 64 128 256; do
  declare -A ba bt
  for r in $(seq 1 $REPS); do
    for C in 3 6 36 c; do
      read a t < <($B -n 200000 -d 128 -k $K -i 20 -r 2 --only $C --no-verify 2>/dev/null |
        awk '/cond +iters +prep +gemm/{f=1;next} f&&/^  \(/{print $5*1000/$2, $10*1000/$2; exit}')
      ba[$C]=$(awk -v a="$a" -v b="${ba[$C]:-99999}" 'BEGIN{print (a<b)?a:b}')
      bt[$C]=$(awk -v a="$t" -v b="${bt[$C]:-99999}" 'BEGIN{print (a<b)?a:b}')
    done
  done
  for C in 3 6 36 c; do
    printf "k=%-5s %-5s argmin %7.1f  step %7.1f\n" "$K" "$C" "${ba[$C]}" "${bt[$C]}"
  done
  echo
  unset ba bt
done
