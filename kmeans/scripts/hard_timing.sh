#!/bin/bash
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build/mpkmeans_bench
printf "%-6s %-9s %9s %9s\n" box cond ms_iter vs_fp32
for BOX in 10 1 0.5 0; do
  declare -A best
  for r in 1 2 3; do
    for C in 3 6 36 c; do
      v=$($B -n 200000 -d 128 -k 64 -i 20 -b $BOX --only $C --no-verify 2>/dev/null |
          awk '/cond +ms\/iter/{f=1;next} f&&/^  \(/{print $2; exit}')
      best[$C]=$(awk -v a="$v" -v b="${best[$C]:-999}" 'BEGIN{print (a<b)?a:b}')
    done
  done
  fp=$($B -n 200000 -d 128 -k 64 -i 20 -b $BOX --only 3 --no-verify 2>/dev/null |
       awk '/cond +ms\/iter/{f=1;next} f&&/^  fp32/{print $2; exit}')
  for C in 3 6 36 c; do
    awk -v bx="$BOX" -v c="$C" -v v="${best[$C]}" -v f="$fp" \
        'BEGIN{printf "%-6s %-9s %9.3f %8.3fx\n", bx, c, v, f/v}'
  done
  awk -v bx="$BOX" -v f="$fp" 'BEGIN{printf "%-6s %-9s %9.3f %8s\n", bx, "fp32", f, "1.000x"}'
  echo
  unset best
done
