#!/bin/bash
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build/mpkmeans_bench
for K in 64 128 256; do
  for C in 3 6 36 c; do
    $B -n 200000 -d 128 -k $K -i 20 -r 3 --only $C --no-verify 2>/dev/null |
      awk -v kk=$K '/cond +ms\/iter/{f=1;next}
        f&&/^  \(/{printf "k=%-4s %-9s %7.3f %8s\n", kk, $1, $2, $3; c=1; next}
        f&&c&&/^  fp32/{printf "k=%-4s %-9s %7.3f\n", kk, "fp32", $2; exit}'
  done
  echo
done
