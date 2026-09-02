#!/bin/bash
# (3)+(6) unconditional vs the (3)->(6) cascade: argmin time and how many
# reference entries each actually evaluates.  Isolated runs (--only) so no
# config is charged for another's warm-up.
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build/mpkmeans_bench
printf "%-5s %-5s %-9s %9s %9s %11s\n" n d/k cond argmin_us total_us ms_per_iter
for D in 64 128 256; do
  for C in 3 6 36 c; do
    $B -n 200000 -d $D -k 64 -i 20 -r 3 --only $C --no-verify 2>/dev/null |
      awk -v d=$D -v c=$C '/cond +iters +prep +gemm/{f=1;next}
        f&&/^  \(/{printf "%-5s %-5s %-9s %9.1f %9.1f\n", "200k", d, $1, $5*1000/$2, $10*1000/$2; exit}'
  done
  echo
done
