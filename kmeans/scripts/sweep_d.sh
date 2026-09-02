#!/bin/bash
# Cost of condition (6) vs (3) as d varies, k fixed.  The argmin column now
# includes the fused exclusion test and survivor enumeration.
cd /global/u2/j/jbellav/mpkmeans/kmeans
echo "d   cond  gemm_us  argmin+cond_us"
for D in 32 64 128 256 512; do
  for C in 3 6; do
    ./build/mpkmeans_bench -n 200000 -d $D -k 64 -i 20 -r 3 --only $C --no-verify 2>/dev/null \
      | awk -v d=$D -v c=$C '/cond +iters +prep +gemm/{f=1;next} f&&/^  \(/{printf "%-3s (%s)  %7.1f  %8.1f\n", d, c, $4*1000/$2, $5*1000/$2; exit}'
  done
done
