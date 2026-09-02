#!/bin/bash
# Single blob (-b 0: every true center at the origin).  Two axes:
#   std -- the blob's scale.  P is shifted to be non-negative by -min(P), which
#          is itself proportional to std, so this is expected to change nothing.
#   d   -- eps grows with d, so this is expected to make the bounds looser.
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build-stats/mpkmeans_bench
row() {  # $1 label  $2.. bench args
  local lab=$1; shift
  local out=$($B "$@" -b 0 -i 15 2>/dev/null)
  local sh=$(echo "$out" | awk '/^shift/{print $5}')
  for C in "(3)" "(6)"; do
    local e=$(echo "$out" | awk -v c="$C" '$1==c && NF==6 && $6 ~ /%$/ {print $4, $6; exit}')
    local v=$(echo "$out" | awk -v c="$C" '/label != fp32 label/{f=1;next} f&&$1==c{print $2; exit}')
    printf "%-14s %-5s %-9s %12s %11s %8s\n" "$lab" "$sh" "$C" $e "$v"
  done
}
printf "%-14s %-5s %-9s %12s %11s %8s\n" axis shift cond survivors eliminated viol
for S in 0.01 0.1 1 10 100; do row "std=$S" -n 200000 -d 128 -k 64 -s $S; done
echo
for D in 32 64 128 256 512; do row "d=$D (std=1)" -n 200000 -d $D -k 64 -s 1; done
