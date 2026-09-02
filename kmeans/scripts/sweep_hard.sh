#!/bin/bash
# How the scheme degrades as the clusters stop being separated.
#
#   -b <box>  half width of the box the k true centers are drawn from.
#             b = 0 puts every center at the origin: one single Gaussian blob,
#             no cluster structure at all, the worst case for the bounds.
#   -s <std>  the blob standard deviation.
#
# Difficulty is set by the ratio std/box, so the sweep holds std = 1 and closes
# the box.  Reports, per configuration, the reference entries and survivors per
# iteration, the fraction of the n*k high precision distances eliminated, the
# ms/iteration of the distance step, and the FP64 oracle's bound violations.
cd /global/u2/j/jbellav/mpkmeans/kmeans
B=./build-stats/mpkmeans_bench
N=${N:-200000}; D=${D:-128}; K=${K:-64}; I=${I:-20}
printf "%-7s %-9s %12s %12s %11s %10s %10s\n" \
       box cond reference survivors eliminated ms/iter violations
for BOX in 10 4 2 1 0.5 0.25 0; do
  out=$($B -n $N -d $D -k $K -i $I -b $BOX -s 1 2>/dev/null)
  for C in "(3)" "(6)" "(3)+(6)" "(3)->(6)"; do
    e=$(echo "$out" | awk -v c="$C" '$1==c && NF==6 && $6 ~ /%$/ {print $3, $4, $6; exit}')
    t=$(echo "$out" | awk -v c="$C" '/cond +ms\/iter/{f=1;next} f&&$1==c{print $2; exit}')
    v=$(echo "$out" | awk -v c="$C" '/FP64 oracle/{f=1;next} f&&$1==c{print $2; exit}')
    printf "%-7s %-9s %12s %12s %11s %10s %10s\n" "$BOX" "$C" $e "$t" "$v"
  done
  echo
done
