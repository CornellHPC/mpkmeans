#!/bin/bash
# CORRECTNESS GATE.  Run this after any change to the kernels or the driver.
#
# Every shape below exercises a different code path, and each run checks every
# row of every iteration against the ordinary FP32 implementation (cublasSgemm
# + row argmin) on the mixed run's own centroids.
#
#   violations  MUST be 0.  Non-zero means the FP32 answer was neither the
#               incumbent nor in the survivor list -- a condition of Theorem 1
#               does not hold, i.e. the bound or the filter is wrong.
#   labeldiff   need NOT be 0.  These are rows where the FP32 answer was
#               reachable but a near-tie ordered differently, because the
#               survivor dot and cuBLAS sum in different orders.  Watch it for
#               sudden jumps, not for being non-zero; a few per 10^5 is normal.
#
# Exits non-zero if any violation or any FAIL is seen.
set -u
BIN=${BIN:-./build-stats/mpkmeans_bench}
cd "$(dirname "$0")/.."
[ -x "$BIN" ] || { echo "no $BIN -- configure with -DMPK_STATS=ON"; exit 1; }

# shape                                        what it covers
SHAPES=(
  "-n 20000  -d 32  -k 16  -i 10"            # small, k<=32: NM=1 register path
  "-n 200000 -d 128 -k 64  -i 20"            # the reference point, NM=2
  "-n 200000 -d 128 -k 256 -i 15"            # k>64: IPT=0 streaming path
  "-n 50000  -d 512 -k 32  -i 10 -b 3"       # large d, long inner products
  "-n 100000 -d 64  -k 128 -i 10 -b 2"       # overlapping: many survivors
  "-n 200000 -d 128 -k 64  -i 20 -b 0"       # single blob: list growth path
  "-n 200000 -d 128 -k 64  -i 20 -e 7"       # a different draw
)
fail=0
printf "%-42s %-9s %11s %11s\n" shape cond violations labeldiff
for sh in "${SHAPES[@]}"; do
  # rt-base is the arXiv:2407.12208 baseline, not an exclusion scheme, so it has
  # no bound to violate and prints no violation counter.  It is still run and
  # still gated: the bench's PASS also requires |inertia - inertia_fp32| /
  # inertia_fp32 < 1e-4, which is the meaningful check for it.
  out=$($BIN $sh --only 36 2>&1) || { echo "RUN FAILED: $sh"; fail=1; continue; }
  out="$out"$'\n'$($BIN $sh --only 3 2>&1)$'\n'$($BIN $sh --only 6 2>&1)$'\n'$($BIN $sh --only c 2>&1)$'\n'$($BIN $sh --only rt 2>&1)
  [ "$(echo "$out" | grep -c '^PASS')" = 5 ] || { echo "NO PASS: $sh"; fail=1; }
  while read -r cond v l; do
    printf "%-42s %-9s %11s %11s\n" "$sh" "$cond" "$v" "$l"
    [ "$v" = "0" ] || fail=1
  done < <(echo "$out" | awk '/label != fp32 label/{f=1;next}
                              f&&/^  \(/{print $1, $2, $5}
                              f&&/^  the oracle/{exit}')
done
echo
if [ $fail -eq 0 ]; then echo "CORRECTNESS OK -- 0 violations everywhere"; else
  echo "CORRECTNESS FAILED"; fi
exit $fail
