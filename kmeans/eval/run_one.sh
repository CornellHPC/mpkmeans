#!/usr/bin/env bash
# Run one problem through both implementations, on identical data, from
# identical initial centroids.
#
# The two cannot be handed the same problem by passing them the same flags.
# --subsample draws with numpy's Generator in the driver and std::mt19937 in
# the benchmark, so one seed gives two different subsets; --blobs is generated
# by two different RNGs and cannot be reproduced across them at all; and the
# initial centroids are drawn independently on each side.  Matching them by
# hand is a three-step dance that is easy to get subtly wrong -- and a wrong
# one does not fail, it just quietly answers a different question.
#
# So this script does it in one direction, once:
#
#   1. the Python driver loads (or generates) the data, applies --subsample,
#      and writes the exact matrix it will cluster plus its shape;
#   2. it draws the initial centroids and writes those too;
#   3. mpkmeans_bench is then pointed at both files -- --bin for the data,
#      --init-centroids for the centroids -- so it cannot draw its own.
#
# The data file is written before --zscore and the centroids after it, which is
# the frame --init-centroids expects, so --zscore stays meaningful and is
# applied by each side to the same input.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

BIN="$REPO/build/mpkmeans_bench"
RT_PY="$HERE/rt_baseline_mp.py"
PYTHON="${RT_PYTHON:-/pscratch/sd/j/jbellav/envs/mpk/bin/python}"
OUT=""
KEEP=0

# common
SRC=""; SRCKIND=""; DIM=""; K=""; N=200000; STD=1.0; BOX=10.0; SEED=1
MAXITERS=400; CONVERGE=0; TOL=1e-8; ZSCORE=0; SUBSAMPLE=0
# per side
KAPPA=5.0; KERNEL=fp16_fp32; BENCH_EXTRA=""; RT_EXTRA=""

usage() {
    cat <<USAGE
usage: $(basename "$0") --out DIR -k K (--bin F -d D | --libsvm F | --blobs) [options]

dataset (pick one):
  --bin FILE -d D    headerless row-major float32 matrix
  --libsvm FILE      LIBSVM text, densified to the largest feature index
  --blobs            synthetic; -n, -d, -s, -b apply

common to both implementations:
  -k K               clusters (required)
  -n N               points, --blobs only          (default $N)
  -d D               dimension; required with --bin, features with --blobs
  -s STD             blob standard deviation       (default $STD)
  -b BOX             blob centre box half width    (default $BOX)
  -e SEED            rng seed                      (default $SEED)
  --maxiters N       iteration ceiling             (default $MAXITERS)
  --convergence      stop at ||C - C_prev||_F < --tol; without it BOTH run
                     exactly --maxiters iterations
  --tol T            that tolerance                (default $TOL)
  --zscore           z-score every feature first
  --subsample N      cluster N rows drawn without replacement

one side only:
  --kappa F          driver: the paper's rho       (default $KAPPA)
  --kernel K         driver: mp-kmeans kernel      (default $KERNEL)
  --bench-extra "…"  passed verbatim to mpkmeans_bench (--accum, --only, …)
  --rt-extra "…"     passed verbatim to the driver

other:
  --out DIR          where the shared data, centroids and CSVs go (required)
  --keep             keep the materialised data file (it can be large)
  --bin-path PATH    mpkmeans_bench            (default $BIN)

Under an existing allocation, run this through srun, or set MPK_SRUN= and let
the two invocations run directly.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin) SRC="$2"; SRCKIND=bin; shift 2 ;;
        --libsvm) SRC="$2"; SRCKIND=libsvm; shift 2 ;;
        --blobs) SRCKIND=blobs; shift ;;
        -d|--dim) DIM="$2"; shift 2 ;;
        -k) K="$2"; shift 2 ;;
        -n) N="$2"; shift 2 ;;
        -s) STD="$2"; shift 2 ;;
        -b) BOX="$2"; shift 2 ;;
        -e) SEED="$2"; shift 2 ;;
        --maxiters) MAXITERS="$2"; shift 2 ;;
        --convergence) CONVERGE=1; shift ;;
        --tol) TOL="$2"; shift 2 ;;
        --zscore) ZSCORE=1; shift ;;
        --subsample) SUBSAMPLE="$2"; shift 2 ;;
        --kappa) KAPPA="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --bench-extra) BENCH_EXTRA="$2"; shift 2 ;;
        --rt-extra) RT_EXTRA="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --bin-path) BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "$OUT" ]] || { echo "--out is required" >&2; usage >&2; exit 1; }
[[ -n "$K"   ]] || { echo "-k is required" >&2; usage >&2; exit 1; }
[[ -n "$SRCKIND" ]] || { echo "pick one of --bin / --libsvm / --blobs" >&2; exit 1; }
[[ "$SRCKIND" == bin && -z "$DIM" ]] && { echo "--bin needs -d" >&2; exit 1; }
[[ -x "$BIN" ]] || { echo "no such binary: $BIN" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "no python at $PYTHON (set RT_PYTHON)" >&2; exit 1; }

mkdir -p "$OUT"
DATA="$OUT/shared_data.bin"
C0="$OUT/shared_c0.bin"
SRUN=${MPK_SRUN-}

# ---- flags common to both, spelled the way each side spells them ---------
common_rt=(-k "$K" -e "$SEED" --maxiters "$MAXITERS" --tol "$TOL")
common_bench=(-k "$K" -e "$SEED" --maxiters "$MAXITERS" --tol "$TOL")
[[ $CONVERGE -eq 1 ]] && { common_rt+=(--convergence); common_bench+=(--convergence); }
[[ $ZSCORE   -eq 1 ]] && { common_rt+=(--zscore);      common_bench+=(--zscore); }
[[ "$SUBSAMPLE" != 0 ]] && common_rt+=(--subsample "$SUBSAMPLE")

case "$SRCKIND" in
    bin)    src_rt=(--bin "$SRC" -d "$DIM") ;;
    libsvm) src_rt=(--libsvm "$SRC") ;;
    blobs)  src_rt=(--blobs -n "$N" -d "${DIM:-64}" -s "$STD" -b "$BOX") ;;
esac

echo "=============================================================="
echo " 1/2  mp-kmeans (arXiv:2407.12208, authors' package)"
echo "=============================================================="
# shellcheck disable=SC2086
$SRUN "$PYTHON" "$RT_PY" "${src_rt[@]}" "${common_rt[@]}" \
    --kappa "$KAPPA" --kernel "$KERNEL" \
    --dump-data "$DATA" --dump-centroids "$C0" \
    --csv "$OUT/rt.csv" $RT_EXTRA

[[ -s "$DATA" && -s "$DATA.meta" ]] || {
    echo "the driver did not write $DATA -- cannot match the benchmark to it" >&2
    exit 1
}
read -r SHARED_N SHARED_D < "$DATA.meta"

echo
echo "=============================================================="
echo " 2/2  mpkmeans_bench, on those same $SHARED_N x $SHARED_D points"
echo "      and those same $K centroids"
echo "=============================================================="
# NOTE: --bin, never --subsample; the rows were already chosen above.
# --csv-header does not touch the GPU, so the benchmark itself runs exactly
# once -- timing it twice to get both the tables and the row would report two
# different measurements of the same configuration.
"$BIN" --csv-header > "$OUT/bench.csv"
# mpkmeans_bench exits 2 when its own correctness check does not pass, with
# the rows still written; on real data that is normally `raw`, which has no
# guarantee by construction.  That is a result, not a failure to abort on.
set +e
# shellcheck disable=SC2086
$SRUN "$BIN" --bin "$DATA" -d "$SHARED_D" "${common_bench[@]}" \
    --init-centroids "$C0" --csv $BENCH_EXTRA >> "$OUT/bench.csv"
bench_rc=$?
set -e
case $bench_rc in
    0) ;;
    2) echo "  (correctness check not passed -- see rel_inertia per cond in"
       echo "   $OUT/bench.csv; normally the 'raw' config, which has none)" ;;
    *) echo "mpkmeans_bench failed with $bench_rc" >&2; exit $bench_rc ;;
esac

echo
echo "=============================================================="
echo " both, per iteration, on the same $SHARED_N x $SHARED_D points"
echo "=============================================================="
awk -F, -v rt="$OUT/rt.csv" '
    BEGIN { printf "  %-18s %6s %10s %10s %10s %16s\n",
                   "impl","iters","dist/iter","tot/iter","total ms","inertia" }
    NR > 1 { printf "  %-18s %6d %10s %10.3f %10.2f %16s\n",
                    $10, $11, ($29+0 > 0 ? sprintf("%.3f", $29/($11?$11:1)) : "-"),
                    $37/($11?$11:1), $37, $26 }
    END {
        while ((getline line < rt) > 0) {
            n = split(line, f, ",")
            if (f[10] == "cond" || f[10] == "") continue
            printf "  %-18s %6d %10s %10.3f %10.2f %16s\n",
                   "rt-mp " f[9], f[11],
                   (f[29]+0 > 0 ? sprintf("%.3f", f[29]/(f[11]?f[11]:1)) : "-"),
                   f[37]/(f[11]?f[11]:1), f[37], f[26]
        }
    }' "$OUT/bench.csv"
awk -F, 'NR==2 { printf "  %-18s %6d %10.3f %10.3f %10.2f %16s\n",
                        "fp32", $39, $30/($39?$39:1), $38/($39?$39:1), $38, $27 }' \
    "$OUT/bench.csv"

[[ $KEEP -eq 1 ]] || { rm -f "$DATA" "$DATA.meta"; echo; echo "(removed $DATA; --keep to retain it)"; }
echo "CSVs: $OUT/bench.csv  $OUT/rt.csv"
