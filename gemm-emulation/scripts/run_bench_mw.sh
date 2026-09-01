#!/bin/bash
#SBATCH -A m1266_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1 -n 1 -c 32 -G 1
#SBATCH -t 00:30:00
#SBATCH -J mpemu_mw
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Multiword fp16 (Fasi/Higham/Lopez/Mary/Mikaitis) vs cublasSgemm and cublasDgemm.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results logs
export NVIDIA_TF32_OVERRIDE=0

STAMP="${SLURM_JOB_ID:-local}"

echo "=== fp16 split correctness ==="
build/mpemu_test_split_fp16 || exit 1

echo "=== main sweep, U(-0.5,0.5] ==="
build/mpemu_bench_mw --reps "${REPS:-5}" ${EXTRA:-} > "results/mw-${STAMP}.csv"

echo "=== wide exponent range ==="
build/mpemu_bench_mw --reps 3 --square --exp-range 12 ${EXTRA:-} \
    > "results/mw-wide-${STAMP}.csv"

echo "=== done ==="
column -s, -t < "results/mw-${STAMP}.csv"
