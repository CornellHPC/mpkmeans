#!/bin/bash
#SBATCH -A m1266_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -t 00:30:00
#SBATCH -J mpemu_bench
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# One A100 out of the shared queue: charged at 1/4 node.
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p results logs

# Force cublasSgemm to be genuine FP32: no TF32 tensor-core substitution.
# (cuBLAS already defaults to this; the override makes it unconditional.)
export NVIDIA_TF32_OVERRIDE=0

BIN=build/mpemu_bench
STAMP="${SLURM_JOB_ID:-local}"

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,persistence_mode --format=csv

echo "=== split correctness ==="
build/mpemu_test_split || exit 1

echo "=== main sweep (uniform [-1,1]) ==="
$BIN --reps "${REPS:-5}" ${EXTRA:-} > "results/bench-${STAMP}.csv"

echo "=== wide exponent range (square only) ==="
$BIN --reps 3 --square --exp-range 20 ${EXTRA:-} > "results/bench-wide-${STAMP}.csv"

echo "=== done ==="
column -s, -t < "results/bench-${STAMP}.csv"
