#!/bin/bash
#SBATCH -A m1266_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1 -n 1 -c 32 -G 1
#SBATCH -t 00:50:00
#SBATCH -J mpemu_all
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results logs
export NVIDIA_TF32_OVERRIDE=0
STAMP="${SLURM_JOB_ID:-local}"
for D in uniform gaussian; do
    echo "=== $D ==="
    ./build/mpemu_bench_all --dist "$D" ${EXTRA:-} > "results/all-${D}-${STAMP}.csv"
done
echo "=== done ==="
