#!/bin/bash
#SBATCH -A m1266_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1 -n 1 -c 32 -G 1
#SBATCH -t 00:15:00
#SBATCH -J mpemu_probe
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p results
export NVIDIA_TF32_OVERRIDE=0
build/mpemu_accum_probe 1024 | tee "results/accum-probe-${SLURM_JOB_ID:-local}.csv"
echo "=== rerun main sweep with fixed CSV error fields ==="
build/mpemu_bench --reps 5 > "results/bench-${SLURM_JOB_ID:-local}.csv"
build/mpemu_bench --reps 3 --square --exp-range 20 > "results/bench-wide-${SLURM_JOB_ID:-local}.csv"
echo done
