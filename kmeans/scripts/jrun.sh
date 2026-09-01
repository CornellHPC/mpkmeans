#!/usr/bin/env bash
# Run a command on the compute node of an existing interactive allocation.
#   JOBID=57821507 ./scripts/jrun.sh ./build/mpkmeans_bench -n 100000
set -uo pipefail
: "${JOBID:?set JOBID to the salloc job id}"
exec srun --jobid="$JOBID" -N 1 -n 1 -c 32 -G 1 --export=ALL "$@"
