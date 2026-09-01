#!/bin/bash
# Configure and build mpemu on Perlmutter.
#   module load cudatoolkit craype-accel-nvidia80 cmake
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The Cray PE default host compiler (/usr/bin/c++) is gcc 7.5, too old for
# CUDA 13; gcc-native/14 is what the loaded PrgEnv-gnu provides.
CXX_HOST="${MPEMU_HOST_CXX:-/opt/cray/pe/gcc-native/14/bin/g++}"

cmake -S "$ROOT" -B "$ROOT/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_CXX_COMPILER="$CXX_HOST" \
      -DCMAKE_CUDA_HOST_COMPILER="$CXX_HOST" \
      "$@"
cmake --build "$ROOT/build" -j "${BUILD_JOBS:-16}"
