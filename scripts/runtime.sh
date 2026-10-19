#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME="$ROOT/toolchains/cuda-12.8"
export CC="$ROOT/toolchains/gcc13/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ROOT/toolchains/gcc13/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export PATH="$ROOT/.venv-train/bin:$CUDA_HOME/bin:$ROOT/toolchains/gcc13/bin:$PATH"
export RAY_TMPDIR=/tmp/ulvr-run-20261018
export TMPDIR="$ROOT/.cache/tmp"
export OMP_NUM_THREADS=4
export MAX_JOBS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
exec "$ROOT/.venv-train/bin/python" -u "$ROOT/scripts/train.py" \
  --run-id ulvr_vopd_4b_seed42_20261018 "$@"
