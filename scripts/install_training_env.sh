#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$ROOT/toolchains/cuda-12.8/bin/nvcc" ]]; then
  export CUDA_HOME="$ROOT/toolchains/cuda-12.8"
fi
if [[ -x "$ROOT/toolchains/gcc13/bin/x86_64-conda-linux-gnu-g++" ]]; then
  export CC="$ROOT/toolchains/gcc13/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$ROOT/toolchains/gcc13/bin/x86_64-conda-linux-gnu-g++"
  export CUDAHOSTCXX="$CXX"
fi
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
export MAX_JOBS="${MAX_JOBS:-12}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
if [[ "${1:-}" != --install ]]; then
  echo "Preparation only. To install the separate training environment later: bash $0 --install"
  exit 0
fi
uv venv --seed --python 3.12 "$ROOT/.venv-train"
uv pip sync --python "$ROOT/.venv-train/bin/python" "$ROOT/config/requirements-train.lock.txt"
uv pip install --python "$ROOT/.venv-train/bin/python" --no-deps -e "$ROOT/vendor/Vision-OPD"
# Build CUDA extensions only on the eventual training host with a matching CUDA toolkit.
FLASH_ATTENTION_FORCE_BUILD=TRUE "$ROOT/.venv-train/bin/pip" install --no-build-isolation \
  "$ROOT/vendor/cuda-extension-sources/flash_attn-2.8.3.post2.tar.gz"
CAUSAL_CONV1D_FORCE_BUILD=TRUE "$ROOT/.venv-train/bin/pip" install --no-build-isolation \
  "$ROOT/vendor/cuda-extension-sources/causal_conv1d-1.6.1.tar.gz"
"$ROOT/.venv-train/bin/python" "$ROOT/scripts/check_training_env.py"
echo "Installed. No training has been started."
