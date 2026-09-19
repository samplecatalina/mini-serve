#!/bin/bash
# Preflight for a cluster job: every check must pass before any measurement.
# Usage: psc/preflight.sh SIF CACHE_DIR EXPECTED_GPU_SUBSTRING
# Checks: the GPU model; the image exists and starts; inside it, python, torch
# (with CUDA), flashinfer, nvcc and gcc work and C++17 compiles with the pinned
# baseline (no -march=native); every JIT and model cache lives in CACHE_DIR,
# outside $HOME; then the container's own environment check.
set -euo pipefail
SIF=$1 CACHE=$2 GPU=$3
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
[ -f "$SIF" ] || fail "no image at $SIF"
mkdir -p "$CACHE"
case "$(realpath "$CACHE")" in "$(realpath "$HOME")"/*) fail "cache $CACHE is under \$HOME";; esac
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$name" == *"$GPU"* ]] || fail "GPU is '$name', expected '$GPU'"
nvidia-smi --query-gpu=name,driver_version,power.max_limit,enforced.power.limit,clocks.max.sm --format=csv
command -v apptainer >/dev/null || fail "apptainer not found"
apptainer exec --nv --cleanenv --no-home --bind "$CACHE:/cache" "$SIF" bash -euc '
  for v in FLASHINFER_WORKSPACE_BASE TRITON_CACHE_DIR TVM_FFI_CACHE_DIR HF_HOME; do
    case "${!v}" in /cache/*) ;; *) echo "PREFLIGHT FAIL: $v=${!v} is not under /cache" >&2; exit 1;; esac
  done
  [ -d "$CUDA_HOME" ] || { echo "PREFLIGHT FAIL: CUDA_HOME=$CUDA_HOME missing" >&2; exit 1; }
  nvcc --version | tail -1
  echo "int main(){return 0;}" > /tmp/pf.cpp && g++ -std=c++17 $CXXFLAGS /tmp/pf.cpp -o /tmp/pf && /tmp/pf && echo "g++ $CXXFLAGS -std=c++17: ok"
  case "$CXXFLAGS $CFLAGS" in *native*) echo "PREFLIGHT FAIL: -march=native in flags" >&2; exit 1;; esac
  python -c "import torch, flashinfer; assert torch.cuda.is_available(); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"flashinfer\", flashinfer.__version__, torch.cuda.get_device_name())"
'
echo "PREFLIGHT OK"
