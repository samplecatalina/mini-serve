#!/bin/bash
# Trained draft heads against a small draft model, in the shapes the heads were built for, on
# natural text, on one L40S.
#
# An earlier job measured one public EAGLE-3 head for Qwen3-8B in a chain of 4 on random
# prompts: 1.8-2.2 tokens per round, against 3.1-3.7 for the same family's 0.6B model as a
# draft. A chain and random prompts are the least favourable conditions for a head; this job
# gives it tree-shaped verification and MT-Bench prompts, and adds a second head.
#
# In sglang 0.5.10, at running caps 1, 8 and 32, the same MT-Bench requests through:
#   plain  |  two-model (STANDALONE, 0.6B, chain of 4)
#   EAGLE-3, Tengyunw head:  chain (4,1,5)  |  tree (3,4,8)  |  tree (5,8,32)
#   EAGLE-3, AngelSlim head: chain (4,1,5)  |  tree (3,4,8)  |  tree (5,8,32)
# (shape = steps, top-k per step, draft tokens verified). Speedups are against the plain arm of
# the same cap in the same job.
#
# Before submitting, from a login node, both heads into the shared cache:
#   apptainer exec --env HF_HOME=/cache/huggingface --bind $PROJECT/miniserve_cache:/cache \
#     $PROJECT/sif/sglang-v0.5.10.sif python -c "from huggingface_hub import snapshot_download as s; \
#     s('Tengyunw/qwen3_8b_eagle3', revision='2a1059d51f622b8cad7d7d72840153ffea5488a0'); \
#     s('AngelSlim/Qwen3-8B_eagle3', revision='9629dfce7a4a10564dd48d3e5485c3976095653c')"
# and the MT-Bench file (see psc/m6_b1_mtbench.sbatch).
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m6_b2_eagle3_tree.sbatch
#SBATCH -N 1 -t 03:30:00 -J miniserve-m6-b2
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
SIZES=${SIZES:-"1 8 32"}
SHAPES=${SHAPES:-"4,1,5 3,4,8 5,8,32"}
POOL=${POOL:-65536}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$OUT" "$TMPDIR" "$TMPDIR/sglang-cache"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), sglang $(sha256sum "$SGLANG_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env PYTHONPATH=/workspace
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --pwd /workspace "$SIF")
SG=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1 --env PYTHONPATH=/workspace
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out"
    --pwd /workspace "$SGLANG_SIF")

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights 0.6B 8B >/dev/null || exit 1
el "heads in the cache"; "${SG[@]}" python -c "
from bench.sglang_spec import EAGLE3_HEADS, eagle3_path
for h in EAGLE3_HEADS: print(h, eagle3_path(h))" || exit 1
FAILED=()
step() {
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}
gpu_free() {
  pkill -u "$(id -u)" -f "sglang" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 1
  done
  el "  WARNING: ${used} MiB still in use"
}
sg_run() {  # sg_run <workload file> <arm> <cap> [more args]
  local wl=$1 arm=$2 n=$3; shift 3
  "${SG[@]}" python -m bench.sglang_spec run --workload "$wl" --arm "$arm" --max-running "$n" \
    --kv-pool-tokens "$POOL" --out m6_6_eagle3 --results-dir /out "$@"
}
sg_try() { sg_run "$@" || { gpu_free; el "  retrying once"; sg_run "$@"; }; }

for n in $SIZES; do
  reqs=$(( n * 4 > 8 ? n * 4 : 8 ))
  wl=/tmp/m6_b2_w$n.json
  step "cap $n: requests" "${MS[@]}" python -m bench.sglang_spec dump --out "$wl" --workload mtbench \
    --max-running "$n" --requests "$reqs" --output-len 256
  step "cap $n: plain" sg_try "$wl" plain "$n"; gpu_free
  step "cap $n: two-model" sg_try "$wl" standalone "$n"; gpu_free
  for head in tengyunw angelslim; do
    for shape in $SHAPES; do
      step "cap $n: eagle3 $head $shape" sg_try "$wl" eagle3 "$n" --eagle-head "$head" --tree "$shape"
      gpu_free
    done
  done
done

el "wrote to $OUT"; ls -l "$OUT" | grep m6_6 || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
