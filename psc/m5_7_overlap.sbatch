#!/bin/bash
# Overlap scheduling against the machine's CPU:GPU ratio: the same ablation, the same
# harness and the same caliber on every GPU, so that what changes between rows is the
# machine and the model, not the measurement.
#
# In one allocation: the denominators (copy, read and GEMM rates, bench/roofline.py);
# then overlap on and off, alternating in one process, for Qwen3-0.6B and Qwen3-8B at
# running caps of 8, 32 and 128, the pool pinned (the laptop GPU runs the same rows
# except the ones it cannot hold). One timeline per model at a cap of 32, both arms.
#
# EXTRAS=1 adds the host-side measurements that only mean something on a node with
# a slow CPU: the block bookkeeping backends end to end (python, cpp, cpp, python),
# the per-component host path of a decode step, and the prefix cache tree at this
# GPU's pool size. They need the C++ extension, built here for this node.
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s,EXTRAS=1 psc/m5_7_overlap.sbatch
#   sbatch -p GPU-shared --gres=gpu:h100-80:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=h100 psc/m5_7_overlap.sbatch
#SBATCH -N 1 -t 04:00:00 -J miniserve-m5-7
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
PROF=${PROF:-$PROJECT/profiling/$DEVICE}
PRIVATE=${PRIVATE:-$PROJECT/miniserve_private}   # host-path scripts that are not part of the repository
CAPS=${CAPS:-"8 32 128"}
EXTRAS=${EXTRAS:-0}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  h100) GPU="H100"; ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$OUT" "$PROF" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), image $(sha256sum "$SIF" | cut -c1-16)"
el "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

NSYS_BIN=${NSYS_BIN:-$(command -v nsys || echo /opt/packages/cuda/v13.3.1/bin/nsys)}
NSYS_ROOT=""
[ -x "$NSYS_BIN" ] && NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$NSYS_BIN")")")
RUN=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
     --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --env NSYS_NVTX_PROFILER_REGISTER_ONLY=0
     --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out"
     --bind "$PROF:/prof" --bind "$PRIVATE:/private" ${NSYS_ROOT:+--bind "$NSYS_ROOT"} --pwd /workspace "$SIF")

el "weights"; "${RUN[@]}" python -m miniserve.tools.fetch_weights 0.6B 8B >/dev/null || exit 1
FAILED=()
step() {
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}

step "roofline" "${RUN[@]}" python -m bench.roofline --results-dir /out

# The caliber: 512-token prompts, 256 output tokens, four times the cap in requests,
# all at once, graphs on, the pool pinned so every GPU runs the same batches.
C="--workload unique --prefix-len 480 --suffix-len 32 --output-len 256 --rounds 3 --ablate overlap --results-dir /out"
for model in 0.6B 8B; do
  extra=""; [ "$model" = 8B ] && extra="--attn-workspace-mb 512 --warmup-max-s 300"
  for n in $CAPS; do
    pool=$(( n * 768 > 32768 ? n * 768 : 32768 ))
    S="--model $model --max-running $n --cuda-graph-max-bs $n --groups $(( n * 4 )) --per-group 1 --kv-pool-tokens $pool"
    step "overlap, $model, cap $n" "${RUN[@]}" python -m bench.offline $C $S $extra --out m5_7_overlap
  done
  if [ -n "$NSYS_ROOT" ]; then
    n=32
    S="--model $model --max-running $n --cuda-graph-max-bs $n --groups $(( n * 4 )) --per-group 1 --kv-pool-tokens 32768"
    for arm in overlap_on overlap_off; do
      step "timeline, $model, cap $n, $arm" "${RUN[@]}" "$NSYS_ROOT/bin/nsys" profile -t cuda,nvtx,osrt \
        --cuda-memory-usage=false --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
        -o "/prof/m5_7_${model}_${arm}" -f true python -m bench.offline $C $S $extra --rounds 1 --arms $arm \
        --out "m5_7_${model}_${arm}_rows" --results-dir /prof
    done
  fi
done

if [ "$EXTRAS" = 1 ]; then
  # The C++ extension for this node: the image's baseline (-march=x86-64-v2), never native.
  step "build minicore" "${RUN[@]}" sh -c 'cmake -S minicore -B /tmp/minicore-build -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$(command -v python) >/dev/null && cmake --build /tmp/minicore-build --parallel >/dev/null \
      && python -c "import miniserve.cache._minicore as m; print(m.BACKEND)"'
  # The block bookkeeping backends end to end, where the host is slow: the main caliber's
  # lengths at 256 running requests, the GPU's own pool size.
  B="--workload unique --prefix-len 992 --suffix-len 32 --output-len 256 --groups 512 --per-group 1 --max-running 256
     --cuda-graph-max-bs 256 --kv-pool-tokens 335000 --rounds 2 --results-dir /out"
  for b in python cpp cpp python; do
    step "block backend $b" "${RUN[@]}" python -m bench.offline $B --block-backend "$b" --out m5_7_block_backend
  done
  for b in python cpp; do
    step "host path, $b" "${RUN[@]}" python /private/host_breakdown.py --batches 32 128 248 --block-backend "$b" \
      --json "/out/m5_7_host_breakdown_$b.json"
  done
  for b in python cpp cpp-walk; do
    step "prefix tree scan, $b" "${RUN[@]}" python /private/radix_host.py scale --block-backend "$b" \
      --out "/out/m5_7_radix_host_scale_$b.json"
  done
  for b in python cpp; do
    step "prefix cache in the engine, $b" "${RUN[@]}" python /private/radix_host.py engine --block-backend "$b" \
      --workloads many_small many_small_big --out "/out/m5_7_radix_host_engine_$b.json"
  done
fi

el "wrote to $OUT and $PROF"; ls -l "$OUT" | grep m5_7 || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
