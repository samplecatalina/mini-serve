#!/bin/bash
# Speculative decoding at the main caliber: Qwen3-8B verified, Qwen3-0.6B
# proposing, one L40S, one allocation.
#
# In order: the correctness anchor of the speculative path on this GPU; the
# acceptance rate on natural and on random prompts, each with the target
# proposing for itself as the ceiling; then throughput against the number of
# requests running at once, 1 to 64, where every size runs the speculative
# engine at gamma 0/2/4/6 in one process (alternating A B B A) and the plain
# engine (decode graphs and overlap on) in another -- the plain engine, not
# gamma 0, is what speculation has to beat. One size is repeated under Nsight
# Systems for the timeline of a round.
#
# Weights: fetch the 8B snapshot (16.4 GB) before submitting, from a login or
# data-transfer node, so the allocation does not spend its time downloading:
#   apptainer exec --bind $PROJECT/miniserve_cache:/cache $PROJECT/sif/miniserve.sif \
#       python -m miniserve.tools.fetch_weights 0.6B 1.7B 8B
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m5_5.sbatch
#SBATCH -N 1 -t 03:00:00 -J miniserve-m5-5
set -euo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
SIZES=${SIZES:-"1 2 4 8 16 32 64"}
POOL=${POOL:-65536}
PROFILE_SIZE=${PROFILE_SIZE:-8}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  h100) GPU="H100"; ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$REPO/results/$DEVICE" "$REPO/profiling/$DEVICE" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), image $(sha256sum "$SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU"

# Nsight Systems runs inside the container, from the host's installation.
NSYS_BIN=$(command -v nsys || echo "${CUDA_HOME:-/nonexistent}/bin/nsys")
NSYS_ROOT=""
[ -x "$NSYS_BIN" ] && NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$NSYS_BIN")")")
RUN=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
     --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --env NSYS_NVTX_PROFILER_REGISTER_ONLY=0
     --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp"
     ${NSYS_ROOT:+--bind "$NSYS_ROOT"} --pwd /workspace "$SIF")

el "weights"; "${RUN[@]}" python -m miniserve.tools.fetch_weights 0.6B 1.7B 8B
FAILED=()
step() {  # step <label> <command...>: one failure must not take the rest of the allocation with it
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}

step "anchor (speculative path)" "${RUN[@]}" python -m pytest -q -p no:cacheprovider tests/test_spec.py
step "acceptance" "${RUN[@]}" python -m bench.spec_alpha --model 8B --draft 0.6B --gammas 2 4 6 \
  --kv-pool-tokens 16384 --out m5_5_alpha

W="--model 8B --workload unique --prefix-len 480 --suffix-len 32 --output-len 128 --kv-pool-tokens $POOL --rounds 2"
for n in $SIZES; do
  reqs=$(( n * 4 > 8 ? n * 4 : 8 ))
  S="--max-running $n --cuda-graph-max-bs $n --groups $reqs --per-group 1"
  step "batch $n: speculative, gamma 0/2/4/6" "${RUN[@]}" python -m bench.offline $W $S \
    --spec-draft 0.6B --spec-gamma 4 --disable-radix --disable-overlap --ablate spec --out m5_5_spec
  step "batch $n: plain engine" "${RUN[@]}" python -m bench.offline $W $S --out m5_5_baseline
done

if [ -n "$NSYS_ROOT" ]; then
  n=$PROFILE_SIZE
  S="--max-running $n --cuda-graph-max-bs $n --groups $(( n * 4 )) --per-group 1"
  step "timeline, batch $n, gamma 4" "${RUN[@]}" "$NSYS_ROOT/bin/nsys" profile -t cuda,nvtx,osrt \
    --cuda-memory-usage=false --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
    -o "profiling/$DEVICE/m5_5_round" -f true python -m bench.offline $W $S --rounds 1 \
    --spec-draft 0.6B --spec-gamma 4 --disable-radix --disable-overlap --ablate spec --arms spec_g4 \
    --out m5_5_round_rows --results-dir "profiling/$DEVICE"
else
  el "no nsys on this node; timeline skipped"; FAILED+=("timeline(no nsys)")
fi

el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
