#!/bin/bash
# Speculative decoding on natural text, and where a small model's decode step goes, on one L40S.
#
# Part 1 -- MT-Bench (bench/mtbench.py): Qwen3-8B verified, at running caps 1 to 64, the same
# requests (written once as token ids) through five arms:
#   this engine     plain (graphs and overlap on)  |  two-model, gamma 2 / 4 / 6 (one process)
#   sglang 0.5.10   plain  |  two-model (STANDALONE, 0.6B, chain of 4)  |  EAGLE-3 (Tengyunw head, chain of 4)
# Random prompts understate speculation on this target (acceptance 0.431 against 0.561 at gamma
# 4); these rows are the natural-text counterpart of the random-prompt ones from an earlier job.
#
# Part 2 -- node-level timelines (nsys --cuda-graph-trace=node), so that every kernel inside a
# graph replay shows up: a plain Qwen3-0.6B decode step at batch 8 in both engines, a plain
# Qwen3-8B one in both, and one speculative round of this engine (draft and verification steps).
#
# Before submitting, from a login node, the MT-Bench file into the shared cache:
#   apptainer exec --env HF_HOME=/cache/huggingface --bind $PROJECT/miniserve_cache:/cache \
#     --bind $PROJECT/miniserve:/workspace --env PYTHONPATH=/workspace $PROJECT/sif/miniserve-m5.sif \
#     python -m bench.mtbench
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m6_b1_mtbench.sbatch
#SBATCH -N 1 -t 04:30:00 -J miniserve-m6-b1
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
PROF=${PROF:-$PROJECT/profiling/$DEVICE}
SIZES=${SIZES:-"1 2 4 8 16 32 64"}
POOL=${POOL:-65536}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$OUT" "$PROF" "$TMPDIR" "$TMPDIR/sglang-cache"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), sglang $(sha256sum "$SGLANG_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

NSYS_BIN=${NSYS_BIN:-$(command -v nsys || echo /opt/packages/cuda/v13.3.1/bin/nsys)}
NSYS_ROOT=""
[ -x "$NSYS_BIN" ] && NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$NSYS_BIN")")")
MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --env NSYS_NVTX_PROFILER_REGISTER_ONLY=0
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --bind "$PROF:/prof"
    ${NSYS_ROOT:+--bind "$NSYS_ROOT"} --pwd /workspace "$SIF")
SG=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1 --env PYTHONPATH=/workspace
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --bind "$PROF:/prof"
    ${NSYS_ROOT:+--bind "$NSYS_ROOT"} --pwd /workspace "$SGLANG_SIF")

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights 0.6B 8B >/dev/null || exit 1
el "mt-bench in the cache"; "${MS[@]}" python -c "from bench import mtbench; print(len(mtbench.questions()))" || exit 1
FAILED=()
step() {  # step <label> <command...>: one failure must not take the rest of the allocation with it
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}
gpu_free() {  # sglang runs its scheduler in child processes; wait until the memory is back
  pkill -u "$(id -u)" -f "sglang" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 1
  done
  el "  WARNING: ${used} MiB still in use"
}
sg_run() {  # sg_run <workload file> <arm> <cap> <out> [more args]
  local wl=$1 arm=$2 n=$3 out=$4; shift 4
  "${SG[@]}" python -m bench.sglang_spec run --workload "$wl" --arm "$arm" --max-running "$n" \
    --kv-pool-tokens "$POOL" --out "$out" --results-dir /out "$@"
}
# sglang's two-model path fails now and then when it merges a running batch with a newly
# prefilled one; once more before the step counts as failed (rows are written only on success).
sg_try() { sg_run "$@" || { gpu_free; el "  retrying once"; sg_run "$@"; }; }

# ---------------------------------------------------------------- part 1: MT-Bench
# Output capped at 256: MT-Bench answers are long, and a cap keeps a cap-1 run under a few minutes.
W="--model 8B --workload mtbench --output-len 256 --kv-pool-tokens $POOL --rounds 2
   --attn-workspace-mb 512 --warmup-max-s 300 --results-dir /out"
for n in $SIZES; do
  reqs=$(( n * 4 > 8 ? n * 4 : 8 ))
  wl=/tmp/m6_b1_w$n.json
  step "cap $n: requests" "${MS[@]}" python -m bench.sglang_spec dump --out "$wl" --workload mtbench \
    --max-running "$n" --requests "$reqs" --output-len 256
  S="--max-running $n --cuda-graph-max-bs $n --groups $reqs --per-group 1"
  step "cap $n: this engine, plain" "${MS[@]}" python -m bench.offline $W $S --out m6_5_mtbench_ours
  step "cap $n: this engine, two-model" "${MS[@]}" python -m bench.offline $W $S --spec-draft 0.6B \
    --disable-radix --disable-overlap --ablate spec --arms spec_g2 spec_g4 spec_g6 --out m6_5_mtbench_ours_spec
  for arm in plain standalone eagle3; do
    step "cap $n: sglang $arm" sg_try "$wl" "$arm" "$n" m6_5_mtbench_sglang
    gpu_free
  done
done

# ---------------------------------------------------------------- part 2: timelines
if [ -n "$NSYS_ROOT" ]; then
  NS="$NSYS_ROOT/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node --cuda-memory-usage=false"
  NR="--capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop"
  D8="--workload unique --groups 8 --per-group 1 --prefix-len 480 --suffix-len 32 --output-len 256 --rounds 1
      --max-running 8 --cuda-graph-max-bs 8 --kv-pool-tokens $POOL --results-dir /prof"
  for model in 0.6B 8B; do
    extra=""; [ "$model" = 8B ] && extra="--attn-workspace-mb 512 --warmup-max-s 300"
    step "timeline: this engine, $model decode at batch 8" "${MS[@]}" $NS $NR -o "/prof/m6_4_ours_${model}_decode8_nodes" -f true \
      python -m bench.offline --model $model $D8 $extra --out "m6_4_ours_${model}_decode8_nodes_rows"
    wl=/tmp/m6_b1_d8_$model.json
    step "timeline: requests for sglang, $model" "${MS[@]}" python -m bench.sglang_spec dump --out "$wl" --model $model \
      --draft 0.6B --max-running 8 --requests 8 --output-len 256
    # No NVTX to cut on in sglang: the whole run is recorded, and decode steps are the graph
    # launches with many kernels (.dev/evidence/m6_4/step_kernels.py, range "*").
    step "timeline: sglang, $model decode at batch 8" "${SG[@]}" $NS -o "/prof/m6_4_sglang_${model}_decode8_nodes" -f true \
      python -m bench.sglang_spec run --workload "$wl" --arm plain --max-running 8 --kv-pool-tokens "$POOL" \
      --rounds 1 --out "m6_4_sglang_${model}_decode8_nodes_rows" --results-dir /prof
    gpu_free
  done
  step "timeline: this engine, a speculative round (8B / 0.6B, cap 8)" "${MS[@]}" $NS $NR -o /prof/m6_4_ours_spec_round_nodes -f true \
    python -m bench.offline --model 8B --spec-draft 0.6B --spec-gamma 4 --disable-radix --disable-overlap \
    --workload unique --groups 32 --per-group 1 --prefix-len 480 --suffix-len 32 --output-len 128 --rounds 1 \
    --max-running 8 --cuda-graph-max-bs 8 --kv-pool-tokens "$POOL" --attn-workspace-mb 512 --warmup-max-s 300 \
    --ablate spec --arms spec_g4 --out m6_4_ours_spec_round_nodes_rows --results-dir /prof
else
  el "no nsys on this node: timelines skipped"; FAILED+=("timelines")
fi

el "wrote to $OUT and $PROF"; ls -l "$OUT" "$PROF" | grep -E "m6_[45]" || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
