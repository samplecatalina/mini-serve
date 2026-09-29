#!/bin/bash
# Fused operators on and off on the L40S, part 1: the serving caliber, overlap, timelines.
#
# In one allocation, in this order:
#   1. the anchor of the fused path on this GPU (tests/test_fused.py, with the speculative case on
#      the 8B target and the 0.6B draft); nothing else runs if it fails
#   2. the main caliber against sglang 0.5.10 (genai-bench, D(1024,256), concurrency 1 / 32 / 256):
#      three servers, this engine on the reference operators, the same on the fused ones, and sglang,
#      rotated A B C C B A A B C, everything else as in the earlier head-to-head job (pool 335,000,
#      chunks of 4096, prefill budget 16,384, 256 running, graphs to 256)
#   3. overlap on and off for Qwen3-0.6B and Qwen3-8B at running caps 8 / 32 / 128 (the earlier
#      overlap job's caliber), once on each operator path, and the operator paths against each other
#      with overlap on, in one process
#   4. node-level timelines (nsys --cuda-graph-trace=node): a decode step at batch 8 for both
#      models on both paths, and one speculative round (8B / 0.6B, gamma 4) on both
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m7_3_serving.sbatch
#SBATCH -N 1 -t 05:00:00 -J miniserve-m7-c1
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
BENCH_SIF=${BENCH_SIF:-$PROJECT/sif/miniserve-bench.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
PROF=${PROF:-$PROJECT/profiling/$DEVICE}
CAPS=${CAPS:-"8 32 128"}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in l40s) GPU="L40S"; ARCH=8.9;; *) echo "unknown DEVICE $DEVICE" >&2; exit 2;; esac
mkdir -p "$OUT" "$PROF" "$TMPDIR" "$TMPDIR/sglang-cache"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), sglang $(sha256sum "$SGLANG_SIF" | cut -c1-16), load $(sha256sum "$BENCH_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

NSYS_BIN=${NSYS_BIN:-$(command -v nsys || echo /opt/packages/cuda/v13.3.1/bin/nsys)}
NSYS_ROOT=""
[ -x "$NSYS_BIN" ] && NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$NSYS_BIN")")")
MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --env NSYS_NVTX_PROFILER_REGISTER_ONLY=0
    --env MINISERVE_SPEC_ANCHOR_TARGET=8B
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --bind "$PROF:/prof"
    ${NSYS_ROOT:+--bind "$NSYS_ROOT"} --pwd /workspace "$SIF")
SG=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" "$SGLANG_SIF")
LOAD=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env PYTHONPATH=/workspace
      --env MINISERVE_BENCH_IMAGE="sha256:$(sha256sum "$BENCH_SIF" | cut -d' ' -f1)"
      --env MINISERVE_HOST="$(hostname)" --env HF_HOME=/cache/huggingface
      --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out"
      --pwd /workspace "$BENCH_SIF")

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights 0.6B 1.7B 8B >/dev/null || exit 1
FAILED=()
step() {  # one failure must not take the rest of the allocation with it
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}

# ---------------------------------------------------------------- 1. the anchor
el "anchor: fused path (tests/test_fused.py; speculative case on 8B / 0.6B)"
ALOG=$PROJECT/logs/m7_3_anchor_${SLURM_JOB_ID:-local}_c1.log
"${MS[@]}" python -m pytest -q -s -p no:cacheprovider tests/test_fused.py > "$ALOG" 2>&1; rc=$?
grep -E "^\[|passed|failed|skipped|^E " "$ALOG"
# A skipped case (the 8B one when memory is short) is an anchor that did not run: stop as well.
if [ $rc -ne 0 ] || grep -q " skipped" "$ALOG"; then el "anchor FAILED or incomplete (exit $rc): nothing else runs"; exit 1; fi

# ---------------------------------------------------------------- 2. the main caliber
MODEL_IN=/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
PORT_MS=${PORT_MS:-8000}; PORT_SG=${PORT_SG:-30000}
POOL=335000; CHUNK=4096; PREFILL_TOKENS=16384; RUNNING=256
SERVER_PID=""
stop_server() {
  [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
  pkill -u "$(id -u)" -f "sglang.launch_server" 2>/dev/null
  pkill -u "$(id -u)" -f "miniserve.server" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 1
  done
  el "    WARNING: ${used} MiB still in use after 60s"
}
trap stop_server EXIT
wait_up() {  # wait_up <port> <log>
  for i in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && { el "    up after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { el "    server exited"; tail -40 "$2"; return 1; }
    sleep 1
  done
  el "    server never answered"; tail -40 "$2"; return 1
}
start_miniserve() {  # start_miniserve [--fused-ops]
  stop_server
  local log=$TMPDIR/ms_${SLURM_JOB_ID}.log
  "${MS[@]}" python -m miniserve.server --host 127.0.0.1 --port "$PORT_MS" \
      --max-running "$RUNNING" --cuda-graph-max-bs 256 --kv-pool-tokens "$POOL" \
      --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$PREFILL_TOKENS" --schedule-policy fcfs "$@" \
      > "$log" 2>&1 &
  SERVER_PID=$!; wait_up "$PORT_MS" "$log"
}
start_sglang() {
  stop_server
  local log=$TMPDIR/sg_${SLURM_JOB_ID}.log
  "${SG[@]}" python -m sglang.launch_server --model-path "$MODEL_IN" \
      --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port "$PORT_SG" \
      --attention-backend flashinfer --max-running-requests "$RUNNING" \
      --cuda-graph-max-bs 256 --max-total-tokens "$POOL" \
      --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$PREFILL_TOKENS" --schedule-policy fcfs \
      > "$log" 2>&1 &
  SERVER_PID=$!; wait_up "$PORT_SG" "$log"
}
el "conditions"; "${LOAD[@]}" python -m bench.sidecar /out/m7_3_serving_preflight.json >/dev/null || exit 1
serve() {  # serve <engine>: miniserve | miniserve-fused | sglang
  local engine=$1 port kind
  el "  $engine"
  case $engine in
    miniserve) port=$PORT_MS; kind=miniserve; start_miniserve || { FAILED+=("$engine"); return 0; };;
    miniserve-fused) port=$PORT_MS; kind=miniserve; start_miniserve --fused-ops || { FAILED+=("$engine"); return 0; };;
    sglang) port=$PORT_SG; kind=sglang; start_sglang || { FAILED+=("$engine"); return 0; };;
  esac
  "${LOAD[@]}" python -m bench.serving --url "http://127.0.0.1:$port" --server-kind "$kind" \
      --label "$engine" --results-dir /out --out m7_3_serving \
      --preflight-json /out/m7_3_serving_preflight.json \
      --concurrency 1 32 256 --scenario "D(1024,256)" \
      --max-time-per-run 2 --max-requests-per-run 4096 --warmup-s 30 \
      $([ "$engine" = sglang ] && echo "--server-engine SGLang") \
    || { el "    load generator failed"; FAILED+=("$engine"); }
}
el "main caliber: A B C C B A A B C"
for round in "miniserve miniserve-fused sglang" "sglang miniserve-fused miniserve" "miniserve miniserve-fused sglang"; do
  for engine in $round; do serve "$engine"; done
done
stop_server

# ---------------------------------------------------------------- 3. overlap on both paths
C="--workload unique --prefix-len 480 --suffix-len 32 --output-len 256 --rounds 3 --results-dir /out"
for model in 0.6B 8B; do
  extra=""; [ "$model" = 8B ] && extra="--attn-workspace-mb 512 --warmup-max-s 300"
  for n in $CAPS; do
    pool=$(( n * 768 > 32768 ? n * 768 : 32768 ))
    S="--model $model --max-running $n --cuda-graph-max-bs $n --groups $(( n * 4 )) --per-group 1 --kv-pool-tokens $pool"
    step "overlap, $model, cap $n, reference operators" "${MS[@]}" python -m bench.offline $C $S $extra \
      --ablate overlap --out m7_3_overlap_ref
    step "overlap, $model, cap $n, fused operators" "${MS[@]}" python -m bench.offline $C $S $extra \
      --ablate overlap --fused-ops --out m7_3_overlap_fused
    step "operators, $model, cap $n" "${MS[@]}" python -m bench.offline $C $S $extra \
      --ablate fused_ops --out m7_3_fused
  done
done

# ---------------------------------------------------------------- 4. timelines
if [ -n "$NSYS_ROOT" ]; then
  NS="$NSYS_ROOT/bin/nsys profile -t cuda,nvtx --cuda-graph-trace=node --cuda-memory-usage=false"
  NR="--capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop"
  D8="--workload unique --groups 8 --per-group 1 --prefix-len 480 --suffix-len 32 --output-len 256 --rounds 1
      --max-running 8 --cuda-graph-max-bs 8 --kv-pool-tokens 65536 --results-dir /prof --ablate fused_ops"
  for model in 0.6B 8B; do
    extra=""; [ "$model" = 8B ] && extra="--attn-workspace-mb 512 --warmup-max-s 300"
    for arm in fused_off fused_on; do
      step "timeline: $model decode at batch 8, $arm" "${MS[@]}" $NS $NR -o "/prof/m7_3_${model}_decode8_${arm}_nodes" -f true \
        python -m bench.offline --model $model $D8 $extra --arms $arm --out "m7_3_${model}_decode8_${arm}_nodes_rows"
    done
  done
  for arm in fused_off fused_on; do
    step "timeline: a speculative round (8B / 0.6B, cap 8), $arm" "${MS[@]}" $NS $NR -o "/prof/m7_3_spec_round_${arm}_nodes" -f true \
      python -m bench.offline --model 8B --spec-draft 0.6B --spec-gamma 4 --disable-radix --disable-overlap \
      --workload unique --groups 32 --per-group 1 --prefix-len 480 --suffix-len 32 --output-len 128 --rounds 1 \
      --max-running 8 --cuda-graph-max-bs 8 --kv-pool-tokens 65536 --attn-workspace-mb 512 --warmup-max-s 300 \
      --ablate fused_ops --arms $arm --out "m7_3_spec_round_${arm}_nodes_rows" --results-dir /prof
  done
else
  el "no nsys on this node: timelines skipped"; FAILED+=("timelines")
fi

el "wrote to $OUT and $PROF"; ls -l "$OUT" "$PROF" | grep m7_3 || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
