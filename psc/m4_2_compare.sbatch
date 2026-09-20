#!/bin/bash
# Head to head against the reference engine, on one GPU, in one allocation.
#
# Written against what the probe job actually found, not against documentation:
# the reference engine's own report of its settings (/get_server_info) named every
# flag used below, and its runtime KV pool came out at 335,379 tokens.
#
# What is made equal, and what cannot be:
#
#   equal   KV pool (both pinned to 335,000 tokens), largest batch with a captured
#           graph, chunked-prefill budget, prefill token budget, maximum running
#           requests, attention backend, dtype, admission policy, prefix caching,
#           the model and its pinned revision, the load generator and its image,
#           the machine, the loopback, the discarded warmup window
#   not     paging granularity (theirs is per token, ours is 16-token blocks),
#           the exact set of captured batch sizes (only the largest is aligned),
#           flashinfer 0.6.7.post2 against post3, torch cu129 against cu128
#
# The pair is run A B B A A B so that clock drift over the hour falls on both
# sides the same way, and the median of three is what gets reported.
#
# Their default largest captured batch is 32, ours is 256. Running only at 256
# would be tuning their engine for them; running only at 32 would hide what ours
# does at the concurrency the whole project is about. So both, and both reported.
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m4_2_compare.sbatch
#SBATCH -N 1 -t 02:30:00 -J miniserve-compare
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
BENCH_SIF=${BENCH_SIF:-$PROJECT/sif/miniserve-bench.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in l40s) ARCH=8.9;; h100) ARCH=9.0;; *) echo "unknown DEVICE" >&2; exit 2;; esac
mkdir -p "$OUT" "$TMPDIR" "$TMPDIR/sglang-cache"
MODEL_IN=/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
PORT_MS=${PORT_MS:-8000}
PORT_SG=${PORT_SG:-30000}

# The settings both engines are pinned to. 335,000 is just under the pool the
# reference engine sizes for itself (335,379), so neither side is asked for more
# than it would have taken; ours would otherwise have taken 353,936.
POOL=335000
CHUNK=4096
PREFILL_TOKENS=16384
RUNNING=256

cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), reference $(sha256sum "$SGLANG_SIF" | cut -c1-16), load $(sha256sum "$BENCH_SIF" | cut -c1-16)"
bash psc/preflight.sh "$SIF" "$CACHE" "$(case $DEVICE in l40s) echo L40S;; h100) echo H100;; esac)" || exit 1

MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --bind "$REPO:/workspace"
    --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")
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

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights >/dev/null || exit 1
# While nothing is loaded; every run below is handed this record.
el "conditions"; "${LOAD[@]}" python -m bench.sidecar /out/m4_2_compare_preflight.json >/dev/null || exit 1

SERVER_PID=""; FAILED=()
# Killing the container does not necessarily kill what it started: the reference
# engine runs a scheduler and a detokenizer as separate processes, and one left
# holding memory would make the next server on this GPU fail to allocate. So the
# stop is not finished until the GPU is actually free again.
stop_server() {
  [ -n "$SERVER_PID" ] && { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
  pkill -u "$(id -u)" -f "sglang.launch_server" 2>/dev/null
  pkill -u "$(id -u)" -f "miniserve.server" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && { [ "$i" -gt 3 ] && el "    GPU free again after ${i}s"; return 0; }
    sleep 1
  done
  el "    WARNING: ${used} MiB still in use after 60s; the next server may fail to allocate"
}
trap stop_server EXIT

wait_up() {  # wait_up <port> <route> <log>
  for i in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$1$2" >/dev/null 2>&1 && { el "    up after ${i}s"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { el "    server exited"; tail -40 "$3"; return 1; }
    sleep 1
  done
  el "    server never answered"; tail -40 "$3"; return 1
}

start_miniserve() {  # start_miniserve <graph max bs>
  stop_server
  local log=$TMPDIR/ms_${SLURM_JOB_ID}.log
  "${MS[@]}" python -m miniserve.server --host 127.0.0.1 --port "$PORT_MS" \
      --max-running "$RUNNING" --cuda-graph-max-bs "$1" --kv-pool-tokens "$POOL" \
      --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$PREFILL_TOKENS" --schedule-policy fcfs \
      > "$log" 2>&1 &
  SERVER_PID=$!; wait_up "$PORT_MS" /health "$log"
}

start_sglang() {  # start_sglang <graph max bs>
  stop_server
  local log=$TMPDIR/sg_${SLURM_JOB_ID}.log
  "${SG[@]}" python -m sglang.launch_server --model-path "$MODEL_IN" \
      --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port "$PORT_SG" \
      --attention-backend flashinfer --max-running-requests "$RUNNING" \
      --cuda-graph-max-bs "$1" --max-total-tokens "$POOL" \
      --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$PREFILL_TOKENS" --schedule-policy fcfs \
      > "$log" 2>&1 &
  SERVER_PID=$!; wait_up "$PORT_SG" /health "$log"
}

# run <engine> <graph max bs> <out name> <concurrency...>
run() {
  local engine=$1 bs=$2 out=$3; shift 3
  el "  $engine, graphs to $bs, concurrency $*"
  local port kind
  if [ "$engine" = miniserve ]; then port=$PORT_MS; kind=miniserve; start_miniserve "$bs" || { FAILED+=("$engine/$bs"); return 0; }
  else port=$PORT_SG; kind=sglang; start_sglang "$bs" || { FAILED+=("$engine/$bs"); return 0; }; fi
  "${LOAD[@]}" python -m bench.serving --url "http://127.0.0.1:$port" --server-kind "$kind" \
      --label "$engine" --results-dir /out --out "$out" \
      --preflight-json /out/m4_2_compare_preflight.json \
      --concurrency "$@" --scenario "D(1024,256)" \
      --max-time-per-run 2 --max-requests-per-run 4096 --warmup-s 30 \
      $([ "$engine" = sglang ] && echo "--server-engine SGLang") \
    || { el "    load generator failed"; FAILED+=("$engine/$bs"); }
}

el "=== both engines with graphs to 256, A B B A A B ==="
for pair in "miniserve sglang" "sglang miniserve" "miniserve sglang"; do
  for engine in $pair; do run "$engine" 256 m4_2_compare 1 32 256; done
done

el "=== both engines at the reference engine's own default of 32, one round ==="
for engine in miniserve sglang; do run "$engine" 32 m4_2_compare_bs32 256; done

stop_server
el "wrote to $OUT"; ls -l "$OUT" | grep compare || true
if [ ${#FAILED[@]} -gt 0 ]; then el "runs that produced nothing: ${FAILED[*]}"; else el "every run produced results"; fi
el "done"
