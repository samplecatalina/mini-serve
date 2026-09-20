#!/bin/bash
# The main caliber and the three ablations, on one GPU, in one allocation.
#
# Everything a comparison needs to be read together is measured here: the same
# node, the same clocks within an hour, the same load generator image, one
# preflight while the GPU is still idle. The comparison against other engines is
# a separate job on purpose -- their runtime has its own way of failing on these
# nodes, and it must not be able to take these numbers down with it.
#
# Two containers: the engine (miniserve.sif, byte-identical to the one the port
# was verified with) serves, the load generator (miniserve-bench.sif, genai-bench
# and nothing else) drives it over this node's loopback.
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m4_2.sbatch
#SBATCH -N 1 -t 02:30:00 -J miniserve-m4-2
set -euo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve.sif}
BENCH_SIF=${BENCH_SIF:-$PROJECT/sif/miniserve-bench.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  h100) GPU="H100"; ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
PORT=${PORT:-8000}
mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "engine image $(sha256sum "$SIF" | cut -c1-16), load generator $(sha256sum "$BENCH_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU"

ENGINE=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
        --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --bind "$REPO:/workspace"
        --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")
LOAD=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env PYTHONPATH=/workspace
      --env MINISERVE_BENCH_IMAGE="sha256:$(sha256sum "$BENCH_SIF" | cut -d' ' -f1)"
      --env MINISERVE_HOST="$(hostname)" --env HF_HOME=/cache/huggingface
      --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out"
      --pwd /workspace "$BENCH_SIF")

el "weights"; "${ENGINE[@]}" python -m miniserve.tools.fetch_weights
# While nothing is loaded. One check for the whole job: the arms below start and
# stop servers, and after the first one "the GPU is idle" is no longer true.
el "conditions"; "${LOAD[@]}" python -m bench.sidecar /out/m4_2_preflight.json > /dev/null

SERVER_PID=""
FAILED=()
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 5   # let the allocator hand the memory back before the next server asks for it
}
trap stop_server EXIT

# start_server <engine flags...>
start_server() {
  stop_server
  "${ENGINE[@]}" python -m miniserve.server --host 127.0.0.1 --port "$PORT" "$@" \
      > "$TMPDIR/server_${SLURM_JOB_ID}.log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { el "  server up after ${i}s: $*"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || { el "  server died:"; tail -30 "$TMPDIR/server_${SLURM_JOB_ID}.log"; return 1; }
    sleep 1
  done
  el "  server did not come up"; tail -30 "$TMPDIR/server_${SLURM_JOB_ID}.log"; return 1
}

# arm <label> <out name> <concurrency...> -- <engine flags...>
arm() {
  local label=$1 out=$2; shift 2
  local conc=() ; while [ "$1" != "--" ]; do conc+=("$1"); shift; done; shift
  el "arm $label: engine $*"
  # One arm that fails must not take the rest of a two-hour allocation with it:
  # the queue wait to redo the others costs far more than this arm's data.
  if ! start_server "$@"; then FAILED+=("$label(server)"); return 0; fi
  # The conditions were recorded above, while the GPU was still idle. Without
  # this the benchmark would check them itself, against a GPU that now holds the
  # server it is about to measure, and refuse every single arm.
  if ! "${LOAD[@]}" python -m bench.serving --url "http://127.0.0.1:$PORT" --results-dir /out \
      --preflight-json /out/m4_2_preflight.json \
      --out "$out" --label "$label" --concurrency "${conc[@]}" \
      --scenario "D(1024,256)" --max-time-per-run 2 --max-requests-per-run 4096 --warmup-s 30
  then
    el "  arm $label FAILED; continuing with the rest"
    FAILED+=("$label")
  fi
}

# Batches of 256 are only real if the engine is allowed to run 256 and has graphs
# for them; the default of 64 would quietly turn the main caliber into a 64 one.
BIG=(--max-running 256 --cuda-graph-max-bs 256)

el "=== main caliber ==="
arm miniserve m4_2_main 1 4 32 256 -- "${BIG[@]}"
# Tolerating a failed arm is for an arm's own trouble. If the first one fails,
# whatever is wrong is wrong for all of them, and running fifteen more of it
# spends the allocation to produce nothing.
if [ ${#FAILED[@]} -gt 0 ]; then
  el "the main caliber failed (${FAILED[*]}); the rest would fail the same way. Stopping."
  exit 1
fi

el "=== ablation 1: decode CUDA Graph ==="
arm graph_on  m4_2_cuda_graph 32 256 -- "${BIG[@]}"
arm graph_off m4_2_cuda_graph 32 256 -- --max-running 256 --disable-cuda-graph

el "=== ablation 2: overlap scheduling ==="
arm overlap_on  m4_2_overlap 32 256 -- "${BIG[@]}"
arm overlap_off m4_2_overlap 32 256 -- "${BIG[@]}" --disable-overlap

el "=== ablation 3a: chunked prefill ==="
arm chunk_2048 m4_2_chunked 32 256 -- "${BIG[@]}" --chunked-prefill-size 2048
arm chunk_off  m4_2_chunked 32 256 -- "${BIG[@]}" --chunked-prefill-size 0

el "=== ablation 3b: schedule policy against three KV pool sizes ==="
# 40k is what this workload gets on the 8 GB card; 100k is the middle; the third
# is whatever the 45 GB card allows, which is where the policies should converge.
for pool in 40960 102400 0; do
  for policy in fcfs sjf cache; do
    if [ "$pool" = 0 ]; then
      arm "${policy}_full" m4_2_policy 256 -- "${BIG[@]}" --schedule-policy "$policy"
    else
      arm "${policy}_${pool}" m4_2_policy 256 -- "${BIG[@]}" --schedule-policy "$policy" --kv-pool-tokens "$pool"
    fi
  done
done

stop_server
el "wrote to $OUT"; ls -l "$OUT"
if [ ${#FAILED[@]} -gt 0 ]; then el "arms that produced nothing: ${FAILED[*]}"; else el "every arm produced results"; fi
el "done"
