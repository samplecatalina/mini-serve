#!/bin/bash
# Can the reference engine run on these GPU nodes at all, and can our harness drive it?
#
# Worth its own short job because the answer is not obvious here. These nodes'
# CPUs have no AVX-512, the module compilers mostly do, and the reference engine
# compiles kernels at run time through three separate paths; outside a container
# that combination has failed three times, in two different ways. Inside its own
# published image it should be fine -- but "should" is what a probe is for, and
# finding out inside a two-hour comparison job costs a queue round trip plus the
# comparison.
#
# Four questions, cheapest first. The job stops at the first one that fails, and
# prints what failed rather than continuing.
#   1. does the image import and see the GPU?
#   2. does the server come up on this node?
#   3. what does /get_server_info actually return? (our fairness record reads it)
#   4. can our own load generator drive it end to end?
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/sglang_probe.sbatch
#SBATCH -N 1 -t 00:30:00 -J sglang-probe
set -uo pipefail          # not -e: a failing step should be reported, not silently end the job
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
BENCH_SIF=${BENCH_SIF:-$PROJECT/sif/miniserve-bench.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
PORT=${PORT:-30000}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
die() { el "PROBE FAILED: $*"; exit 1; }
mkdir -p "$OUT" "$TMPDIR" "$TMPDIR/sglang-cache"
MODEL=$(ls -d "$CACHE"/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ 2>/dev/null | head -1)
[ -n "$MODEL" ] || die "no Qwen3-0.6B snapshot under $CACHE"
MODEL_IN=/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/$(basename "$MODEL")

el "node $(hostname), sglang image $(sha256sum "$SGLANG_SIF" | cut -c1-16)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Everything it compiles or downloads at run time has to land off $HOME, which is
# small, shared, and on these nodes the usual reason a container half-works.
SG=(apptainer exec --nv --cleanenv --no-home
    --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache
    --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor
    --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" "$SGLANG_SIF")

el "--- 1. import and device ---"
"${SG[@]}" python -c "
import torch, sglang
print('sglang', sglang.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
import flashinfer; print('flashinfer', flashinfer.__version__)
" || die "the image cannot import its own stack on this node (this is the AVX-512 / JIT question)"

el "--- 2. server ---"
SERVER_LOG=$TMPDIR/sglang_${SLURM_JOB_ID}.log
"${SG[@]}" python -m sglang.launch_server \
    --model-path "$MODEL_IN" --served-model-name Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port "$PORT" --attention-backend flashinfer \
    --max-running-requests 32 --cuda-graph-max-bs 32 \
    > "$SERVER_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { el "  up after ${i}s"; break; }
  kill -0 $SERVER 2>/dev/null || { echo "--- last 60 lines of the server log ---"; tail -60 "$SERVER_LOG"; die "the server exited while starting"; }
  [ "$i" = 900 ] && { tail -60 "$SERVER_LOG"; die "the server did not answer /health within 900s"; }
  sleep 1
done

el "--- 3. what it reports about itself ---"
curl -sf "http://127.0.0.1:$PORT/get_server_info" -o "$OUT/sglang_probe_server_info.json" \
  || die "/get_server_info did not answer (our fairness record reads it)"
python3 -m json.tool "$OUT/sglang_probe_server_info.json" | head -80
echo "  (full JSON saved to $OUT/sglang_probe_server_info.json)"

el "--- 3b. one completion through the OpenAI route ---"
curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0,\"ignore_eos\":true}" \
  | head -c 600 || die "/v1/completions did not answer"
echo

el "--- 4. our own load generator against it ---"
# The conditions cannot be checked now -- the server already holds the GPU -- so a
# probe records that it skipped the check rather than pretending it passed.
echo '{"idle_gpu": {"name": "PROBE, not a measurement"}, "note": "no preflight: the server was already up"}' \
  > "$TMPDIR/probe_preflight.json"
apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env PYTHONPATH=/workspace \
  --env MINISERVE_BENCH_IMAGE="sha256:$(sha256sum "$BENCH_SIF" | cut -d' ' -f1)" \
  --env MINISERVE_HOST="$(hostname)" --env HF_HOME=/cache/huggingface \
  --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" \
  --pwd /workspace "$BENCH_SIF" \
  python -m bench.serving --url "http://127.0.0.1:$PORT" --server-kind sglang \
    --label sglang_probe --results-dir /out --out sglang_probe \
    --preflight-json /tmp/probe_preflight.json \
    --concurrency 4 --scenario "D(256,64)" --max-time-per-run 1 --max-requests-per-run 16 --warmup-s 10 \
  || die "our load generator could not drive it (see the traceback above)"

el "all four answered. The comparison job can be written against this."
el "server log tail:"; tail -15 "$SERVER_LOG"
el "done"
