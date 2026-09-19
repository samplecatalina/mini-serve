#!/bin/bash
# Port check on a cluster GPU: preflight, correctness anchor, greedy outputs for
# comparison across devices and, with SMOKE=1, the serving smoke test (server up
# with one command, 32 concurrent streaming requests) plus the same workload
# through the offline engine to measure the serving front end's cost.
# Submit from the repository root:
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log --export=ALL,DEVICE=l40s,SMOKE=1 psc/m3_5.sbatch
#   sbatch -p GPU-shared --gres=gpu:h100-80:1 -o $PROJECT/logs/%x_%j.log --export=ALL,DEVICE=h100 psc/m3_5.sbatch
#SBATCH -N 1 -t 00:50:00 -J miniserve-port
set -euo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  h100) GPU="H100"; ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
OUT=$REPO/results/$DEVICE; mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), image $(sha256sum "$SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU"
# FlashInfer and Triton compile for the GPU at run time; the image defaults to sm_89.
RUN=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
     --env TORCH_CUDA_ARCH_LIST=$ARCH --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --pwd /workspace "$SIF")
el "container start + import"; "${RUN[@]}" python -c "import torch, flashinfer, miniserve.engine.engine" && el "imported"
el "environment check"; "${RUN[@]}" python -m miniserve.tools.env_check || el "env_check reported problems (see above)"
el "weights"; "${RUN[@]}" python -m miniserve.tools.fetch_weights
el "anchor tests"
"${RUN[@]}" python -m pytest -q -p no:cacheprovider tests/test_consistency.py tests/test_engine.py \
  -k "consistency or single_request or concurrent_matches_reference or positions_contract or decode_graph" 2>&1 | tail -5
el "golden outputs"; "${RUN[@]}" python -m miniserve.tools.golden --out "results/$DEVICE/m3_5_golden.json"
if [ "${SMOKE:-0}" = 1 ]; then
  W="--workload unique --groups 4 --per-group 8 --prefix-len 1024 --suffix-len 64 --output-len 256"
  el "offline engine, same workload as the smoke test"
  "${RUN[@]}" python -m bench.offline $W --rounds 3 --kv-pool-tokens 65536 --out m3_5_offline
  el "server start"
  "${RUN[@]}" python -m miniserve.server --port 8000 --kv-pool-tokens 65536 > "$TMPDIR/server_$SLURM_JOB_ID.log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 600); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 1; done
  el "server ready after ${i}s"
  for k in 1 2 3; do
    "${RUN[@]}" python -m bench.http_smoke $W --concurrency 32 --label "run $k" --out "results/$DEVICE/m3_5_http_smoke.csv"
  done
  kill $SERVER; wait $SERVER || true
  tail -5 "$TMPDIR/server_$SLURM_JOB_ID.log"
fi
el "done"
