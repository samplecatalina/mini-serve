#!/bin/bash
# The denominators on a cluster GPU: achievable copy bandwidth and BF16 GEMM rate.
# Short on purpose, so it backfills: nothing here loads the model or the engine.
# Submit from the repository root:
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log --export=ALL,DEVICE=l40s psc/roofline.sbatch
#   sbatch -p GPU-shared --gres=gpu:h100-80:1 -o $PROJECT/logs/%x_%j.log --export=ALL,DEVICE=h100 psc/roofline.sbatch
#SBATCH -N 1 -t 00:15:00 -J miniserve-roofline
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
# Results go outside the clone: a job that writes into it leaves the working tree
# dirty, which blocks the next `git pull` and makes the benchmark refuse to run.
OUT=${OUT:-$PROJECT/results/$DEVICE}
mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), image $(sha256sum "$SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU"
RUN=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
     --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --bind "$REPO:/workspace"
     --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")
el "clocks before"; nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,enforced.power.limit,clocks_event_reasons.active --format=csv
el "roofline"; "${RUN[@]}" python -m bench.roofline --results-dir /out "$@"
el "wrote to $OUT"; ls -l "$OUT"
el "clocks after"; nvidia-smi --query-gpu=clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,enforced.power.limit,clocks_event_reasons.active --format=csv
el "done"
