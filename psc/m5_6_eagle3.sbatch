#!/bin/bash
# Two-model speculation against a trained draft head (EAGLE-3), on one L40S, in one allocation.
#
# Qwen3-8B verified throughout. At every running-request cap, 1 to 64, the same
# requests (bench/offline.py's speculative caliber, written once as token ids) go
# through six arms:
#
#   this engine     plain (graphs and overlap on)  |  two-model, gamma 4
#   sglang 0.5.10   plain  |  two-model (STANDALONE, Qwen3-0.6B, chain of 4)
#                   EAGLE-3 (public head, chain of 4)  |  EAGLE-3 with sglang's own draft shape
#
# The two two-model arms are one algorithm in two engines; the sglang arms are
# algorithms in one engine. Acceptance on the natural prompts bench/spec_alpha.py
# uses is recorded once per speculative arm, at the cap of 8.
#
# Before submitting: the EAGLE-3 head in the cache, from a login node --
#   apptainer exec --env HF_HOME=$PROJECT/miniserve_cache/huggingface $PROJECT/sif/sglang-v0.5.10.sif \
#     python -c "from huggingface_hub import snapshot_download as s; \
#                s('Tengyunw/qwen3_8b_eagle3', revision='2a1059d51f622b8cad7d7d72840153ffea5488a0')"
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m5_6_eagle3.sbatch
#SBATCH -N 1 -t 04:30:00 -J miniserve-m5-6
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
SIZES=${SIZES:-"1 2 4 8 16 32 64"}
POOL=${POOL:-65536}
SG_ARMS=${SG_ARMS:-"plain standalone eagle3 eagle3_auto"}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  h100) GPU="H100"; ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
# Results go outside the clone, so the clone stays clean for the next pull.
mkdir -p "$OUT" "$TMPDIR" "$TMPDIR/sglang-cache"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), sglang $(sha256sum "$SGLANG_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --bind "$REPO:/workspace"
    --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")
SG=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1 --env PYTHONPATH=/workspace
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out"
    --pwd /workspace "$SGLANG_SIF")

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights 0.6B 8B >/dev/null || exit 1
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

# The same settings on both engines: 8B, pool of $POOL tokens, chunks of 2048, a prefill
# budget of 8192, graphs to the running cap, FlashInfer's 512 MiB scratch here (D-080's reason).
W="--model 8B --workload unique --prefix-len 480 --suffix-len 32 --output-len 128 --kv-pool-tokens $POOL --rounds 2
   --attn-workspace-mb 512 --warmup-max-s 300 --results-dir /out"
for n in $SIZES; do
  reqs=$(( n * 4 > 8 ? n * 4 : 8 ))
  wl=/tmp/m5_6_w$n.json
  step "batch $n: requests" "${MS[@]}" python -m bench.sglang_spec dump --out "$wl" --max-running "$n" --requests "$reqs"
  S="--max-running $n --cuda-graph-max-bs $n --groups $reqs --per-group 1"
  step "batch $n: this engine, plain" "${MS[@]}" python -m bench.offline $W $S --out m5_6_ours
  for arm in $SG_ARMS; do
    text=""; [ "$n" = 8 ] && text="--text-acceptance"
    step "batch $n: sglang $arm" "${SG[@]}" python -m bench.sglang_spec run --workload "$wl" --arm "$arm" \
      --max-running "$n" --kv-pool-tokens "$POOL" --out m5_6_sglang --results-dir /out $text
    gpu_free
  done
  step "batch $n: this engine, two-model gamma 4" "${MS[@]}" python -m bench.offline $W $S \
    --spec-draft 0.6B --spec-gamma 4 --disable-radix --disable-overlap --ablate spec --arms spec_g4 --out m5_6_ours_spec
done

el "wrote to $OUT"; ls -l "$OUT" | grep m5_6 || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
