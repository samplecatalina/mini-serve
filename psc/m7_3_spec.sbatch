#!/bin/bash
# Fused operators on and off on the L40S, part 2: speculative decoding, Qwen3-8B verified with a
# 0.6B draft, at running caps 1 to 64, on random prompts (the earlier sweep's caliber) and on
# MT-Bench. Per cap and workload:
#   this engine    plain, reference against fused operators in one process (A B B A)
#                  two-model gamma 2 / 4 / 6 on the reference operators (one process)
#                  two-model gamma 2 / 4 / 6 on the fused operators (one process)
#   sglang 0.5.10  plain and two-model (STANDALONE), on the same token ids, as the control
# The fused path's anchor on this GPU runs first; nothing else runs if it fails.
#
# MT-Bench must already be in the shared cache (it is, since the MT-Bench job).
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m7_3_spec.sbatch
#SBATCH -N 1 -t 08:00:00 -J miniserve-m7-c2
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
SGLANG_SIF=${SGLANG_SIF:-$PROJECT/sif/sglang-v0.5.10.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
SIZES=${SIZES:-"1 2 4 8 16 32 64"}
POOL=${POOL:-65536}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in l40s) GPU="L40S"; ARCH=8.9;; *) echo "unknown DEVICE $DEVICE" >&2; exit 2;; esac
mkdir -p "$OUT" "$TMPDIR" "$TMPDIR/sglang-cache"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD)"
el "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
el "images: engine $(sha256sum "$SIF" | cut -c1-16), sglang $(sha256sum "$SGLANG_SIF" | cut -c1-16)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

MS=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --env MINISERVE_SPEC_ANCHOR_TARGET=8B
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")
SG=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/sglang-cache
    --env HF_HOME=/cache/huggingface --env HF_HUB_OFFLINE=1 --env PYTHONPATH=/workspace
    --env FLASHINFER_WORKSPACE_BASE=/tmp/sglang-cache --env TRITON_CACHE_DIR=/tmp/sglang-cache/triton
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache/inductor --env OUTLINES_CACHE_DIR=/tmp/sglang-cache/outlines
    --bind "$REPO:/workspace" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SGLANG_SIF")

el "weights"; "${MS[@]}" python -m miniserve.tools.fetch_weights 0.6B 1.7B 8B >/dev/null || exit 1
el "mt-bench in the cache"; "${MS[@]}" python -c "from bench import mtbench; print(len(mtbench.questions()))" || exit 1
FAILED=()
step() {
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
sg_run() {  # sg_run <workload file> <arm> <cap> <out>
  "${SG[@]}" python -m bench.sglang_spec run --workload "$1" --arm "$2" --max-running "$3" \
    --kv-pool-tokens "$POOL" --out "$4" --results-dir /out
}
# sglang's two-model path fails now and then when it merges batches; once more before it counts.
sg_try() { sg_run "$@" || { gpu_free; el "  retrying once"; sg_run "$@"; }; }

# ---------------------------------------------------------------- the anchor
el "anchor: fused path (tests/test_fused.py; speculative case on 8B / 0.6B)"
ALOG=$PROJECT/logs/m7_3_anchor_${SLURM_JOB_ID:-local}_c2.log
"${MS[@]}" python -m pytest -q -s -p no:cacheprovider tests/test_fused.py > "$ALOG" 2>&1; rc=$?
grep -E "^\[|passed|failed|skipped|^E " "$ALOG"
if [ $rc -ne 0 ] || grep -q " skipped" "$ALOG"; then el "anchor FAILED or incomplete (exit $rc): nothing else runs"; exit 1; fi

# ---------------------------------------------------------------- the sweeps
COMMON="--model 8B --kv-pool-tokens $POOL --rounds 2 --attn-workspace-mb 512 --warmup-max-s 300 --results-dir /out"
SPEC="--spec-draft 0.6B --disable-radix --disable-overlap --ablate spec --arms spec_g2 spec_g4 spec_g6"
for wlname in random mtbench; do
  if [ $wlname = random ]; then
    W="$COMMON --workload unique --prefix-len 480 --suffix-len 32 --output-len 128"; DUMP=""
  else
    W="$COMMON --workload mtbench --output-len 256"; DUMP="--workload mtbench --output-len 256"
  fi
  for n in $SIZES; do
    reqs=$(( n * 4 > 8 ? n * 4 : 8 ))
    wl=/tmp/m7_3_${wlname}_w$n.json
    step "$wlname cap $n: requests" "${MS[@]}" python -m bench.sglang_spec dump --out "$wl" --max-running "$n" \
      --requests "$reqs" $DUMP
    S="--max-running $n --cuda-graph-max-bs $n --groups $reqs --per-group 1"
    step "$wlname cap $n: this engine, plain, operators" "${MS[@]}" python -m bench.offline $W $S \
      --ablate fused_ops --out m7_3_spec_${wlname}_plain
    step "$wlname cap $n: this engine, two-model, reference operators" "${MS[@]}" python -m bench.offline $W $S $SPEC \
      --out m7_3_spec_${wlname}_ref
    step "$wlname cap $n: this engine, two-model, fused operators" "${MS[@]}" python -m bench.offline $W $S $SPEC \
      --fused-ops --out m7_3_spec_${wlname}_fused
    for arm in plain standalone; do
      step "$wlname cap $n: sglang $arm" sg_try "$wl" "$arm" "$n" m7_3_spec_${wlname}_sglang
      gpu_free
    done
  done
done

el "wrote to $OUT"; ls -l "$OUT" | grep m7_3_spec || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
