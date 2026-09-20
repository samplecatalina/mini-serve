#!/bin/bash
# The schedule-policy ablation against KV pool size, on the engine directly.
#
# It cannot be done through the main caliber. A cache-aware policy prefers the
# request whose prefix is already in the cache, so a workload with no shared
# prefixes gives it nothing to prefer -- and genai-bench samples its prompts from
# text, with no way to ask for a shared prefix. Run through it, the three
# policies came out within 3% of each other at every pool size, which measures
# the workload rather than the policies. The offline benchmark builds the
# workload it needs: groups of requests sharing a prefix.
#
# Two workloads on purpose:
#   small  the same one the 8 GB card was measured with, so the two devices can
#          be read row by row, across the pool sizes where that card had pressure;
#   large  four times the requests, so that the pool sizes that matter on a 45 GB
#          card are under pressure at all -- 128 requests occupy about 70k tokens
#          and would leave a 100k pool half empty, which would show convergence
#          for the wrong reason.
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s psc/m4_2_policy.sbatch
#SBATCH -N 1 -t 01:30:00 -J miniserve-policy
set -euo pipefail
: "${PROJECT:?} ${DEVICE:?l40s or h100}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) ARCH=8.9;; h100) ARCH=9.0;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), image $(sha256sum "$SIF" | cut -c1-16)"

RUN=(apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH
     --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace --bind "$REPO:/workspace"
     --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" --pwd /workspace "$SIF")

FAILED=()
# sweep <out name> <extra workload flags...> -- <pool sizes, 0 meaning the whole card>
sweep() {
  local out=$1; shift
  local wl=(); while [ "$1" != "--" ]; do wl+=("$1"); shift; done; shift
  for pool in "$@"; do
    local poolflag=()
    [ "$pool" = 0 ] || poolflag=(--kv-pool-tokens "$pool")
    el "  $out, pool ${pool/0/full}"
    if ! "${RUN[@]}" python -m bench.offline --workload policy --ablate policy --rounds 3 \
        "${wl[@]}" "${poolflag[@]}" --results-dir /out --out "$out"
    then
      el "    FAILED; continuing"
      FAILED+=("$out/$pool")
    fi
  done
}

# The 8 GB card's own workload and its pool sizes, plus the first of this card's:
# 16 groups of 8 requests sharing a 512-token prefix (bench/offline.py defaults).
el "=== same workload as the 8 GB card ==="
sweep m4_2_policy_small -- 6144 8192 16384 40960

# Four times the requests, so 40k and 100k are actually pressured on this card.
el "=== four times the requests ==="
sweep m4_2_policy_large --groups 64 --per-group 8 -- 40960 102400 0

el "wrote to $OUT"; ls -l "$OUT" | grep policy || true
if [ ${#FAILED[@]} -gt 0 ]; then el "sweeps that produced nothing: ${FAILED[*]}"; else el "every sweep produced results"; fi
el "done"
