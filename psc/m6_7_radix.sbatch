#!/bin/bash
# The two prefix-tree backends end to end, where they could differ: a node with slow CPU
# cores, cache-aware admission over a long queue, and a pool small enough that the tree
# evicts all the time.
#
# Three arms, one process each, in the order A B C C B A:
#   A  python backend, this commit
#   B  C++ backend at PRE (the commit before cache-aware ordering moved into C++), run
#      from a worktree of it with its own build of the extension
#   C  C++ backend, this commit
# The admission order is the same function of the tree in all three, so the schedules
# must be identical: every arm has to report the same decode steps, hit rate and prefill
# tokens, or the ordering is wrong somewhere. Then the per-operation host timings of the
# prefix cache for A and C (and B), on this workload and on the M5.7 one.
#
#   sbatch -p GPU-shared --gres=gpu:l40s-48:1 -o $PROJECT/logs/%x_%j.log \
#          --export=ALL,DEVICE=l40s,PRE=<commit> psc/m6_7_radix.sbatch
#SBATCH -N 1 -t 02:00:00 -J miniserve-m6-7
set -uo pipefail
: "${PROJECT:?} ${DEVICE:?l40s} ${PRE:?the commit before cache-aware ordering moved into C++}"
REPO=${REPO:-$PROJECT/miniserve}
SIF=${SIF:-$PROJECT/sif/miniserve-m5.sif}
CACHE=${CACHE:-$PROJECT/miniserve_cache}
OUT=${OUT:-$PROJECT/results/$DEVICE}
PRIVATE=${PRIVATE:-$PROJECT/miniserve_private}   # host-path scripts that are not part of the repository
export PYTHONUNBUFFERED=1 TMPDIR=$PROJECT/tmp
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
case $DEVICE in
  l40s) GPU="L40S"; ARCH=8.9;;
  *) echo "unknown DEVICE $DEVICE" >&2; exit 2;;
esac
mkdir -p "$OUT" "$TMPDIR"
cd "$REPO"
el "node $(hostname), commit $(git rev-parse --short HEAD), pre $PRE, image $(sha256sum "$SIF" | cut -c1-16)"
el "cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
el "preflight"; bash psc/preflight.sh "$SIF" "$CACHE" "$GPU" || exit 1

PRE_TREE=$TMPDIR/m6_7_pre_$SLURM_JOB_ID
git worktree add --detach --quiet "$PRE_TREE" "$PRE" || exit 1
trap 'git -C "$REPO" worktree remove --force "$PRE_TREE"' EXIT

# The repository is also bound at its own path: a worktree's .git is a file naming
# $REPO/.git/worktrees/<name>, which git inside the container must be able to follow for
# the harness to record which commit ran.
run_in() {  # run_in <tree> <command...>: the container with <tree> as /workspace
  local tree=$1; shift
  apptainer exec --nv --cleanenv --no-home --env HOME=/tmp --env FLASHINFER_CUDA_ARCH_LIST=$ARCH \
    --env TORCH_CUDA_ARCH_LIST=$ARCH --env PYTHONPATH=/workspace \
    --bind "$tree:/workspace" --bind "$REPO:$REPO" --bind "$CACHE:/cache" --bind "$TMPDIR:/tmp" --bind "$OUT:/out" \
    --bind "$PRIVATE:/private" --pwd /workspace "$SIF" "$@"
}

el "weights"; run_in "$REPO" python -m miniserve.tools.fetch_weights 0.6B >/dev/null || exit 1
FAILED=()
step() {
  local label=$1; shift
  el "$label"
  "$@" || { el "  $label FAILED; continuing"; FAILED+=("$label"); }
}

# The C++ extension for this node, for each tree: the image's baseline (-march=x86-64-v2),
# never native.
build() {
  run_in "$1" sh -c "cmake -S minicore -B /tmp/minicore-build-$2 -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=\$(command -v python) >/dev/null && cmake --build /tmp/minicore-build-$2 --parallel >/dev/null \
      && python -c 'import miniserve.cache._minicore as m; print(m.BACKEND)'"
}
build "$REPO" head || exit 1
build "$PRE_TREE" pre || exit 1

# 1024 groups of 4 requests: 512 shared tokens, 32 of their own, 32 out -> 786k unique
# tokens through a 335k-token pool, all 4096 requests waiting at once, cache-aware order.
W="--workload shared --groups 1024 --per-group 4 --prefix-len 512 --suffix-len 32 --output-len 32
   --schedule-policy cache --max-running 256 --cuda-graph-max-bs 256 --kv-pool-tokens 335000
   --rounds 2 --warmup-max-s 300 --results-dir /out --out m6_7_radix"
arm() {  # arm A|B|C
  case $1 in
    A) step "arm A: python, this commit" run_in "$REPO" python -m bench.offline $W --block-backend python;;
    B) step "arm B: cpp at $PRE" run_in "$PRE_TREE" python -m bench.offline $W --block-backend cpp;;
    C) step "arm C: cpp, this commit" run_in "$REPO" python -m bench.offline $W --block-backend cpp;;
  esac
}
for a in A B C C B A; do arm $a; done

# Where the host time goes, per operation: this workload for all three arms, and the M5.7
# workload (many_small_big) for the two trees of this commit, which the earlier figures
# (python 2.08, cpp 1.17 ms of ordering per step) were measured on.
step "host timings A" run_in "$REPO" python /private/radix_host.py engine --block-backend python \
  --workloads evict_big many_small_big --policies cache --out /out/m6_7_radix_host_python.json
step "host timings C" run_in "$REPO" python /private/radix_host.py engine --block-backend cpp \
  --workloads evict_big many_small_big --policies cache --out /out/m6_7_radix_host_cpp.json
step "host timings B" run_in "$PRE_TREE" python /private/radix_host.py engine --block-backend cpp \
  --workloads evict_big --policies cache --out /out/m6_7_radix_host_cpp_pre.json

el "wrote to $OUT"; ls -l "$OUT" | grep m6_7 || true
el "done; failed steps: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
