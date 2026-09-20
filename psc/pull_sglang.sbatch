#!/bin/bash
# Fetch the reference engine's own image and convert it to an Apptainer image.
#
# No GPU: this only downloads and converts, so it runs on a CPU partition where
# the queue is short. The registry is reachable from compute nodes, and pulling
# needs no root, so the 16 GiB never crosses the slow path from a laptop.
#
# The image is the project's own published build, which is the point: a
# comparison should run the other engine the way its authors ship it, not the
# way we would have built it.
#
# Memory, not cores, is what this job needs. The last step compresses a ~30 GB
# root filesystem with mksquashfs, which sizes its own memory from the machine's
# physical RAM rather than from the job's cgroup limit: on a 256 GB node it asks
# for about 64 GB however few cores were requested, and a small allocation is
# killed (exit 137) after the download has already succeeded. These nodes give
# 2 GB per core, so the cores below are really a memory request.
#
#   sbatch -p RM-shared -q low -o $PROJECT/logs/%x_%j.log --export=ALL psc/pull_sglang.sbatch
#SBATCH -N 1 -n 40 -t 03:00:00 -J sglang-pull
set -euo pipefail
: "${PROJECT:?}"
TAG=${TAG:-v0.5.10}
SIF=$PROJECT/sif/sglang-$TAG.sif
# Both caches are large and must stay off $HOME, which is small and shared.
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer-cache
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer-tmp
# Both are scratch for this job alone, and a failed run leaves tens of gigabytes
# of half-extracted root filesystem behind, so they are cleared at both ends.
rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$PROJECT/sif"
trap 'rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"' EXIT
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }
el "cores $SLURM_NTASKS, memory $(( SLURM_NTASKS * 2 )) GB (mksquashfs wants about a quarter of the node's 256 GB)"

el "node $(hostname), pulling docker://lmsysorg/sglang:$TAG"
apptainer pull --force "$SIF" "docker://lmsysorg/sglang:$TAG"
el "built $SIF"
ls -lh "$SIF"
sha256sum "$SIF" | tee "$SIF.sha256"
el "what is inside"
apptainer exec "$SIF" python -c "
import sglang, torch, importlib.metadata as md
print('sglang', sglang.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)
for p in ('flashinfer-python', 'sgl-kernel', 'transformers', 'triton'):
    try: print(p, md.version(p))
    except Exception as e: print(p, 'not found', e)
" || el "inspection failed (recorded, not fatal)"
el "done"
