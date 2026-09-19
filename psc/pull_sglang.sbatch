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
#   sbatch -p RM-shared -q low -o $PROJECT/logs/%x_%j.log --export=ALL psc/pull_sglang.sbatch
#SBATCH -N 1 -n 4 -t 02:00:00 -J sglang-pull
set -euo pipefail
: "${PROJECT:?}"
TAG=${TAG:-v0.5.10}
SIF=$PROJECT/sif/sglang-$TAG.sif
# Both caches are large and must stay off $HOME, which is small and shared.
export APPTAINER_CACHEDIR=$PROJECT/tmp/apptainer-cache
export APPTAINER_TMPDIR=$PROJECT/tmp/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$PROJECT/sif"
T0=$(date +%s); el() { echo "[t+$(( $(date +%s) - T0 ))s] $*"; }

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
rm -rf "$APPTAINER_CACHEDIR"
el "done"
