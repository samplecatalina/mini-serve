"""Download the pinned model snapshot into $HF_HOME.

Usage: ``python -m miniserve.tools.fetch_weights``
"""

import sys

from miniserve.model.weights import QWEN3_0_6B, model_path


def main() -> int:
    path = model_path(QWEN3_0_6B, download=True)
    print(f"{QWEN3_0_6B.repo_id}@{QWEN3_0_6B.revision} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
