"""Download pinned model snapshots into $HF_HOME.

Usage: ``python -m miniserve.tools.fetch_weights [SIZE ...]`` (default: 0.6B).
"""

import sys

from miniserve.model.weights import MODELS, model_path, spec_for


def main(argv: list[str] | None = None) -> int:
    names = argv if argv else ["0.6B"]
    if any(n in ("-h", "--help") for n in names):
        print(f"usage: python -m miniserve.tools.fetch_weights [SIZE ...]; sizes: {' '.join(sorted(MODELS))}")
        return 0
    for name in names:
        spec = spec_for(name)
        path = model_path(spec, download=True)
        print(f"{spec.repo_id}@{spec.revision} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
