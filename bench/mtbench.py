"""MT-Bench first-turn questions, the natural-text prompt set for speculative decoding.

Random token prompts understate speculation on a large target: after a random prefix the
target does not degrade and the small draft cannot guess it (acceptance at gamma 4, Qwen3-8B
with a 0.6B draft: 0.431 on random prompts, 0.561 on natural text). MT-Bench is what
speculative decoding work usually reports on: 80 questions, ten in each of eight categories
(writing, roleplay, reasoning, math, coding, extraction, STEM, humanities).

Pinned: HuggingFaceH4/mt_bench_prompts (Apache-2.0) at the revision below, file
raw/question.jsonl. Only the first turn of each question is used.

    python -m bench.mtbench        # download into $HF_HOME (once, where there is a network)
"""

from __future__ import annotations

import json
import sys

REPO = "HuggingFaceH4/mt_bench_prompts"
REVISION = "e3a795c5e9a82ee40611c416b8a7786c73198991"
FILE = "raw/question.jsonl"


def questions(download: bool = False) -> list[str]:
    """The 80 first-turn questions, in the file's order."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION, local_files_only=not download)
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["prompt"][0] for r in rows]


def main() -> int:
    qs = questions(download=True)
    print(f"{REPO}@{REVISION[:12]}: {len(qs)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
