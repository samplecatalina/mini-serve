"""Compare greedy outputs across devices (files written by ``miniserve.tools.golden``).

The verdict is per device: the engine against that device's own reference path, which is the
correctness anchor. At the first token where they differ, the reference's top-1 minus top-2
logit gap says whether it is a near-tie (<= eps: two valid greedy choices at BF16 precision) or
a mismatch to investigate.

Across devices the comparison is descriptive only. The reference path itself is not bitwise
identical on different GPUs (different SM counts lead to different GEMM algorithms, and BF16
logits are coarse enough that exact ties are common), so once two chains differ they condition
on different prefixes and the gaps of either device no longer apply to the other's positions.
Each cross-device line therefore reports where the outputs first differ and where the two
reference paths first differ, without a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--eps", type=float, default=0.5)
    args = ap.parse_args()
    docs = [json.load(open(f)) for f in args.files]
    names = [d["device"] for d in docs]
    bad = 0

    def own(toks_a, toks_b, gaps):
        """The anchor: an engine output against its own device's reference."""
        nonlocal bad
        i = first_divergence(toks_a, toks_b)
        if i is None:
            return "same"
        gap = gaps[i] if i < len(gaps) else float("nan")
        ok = gap <= args.eps
        bad += not ok
        return f"diverge at {i} (reference gap {gap:.4f}{'' if ok else ', MISMATCH: beyond eps'})"

    def across(toks_a, toks_b, ref_split):
        i = first_divergence(toks_a, toks_b)
        if i is None:
            return "same"
        where = "the references are still identical there" if ref_split is None or i < ref_split \
            else f"the references themselves differ from token {ref_split}"
        return f"differ from token {i} ({where})"

    for prompt in docs[0]["prompts"]:
        base = docs[0]["prompts"][prompt]
        print(f"{prompt}:")
        for d in docs:
            p = d["prompts"][prompt]
            for kind in ("engine_alone", "engine_together"):
                print(f"  {d['device']}: {kind} vs own reference: {own(p['reference'], p[kind], p['reference_gaps'])}")
        for d in docs[1:]:
            p = d["prompts"][prompt]
            ref_split = first_divergence(base["reference"], p["reference"])
            for kind in ("reference", "engine_alone", "engine_together"):
                print(f"  {names[0]} vs {d['device']}: {kind}: {across(base[kind], p[kind], None if kind == 'reference' else ref_split)}")
    print(
        "OK: on every device the engine only differs from its own reference at near-ties"
        if not bad else f"{bad} engine-vs-reference difference(s) beyond eps={args.eps}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
