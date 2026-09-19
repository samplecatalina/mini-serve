"""Compare greedy outputs across devices (files written by ``miniserve.tools.golden``).

For each prompt and each output (reference path, engine alone, engine with all prompts), report
where the files first disagree. At a disagreement, the reference top-1 minus top-2 logit gap of
the first file says whether it is a near-tie (<= eps: different but valid greedy choices at
BF16 precision) or a real mismatch. The same test is applied within each file: the engine
against that device's own reference.
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

    def check(label, toks_a, toks_b, gaps):
        nonlocal bad
        i = first_divergence(toks_a, toks_b)
        if i is None:
            return "same"
        gap = gaps[i] if i < len(gaps) else float("nan")
        ok = gap <= args.eps
        bad += not ok
        return f"diverge at {i} (reference gap {gap:.4f}{'' if ok else ', MISMATCH'})"

    for prompt in docs[0]["prompts"]:
        base = docs[0]["prompts"][prompt]
        print(f"{prompt}:")
        for d in docs:
            p = d["prompts"][prompt]
            for kind in ("engine_alone", "engine_together"):
                print(f"  {d['device']}: {kind} vs own reference: {check(kind, p['reference'], p[kind], p['reference_gaps'])}")
        for d in docs[1:]:
            p = d["prompts"][prompt]
            for kind in ("reference", "engine_alone", "engine_together"):
                print(f"  {names[0]} vs {d['device']}: {kind}: {check(kind, base[kind], p[kind], base['reference_gaps'])}")
    print("OK: every disagreement is at a near-tie" if not bad else f"{bad} disagreement(s) beyond eps={args.eps}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
