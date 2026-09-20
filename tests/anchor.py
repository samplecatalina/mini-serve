"""The tolerance rules a non-reference execution path is held to.

The reference path (one sequence, contiguous attention) reproduces Hugging
Face transformers bitwise. Any other path -- batched, paged, graph-replayed,
speculative -- runs different kernels on different shapes, so BF16 logits can
move by an ulp and a sequence can take the other branch of a near-tie. Two
rules keep that from hiding real bugs:

- a path may diverge from the reference only at a position where the
  reference's own top-1/top-2 gap is at most ``EPS``; token comparison stops
  there, because everything after it continues a different sequence;
- since that says nothing about the tokens after a divergence, every token a
  path produced is also checked teacher-forced: running the reference path on
  the path's own output, the chosen token's logit must be within ``EPS`` of
  the reference maximum at that position.
"""

from __future__ import annotations

import torch

# Largest reference logit gap tolerated for a token off the reference path.
# Measured on 64 sequences x 128 tokens: the largest gap at an actual
# divergence was 0.25 (most were exact ties, gap 0.0), and teacher-forced
# margins never exceeded 0.25 either; EPS keeps a 2x margin.
EPS = 0.5
STOP_IDS = frozenset({151645, 151643})


def check_against_reference(name, ours, ref, gaps) -> int | None:
    """Return the divergence position (None if identical); fail if it is not at a near-tie."""
    if ours == ref:
        return None
    pos = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), None)
    assert pos is not None, f"[{name}] same prefix but different lengths: ours={len(ours)} ref={len(ref)}"
    assert gaps[pos] <= EPS, (
        f"[{name}] diverges at position {pos} where the reference top-2 gap is {gaps[pos]:.4f} > EPS={EPS}: "
        f"ours={ours[pos]} ref={ref[pos]}"
    )
    return pos


@torch.inference_mode()
def forced_margins(model, prompt: list[int], tokens: list[int]) -> list[float]:
    """Reference path teacher-forced on ``tokens``: top-1 logit minus the chosen token's logit, per step."""
    cache = model.new_cache(len(prompt) + len(tokens))
    ids = torch.tensor(prompt, device=model.device)
    pos = torch.arange(len(prompt), device=model.device)
    out = []
    for t in tokens:
        logits = model.forward(ids, pos, cache).float()
        out.append(float(logits.max() - logits[t]))
        ids = torch.tensor([t], device=model.device)
        pos = torch.tensor([cache.length], device=model.device)
    return out


def check_margins(name, model, prompt, tokens) -> float:
    m = forced_margins(model, prompt, tokens)
    worst = max(range(len(m)), key=m.__getitem__)
    assert m[worst] <= EPS, (
        f"[{name}] token {tokens[worst]} at position {worst} is {m[worst]:.4f} below the reference maximum (EPS={EPS})"
    )
    return m[worst]
