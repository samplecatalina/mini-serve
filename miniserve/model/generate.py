"""Single-sequence greedy generation on top of the reference model."""

from __future__ import annotations

from collections.abc import Collection

import torch

from miniserve.model.qwen3 import Qwen3ForCausalLM


@torch.inference_mode()
def greedy_generate(
    model: Qwen3ForCausalLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    stop_ids: Collection[int] = (),
    top2_gaps: list[float] | None = None,
) -> list[int]:
    """Prefill ``prompt_ids`` once, then decode one token per step.

    Returns the generated tokens (a stop token, if hit, is included). If
    ``top2_gaps`` is given, the top-1 minus top-2 logit gap of every step is
    appended to it; comparisons against other execution paths use it to tell
    near-ties from real mismatches.
    """
    cache = model.new_cache(len(prompt_ids) + max_new_tokens)
    ids = torch.tensor(prompt_ids, device=model.device, dtype=torch.long)
    positions = torch.arange(len(prompt_ids), device=model.device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(ids, positions, cache)
        if top2_gaps is not None:
            top = torch.topk(logits.float(), 2).values
            top2_gaps.append(float(top[0] - top[1]))
        # argmax returns the first maximal index, same tie-breaking as HF.
        nxt = int(torch.argmax(logits))
        out.append(nxt)
        if nxt in stop_ids:
            break
        ids = torch.tensor([nxt], device=model.device, dtype=torch.long)
        positions = torch.tensor([cache.length], device=model.device)
    return out
