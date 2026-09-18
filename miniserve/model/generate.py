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
) -> list[int]:
    """Prefill ``prompt_ids`` once, then decode one token per step.

    Returns the generated tokens (a stop token, if hit, is included).
    """
    cache = model.new_cache(len(prompt_ids) + max_new_tokens)
    ids = torch.tensor(prompt_ids, device=model.device, dtype=torch.long)
    positions = torch.arange(len(prompt_ids), device=model.device)
    out: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(ids, positions, cache)
        # argmax returns the first maximal index, same tie-breaking as HF.
        nxt = int(torch.argmax(logits))
        out.append(nxt)
        if nxt in stop_ids:
            break
        ids = torch.tensor([nxt], device=model.device, dtype=torch.long)
        positions = torch.tensor([cache.length], device=model.device)
    return out
