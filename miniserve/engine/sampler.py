"""Token selection from logits."""

from __future__ import annotations

import torch


def greedy(logits: torch.Tensor) -> list[int]:
    """``logits [B, vocab]`` -> one token per row.

    ``argmax`` returns the first maximal index, the same tie-breaking as HF.
    ``tolist()`` is a host synchronization point, once per step.
    """
    return torch.argmax(logits, dim=-1).tolist()
