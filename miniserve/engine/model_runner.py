"""Turns a scheduled batch into tensors and runs the model.

KV storage is one contiguous cache per request, sized ``prompt + max_new_tokens``
at admission and dropped on release. Paged storage replaces this later.
"""

from __future__ import annotations

import torch

from miniserve.engine.request import Request
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.qwen3 import Qwen3ForCausalLM


class ModelRunner:
    def __init__(self, model: Qwen3ForCausalLM):
        self.model = model
        self.device = model.device

    def allocate(self, req: Request) -> None:
        if req.cache is not None:
            raise RuntimeError(f"request {req.rid} already holds a cache")
        req.cache = self.model.new_cache(req.max_len)

    def release(self, req: Request) -> None:
        req.cache = None

    def forward(self, batch: Batch) -> torch.Tensor:
        """Logits ``[len(batch.requests), vocab]`` for the last token of each request."""
        ids: list[int] = []
        pos: list[int] = []
        for r in batch.requests:
            if batch.phase is Phase.PREFILL:
                ids += r.prompt_ids
                pos += range(len(r.prompt_ids))
            else:
                ids.append(r.output_ids[-1])
                pos.append(r.cache.length)
        return self.model.forward_batch(
            torch.tensor(ids, device=self.device, dtype=torch.long),
            torch.tensor(pos, device=self.device, dtype=torch.long),
            [r.cache for r in batch.requests],
            batch.seq_lens,
        )
