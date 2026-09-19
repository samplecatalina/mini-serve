"""Turns a scheduled batch into tensors and runs the model.

Two KV storage / attention modes:

- ``paged`` (default): a preallocated KV pool of fixed-size blocks, one
  ``BlockTable`` per request, FlashInfer batch attention.
- ``contiguous``: one contiguous cache per request sized ``prompt +
  max_new_tokens``, PyTorch SDPA per request. This is the reference path.
"""

from __future__ import annotations

import torch

from miniserve.cache.block_allocator import BlockAllocator, OutOfBlocks
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_pool import KVPool
from miniserve.engine.request import Request
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.attention import ContiguousAttention, FlashInferPagedAttention
from miniserve.model.qwen3 import Qwen3ForCausalLM

ATTENTION_MODES = ("paged", "contiguous")


class ModelRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        attention: str = "paged",
        num_kv_blocks: int = 1024,
        block_size: int = 16,
    ):
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.model = model
        self.device = model.device
        self.attention = attention
        if attention == "paged":
            cfg = model.cfg
            self.allocator = BlockAllocator(num_kv_blocks, block_size)
            self.pool = KVPool(
                cfg.num_layers, num_kv_blocks, block_size, cfg.num_kv_heads, cfg.head_dim, model.dtype, model.device
            )
            self.flashinfer = FlashInferPagedAttention(self.pool, cfg.num_heads, model.attn_scale)

    def allocate(self, req: Request) -> None:
        if req.cache is not None:
            raise RuntimeError(f"request {req.rid} already holds a cache")
        req.cache = BlockTable(self.allocator) if self.attention == "paged" else self.model.new_cache(req.max_len)

    def release(self, req: Request) -> None:
        if isinstance(req.cache, BlockTable):
            req.cache.release()
        req.cache = None

    def forward(self, batch: Batch) -> torch.Tensor:
        """Logits ``[len(batch.requests), vocab]`` for the last token of each request."""
        prefill = batch.phase is Phase.PREFILL
        ids: list[int] = []
        pos: list[int] = []
        for r in batch.requests:
            if prefill:
                ids += r.prompt_ids
                pos += range(len(r.prompt_ids))
            else:
                ids.append(r.output_ids[-1])
                pos.append(len(r.prompt_ids) + len(r.output_ids) - 1)
        seq_lens = batch.seq_lens
        caches = [r.cache for r in batch.requests]
        if self.attention == "paged":
            self._reserve(caches, seq_lens)
            slots = [t.slot(p) for t, n in zip(caches, seq_lens) for p in range(t.num_tokens - n, t.num_tokens)]
            self.flashinfer.plan(prefill, seq_lens, caches, slots)
            attn = self.flashinfer
        else:
            attn = ContiguousAttention(caches, seq_lens, self.model.attn_scale)
        return self.model.forward_with(
            torch.tensor(ids, device=self.device, dtype=torch.long),
            torch.tensor(pos, device=self.device, dtype=torch.long),
            attn,
            seq_lens,
        )

    def _reserve(self, tables: list[BlockTable], seq_lens: list[int]) -> None:
        """Extend every table by its new tokens, or change nothing if the pool is short."""
        need = sum(t.blocks_needed(n) for t, n in zip(tables, seq_lens))
        if not self.allocator.can_allocate(need):
            raise OutOfBlocks(f"batch needs {need} KV blocks, {self.allocator.num_free} free")
        for t, n in zip(tables, seq_lens):
            t.append_tokens(n)
