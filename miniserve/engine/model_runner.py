"""Turns a scheduled batch into tensors and runs the model.

Two KV storage / attention modes:

- ``paged`` (default): a preallocated KV pool of fixed-size blocks, one
  ``BlockTable`` per request, FlashInfer batch attention, and optionally a
  radix prefix cache over the blocks (``KVCacheManager``). A prefill then
  runs only the tokens after the cached prefix.
- ``contiguous``: one contiguous cache per request sized ``prompt +
  max_new_tokens``, PyTorch SDPA per request. This is the reference path.
"""

from __future__ import annotations

import gc

import torch

from miniserve.cache.block_allocator import BlockAllocator
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.cache.kv_pool import KVPool
from miniserve.engine.request import Request
from miniserve.engine.sampler import Sampler, SamplingArgs
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.attention import ContiguousAttention, FlashInferPagedAttention
from miniserve.model.qwen3 import Qwen3ForCausalLM

ATTENTION_MODES = ("paged", "contiguous")


class ModelRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        attention: str = "paged",
        kv_pool_tokens: int | None = None,
        block_size: int = 16,
        max_prefill_tokens: int = 8192,
        kv_mem_fraction: float = 0.9,
        max_running: int = 64,
        radix: bool = True,
    ):
        """In paged mode the KV pool is sized from the GPU memory left after the
        weights and the peak memory of a ``max_prefill_tokens`` prefill followed
        by sampling ``max_running`` rows (``kv_mem_fraction`` of it);
        ``kv_pool_tokens`` sets an exact size and must fit in that memory.
        ``radix``: keep a prefix cache over the pool (paged mode only)."""
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.model = model
        self.device = model.device
        self.attention = attention
        self.sampler = Sampler(model.device, model.cfg.vocab_size)
        self.allocator: BlockAllocator | None = None
        self.kv: KVCacheManager | None = None
        if attention == "paged":
            cfg = model.cfg

            def pool(num_blocks: int) -> KVPool:
                return KVPool(
                    cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim, model.dtype, model.device
                )

            self.kv_profile = self._profile(pool, block_size, max_prefill_tokens, max_running, kv_mem_fraction)
            num_blocks = self.kv_profile["max_blocks"]
            if kv_pool_tokens is not None:
                if kv_pool_tokens < block_size:
                    raise ValueError(f"kv_pool_tokens={kv_pool_tokens} is smaller than one block ({block_size})")
                if kv_pool_tokens // block_size > num_blocks:
                    raise ValueError(
                        f"kv_pool_tokens={kv_pool_tokens} does not fit: GPU memory allows "
                        f"{num_blocks * block_size} tokens"
                    )
                num_blocks = kv_pool_tokens // block_size
            self.kv_profile["num_blocks"] = num_blocks
            self.allocator = BlockAllocator(num_blocks, block_size)
            self.kv = KVCacheManager(self.allocator, radix)
            self.pool = pool(num_blocks)
            self.flashinfer.pool = self.pool

    def _profile(self, make_pool, block_size: int, num_tokens: int, num_rows: int, fraction: float) -> dict[str, int]:
        """Peak activation memory of a ``num_tokens`` prefill followed by sampling ``num_rows``
        rows, and the KV blocks that fit next to it.

        Also creates the attention backend (its workspace stays allocated). The
        pass writes into a probe pool just large enough for it, freed before the
        free memory is read.
        """
        cfg = self.model.cfg
        probe = make_pool(-(-num_tokens // block_size))
        self.flashinfer = FlashInferPagedAttention(probe, cfg.num_heads, self.model.attn_scale)
        alloc = BlockAllocator(probe.num_blocks, block_size)
        table = BlockTable(alloc)
        table.append_tokens(num_tokens)
        torch.cuda.synchronize(self.device)
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.flashinfer.plan(True, [num_tokens], [table], [table.slot(p) for p in range(num_tokens)])
        self.model.forward_with(
            torch.zeros(num_tokens, dtype=torch.long, device=self.device),
            torch.arange(num_tokens, device=self.device),
            self.flashinfer,
            [num_tokens],
        )
        # The largest sampling step: every row sampled with a nucleus (the most temporaries).
        logits = torch.zeros(num_rows, cfg.vocab_size, dtype=self.model.dtype, device=self.device)
        ones = torch.ones(num_rows, device=self.device)
        rows = torch.arange(num_rows, device=self.device)
        self.sampler.sample(logits, SamplingArgs(ones, ones * 0.9, rows, rows, any_top_p=True))
        del logits
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) - base
        self.flashinfer.pool = None
        del probe
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        block_bytes = 2 * cfg.num_layers * block_size * cfg.num_kv_heads * cfg.head_dim * self.model.dtype.itemsize
        max_blocks = int(fraction * (free - peak)) // block_bytes
        if max_blocks < 1:
            raise RuntimeError(f"no GPU memory left for the KV pool ({free} B free, {peak} B peak activations)")
        return dict(
            profile_tokens=num_tokens,
            peak_activation_bytes=peak,
            free_bytes=free,
            total_bytes=total,
            block_bytes=block_bytes,
            max_blocks=max_blocks,
        )

    def allocate(self, req: Request) -> None:
        """Contiguous mode only: in paged mode the scheduler acquires KV at admission."""
        if req.cache is not None:
            raise RuntimeError(f"request {req.rid} already holds a cache")
        req.cache = self.model.new_cache(req.max_len)

    def release(self, req: Request) -> None:
        if self.kv is not None:
            self.kv.release(req)
        req.cache = None

    def sample(self, batch: Batch, logits: torch.Tensor) -> torch.Tensor:
        """Next token of each request, ``[len(batch.requests)]`` int64, left on the device."""
        return self.sampler.sample(logits, self.sampler.prepare(batch.requests))

    def forward(self, batch: Batch) -> torch.Tensor:
        """Logits ``[len(batch.requests), vocab]`` for the last token of each request."""
        prefill = batch.phase is Phase.PREFILL
        ids: list[int] = []
        pos: list[int] = []
        for r in batch.requests:
            if prefill:  # prompt + output (a preempted request resumes), after the cached prefix
                ids += (r.prompt_ids + r.output_ids)[r.num_cached_tokens :]
                pos += range(r.num_cached_tokens, r.seq_len)
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
        """Extend every table by its new tokens. If the pool is short even after evicting
        from the prefix cache, raise ``OutOfBlocks`` with the tables unchanged."""
        need = sum(t.blocks_needed(n) for t, n in zip(tables, seq_lens))
        self.kv.reserve(need)  # evicts from the prefix cache if the free pool is short
        for t, n in zip(tables, seq_lens):
            t.append_tokens(n)
