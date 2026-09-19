"""Turns a scheduled batch into tensors and runs the model.

Two KV storage / attention modes:

- ``paged`` (default): a preallocated KV pool of fixed-size blocks, one
  ``BlockTable`` per request, FlashInfer batch attention, and optionally a
  radix prefix cache over the blocks (``KVCacheManager``). A prefill then
  runs only the tokens after the cached prefix.
- ``contiguous``: one contiguous cache per request sized ``prompt +
  max_new_tokens``, PyTorch SDPA per request. This is the reference path.

In paged mode, decode steps replay captured CUDA Graphs (``cuda_graph.py``)
unless disabled or the batch exceeds the largest captured batch size.
"""

from __future__ import annotations

import gc

import torch

from miniserve.cache.block_allocator import BlockAllocator
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.cache.kv_pool import KVPool
from miniserve.engine.cuda_graph import DecodeGraphs, graph_buckets
from miniserve.engine.request import Request
from miniserve.engine.sampler import Sampler, SamplingArgs
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.attention import ContiguousAttention, FlashInferPagedAttention
from miniserve.model.qwen3 import Qwen3ForCausalLM

ATTENTION_MODES = ("paged", "contiguous")


def _tokens(req: Request, start: int, n: int) -> list[int]:
    """Tokens ``start .. start + n`` of ``prompt + output``, without concatenating the two."""
    p = len(req.prompt_ids)
    if start >= p:
        return req.output_ids[start - p : start - p + n]
    if start + n <= p:
        return req.prompt_ids[start : start + n]
    return req.prompt_ids[start:] + req.output_ids[: start + n - p]


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
        cuda_graph: bool = True,
        cuda_graph_max_bs: int | None = None,
    ):
        """In paged mode the KV pool is sized from the GPU memory left after the
        weights and the peak memory of a ``max_prefill_tokens`` prefill followed
        by sampling ``max_running`` rows (``kv_mem_fraction`` of it);
        ``kv_pool_tokens`` sets an exact size and must fit in that memory.
        ``radix``: keep a prefix cache over the pool (paged mode only).
        ``cuda_graph``: capture decode graphs for batch sizes up to ``cuda_graph_max_bs``
        (default ``max_running``) after the pool is allocated (paged mode only); they use the
        memory the pool leaves free. ``use_cuda_graph`` switches them off and on at run time."""
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.model = model
        self.device = model.device
        self.attention = attention
        self.sampler = Sampler(model.device, model.cfg.vocab_size)
        self.allocator: BlockAllocator | None = None
        self.kv: KVCacheManager | None = None
        self.graphs: DecodeGraphs | None = None
        self.use_cuda_graph = False
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
            # One block past the allocator's: padding rows of a decode graph write and read it.
            self.pool = pool(num_blocks + 1)
            self.flashinfer.pool = self.pool
            if cuda_graph:
                self.graphs = DecodeGraphs(
                    model,
                    self.pool,
                    dummy_block=num_blocks,
                    buckets=graph_buckets(cuda_graph_max_bs or max_running),
                    workspace=self.flashinfer.workspace,
                    num_heads=cfg.num_heads,
                    scale=model.attn_scale,
                )
                self.use_cuda_graph = True
                self.kv_profile.update(
                    graph_buckets=self.graphs.buckets,
                    graph_bytes=self.graphs.graph_bytes,
                    graph_capture_s=round(self.graphs.capture_s, 2),
                )

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
        # Any row with more than one new token needs the prefill kernel; decode rows of a mixed
        # batch are causal rows of length 1 to it.
        prefill = batch.phase is not Phase.DECODE
        ids: list[int] = []
        pos: list[int] = []
        for r, start, n in zip(batch.requests, batch.starts, batch.extend_lens):
            # prompt + output (a preempted request resumes), from the first token without KV
            ids += _tokens(r, start, n)
            pos += range(start, start + n)
        seq_lens = batch.seq_lens
        caches = [r.cache for r in batch.requests]
        if self.attention == "paged":
            if any(t.num_tokens != s for t, s in zip(caches, batch.starts)):
                raise RuntimeError("batch starts disagree with the block tables")
            self._reserve(caches, seq_lens)
            slots = [t.slot(p) for t, n in zip(caches, seq_lens) for p in range(t.num_tokens - n, t.num_tokens)]
            if not prefill and self.use_cuda_graph and len(caches) <= self.graphs.max_batch:
                return self.graphs.run(ids, pos, slots, caches)
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
