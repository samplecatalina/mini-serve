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

from miniserve.cache.backend import allocator_class, default_backend
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.cache.kv_pool import KVPool
from miniserve.engine.cuda_graph import DecodeGraphs, graph_buckets
from miniserve.engine.request import PLACEHOLDER, Request
from miniserve.engine.sampler import Sampler, SamplingArgs
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.attention import ContiguousAttention, FlashInferPagedAttention
from miniserve.model.qwen3 import Qwen3ForCausalLM
from miniserve.model.transfer import CopyFence, to_device

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
        cuda_graph: bool = True,
        cuda_graph_max_bs: int | None = None,
        block_backend: str | None = None,
        sample_rows: int | None = None,
        attn_workspace_mb: int | None = None,
    ):
        """In paged mode the KV pool is sized from the GPU memory left after the
        weights and the peak memory of a ``max_prefill_tokens`` prefill followed
        by sampling ``max_running`` rows (``kv_mem_fraction`` of it);
        ``kv_pool_tokens`` sets an exact size and must fit in that memory.
        ``radix``: keep a prefix cache over the pool (paged mode only).
        ``cuda_graph``: capture decode graphs for batch sizes up to ``cuda_graph_max_bs``
        (default ``max_running``) after the pool is allocated (paged mode only); they use the
        memory the pool leaves free. ``use_cuda_graph`` switches them off and on at run time.
        ``block_backend``: implementation of the block bookkeeping (``python`` or ``cpp``);
        default from ``MINISERVE_BLOCK_BACKEND``.
        ``sample_rows``: rows of logits the profiled peak must hold (default ``max_running``);
        a verify pass takes the logits of several positions per request.
        ``attn_workspace_mb``: FlashInfer's scratch buffer, shared by every pass and graph of this
        runner (default ``FlashInferPagedAttention.WORKSPACE_BYTES``)."""
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        self.model = model
        self.device = model.device
        self.attention = attention
        self.block_backend = block_backend if block_backend is not None else default_backend()
        self.sampler = Sampler(model.device, model.cfg.vocab_size)
        self.allocator = None
        self.kv: KVCacheManager | None = None
        self.graphs: DecodeGraphs | None = None
        self.use_cuda_graph = False
        self.fence = CopyFence(model.device)
        self._workspace_bytes = attn_workspace_mb * 1024 * 1024 if attn_workspace_mb else None
        if attention == "paged":
            cfg = model.cfg

            def pool(num_blocks: int) -> KVPool:
                return KVPool(
                    cfg.num_layers, num_blocks, block_size, cfg.num_kv_heads, cfg.head_dim, model.dtype, model.device
                )

            self.kv_profile = self._profile(
                pool, block_size, max_prefill_tokens, sample_rows or max_running, kv_mem_fraction
            )
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
            self.allocator = allocator_class(self.block_backend)(num_blocks, block_size)
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
                    fence=self.fence,
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
        workspace = None
        if self._workspace_bytes:
            workspace = torch.empty(self._workspace_bytes, dtype=torch.uint8, device=self.device)
        self.flashinfer = FlashInferPagedAttention(probe, cfg.num_heads, self.model.attn_scale, workspace=workspace)
        alloc = allocator_class(self.block_backend)(probe.num_blocks, block_size)
        table = alloc.new_table()
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

    def forward(self, batch: Batch, fill: tuple[torch.Tensor, dict[int, int]] | None = None) -> torch.Tensor:
        """Logits ``[len(batch.requests), vocab]`` for the last token of each request.

        ``fill = (tokens, rows)``: the previous step's sampled tokens, still on the device, and
        the row of each request in them. A decode row whose input is a placeholder (a token not
        read back yet) takes its token from there, on the device.

        Nothing here waits for the device, so a step can be launched while the previous one
        is still running."""
        # Any row with more than one new token needs the prefill kernel; decode rows of a mixed
        # batch are causal rows of length 1 to it.
        prefill = batch.phase is not Phase.DECODE
        ids: list[int] = []
        pos: list[int] = []
        for r, start, n in zip(batch.requests, batch.starts, batch.extend_lens):
            # prompt + output (a preempted request resumes), from the first token without KV
            ids += r.token_slice(start, n)
            pos += range(start, start + n)
        seq_lens = batch.seq_lens
        caches = [r.cache for r in batch.requests]
        fill_idx = self._fill_rows(batch, ids, fill)
        # Pinned buffers written below (planning, graph staging) may still feed the previous step's copies.
        self.fence.wait()
        if self.attention == "paged":
            if any(t.num_tokens != s for t, s in zip(caches, batch.starts)):
                raise RuntimeError("batch starts disagree with the block tables")
            self.reserve(caches, seq_lens)
            # One call per request, not one per token: with a C++ table every
            # crossing of the binding costs more than the lookup it performs.
            slots = [s for t, n in zip(caches, seq_lens) for s in t.tail_slots(n)]
            if not prefill and self.use_cuda_graph and len(caches) <= self.graphs.max_batch:
                return self.graphs.run(ids, pos, slots, caches, fill_idx)
            self.flashinfer.plan(prefill, seq_lens, caches, slots)
            attn = self.flashinfer
        else:
            attn = ContiguousAttention(caches, seq_lens, self.model.attn_scale)
        ids_t = to_device(ids, torch.long, self.device)
        if fill_idx is not None:
            ids_t.index_copy_(0, *fill_idx)
        pos_t = to_device(pos, torch.long, self.device)
        self.fence.mark()
        return self.model.forward_with(ids_t, pos_t, attn, seq_lens)

    def _fill_rows(
        self, batch: Batch, ids: list[int], fill: tuple[torch.Tensor, dict[int, int]] | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """(positions in ``ids``, device tokens) replacing placeholders. Only a decode row's
        input can be one; decode rows come first, one token each, so row i is ``ids[i]``."""
        dst = [i for i in range(batch.num_decode) if ids[i] == PLACEHOLDER]
        if len(dst) != ids.count(PLACEHOLDER):
            raise RuntimeError("a placeholder token outside the decode rows")
        if not dst:
            return None
        if fill is None:
            raise RuntimeError("placeholder input tokens without the previous step's samples")
        tokens, rows = fill
        src = [rows[batch.requests[i].rid] for i in dst]
        return to_device(dst, torch.long, self.device), tokens.index_select(0, to_device(src, torch.long, tokens.device))

    def forward_tokens(
        self,
        tables: list,
        ids: list[int],
        pos: list[int],
        qo_lens: list[int],
        fill: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logits ``[sum(qo_lens), vocab]``: one row per new token, not one per sequence.

        ``ids`` and ``pos`` hold the new tokens of all sequences concatenated, ``qo_lens[b]``
        of them for sequence ``b``, whose table already covers them (:meth:`reserve`). The
        pass is always a prefill (extend) one, so it runs eagerly whatever the batch size.

        ``fill = (positions in ids, tokens)``: device tensors replacing those token ids, for
        tokens that were produced on the device and not read back (a round's proposals).
        """
        if self.attention != "paged":
            raise RuntimeError("forward_tokens needs the paged attention path")
        self.fence.wait()
        slots = [s for t, n in zip(tables, qo_lens) for s in t.tail_slots(n)]
        self.flashinfer.plan(True, qo_lens, tables, slots)
        ids_t = to_device(ids, torch.long, self.device)
        if fill is not None:
            ids_t.index_copy_(0, *fill)
        pos_t = to_device(pos, torch.long, self.device)
        self.fence.mark()
        # decode_logits keeps every position's logits; forward_with would gather the last of each.
        return self.model.decode_logits(ids_t, pos_t, self.flashinfer)

    def reserve(self, tables: list, seq_lens: list[int]) -> None:
        """Extend every table by its new tokens. If the pool is short even after evicting
        from the prefix cache, raise ``OutOfBlocks`` with the tables unchanged."""
        need = sum(t.blocks_needed(n) for t, n in zip(tables, seq_lens))
        self.kv.reserve(need)  # evicts from the prefix cache if the free pool is short
        for t, n in zip(tables, seq_lens):
            t.append_tokens(n)
