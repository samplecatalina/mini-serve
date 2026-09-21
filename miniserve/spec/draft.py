"""The draft model's own runtime: its KV pool, block tables and forward passes.

The draft is a second, much smaller model of the same family. It keeps its own
KV pool (its layers and heads differ, so it cannot share the target's), its own
block table per request, and its own captured decode graphs.

Its pool holds as many tokens as the target's. That is exactly enough, and the
argument is short: the draft is fed the same sequences as the target and is
never ahead of it. A round feeds the draft everything the target already has
(positions ``covered .. seq_len``), then its own proposals, one per step; after
the target has verified, the draft is rewound to the accepted length. So at
every instant the draft table holds at most as many tokens as the target one,
and with the prefix cache off (which speculative decoding requires, see
``spec/engine.py``) equal token counts mean equal block counts.

Nothing here reads device memory back to the host: a whole round of proposals
is queued, each step taking the previous step's tokens straight from the device
(the same trick overlap scheduling uses between steps), and the engine reads the
proposals back once, together with the target's verdict.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from miniserve.cache.backend import allocator_class, default_backend
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_pool import KVPool
from miniserve.engine.cuda_graph import DecodeGraphs, graph_buckets
from miniserve.model.attention import FlashInferPagedAttention
from miniserve.model.qwen3 import Qwen3ForCausalLM
from miniserve.model.transfer import CopyFence, to_device


class DraftRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        num_blocks: int,
        block_size: int = 16,
        max_prefill_tokens: int = 8192,
        cuda_graph: bool = True,
        cuda_graph_max_bs: int = 64,
        block_backend: str | None = None,
        workspace: torch.Tensor | None = None,
    ):
        """``num_blocks``: the target pool's block count, which this pool matches.
        ``workspace``: the target's FlashInfer workspace, reused because only one pass
        runs at a time; None allocates a second one.
        ``cuda_graph``: capture decode graphs for the steps that feed a single token
        (all but the first of a round)."""
        cfg = model.cfg
        self.model = model
        self.device = model.device
        self.block_size = block_size
        self.max_prefill_tokens = max_prefill_tokens
        self.allocator = allocator_class(block_backend or default_backend())(num_blocks, block_size)
        # One block past the allocator's, for the padding rows of a decode graph.
        self.pool = KVPool(
            cfg.num_layers, num_blocks + 1, block_size, cfg.num_kv_heads, cfg.head_dim, model.dtype, model.device
        )
        self.attn = FlashInferPagedAttention(self.pool, cfg.num_heads, model.attn_scale, workspace=workspace)
        self.fence = CopyFence(model.device)
        self.tables: dict[int, BlockTable] = {}
        self.graphs: DecodeGraphs | None = None
        if cuda_graph:
            self.graphs = DecodeGraphs(
                model,
                self.pool,
                dummy_block=num_blocks,
                buckets=graph_buckets(cuda_graph_max_bs),
                workspace=self.attn.workspace,
                num_heads=cfg.num_heads,
                scale=model.attn_scale,
                fence=self.fence,
            )

    # ------------------------------------------------------------------ per request

    def admit(self, rid: int) -> None:
        if rid in self.tables:
            raise RuntimeError(f"request {rid} already holds a draft cache")
        self.tables[rid] = self.allocator.new_table()

    def release(self, rid: int) -> None:
        table = self.tables.pop(rid, None)
        if table is not None:
            table.release()

    def covered(self, rid: int) -> int:
        """Tokens of the request whose KV this model holds."""
        return self.tables[rid].num_tokens

    def rewind_to(self, rid: int, num_tokens: int) -> None:
        """Drop the KV of everything past ``num_tokens`` (rejected proposals); keep the blocks."""
        table = self.tables[rid]
        if num_tokens < table.num_tokens:
            table.rewind(table.num_tokens - num_tokens)

    @torch.inference_mode()
    def prefill(self, rid: int, token_ids: Sequence[int]) -> None:
        """Compute the KV of ``token_ids`` (from position 0), in chunks, discarding the logits."""
        table = self.tables[rid]
        if table.num_tokens:
            raise RuntimeError(f"request {rid} already has {table.num_tokens} draft tokens")
        for start in range(0, len(token_ids), self.max_prefill_tokens):
            chunk = list(token_ids[start : start + self.max_prefill_tokens])
            self._extend([table], [chunk], [start])

    # ------------------------------------------------------------------ a round

    @torch.inference_mode()
    def propose(
        self, rids: Sequence[int], pending: Sequence[Sequence[int]], starts: Sequence[int], gamma: int
    ) -> torch.Tensor:
        """``gamma`` proposed tokens per request, ``[B, gamma]`` int64 left on the device.

        ``pending[b]`` are the tokens of request ``b`` that the target has and the draft
        has not, starting at position ``starts[b]``: usually one (the token the target
        sampled last), two after a round in which every proposal was accepted, more after
        a step that ran without speculation.
        """
        tables = [self.tables[r] for r in rids]
        with torch.cuda.nvtx.range("draft_extend"):
            logits = self._extend(tables, pending, starts)
            pos = [s + len(p) for s, p in zip(starts, pending)]
            proposals = [logits.argmax(dim=-1)]
        with torch.cuda.nvtx.range("draft_decode"):
            for _ in range(gamma - 1):
                logits = self._decode(tables, proposals[-1], pos)
                proposals.append(logits.argmax(dim=-1))
                pos = [p + 1 for p in pos]
        return torch.stack(proposals, dim=1)

    def _extend(self, tables: list[BlockTable], ids: Sequence[Sequence[int]], starts: Sequence[int]) -> torch.Tensor:
        """Several new tokens per sequence, eagerly: logits ``[B, vocab]`` of the last of each."""
        qo_lens = [len(x) for x in ids]
        for t, n in zip(tables, qo_lens):
            t.append_tokens(n)
        slots = [s for t, n in zip(tables, qo_lens) for s in t.tail_slots(n)]
        flat = [i for row in ids for i in row]
        pos = [p for s, n in zip(starts, qo_lens) for p in range(s, s + n)]
        self.fence.wait()
        self.attn.plan(True, qo_lens, tables, slots)
        ids_t = to_device(flat, torch.long, self.device)
        pos_t = to_device(pos, torch.long, self.device)
        self.fence.mark()
        return self.model.forward_with(ids_t, pos_t, self.attn, qo_lens)

    def _decode(self, tables: list[BlockTable], tokens: torch.Tensor, pos: Sequence[int]) -> torch.Tensor:
        """One new token per sequence, taken from the device: logits ``[B, vocab]``."""
        for t in tables:
            t.append_tokens(1)
        slots = [t.tail_slots(1)[0] for t in tables]
        rows = torch.arange(len(tables), device=tokens.device)
        # The steps of a round follow each other with nothing in between, so the pinned
        # buffers written below (the graph's staging buffer, the planning ones) may still
        # be feeding the copies the previous step queued.
        self.fence.wait()
        if self.graphs is not None and len(tables) <= self.graphs.max_batch:
            return self.graphs.run([0] * len(tables), list(pos), slots, tables, fill=(rows, tokens))
        self.attn.plan(False, [1] * len(tables), tables, slots)
        ids_t = torch.zeros(len(tables), dtype=torch.long, device=self.device).index_copy_(0, rows, tokens)
        pos_t = to_device(list(pos), torch.long, self.device)
        self.fence.mark()
        return self.model.forward_with(ids_t, pos_t, self.attn, [1] * len(tables))
