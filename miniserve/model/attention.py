"""Attention backends.

The model computes q/k/v (projections, q/k norm, RoPE) and hands them to a
backend, which stores the new K/V and returns the attention output. A backend
instance describes one forward pass: which sequences are in it and where their
KV lives.

- ``ContiguousAttention``: one contiguous cache per sequence, PyTorch SDPA per
  sequence. This is the reference path (mirrors Hugging Face transformers).
- ``FlashInferPagedAttention``: a shared paged KV pool and FlashInfer batch
  kernels, one call per layer for the whole batch.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from miniserve.cache.block_table import BlockTable
    from miniserve.cache.kv_pool import KVPool
    from miniserve.model.qwen3 import ContiguousKVCache


class AttentionBackend(Protocol):
    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Store ``k``/``v`` ``[T, kv_heads, D]`` and attend with ``q`` ``[T, heads, D]``; returns ``[T, heads, D]``."""
        ...

    def finish(self) -> None:
        """Called once after the last layer."""
        ...


class ContiguousAttention:
    def __init__(self, caches: Sequence[ContiguousKVCache], seq_lens: Sequence[int], scale: float):
        if len(caches) != len(seq_lens):
            raise ValueError(f"{len(caches)} caches for {len(seq_lens)} sequences")
        # is_causal for multi-token sequences assumes their cache was empty (a
        # prefill); extending a non-empty cache by several tokens needs an
        # offset mask and is not supported here.
        if any(n > 1 and c.length != 0 for c, n in zip(caches, seq_lens)):
            raise NotImplementedError("multi-token forward on a non-empty cache")
        self.caches = list(caches)
        self.seq_lens = list(seq_lens)
        self.scale = scale

    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outs = []
        for cache, qs, ks, vs in zip(self.caches, q.split(self.seq_lens), k.split(self.seq_lens), v.split(self.seq_lens)):
            k_all, v_all = cache.write(layer, ks, vs)
            # SDPA expects [batch, heads, seq, head_dim].
            o = F.scaled_dot_product_attention(
                qs.transpose(0, 1).unsqueeze(0),
                k_all.transpose(0, 1).unsqueeze(0),
                v_all.transpose(0, 1).unsqueeze(0),
                is_causal=qs.shape[0] > 1,
                scale=self.scale,
                enable_gqa=True,
            )
            outs.append(o.squeeze(0).transpose(0, 1))
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def finish(self) -> None:
        for cache, n in zip(self.caches, self.seq_lens):
            cache.length += n


class FlashInferPagedAttention:
    """FlashInfer batch attention over a paged KV pool.

    Reused across steps: :meth:`plan` describes the next forward pass (it must
    be called before every pass), :meth:`attend` runs one layer.
    """

    WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(self, pool: KVPool, num_heads: int, scale: float):
        import flashinfer

        self.pool = pool
        self.num_heads = num_heads
        self.scale = scale
        workspace = torch.empty(self.WORKSPACE_BYTES, dtype=torch.uint8, device=pool.device)
        self.workspace = workspace  # shared with the decode graphs' wrappers (one pass runs at a time)
        self._prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
        # Tensor-core decode only pays off for large query groups (GQA >= 4).
        self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_tensor_cores=num_heads // pool.num_kv_heads >= 4, backend="fa2"
        )
        self._wrapper = None
        self._slots: torch.Tensor | None = None

    def plan(self, prefill: bool, qo_lens: Sequence[int], tables: Sequence[BlockTable], slots: Sequence[int]) -> None:
        """Describe the next pass: ``qo_lens[b]`` new tokens for sequence ``b``, whose
        table already covers them; ``slots`` are the pool slots of all new tokens in order."""
        if not prefill and any(n != 1 for n in qo_lens):
            raise ValueError("decode passes take one token per sequence")
        kv_indptr = [0]
        indices: list[int] = []
        for t in tables:
            indices += t.blocks
            kv_indptr.append(len(indices))
        i32 = dict(dtype=torch.int32)
        kv_indptr_t = torch.tensor(kv_indptr, **i32)
        indices_t = torch.tensor(indices, **i32)
        last_t = torch.tensor([t.last_block_len for t in tables], **i32)
        p = self.pool
        common = dict(
            num_qo_heads=self.num_heads,
            num_kv_heads=p.num_kv_heads,
            page_size=p.block_size,
            sm_scale=self.scale,
            q_data_type=p.dtype,
            kv_data_type=p.dtype,
            non_blocking=False,
        )
        if prefill:
            qo_indptr = torch.tensor([0, *itertools.accumulate(qo_lens)], **i32)
            self._prefill.plan(
                qo_indptr, kv_indptr_t, indices_t, last_t, head_dim_qk=p.head_dim, causal=True, **common
            )
            self._wrapper = self._prefill
        else:
            self._decode.plan(kv_indptr_t, indices_t, last_t, head_dim=p.head_dim, **common)
            self._wrapper = self._decode
        self._slots = torch.tensor(list(slots), dtype=torch.long, device=p.device)

    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._wrapper is None:
            raise RuntimeError("plan() must be called before each forward pass")
        self.pool.write(layer, self._slots, k, v)
        return self._wrapper.run(q, (self.pool.k(layer), self.pool.v(layer)))

    def finish(self) -> None:
        self._wrapper = None
        self._slots = None
