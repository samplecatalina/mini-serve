"""Preallocated paged KV storage.

One tensor ``[2, layers, num_blocks, block_size, kv_heads, head_dim]``: K and
V for every layer, each layer's cache laid out as ``[num_blocks, block_size,
kv_heads, head_dim]`` (FlashInfer's NHD layout). Which blocks belong to which
request is tracked separately (``BlockAllocator`` / ``BlockTable``).
"""

from __future__ import annotations

import torch


class KVPool:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.buf = torch.zeros(
            (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype, device=device
        )

    @property
    def num_bytes(self) -> int:
        return self.buf.numel() * self.buf.element_size()

    def k(self, layer: int) -> torch.Tensor:
        return self.buf[0, layer]

    def v(self, layer: int) -> torch.Tensor:
        return self.buf[1, layer]

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store ``k``/``v`` ``[T, kv_heads, head_dim]`` at flat slots ``block * block_size + offset``."""
        flat = (-1, self.num_kv_heads, self.head_dim)
        self.buf[0, layer].view(flat)[slots] = k
        self.buf[1, layer].view(flat)[slots] = v
