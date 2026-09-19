"""KVPool storage layout and its agreement with BlockTable slots (CPU)."""

from __future__ import annotations

import torch

from miniserve.cache.block_allocator import BlockAllocator
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_pool import KVPool


def _pool(num_blocks=6, block_size=4):
    return KVPool(num_layers=2, num_blocks=num_blocks, block_size=block_size, num_kv_heads=2, head_dim=3,
                  dtype=torch.float32, device="cpu")


def test_layout_and_size():
    p = _pool()
    assert p.k(1).shape == (6, 4, 2, 3) and p.k(1).is_contiguous() and p.v(0).is_contiguous()
    assert p.num_bytes == 2 * 2 * 6 * 4 * 2 * 3 * 4


def test_write_lands_at_table_slots():
    p = _pool()
    a = BlockAllocator(p.num_blocks, p.block_size)
    a.allocate(3)
    a.free([0, 2])  # table gets blocks 2 then 0: scattered and descending
    t = BlockTable(a)
    t.append_tokens(6)
    assert t.blocks == [2, 0]
    k = torch.arange(6 * 2 * 3, dtype=torch.float32).view(6, 2, 3)
    v = -k
    slots = torch.tensor([t.slot(i) for i in range(6)])
    p.write(1, slots, k, v)
    for pos in range(6):
        blk, off = t.blocks[pos // 4], pos % 4
        assert torch.equal(p.k(1)[blk, off], k[pos]) and torch.equal(p.v(1)[blk, off], v[pos])
    assert p.k(0).abs().sum() == 0  # other layers untouched
    assert p.k(1)[1].abs().sum() == 0  # block not in the table untouched
