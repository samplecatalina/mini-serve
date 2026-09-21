"""BlockAllocator / BlockTable: allocation, release, reference counting, fragmentation.

Parametrized over allocator backends so that every implementation runs the
same tests, including the exact order in which block ids are handed out. The
Python implementation in ``miniserve/cache`` is the specification; the C++ one
built from ``minicore/`` has to agree with it block id for block id and error
for error. That is the whole point of running one suite over both.

Tables are created with ``allocator.new_table(...)`` rather than by naming a
table class, so a table always belongs to the same backend as its allocator.
"""

from __future__ import annotations

import random

import pytest

from miniserve.cache.backend import BACKENDS, OutOfBlocks, allocator_class


@pytest.fixture(params=BACKENDS)
def make(request):
    try:
        return allocator_class(request.param)
    except ImportError as exc:
        pytest.skip(str(exc))


def _snapshot(a):
    return a.num_free, [a.refcount(b) for b in range(a.num_blocks)]


# --------------------------------------------------------------------------- allocator


def test_fresh_allocator(make):
    a = make(8, 16)
    assert (a.num_blocks, a.block_size, a.num_free, a.num_used) == (8, 16, 8, 0)
    assert all(a.refcount(b) == 0 for b in range(8))
    a.check_invariants()


@pytest.mark.parametrize("args", [(0, 16), (8, 0), (-1, 16)])
def test_invalid_construction(make, args):
    with pytest.raises(ValueError):
        make(*args)


def test_allocation_order_and_lifo_reuse(make):
    a = make(8, 16)
    assert a.allocate(3) == [0, 1, 2]
    assert a.allocate(2) == [3, 4]
    a.free([1, 3])
    assert a.allocate(1) == [3]  # most recently freed first
    assert a.allocate(3) == [1, 5, 6]
    assert a.allocate(0) == []
    a.check_invariants()


def test_allocate_all_or_nothing(make):
    a = make(4, 16)
    a.allocate(3)
    before = _snapshot(a)
    with pytest.raises(OutOfBlocks):
        a.allocate(2)
    assert _snapshot(a) == before
    assert not a.can_allocate(2) and a.can_allocate(1)
    with pytest.raises(ValueError):
        a.allocate(-1)


def test_free_returns_blocks(make):
    a = make(4, 16)
    blocks = a.allocate(4)
    assert a.num_free == 0
    a.free(blocks)
    assert a.num_free == 4 and a.num_used == 0
    assert all(a.refcount(b) == 0 for b in blocks)
    a.check_invariants()


def test_double_free_rejected(make):
    a = make(4, 16)
    [b] = a.allocate(1)
    a.free([b])
    with pytest.raises(ValueError):
        a.free([b])
    a.check_invariants()


def test_free_is_validated_before_mutation(make):
    a = make(4, 16)
    x, y = a.allocate(2)
    before = _snapshot(a)
    with pytest.raises(ValueError):
        a.free([x, 3])  # 3 was never allocated
    with pytest.raises(ValueError):
        a.free([y, y])  # two releases, one holder
    with pytest.raises(IndexError):
        a.free([x, 99])
    assert _snapshot(a) == before


def test_incref_rules(make):
    a = make(4, 16)
    [b] = a.allocate(1)
    with pytest.raises(ValueError):
        a.incref([2])  # free block
    with pytest.raises(IndexError):
        a.incref([-1])
    before = _snapshot(a)
    with pytest.raises(ValueError):
        a.incref([b, 3])  # validated before mutation
    assert _snapshot(a) == before


def test_shared_block_freed_by_last_holder(make):
    a = make(4, 16)
    [b] = a.allocate(1)
    a.incref([b])
    a.incref([b])
    assert a.refcount(b) == 3
    a.free([b, b])  # two holders release in one call
    assert a.refcount(b) == 1 and a.num_free == 3
    a.free([b])
    assert a.refcount(b) == 0 and a.num_free == 4
    a.check_invariants()


def test_no_external_fragmentation(make):
    """Blocks are interchangeable: after any interleaving, all free blocks can be taken at once."""
    a = make(64, 16)
    blocks = a.allocate(64)
    a.free(blocks[::2])  # checkerboard: every other block free
    assert a.num_free == 32
    got = a.allocate(32)
    assert sorted(got) == blocks[::2]
    assert a.num_free == 0
    a.check_invariants()


# --------------------------------------------------------------------------- block table


def test_table_append_and_slots(make):
    a = make(8, 4)
    t = a.new_table()
    assert (t.num_tokens, t.capacity, t.last_block_len, t.blocks) == (0, 0, 0, [])
    assert t.append_tokens(5) == [0, 1]  # 5 tokens -> 2 blocks of 4
    assert (t.num_tokens, t.capacity, t.last_block_len) == (5, 8, 1)
    assert [t.slot(p) for p in range(5)] == [0, 1, 2, 3, 4]
    assert t.append_tokens(3) == []  # fills block 1 exactly
    assert t.last_block_len == 4
    assert t.append_tokens(1) == [2]
    assert t.slot(8) == 2 * 4
    with pytest.raises(IndexError):
        t.slot(9)


def test_table_slots_follow_physical_blocks(make):
    a = make(8, 4)
    a.allocate(3)  # blocks 0..2 held elsewhere
    a.free([1])
    t = a.new_table()
    t.append_tokens(6)
    assert t.blocks == [1, 3]
    assert [t.slot(p) for p in range(6)] == [4, 5, 6, 7, 12, 13]


def test_blocks_needed(make):
    t = make(8, 16).new_table()
    assert [t.blocks_needed(n) for n in (0, 1, 16, 17, 32, 33)] == [0, 1, 1, 2, 2, 3]
    t.append_tokens(10)
    assert [t.blocks_needed(n) for n in (0, 6, 7, 22, 23)] == [0, 0, 1, 1, 2]
    with pytest.raises(ValueError):
        t.blocks_needed(-1)


def test_table_append_all_or_nothing(make):
    a = make(2, 4)
    t = a.new_table()
    t.append_tokens(3)
    before = (_snapshot(a), list(t.blocks), t.num_tokens)
    with pytest.raises(OutOfBlocks):
        t.append_tokens(10)  # needs 3 more blocks, 1 free
    assert (_snapshot(a), t.blocks, t.num_tokens) == before


def test_internal_fragmentation_bound(make):
    a = make(64, 16)
    for n in range(1, 200, 7):
        t = a.new_table()
        t.append_tokens(n)
        assert 0 <= t.capacity - t.num_tokens < a.block_size
        t.release()
    assert a.num_free == 64


def test_rewind_keeps_the_blocks(make):
    """Undoing an append must not give the blocks back: the caller recomputes those
    tokens and has to land in the same slots (the decode-graph test relies on this)."""
    a = make(8, 4)
    t = a.new_table()
    t.append_tokens(5)  # 2 blocks
    slots = [t.slot(p) for p in range(5)]
    t.rewind(1)
    assert (t.num_tokens, t.num_blocks, a.num_free) == (4, 2, 6)
    assert t.blocks_needed(1) == 0  # the block for it is still held
    assert t.append_tokens(1) == [] and [t.slot(p) for p in range(5)] == slots
    t.rewind(0)
    assert t.num_tokens == 5
    with pytest.raises(IndexError):
        t.rewind(6)
    with pytest.raises(IndexError):
        t.rewind(-1)
    a.check_invariants()


def test_release(make):
    a = make(8, 4)
    t = a.new_table()
    t.append_tokens(9)
    t.release()
    assert (t.blocks, t.num_tokens, a.num_free) == ([], 0, 8)
    t.append_tokens(1)  # reusable after release
    assert a.num_free == 7


def test_shared_prefix(make):
    a = make(8, 4)
    owner = a.new_table()
    owner.append_tokens(8)  # blocks [0, 1], both full
    t = a.new_table(owner.blocks)
    assert (t.num_tokens, t.last_block_len) == (8, 4)
    assert [a.refcount(b) for b in owner.blocks] == [2, 2]
    assert t.append_tokens(1) == [2]  # new token goes into a fresh private block
    owner.release()
    assert [a.refcount(b) for b in (0, 1, 2)] == [1, 1, 1] and a.num_free == 5
    t.release()
    assert a.num_free == 8
    a.check_invariants()


def test_shared_prefix_validation(make):
    a = make(8, 4)
    [b] = a.allocate(1)
    with pytest.raises(ValueError):
        a.new_table([3])  # free block
    with pytest.raises(ValueError):
        a.new_table([b, b])
    assert a.refcount(b) == 1


# --------------------------------------------------------------------------- randomized


@pytest.mark.parametrize("seed", range(5))
def test_random_operations(make, seed):
    """Random table appends, shares and releases, checked against a plain holder count."""
    rng = random.Random(seed)
    a = make(32, 4)
    tables = []
    for _ in range(2000):
        op = rng.random()
        if op < 0.45 or not tables:
            t = tables[rng.randrange(len(tables))] if tables and rng.random() < 0.7 else a.new_table()
            if t not in tables:
                tables.append(t)
            n = rng.randint(1, 12)
            before = _snapshot(a)
            try:
                t.append_tokens(n)
            except OutOfBlocks:
                assert t.blocks_needed(n) > before[0] and _snapshot(a) == before
        elif op < 0.6:
            src = tables[rng.randrange(len(tables))]
            full = src.num_tokens // a.block_size
            tables.append(a.new_table(src.blocks[: rng.randint(0, full)]))
        else:
            tables.pop(rng.randrange(len(tables))).release()

        a.check_invariants()
        holders = [0] * a.num_blocks
        for t in tables:
            for b in t.blocks:
                holders[b] += 1
            assert len(set(t.blocks)) == len(t.blocks)
            assert 0 <= t.capacity - t.num_tokens < a.block_size or t.num_tokens == 0
        assert holders == [a.refcount(b) for b in range(a.num_blocks)]
    for t in tables:
        t.release()
    assert a.num_free == a.num_blocks


# --------------------------------------------------------------------------- packing


def _naive_pack(tables, pad_rows, pad_block, pad_last):
    """The layout written out the obvious way, as the reference both backends must match."""
    indptr, indices = [0], []
    for t in tables:
        indices += list(t.blocks)
        indptr.append(len(indices))
    for _ in range(pad_rows):
        indices.append(pad_block)
        indptr.append(len(indices))
    return indptr, indices, [t.last_block_len for t in tables] + [pad_last] * pad_rows


@pytest.mark.parametrize("pad_rows", [0, 3])
def test_pack_matches_the_naive_layout(make, pad_rows):
    import numpy as np

    from miniserve.cache.backend import pack_block_tables

    rng = random.Random(pad_rows)
    a = make(256, 4)
    tables = []
    for n in [1, 4, 5, 17, 0, 33]:
        t = a.new_table()
        t.append_tokens(n)
        tables.append(t)
    rng.shuffle(tables)
    want = _naive_pack(tables, pad_rows, 255, 2)
    indptr, indices, last = (np.full(64, -1, dtype=np.int32) for _ in range(3))
    n = pack_block_tables(tables, pad_rows, 255, 2, indptr, indices, last)
    rows = len(tables) + pad_rows
    assert n == len(want[1])
    assert indptr[: rows + 1].tolist() == want[0]
    assert indices[:n].tolist() == want[1]
    assert last[:rows].tolist() == want[2]


def test_pack_refuses_buffers_that_are_too_small(make):
    import numpy as np

    from miniserve.cache.backend import pack_block_tables

    a = make(16, 4)
    t = a.new_table()
    t.append_tokens(12)
    with pytest.raises(ValueError):
        pack_block_tables([t], 0, 0, 1, *(np.zeros(2, dtype=np.int32) for _ in range(2)), np.zeros(1, dtype=np.int32))


def test_packed_tables_grow_and_reuse_their_buffers(make):
    import torch

    from miniserve.cache.packing import PackedTables

    a = make(512, 4)
    p = PackedTables("cpu", rows=2, indices=4)
    tables = [a.new_table() for _ in range(5)]
    for i, t in enumerate(tables):
        t.append_tokens(4 * i + 3)
    indptr, indices, last = p.pack(tables, pad_rows=1, pad_block=511, pad_last=1)
    want = _naive_pack(tables, 1, 511, 1)
    assert (indptr.tolist(), indices.tolist(), last.tolist()) == want
    assert indptr.dtype == indices.dtype == last.dtype == torch.int32
    buf = p.indices.data_ptr()
    small = p.pack(tables[:1])
    assert p.indices.data_ptr() == buf, "a smaller batch must reuse the buffer"
    assert (small[0].tolist(), small[1].tolist(), small[2].tolist()) == _naive_pack(tables[:1], 0, 0, 1)
