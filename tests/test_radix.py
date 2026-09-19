"""RadixTree and KVCacheManager: matching, splitting, locking, LRU eviction, and randomized interleavings.

All CPU. The randomized test plays many requests that acquire cached
prefixes, extend their tables, commit, release, get evicted around, and checks
after every operation that the tree's lock counts agree with the allocator's
reference counts, that no block in use is ever evicted, and that every cached
block holds the tokens its path spells.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from miniserve.cache.block_allocator import BlockAllocator, OutOfBlocks
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.cache.radix_tree import RadixTree

BS = 4


def _tree(num_blocks=32):
    a = BlockAllocator(num_blocks, BS)
    return RadixTree(a), a


def _cache(tree: RadixTree, a: BlockAllocator, tokens: list[int]) -> list[int]:
    """Allocate blocks for the whole blocks of ``tokens``, insert them, drop the caller's reference."""
    n = len(tokens) // BS
    blocks = a.allocate(n)
    tree.insert(tokens[: n * BS], blocks)
    a.free(blocks)
    return blocks


def test_match_empty_and_partial_block():
    t, a = _tree()
    assert t.match([1, 2, 3, 4, 5]) == (t.root, [])
    blocks = _cache(t, a, list(range(1, 9)))  # 2 blocks
    assert t.match([1, 2, 3])[1] == []  # less than a block
    assert t.match([1, 2, 3, 4, 5, 6, 7])[1] == blocks[:1]  # second block incomplete
    assert t.match(list(range(1, 12)))[1] == blocks
    assert t.match([1, 2, 3, 5])[1] == []  # differs inside the first block
    t.check_invariants()


def test_match_splits_at_block_boundary():
    t, a = _tree()
    blocks = _cache(t, a, list(range(1, 13)))  # one node of 3 blocks
    node, got = t.match([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0])
    assert got == blocks[:2] and node.blocks == blocks[:2]
    (lower,) = node.children.values()
    assert lower.blocks == blocks[2:] and lower.tokens == [9, 10, 11, 12]
    assert t.num_cached_blocks == 3
    t.check_invariants()


def test_insert_shares_prefix_and_ignores_duplicates():
    t, a = _tree()
    first = _cache(t, a, [1, 2, 3, 4, 5, 6, 7, 8])
    # Same first block, different second one: branch after one block.
    dup = a.allocate(2)
    leaf = t.insert([1, 2, 3, 4, 9, 9, 9, 9], dup)
    assert leaf.blocks == [dup[1]] and leaf.parent.blocks == [first[0]]
    assert a.refcount(dup[0]) == 1  # the duplicate of the cached first block was not taken
    assert a.refcount(dup[1]) == 2
    a.free(dup)
    assert t.num_cached_blocks == 3
    # Inserting an already cached prefix changes nothing.
    again = a.allocate(1)
    node = t.insert([1, 2, 3, 4], again)
    assert node.blocks == [first[0]] and a.refcount(again[0]) == 1
    a.free(again)
    t.check_invariants()


def test_insert_rejects_partial_blocks():
    t, a = _tree()
    with pytest.raises(ValueError):
        t.insert([1, 2, 3, 4, 5], a.allocate(1))


def test_lru_eviction_order_and_parent_promotion():
    t, a = _tree(8)
    x = _cache(t, a, [1, 1, 1, 1, 2, 2, 2, 2])  # root - [1] - [2]      clock 1
    y = _cache(t, a, [1, 1, 1, 1, 3, 3, 3, 3])  # root - [1] - [3]      clock 2
    z = _cache(t, a, [4, 4, 4, 4])  # root - [4]                        clock 3
    t.match([1, 1, 1, 1, 2, 2, 2, 2])  # [1] and x's leaf touched at clock 4
    assert t.num_evictable == 4 and a.num_free == 4
    assert t.evict(1) == 1 and a.refcount(y[1]) == 0  # least recent leaf: y's (clock 2)
    assert t.evict(1) == 1 and a.refcount(z[0]) == 0  # then z (clock 3)
    assert a.refcount(x[0]) == 1  # [1] is not a leaf yet
    assert t.evict(2) == 2 and a.num_free == 8  # x's leaf, then [1], promoted to a leaf
    assert t.num_cached_blocks == 0 and t.evict(5) == 0
    t.check_invariants()


def test_locked_nodes_are_never_evicted():
    t, a = _tree(8)
    _cache(t, a, [1, 1, 1, 1, 2, 2, 2, 2])
    node, blocks = t.match([1, 1, 1, 1, 2, 2, 2, 2])
    a.incref(blocks)  # a request's table holds them
    t.lock(node)
    assert t.num_evictable == 0 and t.evict(8) == 0
    _cache(t, a, [1, 1, 1, 1, 5, 5, 5, 5])  # a sibling under the locked [1]
    assert t.num_evictable == 1 and t.evict(8) == 1  # only the unlocked sibling
    t.unlock(node)
    a.free(blocks)
    assert t.num_evictable == 2
    t.check_invariants()
    with pytest.raises(ValueError):
        t.unlock(node)


def test_split_keeps_locks():
    t, a = _tree()
    _cache(t, a, list(range(1, 13)))
    node, blocks = t.match(list(range(1, 13)))
    a.incref(blocks)
    t.lock(node)
    upper, part = t.match([1, 2, 3, 4, 0])  # splits the locked node after one block
    assert upper.lock == 1 and t.num_evictable == 0
    t.check_invariants()
    t.unlock(node)  # the lower half's handle unlocks the whole path, upper included
    a.free(blocks)
    assert upper.lock == 0 and t.num_evictable == 3
    t.check_invariants()


def test_clear_refuses_locked_tree():
    t, a = _tree()
    _cache(t, a, [1, 1, 1, 1])
    node, blocks = t.match([1, 1, 1, 1, 0])
    a.incref(blocks)
    t.lock(node)
    with pytest.raises(RuntimeError):
        t.clear()
    t.unlock(node)
    a.free(blocks)
    t.clear()
    assert a.num_free == a.num_blocks


# --------------------------------------------------------------------------- KVCacheManager


@dataclass(eq=False)
class _Req:
    rid: int
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    cache: object = None
    cache_node: object = None
    num_cached_tokens: int = 0

    @property
    def seq_len(self):
        return len(self.prompt_ids) + len(self.output_ids)


def test_manager_acquire_leaves_one_token_to_compute():
    kv = KVCacheManager(BlockAllocator(16, BS))
    x = _Req(0, list(range(1, 9)))  # exactly 2 blocks
    kv.acquire(x)
    x.cache.append_tokens(8)
    kv.release(x)
    y = _Req(1, list(range(1, 9)))  # the same prompt: only the first block may be reused
    assert kv.acquire(y) == 4
    z = _Req(2, list(range(1, 10)))
    assert kv.acquire(z) == 8
    kv.check_invariants()


def test_manager_without_radix_is_plain_accounting():
    kv = KVCacheManager(BlockAllocator(4, BS), radix=False)
    x = _Req(0, list(range(1, 9)))
    assert kv.acquire(x) == 0
    x.cache.append_tokens(8)
    kv.release(x)
    assert kv.allocator.num_free == 4 and kv.num_available == 4


def test_manager_reserve_evicts_and_fails_cleanly():
    kv = KVCacheManager(BlockAllocator(4, BS))
    x = _Req(0, list(range(1, 13)))
    kv.acquire(x)
    x.cache.append_tokens(12)
    kv.release(x)  # 3 blocks cached, 1 free
    assert kv.num_available == 4
    kv.reserve(3)  # evicts 2
    assert kv.allocator.num_free >= 3
    with pytest.raises(OutOfBlocks):
        kv.reserve(5)
    kv.check_invariants()


@pytest.mark.parametrize("seed", range(40))
def test_randomized_interleaving(seed):
    """Random requests over a few shared prefixes: acquire, extend, commit, release, reserve.

    A shadow map remembers which token each (block, offset) holds as tables are
    written; every cached path must spell tokens that its blocks actually hold."""
    rng = random.Random(seed)
    num_blocks = rng.choice([6, 12, 40])
    kv = KVCacheManager(BlockAllocator(num_blocks, BS))
    content: dict[tuple[int, int], int] = {}  # (block, offset) -> token written
    prefixes = [[rng.randrange(1, 9) for _ in range(rng.randint(2, 14))] for _ in range(3)]
    live: list[_Req] = []
    rid = 0

    def write(req, n):
        t = req.cache
        start = t.num_tokens
        kv.reserve(t.blocks_needed(n))
        t.append_tokens(n)
        seq = req.prompt_ids + req.output_ids
        for pos in range(start, start + n):
            content[(t.blocks[pos // BS], pos % BS)] = seq[pos]

    def check():
        kv.check_invariants()
        tree = kv.tree
        stack = [(tree.root, [])]
        while stack:
            node, prefix = stack.pop()
            for child in node.children.values():
                for i, b in enumerate(child.blocks):
                    for off in range(BS):
                        assert content.get((b, off)) == child.tokens[i * BS + off], "cached block holds other tokens"
                stack.append((child, prefix + child.tokens))
        for r in live:  # a live request's table still holds its own tokens
            seq = r.prompt_ids + r.output_ids
            for pos in range(r.cache.num_tokens):
                assert content[(r.cache.blocks[pos // BS], pos % BS)] == seq[pos], "a block in use was overwritten"

    for _ in range(300):
        op = rng.random()
        if op < 0.4 and len(live) < 4:
            prompt = rng.choice(prefixes) + [rng.randrange(1, 9) for _ in range(rng.randint(0, 5))]
            if len(prompt) + 8 > num_blocks * BS:
                continue
            r = _Req(rid, prompt)
            rid += 1
            cached = kv.acquire(r)
            assert cached % BS == 0 and cached < r.seq_len
            need = -(-(r.seq_len + 1) // BS) - len(r.cache.blocks)
            if need > kv.num_available:
                kv.abandon(r)
                continue
            write(r, r.seq_len - cached)  # prefill the uncached part
            kv.commit(r)
            live.append(r)
        elif op < 0.8 and live:
            r = rng.choice(live)
            if r.cache.blocks_needed(1) > kv.num_available:
                continue
            r.output_ids.append(rng.randrange(1, 9))
            write(r, 1)
            if rng.random() < 0.3:
                kv.commit(r)
        elif live:
            r = live.pop(rng.randrange(len(live)))
            kv.release(r)
        check()
    for r in live:
        kv.release(r)
    live.clear()
    check()
    assert kv.num_idle_blocks() == num_blocks
    kv.tree.clear()
    assert kv.allocator.num_free == num_blocks
