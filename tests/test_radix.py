"""RadixTree and KVCacheManager: matching, splitting, locking, LRU eviction, and randomized interleavings.

All CPU, and every test runs against both backends (the Python reference and the
C++ one in minicore/, tree and allocator together). The randomized test plays many requests that acquire cached
prefixes, extend their tables, commit, release, get evicted around, and checks
after every operation that the tree's lock counts agree with the allocator's
reference counts, that no block in use is ever evicted, and that every cached
block holds the tokens its path spells.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from miniserve.cache.backend import BACKENDS, OutOfBlocks, allocator_class, radix_tree_class
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.cache.radix_tree import RadixTree as PythonRadixTree

BS = 4


@pytest.fixture(params=BACKENDS)
def make(request):
    """The allocator class of one backend; the tree follows the allocator."""
    try:
        return allocator_class(request.param)
    except ImportError as exc:
        pytest.skip(str(exc))


def _cpp():
    try:
        from miniserve.cache import _minicore
    except ImportError as exc:
        pytest.skip(str(exc))
    return _minicore


def _tree(make, num_blocks=32):
    a = make(num_blocks, BS)
    return radix_tree_class(a)(a), a


def _cache(tree: RadixTree, a: BlockAllocator, tokens: list[int]) -> list[int]:
    """Allocate blocks for the whole blocks of ``tokens``, insert them, drop the caller's reference."""
    n = len(tokens) // BS
    blocks = a.allocate(n)
    tree.insert(tokens[: n * BS], blocks)
    a.free(blocks)
    return blocks


def test_match_empty_and_partial_block(make):
    t, a = _tree(make)
    assert t.match([1, 2, 3, 4, 5]) == (t.root, [])
    blocks = _cache(t, a, list(range(1, 9)))  # 2 blocks
    assert t.match([1, 2, 3])[1] == []  # less than a block
    assert t.match([1, 2, 3, 4, 5, 6, 7])[1] == blocks[:1]  # second block incomplete
    assert t.match(list(range(1, 12)))[1] == blocks
    assert t.match([1, 2, 3, 5])[1] == []  # differs inside the first block
    t.check_invariants()


def test_match_splits_at_block_boundary(make):
    t, a = _tree(make)
    blocks = _cache(t, a, list(range(1, 13)))  # one node of 3 blocks
    node, got = t.match([1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0])
    assert got == blocks[:2] and node.blocks == blocks[:2]
    (lower,) = node.children.values()
    assert lower.blocks == blocks[2:] and lower.tokens == [9, 10, 11, 12]
    assert t.num_cached_blocks == 3
    t.check_invariants()


def _shape(node):
    return (tuple(node.tokens), tuple(node.blocks), node.last_access, tuple(sorted((k, _shape(c)) for k, c in node.children.items())))


@pytest.mark.parametrize("seed", range(20))
def test_prefix_len_agrees_with_match_and_changes_nothing(make, seed):
    """The read-only lookup schedulers rank requests with: the same length as ``match`` finds, and
    no split node or access time left behind (ranking must not affect eviction)."""
    rng = random.Random(seed)
    t, a = _tree(make, 64)
    for _ in range(4):
        base = [rng.randrange(1, 4) for _ in range(rng.randint(0, 20))]
        _cache(t, a, base + [rng.randrange(1, 4) for _ in range(rng.randint(0, 12))])
    for _ in range(10):
        q = [rng.randrange(1, 4) for _ in range(rng.randint(0, 30))]
        before, clock = _shape(t.root), t._clock
        n = t.prefix_len(q)
        assert _shape(t.root) == before and t._clock == clock
        assert n == len(t.match(q)[1]) * BS
    t.check_invariants()


@pytest.mark.parametrize("seed", range(10))
def test_prefix_lens_is_prefix_len_of_each_slice(make, seed):
    """The batched lookup the cache-aware order uses: the same answer as one ``prefix_len`` per
    sliced sequence, for lists and other sequences, and limits beyond the end."""
    rng = random.Random(seed)
    t, a = _tree(make, 64)
    for _ in range(4):
        _cache(t, a, [rng.randrange(1, 4) for _ in range(rng.randint(0, 24))])
    seqs = [[rng.randrange(1, 4) for _ in range(rng.randint(0, 30))] for _ in range(12)]
    seqs[0] = tuple(seqs[0])
    limits = [rng.randint(0, 35) for _ in seqs]
    clock = t._clock
    assert t.prefix_lens(seqs, limits) == [t.prefix_len(list(s)[:n]) for s, n in zip(seqs, limits)]
    assert t._clock == clock
    with pytest.raises(ValueError):
        t.prefix_lens(seqs, limits[:-1])


def test_evicted_node_stays_readable(make):
    """A request can hold a node handle past its eviction (the C++ backend must not free it)."""
    t, a = _tree(make)
    _cache(t, a, [1, 1, 1, 1, 2, 2, 2, 2])
    node, _ = t.match([1, 1, 1, 1, 2, 2, 2, 2])
    t.clear()
    assert node.parent is None and node.tokens == [1, 1, 1, 1, 2, 2, 2, 2] and node.is_leaf


def test_insert_shares_prefix_and_ignores_duplicates(make):
    t, a = _tree(make)
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


def test_insert_rejects_partial_blocks(make):
    t, a = _tree(make)
    with pytest.raises(ValueError):
        t.insert([1, 2, 3, 4, 5], a.allocate(1))


def test_lru_eviction_order_and_parent_promotion(make):
    t, a = _tree(make, 8)
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


def test_locked_nodes_are_never_evicted(make):
    t, a = _tree(make, 8)
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


def test_split_keeps_locks(make):
    t, a = _tree(make)
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


def test_clear_refuses_locked_tree(make):
    t, a = _tree(make)
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


@pytest.mark.parametrize("seed", range(20))
def test_cpp_eviction_order_matches_the_reference(seed):
    """The C++ tree's eviction index and its full walk evict the same blocks in the same order
    as the Python reference, over random inserts, lookups (which move access times and split
    nodes) and evictions of a few blocks at a time."""
    mc = _cpp()
    rng = random.Random(seed)
    num_blocks = 40
    allocs = [mc.BlockAllocator(num_blocks, BS), mc.BlockAllocator(num_blocks, BS), None]
    from miniserve.cache.block_allocator import BlockAllocator as PyAlloc

    allocs[2] = PyAlloc(num_blocks, BS)
    trees = [mc.RadixTree(allocs[0]), mc.RadixTree(allocs[1], indexed_eviction=False), PythonRadixTree(allocs[2])]
    assert trees[0].indexed_eviction and not trees[1].indexed_eviction
    base = [5, 6, 7, 8]
    for _ in range(300):
        op = rng.random()
        if op < 0.5:
            q = (base if rng.random() < 0.5 else []) + [rng.randrange(1, 4) for _ in range(rng.randint(1, 3) * BS)]
            q = q[: len(q) // BS * BS]
            got = [t.match(q) for t in trees]
            assert len({tuple(b) for _, b in got}) == 1
            for t, (node, _) in zip(trees, got):
                t.lock(node)
            need = len(q) // BS - len(got[0][1])
            short = need - allocs[0].num_free
            if short > 0:
                assert len({t.evict(short) for t in trees}) == 1
            for t, (node, _) in zip(trees, got):
                t.unlock(node)
            if need <= allocs[0].num_free:
                for t, al, (_, have) in zip(trees, allocs, got):
                    fresh = al.allocate(need)
                    t.insert(q, list(have) + fresh)
                    al.free(fresh)
        elif op < 0.8:
            q = [rng.randrange(1, 4) for _ in range(2 * BS)]
            for t in trees:
                t.match(q)
        else:
            k = rng.randint(1, 4)
            assert len({t.evict(k) for t in trees}) == 1
        refs = [[al.refcount(b) for b in range(num_blocks)] for al in allocs]
        assert refs[0] == refs[1] == refs[2]
        assert len({t._clock for t in trees}) == 1
        for t in trees:
            t.check_invariants()


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


def test_manager_acquire_leaves_one_token_to_compute(make):
    kv = KVCacheManager(make(16, BS))
    x = _Req(0, list(range(1, 9)))  # exactly 2 blocks
    kv.acquire(x)
    x.cache.append_tokens(8)
    kv.release(x)
    y = _Req(1, list(range(1, 9)))  # the same prompt: only the first block may be reused
    assert kv.acquire(y) == 4
    z = _Req(2, list(range(1, 10)))
    assert kv.acquire(z) == 8
    kv.check_invariants()


def test_manager_cached_prefix_lens_matches_one_by_one(make):
    kv = KVCacheManager(make(32, BS))
    x = _Req(0, list(range(1, 17)))
    kv.acquire(x)
    x.cache.append_tokens(16)
    kv.release(x)
    reqs = [_Req(1, list(range(1, 17))), _Req(2, list(range(1, 9)), [9, 10, 11, 12, 13]), _Req(3, [7, 7, 7, 7, 7]), _Req(4, [])]
    reqs[3].output_ids = [1]
    assert kv.cached_prefix_lens(reqs) == [kv.cached_prefix_len(r) for r in reqs] == [12, 12, 0, 0]
    assert KVCacheManager(make(4, BS), radix=False).cached_prefix_lens(reqs) == [0, 0, 0, 0]


def test_manager_without_radix_is_plain_accounting(make):
    kv = KVCacheManager(make(4, BS), radix=False)
    x = _Req(0, list(range(1, 9)))
    assert kv.acquire(x) == 0
    x.cache.append_tokens(8)
    kv.release(x)
    assert kv.allocator.num_free == 4 and kv.num_available == 4


def test_manager_reserve_evicts_and_fails_cleanly(make):
    kv = KVCacheManager(make(4, BS))
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
def test_randomized_interleaving(make, seed):
    """Random requests over a few shared prefixes: acquire, extend, commit, release, reserve.

    A shadow map remembers which token each (block, offset) holds as tables are
    written; every cached path must spell tokens that its blocks actually hold."""
    rng = random.Random(seed)
    num_blocks = rng.choice([6, 12, 40])
    kv = KVCacheManager(make(num_blocks, BS))
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


@pytest.mark.parametrize("seed", range(20))
def test_order_by_cached_prefix_matches_sorting_by_prefix_len(make, seed):
    """The cache-aware admission order: by the cached prefix of prompt + output (all but the last
    token), longest first, ties in queue order. The C++ backend reads the two lists in place and
    sorts in the same call; the answer must be the same element for element."""
    from types import SimpleNamespace

    rng = random.Random(seed)
    t, a = _tree(make, num_blocks=256)
    prefixes = [[rng.randrange(1, 9) for _ in range(BS * rng.randrange(1, 6))] for _ in range(6)]
    for p in prefixes:
        _cache(t, a, p + [rng.randrange(1, 9) for _ in range(BS * rng.randrange(0, 3))])
    reqs = []
    for k in range(60):
        base = list(rng.choice(prefixes)) if rng.random() < 0.8 else []
        prompt = base[: rng.randrange(0, len(base) + 1)] + [rng.randrange(1, 9) for _ in range(rng.randrange(1, 12))]
        # Some requests carry output kept across a preemption, which splits the tokens in two lists.
        output = [rng.randrange(1, 9) for _ in range(rng.randrange(0, 6))] if rng.random() < 0.3 else []
        reqs.append(SimpleNamespace(rid=k, prompt_ids=prompt, output_ids=output))
    want = sorted(reqs, key=lambda r: -t.prefix_len((r.prompt_ids + r.output_ids)[:-1]))
    got = t.order_by_cached_prefix(reqs)
    assert [r.rid for r in got] == [r.rid for r in want]
    # The lookup changes nothing in the tree.
    t.check_invariants()


def test_order_by_cached_prefix_through_the_cache_manager(make):
    from types import SimpleNamespace

    t, a = _tree(make)
    _cache(t, a, list(range(1, 9)))
    kv = SimpleNamespace(tree=t)
    reqs = [SimpleNamespace(rid=0, prompt_ids=[9, 9, 9, 9, 9], output_ids=[]),
            SimpleNamespace(rid=1, prompt_ids=list(range(1, 10)), output_ids=[]),
            SimpleNamespace(rid=2, prompt_ids=list(range(1, 5)), output_ids=[5, 6]),
            SimpleNamespace(rid=3, prompt_ids=[1, 2, 3, 4, 7], output_ids=[])]
    got = KVCacheManager.order_by_cached_prefix(kv, reqs)
    # rid 1: 8 cached; rids 2 and 3: 4 each (queue order kept); rid 0: none.
    assert [r.rid for r in got] == [1, 2, 3, 0]
