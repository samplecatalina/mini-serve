"""Block-granular radix tree over KV blocks: the prefix cache.

Each edge holds a run of whole blocks: ``tokens`` (a multiple of
``block_size`` long) and the physical ``blocks`` whose KV holds them. A path
from the root spells a token prefix; the blocks along it hold that prefix's
KV, which depends only on the tokens and their positions, so any request whose
sequence starts with the same tokens can reuse those blocks.

Two counts protect blocks, and they must agree:

- The allocator's per-block reference count. The tree is one holder of each
  of its blocks; every request's block table is another. A block returns to
  the free pool only when nobody holds it, so dropping the tree's reference
  (eviction) never pulls a block out from under a running request.
- ``lock`` on each node: how many running requests use the node's path. The
  tree adds or removes 1 along the whole path, as mini-sglang does with its
  node ``ref_count``, which keeps the number of evictable blocks up to date
  without scanning. A node with ``lock == 0`` is used by no request, so each
  of its blocks is held by the tree alone (reference count exactly 1);
  ``check_invariants`` verifies this. (The converse need not hold: a request
  whose tokens were already cached under other blocks locks those nodes while
  keeping its own duplicate blocks.)

Eviction frees least recently used leaves with ``lock == 0``; a parent whose
last child is evicted becomes a candidate in turn. Candidates are collected by
walking the whole tree, as in mini-sglang: simple, and O(nodes) per eviction.
(The C++ backend keeps them in an ordered index instead and evicts in the same
order; see minicore/include/minicore/radix_tree.hpp.)
Recency is a counter incremented on every match and insert, so eviction order
is reproducible.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence

from miniserve.cache.block_allocator import BlockAllocator


class RadixNode:
    __slots__ = ("tokens", "blocks", "children", "parent", "lock", "last_access", "uid")

    _uids = itertools.count()

    def __init__(self, tokens: list[int], blocks: list[int], parent: RadixNode | None, lock: int, last_access: int):
        self.tokens = tokens
        self.blocks = blocks
        self.children: dict[tuple[int, ...], RadixNode] = {}
        self.parent = parent
        self.lock = lock
        self.last_access = last_access
        self.uid = next(self._uids)  # tie-break for the eviction heap

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not self.children


class RadixTree:
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.root = RadixNode([], [], None, lock=1, last_access=0)  # never evicted
        self.num_cached_blocks = 0
        self.num_evictable = 0  # blocks of nodes with lock == 0
        self._clock = 0

    # ------------------------------------------------------------------ lookup

    def match(self, tokens: Sequence[int]) -> tuple[RadixNode, list[int]]:
        """Longest cached prefix of ``tokens`` in whole blocks: its deepest node and its blocks.

        A match ending inside a node splits it there, so the returned node's
        path covers exactly the matched blocks.
        """
        node, num_blocks = self._walk(tokens)
        return node, self._path_blocks(node, num_blocks)

    def prefix_len(self, tokens: Sequence[int]) -> int:
        """Tokens of the longest cached prefix of ``tokens`` in whole blocks, without changing the
        tree (no split, no access time): a lookup that only ranks requests must not affect eviction."""
        return self._prefix_len(tokens, len(tokens))

    def prefix_lens(self, seqs: Sequence[Sequence[int]], limits: Sequence[int]) -> list[int]:
        """``prefix_len(seqs[i][:limits[i]])`` for every i, without copying the slices. One call for
        a whole waiting queue: the C++ backend answers it in one crossing of the binding."""
        if len(seqs) != len(limits):
            raise ValueError("prefix_lens: one limit per sequence")
        return [self._prefix_len(s, min(n, len(s))) for s, n in zip(seqs, limits)]

    def _prefix_len(self, tokens: Sequence[int], n: int) -> int:
        bs = self.block_size
        limit = n // bs * bs
        node, pos = self.root, 0
        while pos < limit:
            child = node.children.get(tuple(tokens[pos : pos + bs]))
            if child is None:
                break
            m = 1
            while m < len(child.blocks) and pos + (m + 1) * bs <= limit:
                if child.tokens[m * bs : (m + 1) * bs] != list(tokens[pos + m * bs : pos + (m + 1) * bs]):
                    break
                m += 1
            pos += m * bs
            if m < len(child.blocks):
                break
            node = child
        return pos

    def _walk(self, tokens: Sequence[int]) -> tuple[RadixNode, int]:
        bs = self.block_size
        limit = len(tokens) // bs * bs
        self._clock += 1
        node, pos = self.root, 0
        while pos < limit:
            child = node.children.get(tuple(tokens[pos : pos + bs]))
            if child is None:
                break
            m = 1  # blocks of the child that match; the first does, by its key
            while m < len(child.blocks) and pos + (m + 1) * bs <= limit:
                if child.tokens[m * bs : (m + 1) * bs] != list(tokens[pos + m * bs : pos + (m + 1) * bs]):
                    break
                m += 1
            if m < len(child.blocks):
                child = self._split(child, m)
            child.last_access = self._clock
            node, pos = child, pos + m * bs
        return node, pos // bs

    def _path_blocks(self, node: RadixNode, num_blocks: int) -> list[int]:
        parts = []
        while not node.is_root:
            parts.append(node.blocks)
            node = node.parent
        out = [b for part in reversed(parts) for b in part]
        assert len(out) == num_blocks
        return out

    def _split(self, node: RadixNode, m: int) -> RadixNode:
        """Cut ``node`` after ``m`` blocks; the new upper node takes its place and is returned."""
        bs = self.block_size
        upper = RadixNode(node.tokens[: m * bs], node.blocks[:m], node.parent, node.lock, node.last_access)
        node.parent.children[self._key(upper.tokens)] = upper
        node.tokens, node.blocks = node.tokens[m * bs :], node.blocks[m:]
        node.parent = upper
        upper.children[self._key(node.tokens)] = node
        return upper

    def _key(self, tokens: Sequence[int]) -> tuple[int, ...]:
        return tuple(tokens[: self.block_size])

    # ------------------------------------------------------------------ insertion

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> RadixNode:
        """Cache ``blocks`` as holding ``tokens`` (``len(tokens) == len(blocks) * block_size``).

        The part already cached keeps the tree's existing blocks (the caller's
        duplicates are not taken); the rest becomes a new leaf, and the tree
        takes a reference to each of its blocks. Returns the deepest node of
        the inserted prefix.
        """
        bs = self.block_size
        if len(tokens) != len(blocks) * bs:
            raise ValueError(f"{len(tokens)} tokens for {len(blocks)} blocks of {bs}")
        node, matched = self._walk(tokens)
        if matched == len(blocks):
            return node
        new_blocks = list(blocks[matched:])
        self.allocator.incref(new_blocks)
        leaf = RadixNode(list(tokens[matched * bs :]), new_blocks, node, lock=0, last_access=self._clock)
        node.children[self._key(leaf.tokens)] = leaf
        self.num_cached_blocks += len(new_blocks)
        self.num_evictable += len(new_blocks)
        return leaf

    # ------------------------------------------------------------------ locking

    def lock(self, node: RadixNode) -> None:
        while not node.is_root:
            if node.lock == 0:
                self.num_evictable -= len(node.blocks)
            node.lock += 1
            node = node.parent

    def unlock(self, node: RadixNode) -> None:
        while not node.is_root:
            if node.lock <= 0:
                raise ValueError("unlock of an unlocked node")
            node.lock -= 1
            if node.lock == 0:
                self.num_evictable += len(node.blocks)
            node = node.parent

    # ------------------------------------------------------------------ eviction

    def evict(self, num_blocks: int) -> int:
        """Free at least ``num_blocks`` blocks from unlocked leaves, least recently used first,
        or as many as there are. Returns the number freed."""
        heap = [(n.last_access, n.uid, n) for n in self._nodes() if n.is_leaf and n.lock == 0]
        heapq.heapify(heap)
        freed = 0
        while freed < num_blocks and heap:
            _, _, node = heapq.heappop(heap)
            self.allocator.free(node.blocks)
            freed += len(node.blocks)
            self.num_cached_blocks -= len(node.blocks)
            self.num_evictable -= len(node.blocks)
            parent = node.parent
            del parent.children[self._key(node.tokens)]
            node.parent = None
            if not parent.is_root and parent.is_leaf and parent.lock == 0:
                heapq.heappush(heap, (parent.last_access, parent.uid, parent))
        return freed

    def clear(self) -> None:
        """Evict everything. Only valid while no request uses the tree."""
        if self.num_evictable != self.num_cached_blocks:
            raise RuntimeError("cannot clear a tree with locked nodes")
        self.evict(self.num_cached_blocks)

    # ------------------------------------------------------------------ checks

    def _nodes(self):
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def check_invariants(self) -> None:
        """Counts match a full walk, keys match tokens, locks agree with allocator reference counts."""
        bs = self.block_size
        cached = evictable = 0
        for node in self._nodes():
            assert node.blocks and len(node.tokens) == len(node.blocks) * bs, "node is not whole blocks"
            assert node.parent.children.get(self._key(node.tokens)) is node, "child key disagrees with tokens"
            assert node.lock >= 0, "negative lock"
            assert node.parent.is_root or node.parent.lock >= node.lock, "child locked more than its parent"
            cached += len(node.blocks)
            refs = {self.allocator.refcount(b) for b in node.blocks}
            assert len(refs) == 1, f"blocks of one node have different holders: reference counts {refs}"
            if node.lock == 0:
                evictable += len(node.blocks)
                assert refs == {1}, f"unlocked node's blocks held elsewhere: reference counts {refs}"
        assert cached == self.num_cached_blocks, "num_cached_blocks out of date"
        assert evictable == self.num_evictable, "num_evictable out of date"
