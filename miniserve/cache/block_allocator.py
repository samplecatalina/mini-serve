"""Fixed-size KV block allocator with per-block reference counts.

The KV pool is divided into ``num_blocks`` blocks of ``block_size`` tokens.
This class only tracks which blocks are free and how many holders each used
block has; it owns no tensors.

Allocation order is part of the contract, so that other implementations can
be checked block id for block id: a fresh allocator hands out 0, 1, 2, ...;
freed blocks go on top of a stack and are reused first (LIFO).
"""

from __future__ import annotations

from collections.abc import Iterable


class OutOfBlocks(RuntimeError):
    pass


BACKEND = "python"


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks < 1 or block_size < 1:
            raise ValueError(f"num_blocks and block_size must be positive, got {num_blocks}, {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = list(range(num_blocks - 1, -1, -1))  # stack: pop() yields 0 first
        self._ref = [0] * num_blocks  # 0 means free

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int) -> bool:
        return 0 <= n <= len(self._free)

    def refcount(self, block: int) -> int:
        self._check_id(block)
        return self._ref[block]

    def allocate(self, n: int) -> list[int]:
        """Take ``n`` free blocks (reference count 1 each). All or nothing."""
        if n < 0:
            raise ValueError(f"cannot allocate {n} blocks")
        if n > len(self._free):
            raise OutOfBlocks(f"requested {n} blocks, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        for b in out:
            self._ref[b] = 1
        return out

    def incref(self, blocks: Iterable[int]) -> None:
        """Add a holder to each block. Blocks must be allocated."""
        blocks = list(blocks)
        for b in blocks:
            self._check_id(b)
            if self._ref[b] == 0:
                raise ValueError(f"incref of free block {b}")
        for b in blocks:
            self._ref[b] += 1

    def free(self, blocks: Iterable[int]) -> None:
        """Drop one holder from each block; blocks with no holders left return to the pool.

        The whole call is validated before anything changes. A block may appear
        more than once only if it has at least that many holders.
        """
        blocks = list(blocks)
        drops: dict[int, int] = {}
        for b in blocks:
            self._check_id(b)
            drops[b] = drops.get(b, 0) + 1
        for b, k in drops.items():
            if self._ref[b] < k:
                raise ValueError(f"free of block {b}: {k} releases but reference count {self._ref[b]}")
        for b in blocks:
            self._ref[b] -= 1
            if self._ref[b] == 0:
                self._free.append(b)

    def new_table(self, prefix_blocks: Iterable[int] = ()) -> "BlockTable":
        """A block table over this allocator, optionally starting with a shared prefix.

        Callers use this instead of naming a table class, so that a table always
        matches the backend of the allocator it is built on.
        """
        from miniserve.cache.block_table import BlockTable

        return BlockTable(self, list(prefix_blocks))

    def check_invariants(self) -> None:
        """Free stack and reference counts describe the same set; raises AssertionError otherwise."""
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate block in free stack"
        assert free == {b for b, r in enumerate(self._ref) if r == 0}, "free stack disagrees with reference counts"
        assert all(r >= 0 for r in self._ref), "negative reference count"

    def _check_id(self, block: int) -> None:
        if not 0 <= block < self.num_blocks:
            raise IndexError(f"block id {block} out of range [0, {self.num_blocks})")
