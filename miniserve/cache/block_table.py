"""Per-request mapping from logical token positions to physical KV slots.

Token ``pos`` of a request lives in physical block ``blocks[pos // block_size]``
at offset ``pos % block_size``; its slot in a pool flattened to
``[num_blocks * block_size]`` is ``slot(pos)``.

A table may start with blocks shared with other holders (a cached prefix).
Only full blocks can be shared, so every token appended afterwards goes into a
block this table holds exclusively and no copy-on-write is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

from miniserve.cache.block_allocator import BlockAllocator


class BlockTable:
    def __init__(self, allocator: BlockAllocator, prefix_blocks: Sequence[int] = ()):
        self.allocator = allocator
        self.block_size = allocator.block_size
        if len(set(prefix_blocks)) != len(prefix_blocks):
            raise ValueError(f"duplicate block in prefix {list(prefix_blocks)}")
        allocator.incref(prefix_blocks)
        self.blocks: list[int] = list(prefix_blocks)
        self.num_tokens = len(self.blocks) * self.block_size

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.block_size

    @property
    def last_block_len(self) -> int:
        """Tokens in the last block (1..block_size), or 0 for an empty table."""
        if self.num_tokens == 0:
            return 0
        return self.num_tokens - (len(self.blocks) - 1) * self.block_size

    def blocks_needed(self, n: int) -> int:
        """Additional blocks required to append ``n`` tokens."""
        if n < 0:
            raise ValueError(f"cannot append {n} tokens")
        return max(0, -(-(self.num_tokens + n) // self.block_size) - len(self.blocks))

    def append_tokens(self, n: int) -> list[int]:
        """Reserve room for ``n`` more tokens, allocating blocks as needed. All or nothing.

        Returns the newly allocated blocks.
        """
        new = self.allocator.allocate(self.blocks_needed(n))
        self.blocks += new
        self.num_tokens += n
        return new

    @property
    def num_blocks(self) -> int:
        """Blocks the table holds. Cheaper than ``len(table.blocks)``, which a
        C++ backend would have to build a list for."""
        return len(self.blocks)

    def slot(self, pos: int) -> int:
        if not 0 <= pos < self.num_tokens:
            raise IndexError(f"position {pos} out of range [0, {self.num_tokens})")
        return self.blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def rewind(self, n: int) -> None:
        """Undo the append of the last ``n`` tokens, keeping the blocks.

        The blocks stay because the caller is about to recompute those tokens
        into the same slots; freeing and reallocating would hand out different ones.
        """
        if not 0 <= n <= self.num_tokens:
            raise IndexError(f"cannot rewind {n} of {self.num_tokens} tokens")
        self.num_tokens -= n

    def tail_slots(self, n: int) -> list[int]:
        """Slots of the last ``n`` tokens, in order: one call per request per step
        instead of one per token."""
        if not 0 <= n <= self.num_tokens:
            raise IndexError(f"tail of {n} tokens out of range [0, {self.num_tokens}]")
        bs = self.block_size
        return [self.blocks[p // bs] * bs + p % bs for p in range(self.num_tokens - n, self.num_tokens)]

    def release(self) -> None:
        """Give every block back (dropping this table's reference) and empty the table."""
        self.allocator.free(self.blocks)
        self.blocks = []
        self.num_tokens = 0
