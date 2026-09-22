"""Request-level KV cache operations over the block pool and the prefix cache.

The scheduler budgets against this object and the model runner allocates
through it; neither touches the radix tree directly.

Life of a request's KV:

- ``acquire`` (admission): look up the longest cached prefix of the request's
  sequence in whole blocks, excluding its last token (at least one token must
  be computed, for its logits); start the request's block table with those
  blocks and lock their path so they cannot be evicted.
- the model runner extends the table as tokens are computed.
- ``commit`` (after a prefill): insert the table's full blocks into the tree
  so later requests with the same prefix hit while this one is still running.
- ``release`` (finish, abort, preemption): commit, unlock, and drop the
  table's references. Its full blocks stay in the tree; a preempted request
  therefore finds its own computed tokens there when it is admitted again.
- ``abandon`` undoes an ``acquire`` whose admission was then refused.

Blocks held only by the tree count as available: ``reserve`` evicts them
(least recently used first) when the free pool alone is short.

With the prefix cache disabled every operation degrades to plain block
accounting: empty prefixes, nothing inserted.
"""

from __future__ import annotations

from miniserve.cache.backend import radix_tree_class
from miniserve.cache.block_allocator import BlockAllocator, OutOfBlocks


class KVCacheManager:
    def __init__(self, allocator: BlockAllocator, radix: bool = True):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.tree = radix_tree_class(allocator)(allocator) if radix else None

    @property
    def radix(self) -> bool:
        return self.tree is not None

    def set_radix(self, enabled: bool) -> None:
        """Switch the prefix cache on or off (clearing it). Only while no request holds KV."""
        if self.tree is not None:
            self.tree.clear()
        self.tree = radix_tree_class(self.allocator)(self.allocator) if enabled else None

    @property
    def num_available(self) -> int:
        """Blocks that can be handed out: free ones plus those only the prefix cache holds."""
        return self.allocator.num_free + (self.tree.num_evictable if self.tree else 0)

    def acquire(self, req) -> int:
        """Start ``req``'s block table with its longest cached prefix. Returns the cached token count."""
        if req.cache is not None:
            raise RuntimeError(f"request {req.rid} already holds a cache")
        node, blocks = None, []
        if self.tree is not None:
            node, blocks = self.tree.match(self._tokens(req)[: req.seq_len - 1])
            self.tree.lock(node)
        req.cache = self.allocator.new_table(blocks)
        req.cache_node = node
        req.num_cached_tokens = len(blocks) * self.block_size
        return req.num_cached_tokens

    def cached_prefix_len(self, req) -> int:
        """Tokens ``acquire`` would find cached for ``req`` now; changes nothing."""
        if self.tree is None:
            return 0
        return self.tree.prefix_len(self._tokens(req)[: req.seq_len - 1])

    def cached_prefix_lens(self, reqs) -> list[int]:
        """``cached_prefix_len`` of every request, in one lookup call. A request that has no output
        yet (the usual waiting one) passes its prompt as is: no concatenation, no slice."""
        if self.tree is None:
            return [0] * len(reqs)
        seqs = [r.prompt_ids if not r.output_ids else r.prompt_ids + r.output_ids for r in reqs]
        return self.tree.prefix_lens(seqs, [r.seq_len - 1 for r in reqs])

    def abandon(self, req) -> None:
        """Undo ``acquire``: nothing was computed, nothing is inserted."""
        if req.cache_node is not None:
            self.tree.unlock(req.cache_node)
        req.cache.release()
        req.cache, req.cache_node, req.num_cached_tokens = None, None, 0

    def commit(self, req) -> None:
        """Insert the full blocks whose KV ``req`` has computed, and move its lock to cover them."""
        if self.tree is None or req.cache is None:
            return
        table = req.cache
        n = table.num_tokens // self.block_size
        node = self.tree.insert(self._tokens(req)[: n * self.block_size], table.blocks[:n])
        self.tree.lock(node)
        if req.cache_node is not None:
            self.tree.unlock(req.cache_node)
        req.cache_node = node

    def release(self, req) -> None:
        """Give up ``req``'s KV, keeping its full blocks in the prefix cache."""
        if req.cache is None:
            return
        self.commit(req)
        if req.cache_node is not None:
            self.tree.unlock(req.cache_node)
        req.cache.release()
        req.cache, req.cache_node = None, None

    def reserve(self, num_blocks: int) -> None:
        """Make ``num_blocks`` blocks free, evicting from the prefix cache if needed."""
        short = num_blocks - self.allocator.num_free
        if short > 0 and self.tree is not None:
            self.tree.evict(short)
        if num_blocks > self.allocator.num_free:
            raise OutOfBlocks(f"need {num_blocks} free KV blocks, {self.allocator.num_free} after eviction")

    def check_invariants(self) -> None:
        self.allocator.check_invariants()
        if self.tree is not None:
            self.tree.check_invariants()

    def num_idle_blocks(self) -> int:
        """Blocks not held by any request: free plus cached. Equals ``num_blocks`` when idle."""
        return self.allocator.num_free + (self.tree.num_cached_blocks if self.tree else 0)

    @staticmethod
    def _tokens(req) -> list[int]:
        return req.prompt_ids + req.output_ids
