// Per-request mapping from logical token positions to physical KV slots.
//
// Mirrors miniserve/cache/block_table.py. It lives on this side of the boundary
// together with the allocator because append_tokens() calls into the allocator
// on most steps and slot() is called once per token of a batch; splitting the
// two across the binding would add a round trip to each.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "minicore/block_allocator.hpp"

namespace minicore {

class BlockTable {
 public:
  // `prefix_blocks`: whole blocks shared with other holders (a cached prefix);
  // they are increfed. Duplicates are rejected.
  BlockTable(BlockAllocator* allocator, const std::vector<BlockId>& prefix_blocks);

  BlockAllocator* allocator() const { return allocator_; }
  int block_size() const { return allocator_->block_size(); }
  const std::vector<BlockId>& blocks() const { return blocks_; }
  int num_tokens() const { return num_tokens_; }
  int capacity() const { return static_cast<int>(blocks_.size()) * allocator_->block_size(); }

  // Tokens in the last block (1..block_size), or 0 for an empty table.
  int last_block_len() const {
    if (num_tokens_ == 0) return 0;
    return num_tokens_ - (static_cast<int>(blocks_.size()) - 1) * allocator_->block_size();
  }

  // Additional blocks required to append n tokens.
  int blocks_needed(int n) const;

  // Reserve room for n more tokens, allocating as needed. All or nothing.
  // Returns the newly allocated blocks.
  std::vector<BlockId> append_tokens(int n);

  std::int64_t slot(int pos) const {
    if (pos < 0 || pos >= num_tokens_) {
      throw BlockIndexError("position " + std::to_string(pos) + " out of range [0, " +
                            std::to_string(num_tokens_) + ")");
    }
    const int bs = allocator_->block_size();
    return static_cast<std::int64_t>(blocks_[static_cast<std::size_t>(pos / bs)]) * bs + pos % bs;
  }

  // Undo the append of the last n tokens, keeping the blocks. The blocks stay
  // because the caller is about to recompute those tokens into the same slots;
  // freeing and reallocating would hand out different ones.
  void rewind(int n);

  // Slots of the last n tokens, in order. The hot path: one call per request
  // per step instead of one binding crossing per token.
  std::vector<std::int64_t> tail_slots(int n) const;

  // Give every block back (dropping this table's reference) and empty the table.
  void release();

 private:
  BlockAllocator* allocator_;
  std::vector<BlockId> blocks_;
  int num_tokens_ = 0;
};

}  // namespace minicore
