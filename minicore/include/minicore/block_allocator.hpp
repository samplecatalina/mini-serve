// Fixed-size KV block allocator with per-block reference counts.
//
// Behaviour is defined by the Python reference implementation in
// miniserve/cache/block_allocator.py: the same call sequence must give the same
// block ids and raise the same kind of error. In particular the free list is a
// stack, so a fresh allocator hands out 0, 1, 2, ... and freed blocks are reused
// before untouched ones.
//
// Errors are reported as exception types that the bindings map onto the Python
// ones (OutOfBlocks, ValueError, IndexError).

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace minicore {

using BlockId = std::int32_t;

// Mapped to miniserve.cache.block_allocator.OutOfBlocks.
class OutOfBlocks : public std::runtime_error {
 public:
  explicit OutOfBlocks(const std::string& what) : std::runtime_error(what) {}
};

// Mapped to Python's IndexError; ValueError uses std::invalid_argument.
class BlockIndexError : public std::out_of_range {
 public:
  explicit BlockIndexError(const std::string& what) : std::out_of_range(what) {}
};

// Mapped to Python's AssertionError, because the reference implementation states
// its invariants with `assert` and the tests expect that type.
class InvariantError : public std::logic_error {
 public:
  explicit InvariantError(const std::string& what) : std::logic_error(what) {}
};

class BlockAllocator {
 public:
  BlockAllocator(int num_blocks, int block_size);

  int num_blocks() const { return num_blocks_; }
  int block_size() const { return block_size_; }
  int num_free() const { return static_cast<int>(free_.size()); }
  int num_used() const { return num_blocks_ - static_cast<int>(free_.size()); }

  bool can_allocate(int n) const { return n >= 0 && n <= static_cast<int>(free_.size()); }

  int refcount(BlockId block) const {
    check_id(block);
    return ref_[static_cast<std::size_t>(block)];
  }

  // n free blocks, reference count 1 each. All or nothing.
  std::vector<BlockId> allocate(int n);

  // Add a holder to each block; every block must already be allocated.
  // Validated in full before anything changes.
  void incref(const std::vector<BlockId>& blocks);

  // Drop one holder from each; blocks with no holders left return to the pool.
  // A block may appear more than once only if it has at least that many holders.
  // Validated in full before anything changes.
  void free(const std::vector<BlockId>& blocks);

  // Free stack and reference counts describe the same set; throws otherwise.
  void check_invariants() const;

  // --- used by BlockTable, which lives on this side of the boundary too ---
  // Append n freshly allocated blocks to `out` (all or nothing).
  void allocate_into(int n, std::vector<BlockId>& out);

 private:
  void check_id(BlockId block) const {
    if (block < 0 || block >= num_blocks_) {
      throw BlockIndexError("block id " + std::to_string(block) + " out of range [0, " +
                            std::to_string(num_blocks_) + ")");
    }
  }

  int num_blocks_;
  int block_size_;
  std::vector<BlockId> free_;  // stack: back() is handed out first
  std::vector<std::int32_t> ref_;  // 0 means free
  // Scratch for free(): block id -> releases in this call. Sized num_blocks_,
  // reset only where it was touched, so free() stays linear in its argument.
  mutable std::vector<std::int32_t> drops_;
};

}  // namespace minicore
