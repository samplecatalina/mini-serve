#include "minicore/block_allocator.hpp"

#include <algorithm>

namespace minicore {

BlockAllocator::BlockAllocator(int num_blocks, int block_size)
    : num_blocks_(num_blocks), block_size_(block_size) {
  if (num_blocks < 1 || block_size < 1) {
    throw std::invalid_argument("num_blocks and block_size must be positive, got " +
                                std::to_string(num_blocks) + ", " + std::to_string(block_size));
  }
  free_.resize(static_cast<std::size_t>(num_blocks));
  // Reversed, so that back() (popped first) is block 0: same order as the
  // reference implementation's `list(range(num_blocks - 1, -1, -1))`.
  for (int i = 0; i < num_blocks; ++i) free_[static_cast<std::size_t>(i)] = num_blocks - 1 - i;
  ref_.assign(static_cast<std::size_t>(num_blocks), 0);
  drops_.assign(static_cast<std::size_t>(num_blocks), 0);
}

void BlockAllocator::allocate_into(int n, std::vector<BlockId>& out) {
  if (n < 0) throw std::invalid_argument("cannot allocate " + std::to_string(n) + " blocks");
  if (n > static_cast<int>(free_.size())) {
    throw OutOfBlocks("requested " + std::to_string(n) + " blocks, " + std::to_string(free_.size()) +
                      " free");
  }
  out.reserve(out.size() + static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    BlockId b = free_.back();
    free_.pop_back();
    ref_[static_cast<std::size_t>(b)] = 1;
    out.push_back(b);
  }
}

std::vector<BlockId> BlockAllocator::allocate(int n) {
  std::vector<BlockId> out;
  allocate_into(n, out);
  return out;
}

void BlockAllocator::incref(const std::vector<BlockId>& blocks) {
  for (BlockId b : blocks) {
    check_id(b);
    if (ref_[static_cast<std::size_t>(b)] == 0) {
      throw std::invalid_argument("incref of free block " + std::to_string(b));
    }
  }
  for (BlockId b : blocks) ref_[static_cast<std::size_t>(b)] += 1;
}

void BlockAllocator::free(const std::vector<BlockId>& blocks) {
  for (BlockId b : blocks) check_id(b);
  // Count releases per block, then check each against its reference count.
  // Touched entries are reset below whatever happens, so `drops_` is all zeros
  // on entry and on exit, including when this throws.
  for (BlockId b : blocks) drops_[static_cast<std::size_t>(b)] += 1;
  BlockId offender = -1;
  std::int32_t releases = 0;
  for (BlockId b : blocks) {
    std::int32_t k = drops_[static_cast<std::size_t>(b)];
    if (k > 0 && ref_[static_cast<std::size_t>(b)] < k) {
      offender = b;
      releases = k;
      break;
    }
  }
  for (BlockId b : blocks) drops_[static_cast<std::size_t>(b)] = 0;
  if (offender >= 0) {
    throw std::invalid_argument("free of block " + std::to_string(offender) + ": " +
                                std::to_string(releases) + " releases but reference count " +
                                std::to_string(ref_[static_cast<std::size_t>(offender)]));
  }
  for (BlockId b : blocks) {
    if (--ref_[static_cast<std::size_t>(b)] == 0) free_.push_back(b);
  }
}

void BlockAllocator::check_invariants() const {
  std::vector<char> seen(static_cast<std::size_t>(num_blocks_), 0);
  for (BlockId b : free_) {
    if (seen[static_cast<std::size_t>(b)]) {
      throw InvariantError("duplicate block in free stack");
    }
    seen[static_cast<std::size_t>(b)] = 1;
  }
  for (int b = 0; b < num_blocks_; ++b) {
    bool is_free = seen[static_cast<std::size_t>(b)] != 0;
    if (is_free != (ref_[static_cast<std::size_t>(b)] == 0)) {
      throw InvariantError("free stack disagrees with reference counts");
    }
    if (ref_[static_cast<std::size_t>(b)] < 0) throw InvariantError("negative reference count");
  }
}

}  // namespace minicore
