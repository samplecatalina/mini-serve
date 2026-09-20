#include "minicore/block_table.hpp"

namespace minicore {

BlockTable::BlockTable(BlockAllocator* allocator, const std::vector<BlockId>& prefix_blocks)
    : allocator_(allocator) {
  std::unordered_set<BlockId> seen(prefix_blocks.size() * 2);
  for (BlockId b : prefix_blocks) {
    if (!seen.insert(b).second) {
      std::string listed = "[";
      for (std::size_t i = 0; i < prefix_blocks.size(); ++i) {
        if (i) listed += ", ";
        listed += std::to_string(prefix_blocks[i]);
      }
      throw std::invalid_argument("duplicate block in prefix " + listed + "]");
    }
  }
  allocator_->incref(prefix_blocks);
  blocks_ = prefix_blocks;
  num_tokens_ = static_cast<int>(blocks_.size()) * allocator_->block_size();
}

int BlockTable::blocks_needed(int n) const {
  if (n < 0) throw std::invalid_argument("cannot append " + std::to_string(n) + " tokens");
  const int bs = allocator_->block_size();
  const int need = (num_tokens_ + n + bs - 1) / bs - static_cast<int>(blocks_.size());
  return need > 0 ? need : 0;
}

std::vector<BlockId> BlockTable::append_tokens(int n) {
  const int need = blocks_needed(n);
  const std::size_t before = blocks_.size();
  allocator_->allocate_into(need, blocks_);  // throws before touching blocks_ if short
  num_tokens_ += n;
  return std::vector<BlockId>(blocks_.begin() + static_cast<std::ptrdiff_t>(before), blocks_.end());
}

std::vector<std::int64_t> BlockTable::tail_slots(int n) const {
  if (n < 0 || n > num_tokens_) {
    throw BlockIndexError("tail of " + std::to_string(n) + " tokens out of range [0, " +
                          std::to_string(num_tokens_) + "]");
  }
  std::vector<std::int64_t> out;
  out.reserve(static_cast<std::size_t>(n));
  const int bs = allocator_->block_size();
  for (int pos = num_tokens_ - n; pos < num_tokens_; ++pos) {
    out.push_back(static_cast<std::int64_t>(blocks_[static_cast<std::size_t>(pos / bs)]) * bs +
                  pos % bs);
  }
  return out;
}

void BlockTable::release() {
  allocator_->free(blocks_);
  blocks_.clear();
  num_tokens_ = 0;
}

}  // namespace minicore
