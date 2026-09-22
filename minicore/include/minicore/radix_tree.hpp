// Block-granular radix tree over KV blocks: the prefix cache.
//
// Behaviour is defined by the Python reference implementation in
// miniserve/cache/radix_tree.py: the same call sequence splits the same nodes,
// advances the same integer clock, and evicts the same blocks in the same order
// (ties in last access broken by creation order). tests/test_radix.py runs
// against both.
//
// One difference is internal: which leaves can be evicted is kept in an ordered
// index, (last_access, uid) -> node, updated wherever a node's leafness, lock or
// access time changes. Eviction then takes the minimum k times, O(k log n),
// instead of walking the whole tree to collect leaves and heapifying them,
// O(nodes). The full walk is kept behind `indexed_eviction = false` so that the
// two can be compared on the same tree; it gives the same order.
//
// The lookups are templates over a token source with size() and operator[], so
// that the bindings can read a Python list element by element, as far as the
// walk goes, instead of converting all of it first.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minicore/block_allocator.hpp"

namespace minicore {

using Token = std::int32_t;

// Children are keyed by the first block of their tokens.
struct BlockKeyHash {
  std::size_t operator()(const std::vector<Token>& k) const noexcept {
    std::uint64_t h = 1469598103934665603ull;  // FNV-1a over the token values
    for (Token t : k) {
      h ^= static_cast<std::uint32_t>(t);
      h *= 1099511628211ull;
    }
    return static_cast<std::size_t>(h);
  }
};

class RadixTree;

struct RadixNode : std::enable_shared_from_this<RadixNode> {
  std::vector<Token> tokens;
  std::vector<BlockId> blocks;
  std::unordered_map<std::vector<Token>, std::shared_ptr<RadixNode>, BlockKeyHash> children;
  RadixNode* parent = nullptr;  // owned by it; null for the root and for evicted nodes
  int lock = 0;
  std::uint64_t last_access = 0;
  std::uint64_t uid = 0;  // creation order: the tie-break for eviction
  bool indexed = false;   // currently in the tree's eviction index

  bool is_root() const { return parent == nullptr; }
  bool is_leaf() const { return children.empty(); }
};

using NodePtr = std::shared_ptr<RadixNode>;

class RadixTree {
 public:
  explicit RadixTree(BlockAllocator* allocator, bool indexed_eviction = true);

  BlockAllocator* allocator() const { return allocator_; }
  int block_size() const { return bs_; }
  const NodePtr& root() const { return root_; }
  int num_cached_blocks() const { return num_cached_blocks_; }
  int num_evictable() const { return num_evictable_; }
  std::uint64_t clock() const { return clock_; }
  bool indexed_eviction() const { return indexed_; }

  // Longest cached prefix of tokens[0, n) in whole blocks: its deepest node (a match
  // ending inside a node splits it there) and the blocks along the path.
  template <class Tokens>
  std::pair<NodePtr, std::vector<BlockId>> match(const Tokens& tokens) {
    auto [node, num_blocks] = walk(tokens);
    return {node, path_blocks(node.get(), num_blocks)};
  }

  // Tokens of that prefix, changing nothing (no split, no access time).
  template <class Tokens>
  int prefix_len(const Tokens& tokens) const;

  // Cache `blocks` as holding `tokens` (size() == blocks.size() * block_size). The part
  // already cached keeps the tree's blocks; the rest becomes a new leaf and the tree takes
  // a reference to each of its blocks. Returns the deepest node of the inserted prefix.
  template <class Tokens>
  NodePtr insert(const Tokens& tokens, const std::vector<BlockId>& blocks);

  void lock(RadixNode* node);
  void unlock(RadixNode* node);

  // Free at least num_blocks blocks from unlocked leaves, least recently used first, or as
  // many as there are. Returns the number freed.
  int evict(int num_blocks);
  void clear();

  std::vector<NodePtr> nodes() const;  // all but the root
  void check_invariants() const;

 private:
  using IndexKey = std::tuple<std::uint64_t, std::uint64_t, RadixNode*>;

  template <class Tokens>
  std::pair<NodePtr, int> walk(const Tokens& tokens);

  // Blocks [0, m) of child match tokens[pos, ...) up to `limit`; the first does, by its key.
  template <class Tokens>
  std::size_t matched_blocks(const RadixNode& child, const Tokens& tokens, std::size_t pos, std::size_t limit) const;

  template <class Tokens>
  const std::vector<Token>& read_key(const Tokens& tokens, std::size_t pos) const;

  NodePtr split(RadixNode* node, std::size_t m);
  std::vector<BlockId> path_blocks(const RadixNode* node, int num_blocks) const;
  NodePtr new_node(std::vector<Token> tokens, std::vector<BlockId> blocks, RadixNode* parent, int lock,
                   std::uint64_t last_access);
  std::vector<Token> key_of(const std::vector<Token>& tokens) const {
    return {tokens.begin(), tokens.begin() + bs_};
  }

  // Eviction index upkeep: remove before changing a node's key or state, re-add after.
  bool evictable(const RadixNode* n) const { return !n->is_root() && n->is_leaf() && n->lock == 0; }
  void unindex(RadixNode* n);
  void reindex(RadixNode* n);
  void touch(RadixNode* n, std::uint64_t t) {
    if (n->last_access == t) return;
    unindex(n);
    n->last_access = t;
    reindex(n);
  }
  void remove_leaf(RadixNode* node);  // free its blocks, detach it; for evict

  BlockAllocator* allocator_;
  int bs_;
  bool indexed_;
  NodePtr root_;
  int num_cached_blocks_ = 0;
  int num_evictable_ = 0;  // blocks of nodes with lock == 0
  std::uint64_t clock_ = 0;
  std::uint64_t next_uid_ = 0;
  std::set<IndexKey> index_;
  mutable std::vector<Token> key_buf_;  // scratch for the lookup key
};

// ------------------------------------------------------------------ templates

template <class Tokens>
const std::vector<Token>& RadixTree::read_key(const Tokens& tokens, std::size_t pos) const {
  for (int i = 0; i < bs_; ++i) key_buf_[static_cast<std::size_t>(i)] = tokens[pos + static_cast<std::size_t>(i)];
  return key_buf_;
}

template <class Tokens>
std::size_t RadixTree::matched_blocks(const RadixNode& child, const Tokens& tokens, std::size_t pos,
                                      std::size_t limit) const {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  std::size_t m = 1;
  while (m < child.blocks.size() && pos + (m + 1) * bs <= limit) {
    const Token* c = child.tokens.data() + m * bs;
    const std::size_t base = pos + m * bs;
    bool same = true;
    for (std::size_t i = 0; i < bs; ++i) {
      if (c[i] != tokens[base + i]) {
        same = false;
        break;
      }
    }
    if (!same) break;
    ++m;
  }
  return m;
}

template <class Tokens>
int RadixTree::prefix_len(const Tokens& tokens) const {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  const std::size_t limit = tokens.size() / bs * bs;
  const RadixNode* node = root_.get();
  std::size_t pos = 0;
  while (pos < limit) {
    auto it = node->children.find(read_key(tokens, pos));
    if (it == node->children.end()) break;
    const RadixNode& child = *it->second;
    const std::size_t m = matched_blocks(child, tokens, pos, limit);
    pos += m * bs;
    if (m < child.blocks.size()) break;
    node = &child;
  }
  return static_cast<int>(pos);
}

template <class Tokens>
std::pair<NodePtr, int> RadixTree::walk(const Tokens& tokens) {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  const std::size_t limit = tokens.size() / bs * bs;
  ++clock_;
  NodePtr node = root_;
  std::size_t pos = 0;
  while (pos < limit) {
    auto it = node->children.find(read_key(tokens, pos));
    if (it == node->children.end()) break;
    NodePtr child = it->second;
    const std::size_t m = matched_blocks(*child, tokens, pos, limit);
    if (m < child->blocks.size()) child = split(child.get(), m);
    touch(child.get(), clock_);
    node = std::move(child);
    pos += m * bs;
  }
  return {node, static_cast<int>(pos / bs)};
}

template <class Tokens>
NodePtr RadixTree::insert(const Tokens& tokens, const std::vector<BlockId>& blocks) {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  if (tokens.size() != blocks.size() * bs) {
    throw std::invalid_argument(std::to_string(tokens.size()) + " tokens for " + std::to_string(blocks.size()) +
                                " blocks of " + std::to_string(bs_));
  }
  auto [node, matched] = walk(tokens);
  if (static_cast<std::size_t>(matched) == blocks.size()) return node;
  std::vector<BlockId> fresh(blocks.begin() + matched, blocks.end());
  allocator_->incref(fresh);
  std::vector<Token> toks(tokens.size() - static_cast<std::size_t>(matched) * bs);
  for (std::size_t i = 0; i < toks.size(); ++i) toks[i] = tokens[static_cast<std::size_t>(matched) * bs + i];
  const int n = static_cast<int>(fresh.size());
  NodePtr leaf = new_node(std::move(toks), std::move(fresh), node.get(), 0, clock_);
  unindex(node.get());  // it is about to stop being a leaf
  node->children.emplace(key_of(leaf->tokens), leaf);
  reindex(node.get());
  reindex(leaf.get());
  num_cached_blocks_ += n;
  num_evictable_ += n;
  return leaf;
}

// A contiguous token range, for callers on this side of the boundary (and the gtest).
struct TokenSpan {
  const Token* data;
  std::size_t n;
  std::size_t size() const { return n; }
  Token operator[](std::size_t i) const { return data[i]; }
};

inline TokenSpan span_of(const std::vector<Token>& v) { return {v.data(), v.size()}; }

}  // namespace minicore
