#include "minicore/radix_tree.hpp"

#include <algorithm>
#include <queue>
#include <set>

namespace minicore {

RadixTree::RadixTree(BlockAllocator* allocator, bool indexed_eviction)
    : allocator_(allocator), bs_(allocator->block_size()), indexed_(indexed_eviction), key_buf_(static_cast<std::size_t>(bs_)) {
  root_ = new_node({}, {}, nullptr, 1, 0);  // never evicted
}

NodePtr RadixTree::new_node(std::vector<Token> tokens, std::vector<BlockId> blocks, RadixNode* parent, int lock,
                            std::uint64_t last_access) {
  auto n = std::make_shared<RadixNode>();
  n->tokens = std::move(tokens);
  n->blocks = std::move(blocks);
  n->parent = parent;
  n->lock = lock;
  n->last_access = last_access;
  n->uid = next_uid_++;
  return n;
}

void RadixTree::unindex(RadixNode* n) {
  if (n->indexed) {
    index_.erase({n->last_access, n->uid, n});
    n->indexed = false;
  }
}

void RadixTree::reindex(RadixNode* n) {
  if (!indexed_) return;
  const bool want = evictable(n);
  if (want && !n->indexed) {
    index_.insert({n->last_access, n->uid, n});
    n->indexed = true;
  } else if (!want && n->indexed) {
    unindex(n);
  }
}

NodePtr RadixTree::split(RadixNode* node, std::size_t m) {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  RadixNode* parent = node->parent;
  NodePtr upper = new_node({node->tokens.begin(), node->tokens.begin() + static_cast<std::ptrdiff_t>(m * bs)},
                           {node->blocks.begin(), node->blocks.begin() + static_cast<std::ptrdiff_t>(m)}, parent,
                           node->lock, node->last_access);
  auto slot = parent->children.find(key_of(node->tokens));
  NodePtr self = std::move(slot->second);  // same key: the upper half starts with the same block
  slot->second = upper;
  node->tokens.erase(node->tokens.begin(), node->tokens.begin() + static_cast<std::ptrdiff_t>(m * bs));
  node->blocks.erase(node->blocks.begin(), node->blocks.begin() + static_cast<std::ptrdiff_t>(m));
  node->parent = upper.get();
  upper->children.emplace(key_of(node->tokens), std::move(self));
  // The lower half keeps its leafness, lock and access time; the upper one has a child.
  return upper;
}

std::vector<BlockId> RadixTree::path_blocks(const RadixNode* node, int num_blocks) const {
  std::vector<const RadixNode*> parts;
  for (; !node->is_root(); node = node->parent) parts.push_back(node);
  std::vector<BlockId> out;
  out.reserve(static_cast<std::size_t>(num_blocks));
  for (auto it = parts.rbegin(); it != parts.rend(); ++it) out.insert(out.end(), (*it)->blocks.begin(), (*it)->blocks.end());
  if (static_cast<int>(out.size()) != num_blocks) throw InvariantError("path blocks disagree with the match length");
  return out;
}

void RadixTree::lock(RadixNode* node) {
  for (; !node->is_root(); node = node->parent) {
    if (node->lock == 0) num_evictable_ -= static_cast<int>(node->blocks.size());
    ++node->lock;
    reindex(node);
  }
}

void RadixTree::unlock(RadixNode* node) {
  for (; !node->is_root(); node = node->parent) {
    if (node->lock <= 0) throw std::invalid_argument("unlock of an unlocked node");
    --node->lock;
    if (node->lock == 0) num_evictable_ += static_cast<int>(node->blocks.size());
    reindex(node);
  }
}

void RadixTree::remove_leaf(RadixNode* node) {
  unindex(node);
  allocator_->free(node->blocks);
  const int n = static_cast<int>(node->blocks.size());
  num_cached_blocks_ -= n;
  num_evictable_ -= n;
  RadixNode* parent = node->parent;
  auto it = parent->children.find(key_of(node->tokens));
  NodePtr keep = std::move(it->second);  // a caller may still hold the node; it stays readable
  parent->children.erase(it);
  node->parent = nullptr;
  reindex(parent);
}

int RadixTree::evict(int num_blocks) {
  int freed = 0;
  if (indexed_) {
    while (freed < num_blocks && !index_.empty()) {
      RadixNode* node = std::get<2>(*index_.begin());
      freed += static_cast<int>(node->blocks.size());
      remove_leaf(node);  // re-indexes the parent, which may now be an evictable leaf
    }
    return freed;
  }
  // The reference algorithm: collect every unlocked leaf, heapify, pop; a parent left
  // as an unlocked leaf joins the heap.
  using Entry = std::pair<std::pair<std::uint64_t, std::uint64_t>, RadixNode*>;
  auto later = [](const Entry& a, const Entry& b) { return a.first > b.first; };
  std::vector<Entry> leaves;
  std::vector<RadixNode*> stack;
  for (auto& [k, c] : root_->children) stack.push_back(c.get());
  while (!stack.empty()) {
    RadixNode* n = stack.back();
    stack.pop_back();
    if (n->is_leaf()) {
      if (n->lock == 0) leaves.push_back({{n->last_access, n->uid}, n});
    } else {
      for (auto& [k, c] : n->children) stack.push_back(c.get());
    }
  }
  std::priority_queue<Entry, std::vector<Entry>, decltype(later)> heap(later, std::move(leaves));
  while (freed < num_blocks && !heap.empty()) {
    RadixNode* node = heap.top().second;
    heap.pop();
    freed += static_cast<int>(node->blocks.size());
    RadixNode* parent = node->parent;
    remove_leaf(node);
    if (evictable(parent)) heap.push({{parent->last_access, parent->uid}, parent});
  }
  return freed;
}

void RadixTree::clear() {
  if (num_evictable_ != num_cached_blocks_) throw std::runtime_error("cannot clear a tree with locked nodes");
  evict(num_cached_blocks_);
}

std::vector<NodePtr> RadixTree::nodes() const {
  std::vector<NodePtr> out;
  std::vector<const RadixNode*> stack{root_.get()};
  while (!stack.empty()) {
    const RadixNode* n = stack.back();
    stack.pop_back();
    for (auto& [k, c] : n->children) {
      out.push_back(c);
      stack.push_back(c.get());
    }
  }
  return out;
}

namespace {
void require(bool ok, const char* what) {
  if (!ok) throw InvariantError(what);
}
}  // namespace

void RadixTree::check_invariants() const {
  const std::size_t bs = static_cast<std::size_t>(bs_);
  int cached = 0, evictable_blocks = 0;
  std::set<IndexKey> want;
  for (const NodePtr& n : nodes()) {
    require(!n->blocks.empty() && n->tokens.size() == n->blocks.size() * bs, "node is not whole blocks");
    auto it = n->parent->children.find(key_of(n->tokens));
    require(it != n->parent->children.end() && it->second.get() == n.get(), "child key disagrees with tokens");
    require(n->lock >= 0, "negative lock");
    require(n->parent->is_root() || n->parent->lock >= n->lock, "child locked more than its parent");
    cached += static_cast<int>(n->blocks.size());
    const int ref = allocator_->refcount(n->blocks[0]);
    for (BlockId b : n->blocks) {
      require(allocator_->refcount(b) == ref, "blocks of one node have different holders");
    }
    if (n->lock == 0) {
      evictable_blocks += static_cast<int>(n->blocks.size());
      require(ref == 1, "unlocked node's blocks held elsewhere");
    }
    if (evictable(n.get())) want.insert({n->last_access, n->uid, n.get()});
    require(n->indexed == (indexed_ && evictable(n.get())), "node's index flag out of date");
  }
  require(cached == num_cached_blocks_, "num_cached_blocks out of date");
  require(evictable_blocks == num_evictable_, "num_evictable out of date");
  if (indexed_) require(want == index_, "eviction index disagrees with the tree");
}

}  // namespace minicore
