// gtest for the prefix cache tree. The Python suite (tests/test_radix.py) is the
// behavioural contract for both backends; these cover the C++ side on its own,
// including the eviction index, which Python cannot see.

#include "minicore/radix_tree.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

#include "minicore/block_allocator.hpp"

using minicore::BlockAllocator;
using minicore::BlockId;
using minicore::RadixTree;
using minicore::span_of;
using minicore::Token;

namespace {

constexpr int BS = 4;

// Allocate blocks for the whole blocks of `tokens`, insert them, drop the caller's reference.
std::vector<BlockId> cache(RadixTree& t, BlockAllocator& a, const std::vector<Token>& tokens) {
  const int n = static_cast<int>(tokens.size()) / BS;
  auto blocks = a.allocate(n);
  std::vector<Token> whole(tokens.begin(), tokens.begin() + n * BS);
  t.insert(span_of(whole), blocks);
  a.free(blocks);
  return blocks;
}

std::vector<Token> seq(std::initializer_list<Token> v) { return v; }

}  // namespace

TEST(RadixTree, MatchSplitsAtBlockBoundary) {
  BlockAllocator a(32, BS);
  RadixTree t(&a);
  auto blocks = cache(t, a, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  auto q = seq({1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0});
  auto [node, got] = t.match(span_of(q));
  EXPECT_EQ(got, std::vector<BlockId>(blocks.begin(), blocks.begin() + 2));
  ASSERT_EQ(node->children.size(), 1u);
  EXPECT_EQ(node->children.begin()->second->tokens, seq({9, 10, 11, 12}));
  EXPECT_EQ(t.num_cached_blocks(), 3);
  t.check_invariants();
}

TEST(RadixTree, PrefixLenChangesNothing) {
  BlockAllocator a(32, BS);
  RadixTree t(&a);
  cache(t, a, {1, 2, 3, 4, 5, 6, 7, 8});
  const auto clock = t.clock();
  auto q = seq({1, 2, 3, 4, 9, 9, 9, 9});
  EXPECT_EQ(t.prefix_len(span_of(q)), 4);
  EXPECT_EQ(t.clock(), clock);
  EXPECT_EQ(t.nodes().size(), 1u);  // not split
}

TEST(RadixTree, LruOrderAndParentPromotion) {
  BlockAllocator a(8, BS);
  RadixTree t(&a);
  auto x = cache(t, a, {1, 1, 1, 1, 2, 2, 2, 2});
  auto y = cache(t, a, {1, 1, 1, 1, 3, 3, 3, 3});
  auto z = cache(t, a, {4, 4, 4, 4});
  auto q = seq({1, 1, 1, 1, 2, 2, 2, 2});
  t.match(span_of(q));
  EXPECT_EQ(t.evict(1), 1);
  EXPECT_EQ(a.refcount(y[1]), 0);
  EXPECT_EQ(t.evict(1), 1);
  EXPECT_EQ(a.refcount(z[0]), 0);
  EXPECT_EQ(a.refcount(x[0]), 1);
  EXPECT_EQ(t.evict(2), 2);
  EXPECT_EQ(a.num_free(), 8);
  EXPECT_EQ(t.evict(5), 0);
  t.check_invariants();
}

TEST(RadixTree, LockedNodesAreNeverEvicted) {
  BlockAllocator a(8, BS);
  RadixTree t(&a);
  cache(t, a, {1, 1, 1, 1, 2, 2, 2, 2});
  auto q = seq({1, 1, 1, 1, 2, 2, 2, 2});
  auto [node, blocks] = t.match(span_of(q));
  a.incref(blocks);
  t.lock(node.get());
  EXPECT_EQ(t.num_evictable(), 0);
  EXPECT_EQ(t.evict(8), 0);
  t.check_invariants();
  t.unlock(node.get());
  EXPECT_THROW(t.unlock(node.get()), std::invalid_argument);
  a.free(blocks);
  t.check_invariants();
}

TEST(RadixTree, EvictedNodeStaysReadable) {
  BlockAllocator a(8, BS);
  RadixTree t(&a);
  cache(t, a, {1, 1, 1, 1});
  auto q = seq({1, 1, 1, 1});
  auto held = t.match(span_of(q)).first;
  t.clear();
  EXPECT_EQ(held->parent, nullptr);
  EXPECT_EQ(held->tokens, q);
}

TEST(RadixTree, InsertRejectsPartialBlocks) {
  BlockAllocator a(8, BS);
  RadixTree t(&a);
  auto q = seq({1, 2, 3, 4, 5});
  EXPECT_THROW(t.insert(span_of(q), a.allocate(1)), std::invalid_argument);
}

// The index and the reference full walk evict the same blocks in the same order.
TEST(RadixTree, IndexedEvictionMatchesFullWalk) {
  for (unsigned seed = 0; seed < 20; ++seed) {
    BlockAllocator a1(40, BS), a2(40, BS);
    RadixTree fast(&a1, true), walk(&a2, false);
    std::mt19937 rng(seed);
    auto rand_tokens = [&](int blocks) {
      std::vector<Token> v;
      std::vector<Token> base = {1, 2, 3, 4, 5, 6, 7, 8};  // a shared first block for some
      if (rng() % 2) v.insert(v.end(), base.begin(), base.begin() + BS);
      while (static_cast<int>(v.size()) < blocks * BS) v.push_back(static_cast<Token>(rng() % 3 + 1));
      return v;
    };
    for (int step = 0; step < 200; ++step) {
      const unsigned op = rng() % 10;
      if (op < 6) {
        // As the engine does: match and lock the cached part, then evict for the rest.
        auto q = rand_tokens(1 + static_cast<int>(rng() % 3));
        auto [n1, m1] = fast.match(span_of(q));
        auto [n2, m2] = walk.match(span_of(q));
        ASSERT_EQ(m1, m2);
        fast.lock(n1.get());
        walk.lock(n2.get());
        const int need = static_cast<int>(q.size()) / BS - static_cast<int>(m1.size());
        if (need > a1.num_free()) {
          ASSERT_EQ(fast.evict(need - a1.num_free()), walk.evict(need - a2.num_free()));
        }
        fast.unlock(n1.get());
        walk.unlock(n2.get());
        if (need > a1.num_free()) continue;
        auto b1 = a1.allocate(need), b2 = a2.allocate(need);
        ASSERT_EQ(b1, b2);
        m1.insert(m1.end(), b1.begin(), b1.end());
        m2.insert(m2.end(), b2.begin(), b2.end());
        fast.insert(span_of(q), m1);
        walk.insert(span_of(q), m2);
        a1.free(b1);
        a2.free(b2);
      } else if (op < 8) {
        auto q = rand_tokens(2);
        fast.match(span_of(q));
        walk.match(span_of(q));
      } else {
        const int k = 1 + static_cast<int>(rng() % 4);
        ASSERT_EQ(fast.evict(k), walk.evict(k));
      }
      ASSERT_EQ(a1.num_free(), a2.num_free());
      for (BlockId b = 0; b < 40; ++b) ASSERT_EQ(a1.refcount(b), a2.refcount(b)) << "seed " << seed << " step " << step;
      fast.check_invariants();
      walk.check_invariants();
    }
  }
}

TEST(RadixTree, OrderByPrefixLensIsStableAndLongestFirst) {
  const std::vector<int> lens{16, 0, 32, 16, 0, 32};
  const auto order = minicore::order_by_prefix_lens(lens);
  EXPECT_EQ(order, (std::vector<std::size_t>{2, 5, 0, 3, 1, 4}));
  EXPECT_TRUE(minicore::order_by_prefix_lens({}).empty());
}
