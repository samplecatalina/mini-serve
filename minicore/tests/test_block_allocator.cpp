// gtest for the allocator. These cover what the Python parametrized suite
// cannot reach from the binding: internal invariants after a throw, and the
// exception types themselves.

#include "minicore/block_allocator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <set>

using minicore::BlockAllocator;
using minicore::BlockId;
using minicore::OutOfBlocks;

TEST(BlockAllocator, FreshAllocatorHandsOutAscendingIds) {
  BlockAllocator a(8, 16);
  EXPECT_EQ(a.num_blocks(), 8);
  EXPECT_EQ(a.block_size(), 16);
  EXPECT_EQ(a.num_free(), 8);
  EXPECT_EQ(a.num_used(), 0);
  EXPECT_EQ(a.allocate(3), (std::vector<BlockId>{0, 1, 2}));
  a.check_invariants();
}

TEST(BlockAllocator, RejectsNonPositiveSizes) {
  EXPECT_THROW(BlockAllocator(0, 16), std::invalid_argument);
  EXPECT_THROW(BlockAllocator(8, 0), std::invalid_argument);
  EXPECT_THROW(BlockAllocator(-1, 16), std::invalid_argument);
}

TEST(BlockAllocator, FreedBlocksAreReusedLastInFirstOut) {
  BlockAllocator a(8, 16);
  a.allocate(5);
  a.free({1, 3});
  EXPECT_EQ(a.allocate(1), (std::vector<BlockId>{3}));
  EXPECT_EQ(a.allocate(3), (std::vector<BlockId>{1, 5, 6}));
  a.check_invariants();
}

TEST(BlockAllocator, AllocationIsAllOrNothing) {
  BlockAllocator a(4, 16);
  a.allocate(3);
  EXPECT_THROW(a.allocate(2), OutOfBlocks);
  EXPECT_EQ(a.num_free(), 1);  // nothing was taken
  EXPECT_EQ(a.refcount(3), 0);
  a.check_invariants();
}

TEST(BlockAllocator, NegativeAllocationIsAValueError) {
  BlockAllocator a(4, 16);
  EXPECT_THROW(a.allocate(-1), std::invalid_argument);
}

TEST(BlockAllocator, IncrefAndFreeTrackHolders) {
  BlockAllocator a(4, 16);
  auto blocks = a.allocate(2);
  a.incref(blocks);
  EXPECT_EQ(a.refcount(blocks[0]), 2);
  a.free(blocks);
  EXPECT_EQ(a.refcount(blocks[0]), 1);
  EXPECT_EQ(a.num_free(), 2);
  a.free(blocks);
  EXPECT_EQ(a.num_free(), 4);
  a.check_invariants();
}

TEST(BlockAllocator, FreeValidatesTheWholeCallBeforeChangingAnything) {
  BlockAllocator a(4, 16);
  auto blocks = a.allocate(2);  // {0, 1}, one holder each
  // Two releases of block 0, which has one holder: the whole call must fail.
  EXPECT_THROW(a.free({blocks[0], blocks[0], blocks[1]}), std::invalid_argument);
  EXPECT_EQ(a.refcount(blocks[0]), 1);
  EXPECT_EQ(a.refcount(blocks[1]), 1);
  EXPECT_EQ(a.num_free(), 2);
  a.check_invariants();
  // And the scratch counters it used are clean, so the next call is unaffected.
  a.free({blocks[0], blocks[1]});
  EXPECT_EQ(a.num_free(), 4);
  a.check_invariants();
}

TEST(BlockAllocator, RepeatedFreeIsAllowedWithEnoughHolders) {
  BlockAllocator a(4, 16);
  auto blocks = a.allocate(1);
  a.incref(blocks);
  a.free({blocks[0], blocks[0]});
  EXPECT_EQ(a.num_free(), 4);
  a.check_invariants();
}

TEST(BlockAllocator, IncrefOfAFreeBlockIsRejected) {
  BlockAllocator a(4, 16);
  EXPECT_THROW(a.incref({0}), std::invalid_argument);
  a.allocate(1);
  a.incref({0});
  EXPECT_THROW(a.incref({0, 1}), std::invalid_argument);  // 1 is free
  EXPECT_EQ(a.refcount(0), 2);                            // and 0 was left alone
  a.check_invariants();
}

TEST(BlockAllocator, OutOfRangeIdsAreIndexErrors) {
  BlockAllocator a(4, 16);
  EXPECT_THROW(a.refcount(4), minicore::BlockIndexError);
  EXPECT_THROW(a.refcount(-1), minicore::BlockIndexError);
  EXPECT_THROW(a.free({4}), minicore::BlockIndexError);
  EXPECT_THROW(a.incref({-1}), minicore::BlockIndexError);
}

TEST(BlockAllocator, NoExternalFragmentationUnderRandomChurn) {
  BlockAllocator a(64, 16);
  std::mt19937 rng(7);
  std::vector<std::vector<BlockId>> held;
  for (int i = 0; i < 20000; ++i) {
    if (!held.empty() && (held.size() > 20 || rng() % 2 == 0)) {
      std::size_t k = rng() % held.size();
      a.free(held[k]);
      held.erase(held.begin() + static_cast<std::ptrdiff_t>(k));
    } else {
      int n = static_cast<int>(rng() % 8);
      if (a.can_allocate(n)) held.push_back(a.allocate(n));
    }
  }
  for (auto& h : held) a.free(h);
  a.check_invariants();
  // Every block is free, and they can all be taken in one call.
  EXPECT_EQ(a.num_free(), 64);
  auto all = a.allocate(64);
  EXPECT_EQ(std::set<BlockId>(all.begin(), all.end()).size(), 64u);
}
