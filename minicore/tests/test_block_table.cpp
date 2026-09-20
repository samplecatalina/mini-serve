// gtest for the per-request table.

#include "minicore/block_table.hpp"

#include <gtest/gtest.h>

#include "minicore/block_allocator.hpp"

using minicore::BlockAllocator;
using minicore::BlockId;
using minicore::BlockTable;
using minicore::OutOfBlocks;

TEST(BlockTable, EmptyTable) {
  BlockAllocator a(8, 4);
  BlockTable t(&a, {});
  EXPECT_EQ(t.num_tokens(), 0);
  EXPECT_EQ(t.capacity(), 0);
  EXPECT_EQ(t.last_block_len(), 0);
  EXPECT_TRUE(t.blocks().empty());
}

TEST(BlockTable, AppendAllocatesOnlyWhenTheLastBlockIsFull) {
  BlockAllocator a(8, 4);
  BlockTable t(&a, {});
  EXPECT_EQ(t.blocks_needed(4), 1);
  EXPECT_EQ(t.append_tokens(4).size(), 1u);
  EXPECT_EQ(t.last_block_len(), 4);
  EXPECT_EQ(t.blocks_needed(0), 0);
  EXPECT_EQ(t.append_tokens(1).size(), 1u);  // spills into a second block
  EXPECT_EQ(t.num_tokens(), 5);
  EXPECT_EQ(t.last_block_len(), 1);
  EXPECT_EQ(t.blocks_needed(3), 0);  // room in the second block
}

TEST(BlockTable, AppendIsAllOrNothing) {
  BlockAllocator a(2, 4);
  BlockTable t(&a, {});
  EXPECT_THROW(t.append_tokens(12), OutOfBlocks);
  EXPECT_EQ(t.num_tokens(), 0);
  EXPECT_TRUE(t.blocks().empty());
  EXPECT_EQ(a.num_free(), 2);
  a.check_invariants();
}

TEST(BlockTable, SlotsFollowTheBlockOrder) {
  BlockAllocator a(8, 4);
  a.allocate(2);      // take 0 and 1 so the table gets 2, 3
  BlockTable t(&a, {});
  t.append_tokens(6);
  EXPECT_EQ(t.blocks(), (std::vector<BlockId>{2, 3}));
  EXPECT_EQ(t.slot(0), 8);
  EXPECT_EQ(t.slot(3), 11);
  EXPECT_EQ(t.slot(4), 12);
  EXPECT_THROW(t.slot(6), minicore::BlockIndexError);
  EXPECT_THROW(t.slot(-1), minicore::BlockIndexError);
}

TEST(BlockTable, TailSlotsMatchSlotOneByOne) {
  BlockAllocator a(8, 4);
  BlockTable t(&a, {});
  t.append_tokens(10);
  auto tail = t.tail_slots(3);
  ASSERT_EQ(tail.size(), 3u);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(tail[static_cast<std::size_t>(i)], t.slot(7 + i));
  EXPECT_TRUE(t.tail_slots(0).empty());
  EXPECT_THROW(t.tail_slots(11), minicore::BlockIndexError);
}

TEST(BlockTable, RewindKeepsTheBlocks) {
  BlockAllocator a(8, 4);
  BlockTable t(&a, {});
  t.append_tokens(5);
  const auto slot4 = t.slot(4);
  t.rewind(1);
  EXPECT_EQ(t.num_tokens(), 4);
  EXPECT_EQ(t.blocks().size(), 2u);  // the second block is still held
  EXPECT_EQ(a.num_free(), 6);
  EXPECT_EQ(t.blocks_needed(1), 0);
  EXPECT_TRUE(t.append_tokens(1).empty());
  EXPECT_EQ(t.slot(4), slot4);  // and the token lands in the same slot
  EXPECT_THROW(t.rewind(6), minicore::BlockIndexError);
  EXPECT_THROW(t.rewind(-1), minicore::BlockIndexError);
  a.check_invariants();
}

TEST(BlockTable, SharedPrefixIsIncrefedAndReleasedIndependently) {
  BlockAllocator a(8, 4);
  BlockTable owner(&a, {});
  owner.append_tokens(8);
  BlockTable t(&a, owner.blocks());
  EXPECT_EQ(a.refcount(owner.blocks()[0]), 2);
  EXPECT_EQ(t.num_tokens(), 8);
  t.release();
  EXPECT_EQ(a.refcount(owner.blocks()[0]), 1);
  EXPECT_EQ(t.num_tokens(), 0);
  owner.release();
  EXPECT_EQ(a.num_free(), 8);
  a.check_invariants();
}

TEST(BlockTable, DuplicateOrFreePrefixBlocksAreRejected) {
  BlockAllocator a(8, 4);
  auto blocks = a.allocate(2);
  EXPECT_THROW(BlockTable(&a, {blocks[0], blocks[0]}), std::invalid_argument);
  EXPECT_THROW(BlockTable(&a, {7}), std::invalid_argument);  // block 7 is free
  EXPECT_EQ(a.refcount(blocks[0]), 1);
  a.check_invariants();
}

TEST(BlockTable, NegativeAppendIsAValueError) {
  BlockAllocator a(8, 4);
  BlockTable t(&a, {});
  EXPECT_THROW(t.blocks_needed(-1), std::invalid_argument);
  EXPECT_THROW(t.append_tokens(-1), std::invalid_argument);
}
