// gtest for the batch packing into FlashInfer's paged-KV layout.

#include "minicore/packing.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "minicore/block_allocator.hpp"
#include "minicore/block_table.hpp"

using minicore::BlockAllocator;
using minicore::BlockTable;
using minicore::PackedOut;

TEST(Packing, TablesThenPaddingRows) {
  BlockAllocator a(16, 4);
  BlockTable t0(&a, {}), t1(&a, {});
  t0.append_tokens(9);  // three blocks, one token in the last
  t1.append_tokens(4);  // one full block
  std::vector<std::int32_t> indptr(5, -1), indices(8, -1), last(4, -1);
  PackedOut out{indptr.data(), indptr.size(), indices.data(), indices.size(), last.data(), last.size()};
  const std::size_t n = minicore::pack_block_tables({&t0, &t1}, 2, 15, 3, out);
  ASSERT_EQ(n, 6u);
  EXPECT_EQ(std::vector<std::int32_t>(indptr.begin(), indptr.end()), (std::vector<std::int32_t>{0, 3, 4, 5, 6}));
  std::vector<std::int32_t> want(t0.blocks().begin(), t0.blocks().end());
  want.insert(want.end(), t1.blocks().begin(), t1.blocks().end());
  want.insert(want.end(), {15, 15});
  EXPECT_EQ(std::vector<std::int32_t>(indices.begin(), indices.begin() + 6), want);
  EXPECT_EQ(last, (std::vector<std::int32_t>{1, 4, 3, 3}));
}

TEST(Packing, RefusesBuffersThatAreTooSmall) {
  BlockAllocator a(16, 4);
  BlockTable t(&a, {});
  t.append_tokens(12);
  std::vector<std::int32_t> indptr(2), indices(2), last(1);
  PackedOut out{indptr.data(), indptr.size(), indices.data(), indices.size(), last.data(), last.size()};
  EXPECT_THROW(minicore::pack_block_tables({&t}, 0, 0, 1, out), std::invalid_argument);
}
