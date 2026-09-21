// A batch of block tables in the layout FlashInfer's paged-KV kernels read.
//
// Every step hands the attention kernels three int32 arrays: where each
// sequence's page list starts (indptr), the page lists themselves (indices),
// and how many tokens the last page of each sequence holds. Building them is
// a walk over every block of every sequence in the batch, which is tens of
// thousands of entries at a large batch. It lives here, next to the tables,
// because a table's blocks crossing into Python one list at a time costs more
// than the walk itself.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "minicore/block_table.hpp"

namespace minicore {

struct PackedOut {
  std::int32_t* indptr;     // rows + 1 entries
  std::size_t indptr_cap;
  std::int32_t* indices;    // one entry per page
  std::size_t indices_cap;
  std::int32_t* last;       // rows entries
  std::size_t last_cap;
};

// Writes `tables` followed by `pad_rows` padding rows, each a single page
// `pad_block` holding `pad_last` tokens, into `out`. Returns the number of
// page indices written. Throws std::invalid_argument if a buffer is too small
// (nothing is guaranteed about its contents then).
std::size_t pack_block_tables(const std::vector<const BlockTable*>& tables, int pad_rows, BlockId pad_block,
                              int pad_last, const PackedOut& out);

}  // namespace minicore
