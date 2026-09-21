#include "minicore/packing.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace minicore {

std::size_t pack_block_tables(const std::vector<const BlockTable*>& tables, int pad_rows, BlockId pad_block,
                              int pad_last, const PackedOut& out) {
  if (pad_rows < 0) throw std::invalid_argument("pad_rows must be >= 0");
  const std::size_t rows = tables.size() + static_cast<std::size_t>(pad_rows);
  std::size_t total = static_cast<std::size_t>(pad_rows);
  for (const BlockTable* t : tables) total += t->blocks().size();
  if (out.indptr_cap < rows + 1 || out.last_cap < rows || out.indices_cap < total) {
    throw std::invalid_argument("packing buffers too small: need " + std::to_string(rows + 1) + " indptr, " +
                                std::to_string(total) + " indices, " + std::to_string(rows) + " last entries");
  }
  std::size_t n = 0;
  std::size_t r = 0;
  out.indptr[0] = 0;
  for (const BlockTable* t : tables) {
    const std::vector<BlockId>& b = t->blocks();
    std::copy(b.begin(), b.end(), out.indices + n);
    n += b.size();
    out.last[r] = t->last_block_len();
    out.indptr[++r] = static_cast<std::int32_t>(n);
  }
  for (int i = 0; i < pad_rows; ++i) {
    out.indices[n++] = pad_block;
    out.last[r] = pad_last;
    out.indptr[++r] = static_cast<std::int32_t>(n);
  }
  return n;
}

}  // namespace minicore
