// Python bindings for the C++ block bookkeeping, built in place as
// miniserve/cache/_minicore*.so and selected by miniserve/cache/backend.py.
//
// The module is a drop-in replacement for miniserve.cache.block_allocator and
// miniserve.cache.block_table: same method names, same block ids for the same
// call sequence, same exception types. tests/test_block_allocator.py runs
// against both backends, which is what keeps the two honest.
//
// OutOfBlocks is not defined here. It is imported from the Python reference
// module at import time, so that a caller's `except OutOfBlocks` matches
// whichever backend raised, and so that KVCacheManager (which raises it from
// Python) and this module agree on one class.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <string>
#include <utility>

#include "minicore/block_allocator.hpp"
#include "minicore/block_table.hpp"
#include "minicore/packing.hpp"

namespace py = pybind11;
using minicore::BlockAllocator;
using minicore::BlockId;
using minicore::BlockTable;

namespace {

// The Python OutOfBlocks class, resolved once at import.
py::handle out_of_blocks;

// A writable one-dimensional int32 buffer (a numpy view of a pinned tensor, in practice).
std::pair<std::int32_t*, std::size_t> int32_buffer(const py::buffer& b, const char* name) {
  py::buffer_info info = b.request(true);
  if (info.ndim != 1 || info.itemsize != 4 || (info.format != "i" && info.format != "<i" && info.format != "=i") ||
      info.strides[0] != 4) {
    throw std::invalid_argument(std::string(name) + " must be a contiguous one-dimensional int32 buffer");
  }
  return {static_cast<std::int32_t*>(info.ptr), static_cast<std::size_t>(info.shape[0])};
}

}  // namespace

PYBIND11_MODULE(_minicore, m) {
  m.doc() = "C++ KV block bookkeeping: reference-counted block allocator and per-request tables.";

  out_of_blocks = py::object(py::module_::import("miniserve.cache.block_allocator").attr("OutOfBlocks")).release();
  m.attr("OutOfBlocks") = out_of_blocks;

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const minicore::OutOfBlocks& e) {
      PyErr_SetString(out_of_blocks.ptr(), e.what());
    } catch (const minicore::BlockIndexError& e) {
      PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const minicore::InvariantError& e) {
      PyErr_SetString(PyExc_AssertionError, e.what());
    } catch (const std::invalid_argument& e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });

  py::class_<BlockTable>(m, "BlockTable")
      .def(py::init([](BlockAllocator& a, const std::vector<BlockId>& prefix) {
             return new BlockTable(&a, prefix);
           }),
           py::arg("allocator"), py::arg("prefix_blocks") = std::vector<BlockId>{},
           // The table borrows the allocator; keep it alive at least as long.
           py::keep_alive<1, 2>())
      .def_property_readonly("allocator", [](BlockTable& t) { return t.allocator(); },
                             py::return_value_policy::reference_internal)
      .def_property_readonly("block_size", &BlockTable::block_size)
      .def_property_readonly("blocks", &BlockTable::blocks)
      .def_property_readonly("num_tokens", &BlockTable::num_tokens)
      .def_property_readonly("num_blocks", [](const BlockTable& t) { return t.blocks().size(); })
      .def_property_readonly("capacity", &BlockTable::capacity)
      .def_property_readonly("last_block_len", &BlockTable::last_block_len)
      .def("blocks_needed", &BlockTable::blocks_needed, py::arg("n"))
      .def("append_tokens", &BlockTable::append_tokens, py::arg("n"))
      .def("slot", &BlockTable::slot, py::arg("pos"))
      .def("rewind", &BlockTable::rewind, py::arg("n"))
      .def("tail_slots", &BlockTable::tail_slots, py::arg("n"))
      .def("release", &BlockTable::release);

  py::class_<BlockAllocator>(m, "BlockAllocator")
      .def(py::init<int, int>(), py::arg("num_blocks"), py::arg("block_size"))
      .def_property_readonly("num_blocks", &BlockAllocator::num_blocks)
      .def_property_readonly("block_size", &BlockAllocator::block_size)
      .def_property_readonly("num_free", &BlockAllocator::num_free)
      .def_property_readonly("num_used", &BlockAllocator::num_used)
      .def("can_allocate", &BlockAllocator::can_allocate, py::arg("n"))
      .def("refcount", &BlockAllocator::refcount, py::arg("block"))
      .def("allocate", &BlockAllocator::allocate, py::arg("n"))
      .def("incref", &BlockAllocator::incref, py::arg("blocks"))
      .def("free", &BlockAllocator::free, py::arg("blocks"))
      .def("check_invariants", &BlockAllocator::check_invariants)
      .def(
          "new_table",
          [](BlockAllocator& a, const std::vector<BlockId>& prefix) {
            return new BlockTable(&a, prefix);
          },
          py::arg("prefix_blocks") = std::vector<BlockId>{}, py::keep_alive<0, 1>(),
          "A block table over this allocator. Callers use this instead of naming a table class, "
          "so that a table always matches the backend of its allocator.");

  m.def(
      "pack_block_tables",
      [](const std::vector<const BlockTable*>& tables, int pad_rows, BlockId pad_block, int pad_last,
         const py::buffer& indptr, const py::buffer& indices, const py::buffer& last) {
        auto [ip, ipn] = int32_buffer(indptr, "indptr");
        auto [ix, ixn] = int32_buffer(indices, "indices");
        auto [la, lan] = int32_buffer(last, "last");
        return minicore::pack_block_tables(tables, pad_rows, pad_block, pad_last, {ip, ipn, ix, ixn, la, lan});
      },
      py::arg("tables"), py::arg("pad_rows"), py::arg("pad_block"), py::arg("pad_last"), py::arg("indptr"),
      py::arg("indices"), py::arg("last"),
      "Write a batch of tables (then pad_rows single-page padding rows) in FlashInfer's paged-KV "
      "layout into three int32 buffers; returns the number of page indices. One crossing per batch.");

  m.attr("BACKEND") = "cpp";
}
