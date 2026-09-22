// Python bindings for the C++ block bookkeeping, built in place as
// miniserve/cache/_minicore*.so and selected by miniserve/cache/backend.py.
//
// The module is a drop-in replacement for miniserve.cache.block_allocator,
// miniserve.cache.block_table and miniserve.cache.radix_tree: same method names,
// same block ids and eviction order for the same call sequence, same exception
// types. tests/test_block_allocator.py and tests/test_radix.py run against both
// backends, which is what keeps the two honest.
//
// OutOfBlocks is not defined here. It is imported from the Python reference
// module at import time, so that a caller's `except OutOfBlocks` matches
// whichever backend raised, and so that KVCacheManager (which raises it from
// Python) and this module agree on one class.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <string>
#include <utility>

#include "minicore/block_allocator.hpp"
#include "minicore/block_table.hpp"
#include "minicore/packing.hpp"
#include "minicore/radix_tree.hpp"

namespace py = pybind11;
using minicore::BlockAllocator;
using minicore::BlockId;
using minicore::BlockTable;
using minicore::NodePtr;
using minicore::RadixNode;
using minicore::RadixTree;
using minicore::Token;

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

// A Python list of token ids read in place, one element at a time, only as far as the
// tree walk goes: a lookup that stops in the first block should not pay for converting
// a thousand-token prompt.
struct ListTokens {
  PyObject* list;
  std::size_t n;
  std::size_t size() const { return n; }
  Token operator[](std::size_t i) const {
    const long v = PyLong_AsLong(PyList_GET_ITEM(list, static_cast<Py_ssize_t>(i)));
    if (v == -1 && PyErr_Occurred()) throw py::error_already_set();
    return static_cast<Token>(v);
  }
};

// Call f with the first `limit` tokens of `seq` (all of them if limit < 0): read in place
// from a list, converted once from any other sequence.
template <class F>
auto with_tokens(const py::handle& seq, long limit, F&& f) {
  if (PyList_Check(seq.ptr())) {
    std::size_t n = static_cast<std::size_t>(PyList_GET_SIZE(seq.ptr()));
    if (limit >= 0) n = std::min(n, static_cast<std::size_t>(limit));
    return f(ListTokens{seq.ptr(), n});
  }
  auto v = seq.cast<std::vector<Token>>();
  if (limit >= 0 && static_cast<std::size_t>(limit) < v.size()) v.resize(static_cast<std::size_t>(limit));
  return f(minicore::span_of(v));
}

py::object node_or_none(RadixNode* n) {
  if (n == nullptr) return py::none();
  return py::cast(n->shared_from_this());
}

}  // namespace

PYBIND11_MODULE(_minicore, m) {
  m.doc() = "C++ KV block bookkeeping: reference-counted block allocator, per-request tables, prefix cache tree.";

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

  py::class_<RadixNode, NodePtr>(m, "RadixNode")
      .def_property_readonly("tokens", [](const RadixNode& n) { return n.tokens; })
      .def_property_readonly("blocks", [](const RadixNode& n) { return n.blocks; })
      .def_property_readonly("children",
                             [](const RadixNode& n) {
                               py::dict d;
                               for (auto& [k, c] : n.children) d[py::tuple(py::cast(k))] = c;
                               return d;
                             })
      .def_property_readonly("parent", [](const RadixNode& n) { return node_or_none(n.parent); })
      .def_property_readonly("lock", [](const RadixNode& n) { return n.lock; })
      .def_property_readonly("last_access", [](const RadixNode& n) { return n.last_access; })
      .def_property_readonly("uid", [](const RadixNode& n) { return n.uid; })
      .def_property_readonly("is_root", &RadixNode::is_root)
      .def_property_readonly("is_leaf", &RadixNode::is_leaf)
      .def("__eq__", [](const RadixNode& a, const RadixNode& b) { return &a == &b; })
      .def("__hash__", [](const RadixNode& n) { return std::hash<const RadixNode*>{}(&n); });

  py::class_<RadixTree>(m, "RadixTree")
      .def(py::init([](BlockAllocator& a, bool indexed) { return new RadixTree(&a, indexed); }), py::arg("allocator"),
           py::arg("indexed_eviction") = true,
           // The tree borrows the allocator; keep it alive at least as long.
           py::keep_alive<1, 2>())
      .def_property_readonly("allocator", &RadixTree::allocator, py::return_value_policy::reference_internal)
      .def_property_readonly("block_size", &RadixTree::block_size)
      .def_property_readonly("root", &RadixTree::root)
      .def_property_readonly("num_cached_blocks", &RadixTree::num_cached_blocks)
      .def_property_readonly("num_evictable", &RadixTree::num_evictable)
      .def_property_readonly("indexed_eviction", &RadixTree::indexed_eviction)
      .def_property_readonly("_clock", &RadixTree::clock)
      .def(
          "match",
          [](RadixTree& t, const py::handle& tokens) {
            return with_tokens(tokens, -1, [&](const auto& tok) { return t.match(tok); });
          },
          py::arg("tokens"))
      .def(
          "prefix_len",
          [](const RadixTree& t, const py::handle& tokens) {
            return with_tokens(tokens, -1, [&](const auto& tok) { return t.prefix_len(tok); });
          },
          py::arg("tokens"))
      .def(
          "prefix_lens",
          [](const RadixTree& t, const py::sequence& seqs, const std::vector<long>& limits) {
            if (static_cast<std::size_t>(py::len(seqs)) != limits.size()) {
              throw std::invalid_argument("prefix_lens: one limit per sequence");
            }
            std::vector<int> out(limits.size());
            for (std::size_t i = 0; i < limits.size(); ++i) {
              out[i] = with_tokens(seqs[i], limits[i], [&](const auto& tok) { return t.prefix_len(tok); });
            }
            return out;
          },
          py::arg("seqs"), py::arg("limits"),
          "prefix_len of seqs[i][:limits[i]] for every i, in one call across the binding.")
      .def(
          "insert",
          [](RadixTree& t, const py::handle& tokens, const std::vector<minicore::BlockId>& blocks) {
            return with_tokens(tokens, -1, [&](const auto& tok) { return t.insert(tok, blocks); });
          },
          py::arg("tokens"), py::arg("blocks"))
      .def("lock", [](RadixTree& t, RadixNode& n) { t.lock(&n); }, py::arg("node"))
      .def("unlock", [](RadixTree& t, RadixNode& n) { t.unlock(&n); }, py::arg("node"))
      .def("evict", &RadixTree::evict, py::arg("num_blocks"))
      .def("clear", &RadixTree::clear)
      .def("_nodes", &RadixTree::nodes)
      .def("check_invariants", &RadixTree::check_invariants);

  m.attr("BACKEND") = "cpp";
}
