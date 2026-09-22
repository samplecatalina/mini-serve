"""Which implementation of the block bookkeeping is in use.

Two backends implement the same contract: the Python reference implementation in
this package, and the C++ one built from ``minicore/`` by ``make minicore`` into
``miniserve/cache/_minicore*.so``. They
must hand out the same block ids for the same call sequence and raise the same
exceptions; ``tests/test_block_allocator.py`` runs the whole suite against both.

The default is the Python backend: every number published so far was measured
with it, and silently changing the default would break comparisons across
milestones. ``MINISERVE_BLOCK_BACKEND=cpp`` selects the other one, and
``--block-backend`` does the same on the command line.

Only the allocator class is selected here. A table is created through
``allocator.new_table(...)``, so a table always matches its allocator's backend
and no caller has to name a table class. The prefix cache tree follows the
allocator the same way (``radix_tree_class``): a C++ tree frees and increfs
blocks of a C++ allocator without calling back into Python, and a Python tree
works on a Python allocator.
"""

from __future__ import annotations

import os

from miniserve.cache.block_allocator import BlockAllocator as PythonBlockAllocator
from miniserve.cache.block_allocator import OutOfBlocks

BACKENDS = ("python", "cpp")
ENV_VAR = "MINISERVE_BLOCK_BACKEND"

from miniserve.cache.block_table import BlockTable as PythonBlockTable
from miniserve.cache.block_table import pack_block_tables as _pack_python

__all__ = [
    "BACKENDS",
    "ENV_VAR",
    "OutOfBlocks",
    "allocator_class",
    "default_backend",
    "pack_block_tables",
    "radix_tree_class",
]


def default_backend() -> str:
    """The backend selected by the environment (``python`` unless set)."""
    name = os.environ.get(ENV_VAR, "python")
    if name not in BACKENDS:
        raise ValueError(f"{ENV_VAR} must be one of {BACKENDS}, got {name!r}")
    return name


def allocator_class(backend: str | None = None):
    """The allocator class of ``backend`` (default: the environment's).

    Raises ImportError with a usable message if the C++ backend is asked for but
    not built.
    """
    name = backend if backend is not None else default_backend()
    if name == "python":
        return PythonBlockAllocator
    if name == "cpp":
        try:
            from miniserve.cache._minicore import BlockAllocator as CppBlockAllocator
        except ImportError as exc:
            raise ImportError(
                "the C++ block backend is not built for this interpreter; run `make minicore`"
            ) from exc

        return CppBlockAllocator
    raise ValueError(f"unknown block backend {name!r}, expected one of {BACKENDS}")


def radix_tree_class(allocator):
    """The prefix cache tree class that works on ``allocator``: the one of the same backend."""
    if isinstance(allocator, PythonBlockAllocator):
        from miniserve.cache.radix_tree import RadixTree

        return RadixTree
    from miniserve.cache._minicore import RadixTree as CppRadixTree

    return CppRadixTree


def pack_block_tables(tables, pad_rows: int, pad_block: int, pad_last: int, indptr, indices, last) -> int:
    """The batch in FlashInfer's paged-KV layout, written by the backend the tables belong to
    (see ``block_table.pack_block_tables`` for the contract). For C++ tables this is one call
    across the binding for the whole batch, instead of one list per table."""
    if not tables or isinstance(tables[0], PythonBlockTable):
        return _pack_python(tables, pad_rows, pad_block, pad_last, indptr, indices, last)
    from miniserve.cache._minicore import pack_block_tables as pack_cpp

    return pack_cpp(tables, pad_rows, pad_block, pad_last, indptr, indices, last)
