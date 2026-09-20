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
and no caller has to name a table class.
"""

from __future__ import annotations

import os

from miniserve.cache.block_allocator import BlockAllocator as PythonBlockAllocator
from miniserve.cache.block_allocator import OutOfBlocks

BACKENDS = ("python", "cpp")
ENV_VAR = "MINISERVE_BLOCK_BACKEND"

__all__ = ["BACKENDS", "ENV_VAR", "OutOfBlocks", "allocator_class", "default_backend"]


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
