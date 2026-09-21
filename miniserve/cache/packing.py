"""Pinned host buffers for a batch's paged-KV metadata, reused every step.

FlashInfer plans a pass from three int32 arrays: where each sequence's page list
starts, the page lists, and how full each last page is. Built from Python lists,
that is one ``torch.tensor`` and one pinned allocation per array per step, over
every block of every sequence (tens of thousands of entries at a large batch).
Here the arrays are written in place into pinned buffers that live as long as
their owner, by whichever block backend the tables belong to, and handed to the
planner as views.

Reusing a pinned buffer is only safe once the device has read the previous
step's copy out of it. Every owner already waits on its ``CopyFence`` before it
plans, which is exactly that condition.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from miniserve.cache.backend import pack_block_tables


class PackedTables:
    def __init__(self, device: torch.device | str, rows: int = 64, indices: int = 4096):
        self.pin = torch.device(device).type == "cuda"
        self._rows = self._indices = 0
        self._grow(rows, indices)

    def _grow(self, rows: int, indices: int) -> None:
        def buf(n: int) -> torch.Tensor:
            t = torch.zeros(n, dtype=torch.int32)
            return t.pin_memory() if self.pin else t

        if rows > self._rows:
            self._rows = max(rows, 2 * self._rows)
            self.indptr, self.last = buf(self._rows + 1), buf(self._rows)
            self._indptr_np, self._last_np = self.indptr.numpy(), self.last.numpy()
        if indices > self._indices:
            self._indices = max(indices, 2 * self._indices)
            self.indices = buf(self._indices)
            self._indices_np = self.indices.numpy()

    def pack(
        self, tables: Sequence, pad_rows: int = 0, pad_block: int = 0, pad_last: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(indptr, indices, last) of ``tables`` followed by ``pad_rows`` padding rows (each
        the single page ``pad_block`` holding ``pad_last`` tokens): views of the pinned
        buffers, valid until the next ``pack``."""
        rows = len(tables) + pad_rows
        self._grow(rows, sum(t.num_blocks for t in tables) + pad_rows)
        n = pack_block_tables(
            list(tables), pad_rows, pad_block, pad_last, self._indptr_np, self._indices_np, self._last_np
        )
        return self.indptr[: rows + 1], self.indices[:n], self.last[:rows]
