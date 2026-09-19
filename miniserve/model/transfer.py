"""Host-to-device copies that do not make the host wait for the device.

A copy from pageable host memory to the device first waits for every earlier
operation on the stream to finish (CUDA semantics), so one such copy while the
previous step is still running on the GPU stalls the CPU until it ends, and
nothing overlaps. Copies from pinned memory with ``non_blocking=True`` are
queued like kernels.

A pinned buffer that is reused (FlashInfer's planning buffers, the decode
graphs' staging buffer) must not be rewritten while a queued copy still reads
from it: ``CopyFence`` records an event after a step's input copies are
queued, and the next step waits on it before writing pinned memory again. By
then the device is running the step after the previous one, so the wait is
normally a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pinned(values: Sequence, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``values`` as a host tensor, pinned if it is meant for a CUDA device."""
    t = torch.tensor(values, dtype=dtype)
    return t.pin_memory() if device.type == "cuda" else t


def to_device(values: Sequence, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Copy ``values`` to ``device`` without waiting for the device."""
    device = torch.device(device)
    return pinned(values, dtype, device).to(device, non_blocking=True)


class CopyFence:
    """Orders host writes into reused pinned buffers after the device read the previous contents."""

    def __init__(self, device: torch.device | str):
        self.cuda = torch.device(device).type == "cuda"
        self._event: torch.cuda.Event | None = None

    def wait(self) -> None:
        """Before writing pinned buffers: the copies queued at the last ``mark`` have run."""
        if self._event is not None:
            self._event.synchronize()
            self._event = None

    def mark(self) -> None:
        """After queuing copies out of pinned buffers."""
        if self.cuda:
            self._event = torch.cuda.Event()
            self._event.record()
