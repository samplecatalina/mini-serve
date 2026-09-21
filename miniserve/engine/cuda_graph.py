"""CUDA Graphs for decode steps, and for passes of a fixed number of tokens per row.

An eager decode forward is several hundred kernel launches (about two dozen
per layer), each a round trip through Python and the CUDA driver. For a small
model the kernels themselves take microseconds, so the GPU often waits for the
CPU to issue the next one. A CUDA Graph records the whole forward once and
replays it with a single launch.

A graph replays fixed kernels on fixed addresses, so:

- Only decode steps are captured (one token per sequence); prefill and mixed
  steps vary in shape and run eagerly.
- One graph per batch-size bucket. A batch of B sequences runs the graph of
  the smallest bucket G >= B; the G - B padding rows read and write a dummy KV
  block that no request owns.
- Inputs live in static buffers that each step overwrites: token ids,
  positions and KV slots (one host-to-device copy from a pinned staging
  buffer), and FlashInfer's paged-KV metadata, which ``plan`` copies into
  buffers fixed at wrapper construction.
- Sampling stays outside the graph: its parameters change per step, and an
  all-greedy batch only needs an argmax.

The same machinery captures passes of ``width > 1`` new tokens per row, all
rows alike: a speculative round verifies ``gamma + 1`` positions per request,
and the draft model's first step feeds it two. Those are extend passes, run
by FlashInfer's prefill wrapper in its CUDA Graph mode; the query layout is
fixed (``width`` rows per sequence), the KV layout is planned every step as
for decode.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from miniserve.model.transfer import CopyFence, pinned, to_device

if TYPE_CHECKING:
    from miniserve.cache.block_table import BlockTable
    from miniserve.cache.kv_pool import KVPool
    from miniserve.model.qwen3 import Qwen3ForCausalLM


def graph_buckets(max_bs: int) -> list[int]:
    """Batch sizes to capture: powers of two below ``max_bs``, and ``max_bs`` itself."""
    if max_bs < 1:
        raise ValueError(f"max_bs must be positive, got {max_bs}")
    out, b = [], 1
    while b < max_bs:
        out.append(b)
        b *= 2
    return out + [max_bs]


def bucket_for(buckets: Sequence[int], batch_size: int) -> int:
    """The smallest bucket that holds ``batch_size`` rows (``buckets`` ascending)."""
    for g in buckets:
        if g >= batch_size:
            return g
    raise ValueError(f"batch of {batch_size} exceeds the largest graph bucket {buckets[-1]}")


@functools.cache
def _warmup_stream(device: torch.device) -> torch.cuda.Stream:
    """The side stream every capture warms up on. One per device for the whole process:
    cuBLAS keeps a workspace for each stream it has run on, for as long as the process
    lives, so a new stream per capture leaks one workspace per captured graph."""
    return torch.cuda.Stream(device)


class _GraphAttention:
    """Attention backend used inside a captured decode graph: fixed slot buffer, fixed wrapper."""

    def __init__(self, pool: KVPool, slots: torch.Tensor, wrapper):
        self.pool = pool
        self.slots = slots
        self.wrapper = wrapper

    def attend(self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.pool.write(layer, self.slots, k, v)
        return self.wrapper.run(q, (self.pool.k(layer), self.pool.v(layer)))

    def finish(self) -> None:
        pass


class DecodeGraphs:
    """One captured forward per bucket, sharing input buffers and a memory pool.

    ``dummy_block`` is a block of ``pool`` outside the allocator: padding rows
    attend to it and write their K/V into it. ``width``: new tokens per row
    (1 for decode steps); a padding row writes ``width`` slots of the dummy
    block, so it may not exceed the block size.
    """

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        pool: KVPool,
        dummy_block: int,
        buckets: Sequence[int],
        workspace: torch.Tensor,
        num_heads: int,
        scale: float,
        fence: CopyFence,
        width: int = 1,
    ):
        """``fence``: the model runner's; marked once this step's inputs are queued for copying."""
        import flashinfer

        if not 1 <= width <= pool.block_size:
            raise ValueError(f"width must be in [1, {pool.block_size}] (the block size), got {width}")
        self.width = width
        self.model = model
        self.pool = pool
        self.dummy_block = dummy_block
        self.buckets = sorted(buckets)
        self.fence = fence
        dev = pool.device
        gmax = self.buckets[-1]
        # ids, positions, slots: one pinned staging tensor, one copy per step.
        self._staging = torch.zeros(3, gmax * width, dtype=torch.long).pin_memory()
        self._inputs = torch.zeros(3, gmax * width, dtype=torch.long, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        self._indptr = torch.zeros(gmax + 1, **i32)
        # Page indices of a whole batch. With prefix caching several sequences list the same
        # block, so the total is bounded by rows x blocks per sequence, not by the pool size.
        self._indices = torch.zeros(gmax * pool.num_blocks, **i32)
        self._last = torch.zeros(gmax, **i32)
        self._plan_args = dict(
            num_qo_heads=num_heads,
            num_kv_heads=pool.num_kv_heads,
            head_dim=pool.head_dim,
            page_size=pool.block_size,
            sm_scale=scale,
            q_data_type=pool.dtype,
            kv_data_type=pool.dtype,
            non_blocking=True,
        )
        tensor_cores = num_heads // pool.num_kv_heads >= 4
        if width == 1:
            self._wrappers = {
                g: flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=tensor_cores,
                    paged_kv_indptr_buffer=self._indptr[: g + 1],
                    paged_kv_indices_buffer=self._indices,
                    paged_kv_last_page_len_buffer=self._last[:g],
                    backend="fa2",
                )
                for g in self.buckets
            }
        else:
            # Every row has ``width`` queries, so the query layout is the same at every step.
            self._qo_indptr = torch.arange(0, (gmax + 1) * width, width, **i32)
            self._qo_indptr_host = self._qo_indptr.cpu().pin_memory()
            self._wrappers = {
                g: flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self._qo_indptr[: g + 1],
                    paged_kv_indptr_buf=self._indptr[: g + 1],
                    paged_kv_indices_buf=self._indices,
                    paged_kv_last_page_len_buf=self._last[:g],
                    backend="fa2",
                )
                for g in self.buckets
            }
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._logits: dict[int, torch.Tensor] = {}

        torch.cuda.synchronize(dev)
        before = torch.cuda.memory_allocated(dev)
        t0 = time.perf_counter()
        self._capture_all()
        torch.cuda.synchronize(dev)
        self.capture_s = time.perf_counter() - t0
        self.graph_bytes = torch.cuda.memory_allocated(dev) - before

    @property
    def max_batch(self) -> int:
        return self.buckets[-1]

    def _forward(self, g: int) -> torch.Tensor:
        n = g * self.width
        attn = _GraphAttention(self.pool, self._inputs[2, :n], self._wrappers[g])
        return self.model.decode_logits(self._inputs[0, :n], self._inputs[1, :n], attn)

    @torch.inference_mode()
    def _capture_all(self) -> None:
        mempool = torch.cuda.graph_pool_handle()
        dummy = [self.dummy_block]
        w = self.width
        for g in reversed(self.buckets):  # largest first: smaller graphs reuse its pool memory
            self._stage([], [], [], g)
            self._plan(g, [dummy] * g, [w] * g)
            # Lazy initialization (cuBLAS handles, kernel selection) must not happen during capture.
            side = _warmup_stream(self.pool.device)
            side.wait_stream(torch.cuda.current_stream(self.pool.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward(g)
            torch.cuda.current_stream(self.pool.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=mempool):
                self._logits[g] = self._forward(g)
            self._graphs[g] = graph
            torch.cuda.synchronize(self.pool.device)  # the staging buffer is rewritten next round

    def _stage(self, ids: Sequence[int], pos: Sequence[int], slots: Sequence[int], g: int) -> None:
        """Write the per-token inputs (``width`` per row), padded to ``g`` rows, into the static
        input buffer. A padding row takes positions and dummy-block slots ``0 .. width - 1``."""
        n, w = len(ids), self.width
        s = self._staging
        s[0, :n] = torch.tensor(ids, dtype=torch.long)
        s[1, :n] = torch.tensor(pos, dtype=torch.long)
        s[2, :n] = torch.tensor(slots, dtype=torch.long)
        if g * w > n:
            pad = torch.arange(w, dtype=torch.long).repeat(g - n // w)
            s[0, n : g * w] = 0
            s[1, n : g * w] = pad
            s[2, n : g * w] = pad + self.dummy_block * self.pool.block_size
        # The whole buffer: a copy out of a slice of it would go through a pageable temporary
        # (non-contiguous), which waits for the device. Rows past g are never read by graph g.
        self._inputs.copy_(s, non_blocking=True)

    def _plan(self, g: int, blocks: Sequence[Sequence[int]], last_lens: Sequence[int]) -> None:
        indptr = [0]
        indices: list[int] = []
        for bl in blocks:
            indices += bl
            indptr.append(len(indices))
        dev = self.pool.device
        i32 = dict(dtype=torch.int32, device=dev)
        # FlashInfer copies ``indices`` into its buffer non-blocking only from the same device
        # (a host tensor, even pinned, is copied synchronously), so it goes up first.
        kv = (pinned(indptr, **i32), to_device(indices, torch.int32, dev), pinned(list(last_lens), **i32))
        if self.width == 1:
            self._wrappers[g].plan(*kv, **self._plan_args)
            return
        args = dict(self._plan_args)
        head_dim = args.pop("head_dim")
        self._wrappers[g].plan(
            self._qo_indptr_host[: g + 1], *kv, head_dim_qk=head_dim, causal=True, **args
        )

    def run(
        self,
        ids: Sequence[int],
        pos: Sequence[int],
        slots: Sequence[int],
        tables: Sequence[BlockTable],
        fill: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Logits ``[B * width, vocab]`` for B sequences, ``width`` new tokens each (one row of
        logits per token); ``tables`` already cover the new tokens, and ``ids``, ``pos`` and
        ``slots`` list them sequence by sequence. ``fill = (rows, tokens)``: device tensors
        overriding the token ids at those positions (tokens produced on the device, not known
        on the host). The returned tensor is a view of the graph's output buffer, valid until
        the next replay."""
        b = len(tables)
        if len(ids) != b * self.width:
            raise ValueError(f"{len(ids)} tokens for {b} sequences of width {self.width}")
        g = bucket_for(self.buckets, b)
        self._stage(ids, pos, slots, g)
        if fill is not None:
            self._inputs[0].index_copy_(0, *fill)
        pad = g - b
        self._plan(
            g,
            [t.blocks for t in tables] + [[self.dummy_block]] * pad,
            [t.last_block_len for t in tables] + [self.width] * pad,
        )
        self.fence.mark()
        self._graphs[g].replay()
        return self._logits[g][: b * self.width]
