"""The engine loop: ``step()`` = schedule -> forward -> sample -> postprocess.

Synchronous: the caller drives ``step()``. Requests can be added or aborted
between steps, and each step re-forms the batch from whatever is running.

Sampling stays on the device; the host reads the sampled tokens back once per
step. Nothing else in a step waits for the device.

Overlap scheduling (default): a step launches the next batch before it reads
back the previous one, so the CPU schedules and prepares batch t+1 while the
GPU still runs batch t. Batch t+1 is formed before t's tokens are known:

- each request that sampled a token in t gets a placeholder output token,
  and its input token in t+1 is taken from t's sampled tokens on the device;
- a request that sampled its last allowed token is not scheduled again;
- a request that turns out to have sampled a stop token in t was already
  given a row in t+1; that token is discarded when t+1 is read back.

Freeing the KV of a request that ended in t while t+1 still uses it is safe:
the blocks can only be handed out when t+2 is formed, and t+2 runs after t+1
on the same stream.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import torch

from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.policy import make_policy
from miniserve.engine.request import PLACEHOLDER, Request, RequestState, SamplingParams
from miniserve.engine.scheduler import Batch, Phase, Scheduler
from miniserve.model.qwen3 import Qwen3ForCausalLM

DEFAULT_CHUNKED_PREFILL = 2048


class Engine:
    def __init__(
        self,
        model: Qwen3ForCausalLM | None,
        max_running: int = 64,
        max_prefill_tokens: int = 8192,
        attention: str = "paged",
        kv_pool_tokens: int | None = None,
        seed: int = 0,
        runner: ModelRunner | None = None,
        radix: bool = True,
        chunked_prefill_size: int | None = None,
        cuda_graph: bool = True,
        cuda_graph_max_bs: int | None = None,
        overlap: bool = True,
        schedule_policy: str = "fcfs",
        block_backend: str | None = None,
    ):
        """``kv_pool_tokens``: exact KV pool size (default: as large as GPU memory allows).
        ``radix``: reuse cached KV of shared prefixes (paged attention only).
        ``chunked_prefill_size``: token budget per step with prefills cut into chunks and batched
        with decodes; 0 disables; default 2048 with paged attention, 0 with contiguous.
        ``cuda_graph``: replay captured CUDA Graphs for decode steps of up to ``cuda_graph_max_bs``
        requests (default ``max_running``; paged attention only).
        ``overlap``: launch each step before reading back the previous one.
        ``schedule_policy``: admission order and preemption choice (``policy.py``).
        ``block_backend``: implementation of the KV block bookkeeping (``python`` or ``cpp``).
        ``seed``: seeds the sampling of requests submitted without a seed of their own, in
        submission order. ``runner``: use this model runner instead of building one for ``model``."""
        if runner is None:
            runner = ModelRunner(
                model,
                attention=attention,
                kv_pool_tokens=kv_pool_tokens,
                max_prefill_tokens=max_prefill_tokens,
                max_running=max_running,
                radix=radix,
                cuda_graph=cuda_graph,
                cuda_graph_max_bs=cuda_graph_max_bs,
                block_backend=block_backend,
            )
        self.runner = runner
        if chunked_prefill_size is None:
            chunked_prefill_size = DEFAULT_CHUNKED_PREFILL if runner.kv is not None else 0
        # With a paged pool the scheduler budgets KV blocks (and acquires cached prefixes) through it.
        self.scheduler = Scheduler(
            max_running,
            max_prefill_tokens,
            kv=runner.kv,
            chunked_prefill_size=chunked_prefill_size,
            policy=make_policy(schedule_policy),
        )
        self.requests: dict[int, Request] = {}  # unfinished requests by id
        self._rids = itertools.count()
        self._seeds = random.Random(seed)
        # Called with (batch, logits) before sampling; used by diagnostics.
        self.logits_hook = None
        self.overlap = overlap
        self.launched: Batch | None = None  # the batch the last step() launched
        self._inflight: _Launched | None = None  # launched, not read back yet

    def add_request(self, prompt_ids: Sequence[int], params: SamplingParams) -> Request:
        req = Request(next(self._rids), list(prompt_ids), params)
        req.seed = params.seed if params.seed is not None else self._seeds.randrange(2**32)
        self.scheduler.add(req)
        self.requests[req.rid] = req
        return req

    def abort(self, rid: int) -> None:
        req = self.requests.get(rid)
        if req is None:
            return  # already finished or never existed
        req.transition(RequestState.ABORTED)
        self._retire(req)

    @property
    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished or self._inflight is not None

    def step(self) -> Batch | None:
        """Run one step. Returns the batch whose sampled tokens this step read back (its requests
        now carry them), or None if there is none: idle, or (with overlap) the first step after
        idle, which only launches."""
        batch = self.scheduler.schedule()
        prev = self._inflight
        self.launched = batch
        self._inflight = self._launch(batch, prev) if batch is not None else None
        if not self.overlap:
            prev, self._inflight = self._inflight, None
        return self._process(prev) if prev is not None else None

    def _launch(self, batch: Batch, prev: _Launched | None) -> _Launched:
        """Queue the forward pass and sampling of ``batch``, and the copy of its tokens to the host,
        without waiting for the device. Rows that complete their sequence get a placeholder token."""
        kv = self.runner.kv
        if batch.phase is Phase.PREFILL and kv is None:
            for r in batch.requests:
                self.runner.allocate(r)
        completes = batch.completes()
        fill = (prev.tokens, prev.rows) if prev is not None else None
        logits = self.runner.forward(batch, fill)
        if self.logits_hook is not None:
            self.logits_hook(batch, logits)
        tokens = self.runner.sample(batch, logits)
        host, done = tokens, None
        if tokens.device.type == "cuda":
            host = tokens.to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        slots = []
        for r, complete in zip(batch.requests, completes):
            if not complete:  # a prefill chunk short of the end: nothing to sample yet
                slots.append(None)
                kv.commit(r)
                continue
            slots.append(len(r.output_ids))
            r.output_ids.append(PLACEHOLDER)
            r.num_pending += 1
            if r.state is RequestState.PREFILL:
                r.transition(RequestState.DECODE)
                if kv is not None:
                    kv.commit(r)  # later requests with this prefix can hit now
        rows = {r.rid: i for i, r in enumerate(batch.requests)}
        return _Launched(batch, tokens, rows, host, done, slots)

    def _process(self, launched: _Launched) -> Batch:
        """Wait for ``launched``'s tokens (the one device-to-host synchronization of a step), put
        them in place of the placeholders, and retire the requests that ended."""
        if launched.done is not None:
            launched.done.synchronize()
        values = launched.host.tolist()
        for r, tok, slot in zip(launched.batch.requests, values, launched.slots):
            if slot is None or r.is_done:  # a chunk; or ended before (a token past its end)
                continue
            r.output_ids[slot] = tok
            r.num_pending -= 1
            if slot + 1 >= r.params.max_new_tokens or tok in r.params.stop_token_ids:
                r.drop_pending()  # tokens launched after this one
                r.transition(RequestState.FINISHED)
                self._retire(r)
        return launched.batch

    def _retire(self, req: Request) -> None:
        req.drop_pending()
        self.runner.release(req)
        self.scheduler.retire(req)
        del self.requests[req.rid]

    def generate(self, prompts: Sequence[Sequence[int]], params: SamplingParams) -> list[list[int]]:
        """Offline helper: submit all prompts at once and run to completion."""
        reqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished:
            self.step()
        return [r.output_ids for r in reqs]


class _Launched:
    """A launched step: its batch, sampled tokens (device, and their host copy once ``done``),
    each request's row, and each row's output slot (None for a prefill chunk)."""

    __slots__ = ("batch", "tokens", "rows", "host", "done", "slots")

    def __init__(self, batch, tokens, rows, host, done, slots):
        self.batch, self.tokens, self.rows, self.host, self.done, self.slots = batch, tokens, rows, host, done, slots
