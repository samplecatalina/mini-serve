"""The engine loop: ``step()`` = schedule -> forward -> sample -> postprocess.

Synchronous: the caller drives ``step()``. Requests can be added or aborted
between steps, and each step re-forms the batch from whatever is running.

Sampling stays on the device; the host waits for the device once per step,
when the sampled tokens are copied back (``_read_back``). Input staging in
the model runner still uses synchronous host-to-device copies.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import torch

from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.request import Request, RequestState, SamplingParams
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
    ):
        """``kv_pool_tokens``: exact KV pool size (default: as large as GPU memory allows).
        ``radix``: reuse cached KV of shared prefixes (paged attention only).
        ``chunked_prefill_size``: token budget per step with prefills cut into chunks and batched
        with decodes; 0 disables; default 2048 with paged attention, 0 with contiguous.
        ``cuda_graph``: replay captured CUDA Graphs for decode steps of up to ``cuda_graph_max_bs``
        requests (default ``max_running``; paged attention only).
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
            )
        self.runner = runner
        if chunked_prefill_size is None:
            chunked_prefill_size = DEFAULT_CHUNKED_PREFILL if runner.kv is not None else 0
        # With a paged pool the scheduler budgets KV blocks (and acquires cached prefixes) through it.
        self.scheduler = Scheduler(max_running, max_prefill_tokens, kv=runner.kv, chunked_prefill_size=chunked_prefill_size)
        self.requests: dict[int, Request] = {}  # unfinished requests by id
        self._rids = itertools.count()
        self._seeds = random.Random(seed)
        # Called with (batch, logits) before sampling; used by diagnostics.
        self.logits_hook = None

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
        return self.scheduler.has_unfinished

    def step(self) -> Batch | None:
        """Run one batch. Returns it (its requests now carry one more token), or None if idle."""
        batch = self.scheduler.schedule()
        if batch is None:
            return None
        kv = self.runner.kv
        if batch.phase is Phase.PREFILL and kv is None:
            for r in batch.requests:
                self.runner.allocate(r)
        completes = batch.completes()
        logits = self.runner.forward(batch)
        if self.logits_hook is not None:
            self.logits_hook(batch, logits)
        tokens = self._read_back(self.runner.sample(batch, logits))
        for r, tok, done in zip(batch.requests, tokens, completes):
            if not done:  # a prefill chunk short of the end: nothing to sample yet
                kv.commit(r)
                continue
            r.output_ids.append(tok)
            if r.should_stop():
                r.transition(RequestState.FINISHED)
                self._retire(r)
            elif r.state is RequestState.PREFILL:
                r.transition(RequestState.DECODE)
                if kv is not None:
                    kv.commit(r)  # later requests with this prefix can hit now
        return batch

    @staticmethod
    def _read_back(tokens: torch.Tensor) -> list[int]:
        """Copy sampled tokens to the host. The one device-to-host synchronization of a step:
        everything before it is queued on the device without waiting."""
        if tokens.device.type != "cuda":
            return tokens.tolist()
        host = tokens.to("cpu", non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        done.synchronize()
        return host.tolist()

    def _retire(self, req: Request) -> None:
        self.runner.release(req)
        self.scheduler.retire(req)
        del self.requests[req.rid]

    def generate(self, prompts: Sequence[Sequence[int]], params: SamplingParams) -> list[list[int]]:
        """Offline helper: submit all prompts at once and run to completion."""
        reqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished:
            self.step()
        return [r.output_ids for r in reqs]
