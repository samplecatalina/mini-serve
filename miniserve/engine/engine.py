"""The engine loop: ``step()`` = schedule -> forward -> sample -> postprocess.

Synchronous: the caller drives ``step()``. Requests can be added or aborted
between steps, and each step re-forms the batch from whatever is running.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from miniserve.engine import sampler
from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.request import Request, RequestState, SamplingParams
from miniserve.engine.scheduler import Batch, Phase, Scheduler
from miniserve.model.qwen3 import Qwen3ForCausalLM


class Engine:
    def __init__(self, model: Qwen3ForCausalLM, max_running: int = 64, max_prefill_tokens: int = 8192):
        self.runner = ModelRunner(model)
        self.scheduler = Scheduler(max_running, max_prefill_tokens)
        self.requests: dict[int, Request] = {}  # unfinished requests by id
        self._rids = itertools.count()
        # Called with (batch, logits) before sampling; used by diagnostics.
        self.logits_hook = None

    def add_request(self, prompt_ids: Sequence[int], params: SamplingParams) -> Request:
        req = Request(next(self._rids), list(prompt_ids), params)
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
        if batch.phase is Phase.PREFILL:
            for r in batch.requests:
                self.runner.allocate(r)
        logits = self.runner.forward(batch)
        if self.logits_hook is not None:
            self.logits_hook(batch, logits)
        for r, tok in zip(batch.requests, sampler.greedy(logits)):
            r.output_ids.append(tok)
            if r.should_stop():
                r.transition(RequestState.FINISHED)
                self._retire(r)
            elif r.state is RequestState.PREFILL:
                r.transition(RequestState.DECODE)
        return batch

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
