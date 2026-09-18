"""Batch assembly: which requests run in the next step.

Policy (same as mini-sglang's default): prefill first. If requests are waiting
and there is room, the next batch admits waiting requests in arrival order
until the prefill token budget is spent; otherwise every running request
decodes one token. A batch holds a single phase.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass

from miniserve.engine.request import Request, RequestState


class Phase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Batch:
    phase: Phase
    requests: list[Request]

    @property
    def seq_lens(self) -> list[int]:
        """New tokens each request contributes to this step's forward pass."""
        if self.phase is Phase.PREFILL:
            return [len(r.prompt_ids) for r in self.requests]
        return [1] * len(self.requests)


class Scheduler:
    def __init__(self, max_running: int = 64, max_prefill_tokens: int = 8192):
        if max_running < 1 or max_prefill_tokens < 1:
            raise ValueError("max_running and max_prefill_tokens must be positive")
        self.max_running = max_running
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order

    def add(self, req: Request) -> None:
        if req.state is not RequestState.WAITING:
            raise ValueError(f"request {req.rid} is {req.state.name}, expected WAITING")
        self.waiting.append(req)

    def schedule(self) -> Batch | None:
        """Pick the next batch and move admitted requests WAITING -> PREFILL."""
        admitted: list[Request] = []
        budget = self.max_prefill_tokens
        while self.waiting and len(self.running) + len(admitted) < self.max_running:
            n = len(self.waiting[0].prompt_ids)
            # The first request is admitted even if it alone exceeds the budget;
            # otherwise a long prompt would wait forever.
            if admitted and n > budget:
                break
            req = self.waiting.popleft()
            req.transition(RequestState.PREFILL)
            admitted.append(req)
            budget -= n
        if admitted:
            self.running.extend(admitted)
            return Batch(Phase.PREFILL, admitted)
        decoding = [r for r in self.running if r.state is RequestState.DECODE]
        return Batch(Phase.DECODE, decoding) if decoding else None

    def retire(self, req: Request) -> None:
        """Drop a finished or aborted request from the queues."""
        if not req.is_done:
            raise ValueError(f"request {req.rid} is {req.state.name}, not done")
        if req in self.running:
            self.running.remove(req)
        elif req in self.waiting:
            self.waiting.remove(req)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
