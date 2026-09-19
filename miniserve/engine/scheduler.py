"""Batch assembly: which requests run in the next step.

Policy (same as mini-sglang's default): prefill first. If requests are waiting
and there is room, the next batch admits waiting requests in arrival order
until the prefill token budget is spent; otherwise every running request
decodes one token. A batch holds a single phase.

With a paged KV pool (an ``allocator`` is given), the scheduler also keeps
the batch within the pool:

- Admission. A waiting request of ``L`` tokens needs the blocks for its
  prefill plus one decode token, ``ceil((L + 1) / block_size)``, on top of the
  blocks every decoding request needs for its next token. Admission stops at
  the first request that does not fit (no skipping ahead). Right after an
  admission the next decode step therefore always fits.
- Preemption. If a decode step needs more blocks than are free, the most
  recently admitted requests are preempted one by one until it fits: their
  blocks are released, they keep their output and go back to the front of the
  waiting queue, to be prefilled again later (recomputation).

No deadlock: a request that could not fit in the whole pool is rejected when
it is submitted, and the oldest running request is never preempted, so it
always makes progress and eventually finishes.

Unlike mini-sglang, admission does not reserve KV for the full
``max_new_tokens`` of every request; mini-sglang does, and so never needs to
preempt, at the cost of pool space that sits reserved but unused whenever a
request stops early.
"""

from __future__ import annotations

import enum
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from miniserve.cache.block_allocator import BlockAllocator
from miniserve.engine.request import Request, RequestState


class Phase(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Batch:
    phase: Phase
    requests: list[Request]
    # Requests preempted while this batch was formed (already back in the waiting queue).
    preempted: list[Request] = field(default_factory=list)

    @property
    def seq_lens(self) -> list[int]:
        """New tokens each request contributes to this step's forward pass."""
        if self.phase is Phase.PREFILL:
            return [r.seq_len for r in self.requests]
        return [1] * len(self.requests)


class Scheduler:
    def __init__(
        self,
        max_running: int = 64,
        max_prefill_tokens: int = 8192,
        allocator: BlockAllocator | None = None,
        release: Callable[[Request], None] | None = None,
    ):
        """``allocator``: the paged KV pool to budget against (None: no KV budget).
        ``release``: frees a request's KV; called on preemption, required with an allocator."""
        if max_running < 1 or max_prefill_tokens < 1:
            raise ValueError("max_running and max_prefill_tokens must be positive")
        if allocator is not None and release is None:
            raise ValueError("a KV budget needs a release callback for preemption")
        self.max_running = max_running
        self.max_prefill_tokens = max_prefill_tokens
        self.allocator = allocator
        self.release = release
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        if req.state is not RequestState.WAITING:
            raise ValueError(f"request {req.rid} is {req.state.name}, expected WAITING")
        a = self.allocator
        if a is not None and self._blocks_for(req.max_len) > a.num_blocks:
            raise ValueError(
                f"request {req.rid} needs up to {req.max_len} tokens of KV, "
                f"the pool holds {a.num_blocks * a.block_size}"
            )
        self.waiting.append(req)

    def schedule(self) -> Batch | None:
        """Pick the next batch; moves admitted requests WAITING -> PREFILL and preempted ones DECODE -> WAITING."""
        admitted = self._admit()
        if admitted:
            self.running.extend(admitted)
            return Batch(Phase.PREFILL, admitted)
        decoding = [r for r in self.running if r.state is RequestState.DECODE]
        preempted = self._make_room(decoding)
        return Batch(Phase.DECODE, decoding, preempted) if decoding else None

    def _admit(self) -> list[Request]:
        admitted: list[Request] = []
        budget = self.max_prefill_tokens
        if self.allocator is not None:
            free = self.allocator.num_free - self._decode_need(self.running)
        while self.waiting and len(self.running) + len(admitted) < self.max_running:
            req = self.waiting[0]
            n = req.seq_len
            # The first request is admitted even if it alone exceeds the budget;
            # otherwise a long prompt would wait forever.
            if admitted and n > budget:
                break
            if self.allocator is not None:
                need = self._blocks_for(n + 1)  # prefill + the next decode token
                if need > free:
                    break
                free -= need
            self.waiting.popleft()
            req.transition(RequestState.PREFILL)
            admitted.append(req)
            budget -= n
        return admitted

    def _make_room(self, decoding: list[Request]) -> list[Request]:
        """Preempt from ``decoding`` (in place) until one more token each fits in the pool."""
        preempted: list[Request] = []
        if self.allocator is None:
            return preempted
        while self._decode_need(decoding) > self.allocator.num_free:
            # A single request always fits: add() rejects requests larger than the pool.
            assert len(decoding) > 1, "a lone decoding request does not fit in the pool"
            victim = self._pick_victim(decoding)
            decoding.remove(victim)
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    def _pick_victim(self, decoding: list[Request]) -> Request:
        """The most recently admitted request. Never the oldest, which guarantees progress."""
        return decoding[-1]

    def _preempt(self, req: Request) -> None:
        req.transition(RequestState.WAITING)
        self.release(req)
        self.running.remove(req)
        # Victims are taken newest first, so pushing each to the front keeps arrival order.
        self.waiting.appendleft(req)
        req.num_preemptions += 1
        self.num_preemptions += 1

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.allocator.block_size)

    @staticmethod
    def _decode_need(reqs: list[Request]) -> int:
        """Blocks the next decode step of ``reqs`` would allocate."""
        return sum(r.cache.blocks_needed(1) for r in reqs if r.state is RequestState.DECODE)

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
