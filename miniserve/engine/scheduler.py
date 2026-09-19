"""Batch assembly: which requests run in the next step.

Two policies:

- Without chunked prefill (same as mini-sglang's default): prefill first. If
  requests are waiting and there is room, the next batch admits waiting
  requests in arrival order until the prefill token budget is spent;
  otherwise every running request decodes one token. A batch holds a single
  phase, so a long prefill stalls every decoding request for its duration.
- With chunked prefill (``chunked_prefill_size`` tokens per step): decode
  first. Every decoding request gets its token; the rest of the step's token
  budget goes to prefill chunks, so a long prompt is prefilled over several
  steps alongside the decodes, and no step runs more than the budget.
  mini-sglang also cuts prefills into chunks, but keeps prefill-only batches
  ahead of decode, so its chunks bound the step size without letting decodes
  through.

With a paged KV pool (a ``KVCacheManager`` is given), the scheduler also
keeps the batch within the pool. "Available" blocks are the free ones plus
those only the prefix cache holds, which can be evicted at any time.

- Admission. A waiting request of ``L`` tokens first looks up its cached
  prefix (``C`` tokens, whole blocks). It needs the blocks for the rest of its
  prefill plus one decode token, ``ceil((L + 1) / block_size) - C / block_size``,
  on top of the blocks every decoding request needs for its next token, and
  ``L - C`` tokens of the prefill budget. Admission stops at the first request
  that does not fit (no skipping ahead); its lookup is undone. Right after an
  admission the next decode step therefore always fits.
- Preemption. If a decode step needs more blocks than are available, the most
  recently admitted requests are preempted one by one until it fits: their
  blocks are released (full blocks stay in the prefix cache), they keep their
  output and go back to the front of the waiting queue, to be prefilled again
  later; the prefix cache usually holds most of what they had computed.

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
from dataclasses import dataclass, field

from miniserve.cache.kv_cache import KVCacheManager
from miniserve.engine.request import Request, RequestState


class Phase(enum.Enum):
    PREFILL = "prefill"  # prefill rows only (whole prompts, or chunks of them)
    DECODE = "decode"  # one token per request
    MIXED = "mixed"  # decode rows and prefill chunks in one pass (chunked prefill)


@dataclass
class Batch:
    phase: Phase
    requests: list[Request]
    # Requests preempted while this batch was formed (already back in the waiting queue).
    preempted: list[Request] = field(default_factory=list)
    # Per request: tokens whose KV exists before this step, and new tokens this step.
    # Defaults: a whole prefill after the cached prefix, or one decode token.
    starts: list[int] | None = None
    extend_lens: list[int] | None = None
    # Rows ``[0, num_decode)`` are decode rows (one token each); the rest are prefill rows.
    num_decode: int | None = None

    def __post_init__(self):
        if self.num_decode is None:
            if self.phase is Phase.MIXED:
                raise ValueError("a mixed batch needs num_decode")
            self.num_decode = len(self.requests) if self.phase is Phase.DECODE else 0
        if self.starts is None:
            if self.phase is Phase.PREFILL:
                self.starts = [r.num_cached_tokens for r in self.requests]
                self.extend_lens = [r.seq_len - r.num_cached_tokens for r in self.requests]
            elif self.phase is Phase.DECODE:
                self.starts = [r.seq_len - 1 for r in self.requests]
                self.extend_lens = [1] * len(self.requests)
            else:
                raise ValueError("a mixed batch needs explicit starts and extend_lens")

    @property
    def seq_lens(self) -> list[int]:
        """New tokens each request contributes to this step's forward pass."""
        return self.extend_lens

    def completes(self) -> list[bool]:
        """Which rows reach the end of their sequence this step, and so sample a token.
        (A prefill chunk that stops short does not.) Valid until the step appends tokens."""
        return [s + n == r.seq_len for r, s, n in zip(self.requests, self.starts, self.extend_lens)]


class Scheduler:
    def __init__(
        self,
        max_running: int = 64,
        max_prefill_tokens: int = 8192,
        kv: KVCacheManager | None = None,
        chunked_prefill_size: int = 0,
    ):
        """``kv``: the paged KV pool to budget against, with its prefix cache (None: no KV budget).
        ``chunked_prefill_size``: tokens per step, decode rows included, with prefills cut into
        chunks and batched with decodes; 0 keeps whole prefills in prefill-only batches."""
        if max_running < 1 or max_prefill_tokens < 1:
            raise ValueError("max_running and max_prefill_tokens must be positive")
        if chunked_prefill_size < 0:
            raise ValueError(f"chunked_prefill_size must be >= 0, got {chunked_prefill_size}")
        if chunked_prefill_size and kv is None:
            raise ValueError("chunked prefill needs a paged KV pool")
        self.chunked_prefill_size = chunked_prefill_size
        self.max_running = max_running
        self.max_prefill_tokens = max_prefill_tokens
        self.kv = kv
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order
        self.num_preemptions = 0
        # Prefill tokens asked for at admission and how many of them came from the
        # prefix cache; first admissions and readmissions after preemption apart.
        self.stats = dict(first_tokens=0, first_cached=0, re_tokens=0, re_cached=0)

    def add(self, req: Request) -> None:
        if req.state is not RequestState.WAITING:
            raise ValueError(f"request {req.rid} is {req.state.name}, expected WAITING")
        if self.kv is not None and self._blocks_for(req.max_len) > self.kv.allocator.num_blocks:
            raise ValueError(
                f"request {req.rid} needs up to {req.max_len} tokens of KV, "
                f"the pool holds {self.kv.allocator.num_blocks * self.kv.block_size}"
            )
        self.waiting.append(req)

    def schedule(self) -> Batch | None:
        """Pick the next batch; moves admitted requests WAITING -> PREFILL and preempted ones back to WAITING."""
        if self.chunked_prefill_size:
            return self._schedule_chunked()
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
        reserved = 0  # blocks promised to requests admitted in this call
        kv = self.kv
        while self.waiting and len(self.running) + len(admitted) < self.max_running:
            req = self.waiting[0]
            cached = kv.acquire(req) if kv is not None else 0
            n = req.seq_len - cached
            # The first request is admitted even if it alone exceeds the budget;
            # otherwise a long prompt would wait forever.
            fits = not admitted or n <= budget
            if fits and kv is not None:
                # The rest of the prefill + the next decode token. Acquiring locked the
                # cached prefix, so kv.num_available no longer counts it as evictable.
                need = self._blocks_for(req.seq_len + 1) - len(req.cache.blocks)
                fits = need <= kv.num_available - self._decode_need(self.running) - reserved
            if not fits:
                if kv is not None:
                    kv.abandon(req)
                break
            if kv is not None:
                reserved += need
            self.waiting.popleft()
            req.transition(RequestState.PREFILL)
            admitted.append(req)
            budget -= n
            first = req.num_preemptions == 0
            self.stats["first_tokens" if first else "re_tokens"] += req.seq_len
            self.stats["first_cached" if first else "re_cached"] += cached
        return admitted

    def _make_room(self, decoding: list[Request]) -> list[Request]:
        """Preempt from ``decoding`` (in place) until one more token each fits in the pool."""
        preempted: list[Request] = []
        if self.kv is None:
            return preempted
        while self._decode_need(decoding) > self.kv.num_available:
            # A single request always fits: add() rejects requests larger than the pool.
            assert len(decoding) > 1, "a lone decoding request does not fit in the pool"
            victim = self._pick_victim(decoding)
            decoding.remove(victim)
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    # ------------------------------------------------------------------ chunked prefill

    def _schedule_chunked(self) -> Batch | None:
        """Decode first: every decoding request gets its token, then the rest of the step's token
        budget goes to prefill chunks, first of requests already part-way through their prefill
        (in admission order), then of newly admitted ones (in arrival order).

        Admission reserves blocks for the whole prefill plus one token, as without chunking;
        chunks allocate them step by step. So the pool always holds what every running request
        still needs for its next token or the rest of its prefill (``_outstanding``)."""
        kv = self.kv
        preempted = self._make_room_chunked()
        budget = self.chunked_prefill_size
        rows: list[tuple[Request, int, int]] = []  # (request, start, new tokens)
        for r in self.running:
            if r.state is RequestState.DECODE:
                rows.append((r, r.seq_len - 1, 1))
        num_decode = len(rows)
        budget -= num_decode
        for r in self.running:
            if r.state is RequestState.PREFILL and budget > 0:
                start = r.cache.num_tokens
                n = min(r.seq_len - start, budget)
                rows.append((r, start, n))
                budget -= n
        while budget > 0 and self.waiting and len(self.running) < self.max_running:
            req = self.waiting[0]
            cached = kv.acquire(req)
            need = self._blocks_for(req.seq_len + 1) - len(req.cache.blocks)
            if need > kv.num_available - self._outstanding(self.running):
                kv.abandon(req)
                break
            self.waiting.popleft()
            req.transition(RequestState.PREFILL)
            self.running.append(req)
            first = req.num_preemptions == 0
            self.stats["first_tokens" if first else "re_tokens"] += req.seq_len
            self.stats["first_cached" if first else "re_cached"] += cached
            n = min(req.seq_len - cached, budget)
            rows.append((req, cached, n))
            budget -= n
        if not rows:
            return None
        phase = Phase.DECODE if num_decode == len(rows) else Phase.PREFILL if num_decode == 0 else Phase.MIXED
        return Batch(
            phase,
            [r for r, _, _ in rows],
            preempted,
            starts=[s for _, s, _ in rows],
            extend_lens=[n for _, _, n in rows],
            num_decode=num_decode,
        )

    def _make_room_chunked(self) -> list[Request]:
        """Preempt the most recently admitted requests (decoding or part-way through a chunked
        prefill) until what the running requests still need fits in the available blocks."""
        preempted: list[Request] = []
        while self._outstanding(self.running) > self.kv.num_available:
            # A single request always fits: add() rejects requests larger than the pool.
            assert len(self.running) > 1, "a lone request does not fit in the pool"
            victim = self._pick_victim(self.running)
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    def _outstanding(self, reqs: list[Request]) -> int:
        """Blocks ``reqs`` may still allocate before any of them finishes: the next token of each
        decoding request, and the rest of the prefill plus one token of each prefilling one."""
        need = 0
        for r in reqs:
            if r.state is RequestState.DECODE:
                need += r.cache.blocks_needed(1)
            elif r.state is RequestState.PREFILL:
                need += self._blocks_for(r.seq_len + 1) - len(r.cache.blocks)
        return need

    # ------------------------------------------------------------------ preemption

    def _pick_victim(self, decoding: list[Request]) -> Request:
        """The most recently admitted request. Never the oldest, which guarantees progress."""
        return decoding[-1]

    def _preempt(self, req: Request) -> None:
        req.transition(RequestState.WAITING)
        self.kv.release(req)
        self.running.remove(req)
        # Victims are taken newest first, so pushing each to the front keeps arrival order.
        self.waiting.appendleft(req)
        req.num_preemptions += 1
        self.num_preemptions += 1

    def _blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.kv.block_size)

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
