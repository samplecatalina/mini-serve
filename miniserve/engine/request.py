"""Request state machine.

    WAITING --> PREFILL --> DECODE --> FINISHED
       ^           |          |
       +-------- preempted ---+

    WAITING / PREFILL / DECODE --> ABORTED

PREFILL is the step in which the prompt is run (with chunked prefill, the
steps: a long prompt is run a chunk per step); the first output token is
sampled at the end of it, so a request can finish straight from PREFILL (stop
token or ``max_new_tokens == 1``). A request can be preempted between chunks
(PREFILL -> WAITING).

A preempted request (DECODE -> WAITING) loses its KV cache but keeps its
output. When it is admitted again, its prefill runs prompt + output, and the
logits of the last position are exactly those the interrupted decode step
would have produced, so generation continues where it stopped.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from miniserve.cache.block_table import BlockTable
from miniserve.model.qwen3 import ContiguousKVCache


class RequestState(enum.Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    ABORTED = "aborted"


_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    # WAITING -> FINISHED: with overlap scheduling, a request preempted before its last sampled
    # token was read back, which then turns out to end it.
    RequestState.WAITING: frozenset({RequestState.PREFILL, RequestState.ABORTED, RequestState.FINISHED}),
    # PREFILL -> WAITING: preempted between chunks of a chunked prefill.
    RequestState.PREFILL: frozenset(
        {RequestState.DECODE, RequestState.FINISHED, RequestState.ABORTED, RequestState.WAITING}
    ),
    RequestState.DECODE: frozenset({RequestState.FINISHED, RequestState.ABORTED, RequestState.WAITING}),
    RequestState.FINISHED: frozenset(),
    RequestState.ABORTED: frozenset(),
}


# Output slot of a token that was sampled on the device but not read back yet (overlap scheduling).
PLACEHOLDER = -1


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    stop_token_ids: frozenset[int] = frozenset()
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0  # nucleus: sample from the smallest set of tokens with probability mass >= top_p
    seed: int | None = None  # None: the engine assigns one

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if not self.temperature >= 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.seed is not None and not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be in [0, 2**32), got {self.seed}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0


@dataclass(eq=False)
class Request:
    rid: int
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    # KV storage (a block table, or a contiguous cache on the reference path);
    # owned by the model runner between admission and release.
    cache: BlockTable | ContiguousKVCache | None = None
    # Prefix-cache bookkeeping, set at admission: the locked tree node, and how
    # many leading tokens of this prefill come from the cache.
    cache_node: object = None
    num_cached_tokens: int = 0
    num_preemptions: int = 0
    # Seed of the sampling noise; fixed when the request is submitted.
    seed: int = 0
    # Trailing PLACEHOLDER entries of output_ids: sampled, not yet read back (overlap scheduling).
    num_pending: int = 0

    def __post_init__(self):
        if not self.prompt_ids:
            raise ValueError("empty prompt")

    def transition(self, new: RequestState) -> None:
        if new not in _TRANSITIONS[self.state]:
            raise InvalidTransition(f"request {self.rid}: {self.state.name} -> {new.name}")
        self.state = new

    @property
    def is_done(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    @property
    def seq_len(self) -> int:
        """Tokens a prefill of this request runs: the prompt plus any output kept across a preemption."""
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def ready_ids(self) -> list[int]:
        """Output tokens read back so far (``output_ids`` without trailing placeholders)."""
        return self.output_ids[: len(self.output_ids) - self.num_pending]

    @property
    def reached_max_tokens(self) -> bool:
        """Every token this request may produce has been sampled (some maybe not read back yet)."""
        return len(self.output_ids) >= self.params.max_new_tokens

    def drop_pending(self) -> None:
        """Forget tokens sampled after the request ended (they were launched before its end was known)."""
        if self.num_pending:
            del self.output_ids[-self.num_pending :]
            self.num_pending = 0

    @property
    def max_len(self) -> int:
        """Upper bound on the sequence length, used to size the KV cache."""
        return len(self.prompt_ids) + self.params.max_new_tokens

    def should_stop(self) -> bool:
        """Same rule as single-request greedy generation: the stop token is kept in the output."""
        return len(self.output_ids) >= self.params.max_new_tokens or (
            bool(self.output_ids) and self.output_ids[-1] in self.params.stop_token_ids
        )
