"""Request state machine.

    WAITING --> PREFILL --> DECODE --> FINISHED
       |           |          |
       +-----------+----------+-----> ABORTED

PREFILL is the step in which the prompt is run; the first output token is
sampled at the end of it, so a request can finish straight from PREFILL (stop
token or ``max_new_tokens == 1``).
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
    RequestState.WAITING: frozenset({RequestState.PREFILL, RequestState.ABORTED}),
    RequestState.PREFILL: frozenset({RequestState.DECODE, RequestState.FINISHED, RequestState.ABORTED}),
    RequestState.DECODE: frozenset({RequestState.FINISHED, RequestState.ABORTED}),
    RequestState.FINISHED: frozenset(),
    RequestState.ABORTED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    stop_token_ids: frozenset[int] = frozenset()

    def __post_init__(self):
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


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
    def max_len(self) -> int:
        """Upper bound on the sequence length, used to size the KV cache."""
        return len(self.prompt_ids) + self.params.max_new_tokens

    def should_stop(self) -> bool:
        """Same rule as single-request greedy generation: the stop token is kept in the output."""
        return len(self.output_ids) >= self.params.max_new_tokens or (
            bool(self.output_ids) and self.output_ids[-1] in self.params.stop_token_ids
        )
