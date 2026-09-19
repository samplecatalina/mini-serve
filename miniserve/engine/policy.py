"""Scheduling policies: the order in which waiting requests are admitted, and which running
request is preempted when the KV pool runs short.

The two decisions matter only when requests queue, that is when the KV pool (or the running
limit) cannot hold everything at once; with a large enough pool every request is admitted as it
arrives and the policies coincide.

- ``FCFS``: arrival order (preempted requests first, as they arrived earlier); the most recently
  admitted request is preempted.
- ``ShortestJobFirst``: least remaining work first, estimated as the prefill tokens still to
  compute plus the tokens the request may still generate (``max_new_tokens`` minus what it has
  produced; its actual length is unknown). The request with the most remaining work is
  preempted. It lowers mean latency at the cost of long requests, which can wait arbitrarily
  long under a steady stream of short ones (no aging).
- ``CacheAware``: the request with the longest prefix already in the prefix cache first (ties in
  arrival order), so requests sharing a prefix are admitted while it is cached; preemption as
  FCFS.

Admission tries requests in the policy's order and stops at the first that does not fit.
Preemption candidates never include the earliest admitted running request, which keeps the
scheduler's progress guarantee under any policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from miniserve.engine.request import Request


class SchedulePolicy(Protocol):
    name: str

    def order(self, waiting: Sequence[Request], kv) -> list[Request]:
        """Waiting requests in the order admission should try them."""
        ...

    def pick_victim(self, candidates: Sequence[Request]) -> Request:
        """The request to preempt among ``candidates`` (running requests in admission order)."""
        ...


class FCFS:
    name = "fcfs"

    def order(self, waiting: Sequence[Request], kv) -> list[Request]:
        return list(waiting)

    def pick_victim(self, candidates: Sequence[Request]) -> Request:
        return candidates[-1]


def remaining_work(r: Request) -> int:
    """Prefill tokens still to compute plus tokens the request may still generate."""
    prefill = r.seq_len - (r.cache.num_tokens if r.cache is not None else 0)
    return prefill + r.params.max_new_tokens - len(r.output_ids)


class ShortestJobFirst:
    name = "sjf"

    def order(self, waiting: Sequence[Request], kv) -> list[Request]:
        return sorted(waiting, key=remaining_work)  # stable: ties keep arrival order

    def pick_victim(self, candidates: Sequence[Request]) -> Request:
        # the last of the largest: among equals, the most recently admitted
        return max(reversed(candidates), key=remaining_work)


class CacheAware:
    name = "cache"

    def order(self, waiting: Sequence[Request], kv) -> list[Request]:
        if kv is None or not kv.radix:
            return list(waiting)
        cached = {r.rid: kv.cached_prefix_len(r) for r in waiting}
        return sorted(waiting, key=lambda r: -cached[r.rid])

    def pick_victim(self, candidates: Sequence[Request]) -> Request:
        return candidates[-1]


POLICIES = {p.name: p for p in (FCFS, ShortestJobFirst, CacheAware)}


def make_policy(name: str) -> SchedulePolicy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown schedule policy {name!r}; choose from {sorted(POLICIES)}") from None
