"""Token selection on the device: greedy, temperature and top-p (nucleus) sampling.

Nothing here reads device memory back to the host: the sampled tokens stay on
the device and the engine copies them back once per step.

Greedy rows (temperature 0) take ``argmax`` of the logits, whatever else is in
the batch, so greedy output does not depend on batch composition through the
sampler.

Sampled rows use the Gumbel-max trick: ``argmax(logits / T + g)`` with i.i.d.
Gumbel noise ``g`` is a sample from ``softmax(logits / T)``; masking the tokens
outside the nucleus to -inf first samples from the renormalized nucleus. The
noise is not drawn from a stateful generator but hashed from (request seed,
position of the token in the sequence, vocabulary index). A request therefore
draws the same noise wherever it sits in a batch, whoever it shares the batch
with, and whether or not it was preempted and recomputed: given the same
logits, it samples the same tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from miniserve.engine.request import Request

_M32 = 0xFFFFFFFF
_MIX = 0x45D9F3B  # < 2**31, so products of 32-bit values fit in int64
_GOLDEN = 0x9E3779B9


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash (xor-shift-multiply) on int64 tensors holding values in [0, 2**32)."""
    x = ((x >> 16) ^ x) * _MIX & _M32
    x = ((x >> 16) ^ x) * _MIX & _M32
    return (x >> 16) ^ x


def uniform_noise(seeds: torch.Tensor, counters: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """``[B, vocab]`` float32 in (0, 1), a pure function of (seed, counter, vocabulary index)."""
    key = _mix32(seeds ^ _mix32(counters))  # [B]
    v = torch.arange(vocab_size, device=seeds.device, dtype=torch.int64) * _GOLDEN
    h = _mix32((key[:, None] + v[None, :]) & _M32)
    return ((h >> 8).to(torch.float32) + 0.5) * (1.0 / (1 << 24))


@dataclass
class SamplingArgs:
    """Per-row sampling parameters of one batch, on the device."""

    temperature: torch.Tensor  # [B] float32, 0 for greedy rows
    top_p: torch.Tensor  # [B] float32
    seed: torch.Tensor  # [B] int64
    counter: torch.Tensor  # [B] int64: position of the sampled token in its sequence
    any_top_p: bool  # some row has top_p < 1 (known on the host, so no device check is needed)


class Sampler:
    def __init__(self, device: torch.device | str, vocab_size: int):
        self.device = torch.device(device)
        self.vocab_size = vocab_size

    def prepare(self, requests: Sequence[Request]) -> SamplingArgs | None:
        """Sampling parameters for the next token of each request; None if all are greedy."""
        params = [r.params for r in requests]
        if all(p.is_greedy for p in params):
            return None
        return SamplingArgs(
            temperature=self._to_device([p.temperature for p in params], torch.float32),
            top_p=self._to_device([p.top_p for p in params], torch.float32),
            seed=self._to_device([r.seed for r in requests], torch.int64),
            # The token about to be sampled sits at index seq_len, both for a
            # decode step and for the prefill of a preempted request.
            counter=self._to_device([r.seq_len for r in requests], torch.int64),
            any_top_p=any(p.top_p < 1 for p in params),
        )

    def _to_device(self, values: list, dtype: torch.dtype) -> torch.Tensor:
        t = torch.tensor(values, dtype=dtype)
        if self.device.type == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def sample(self, logits: torch.Tensor, args: SamplingArgs | None) -> torch.Tensor:
        """``logits [B, vocab]`` -> ``[B]`` int64 token ids, left on the device.

        ``argmax`` returns the first maximal index, the same tie-breaking as HF.
        """
        greedy = torch.argmax(logits, dim=-1)
        if args is None:
            return greedy
        t = args.temperature
        z = logits.float() / torch.where(t > 0, t, 1.0)[:, None]
        if args.any_top_p:
            z = self._mask_outside_nucleus(z, args.top_p)
        g = -torch.log(-torch.log(uniform_noise(args.seed, args.counter, z.shape[-1])))
        sampled = torch.argmax(z + g, dim=-1)
        return torch.where(t > 0, sampled, greedy)

    @staticmethod
    def _mask_outside_nucleus(z: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Keep the smallest set of most likely tokens whose probability mass reaches ``top_p``."""
        probs = torch.softmax(z, dim=-1)
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True, stable=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        # The first token always stays (mass before it is 0); rows with top_p = 1 keep everything
        # even if rounding lets the running sum reach 1 early.
        keep_sorted = (mass_before < top_p[:, None]) | (top_p[:, None] >= 1)
        keep = torch.empty_like(keep_sorted).scatter_(-1, order, keep_sorted)
        return z.masked_fill(~keep, float("-inf"))
