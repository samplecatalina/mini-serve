"""Sampler: greedy equivalence, sampling distributions, per-request reproducibility, no host sync."""

from __future__ import annotations

import pytest
import torch

from miniserve.engine.request import Request, SamplingParams
from miniserve.engine.sampler import Sampler, SamplingArgs, uniform_noise


def _args(n: int, temperature: float, top_p: float = 1.0, seeds=None, counter: int = 7) -> SamplingArgs:
    seeds = torch.arange(n) if seeds is None else torch.as_tensor(seeds, dtype=torch.int64)
    return SamplingArgs(
        temperature=torch.full((n,), temperature),
        top_p=torch.full((n,), top_p),
        seed=seeds,
        counter=torch.full((n,), counter, dtype=torch.int64),
        any_top_p=top_p < 1,
    )


def _freq(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(tokens, minlength=vocab).double() / tokens.numel()


def _logits_with_ties() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 32, generator=g)
    logits[0, [3, 7]] = 9.0  # exact tie: argmax must pick the first
    logits[3, [30, 5]] = 9.0
    return logits.to(torch.bfloat16)


# --------------------------------------------------------------------------- greedy


def test_all_greedy_is_argmax():
    s = Sampler("cpu", 32)
    reqs = [Request(i, [1, 2], SamplingParams(4)) for i in range(6)]
    args = s.prepare(reqs)
    assert args is None
    logits = _logits_with_ties()
    out = s.sample(logits, args)
    assert torch.equal(out, torch.argmax(logits, dim=-1))
    assert out[0] == 3 and out[3] == 5


def test_greedy_rows_unaffected_by_sampled_rows():
    s = Sampler("cpu", 32)
    params = [SamplingParams(4, temperature=0.0 if i % 2 == 0 else 1.5, top_p=0.8) for i in range(6)]
    reqs = [Request(i, [1, 2], p) for i, p in enumerate(params)]
    logits = _logits_with_ties()
    out = s.sample(logits, s.prepare(reqs))
    assert torch.equal(out[0::2], torch.argmax(logits, dim=-1)[0::2])
    assert out[0] == 3


# --------------------------------------------------------------------------- distributions

N = 50_000  # rows; one noise draw each. Frequency standard error <= 0.0023.


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_distribution(temperature):
    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, -3.0, 1.5, 0.2])
    out = Sampler("cpu", 8).sample(logits.expand(N, -1), _args(N, temperature))
    expected = torch.softmax(logits.double() / temperature, -1)
    err = (_freq(out, 8) - expected).abs().max().item()
    assert err < 0.01, err


def test_top_p_samples_the_renormalized_nucleus():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.06, 0.04])
    logits = probs.log()
    # mass before each token: 0, 0.4, 0.7, 0.9, 0.96 -> top_p 0.8 keeps the first three (0.9 >= 0.8)
    out = Sampler("cpu", 5).sample(logits.expand(N, -1), _args(N, 1.0, top_p=0.8))
    f = _freq(out, 5)
    assert f[3] == 0 and f[4] == 0  # outside the nucleus: never
    expected = probs[:3].double() / probs[:3].sum()
    assert (f[:3] - expected).abs().max().item() < 0.01


def test_tiny_top_p_is_the_most_likely_token():
    logits = torch.tensor([0.1, 2.0, 1.9, -1.0])
    out = Sampler("cpu", 4).sample(logits.expand(1000, -1), _args(1000, 3.0, top_p=1e-6))
    assert (out == 1).all()


def test_top_p_order_independent_of_token_ids():
    """The nucleus is taken over tokens sorted by probability, not by id."""
    probs = torch.tensor([0.04, 0.2, 0.06, 0.4, 0.3])
    out = Sampler("cpu", 5).sample(probs.log().expand(N, -1), _args(N, 1.0, top_p=0.8))
    f = _freq(out, 5)
    assert f[0] == 0 and f[2] == 0 and (f[[1, 3, 4]] > 0).all()


# --------------------------------------------------------------------------- reproducibility


def test_same_seed_and_position_same_token_anywhere_in_a_batch():
    """A row's sample depends only on its (seed, counter, logits), not on batch size or position."""
    g = torch.Generator().manual_seed(1)
    vocab = 1000
    s = Sampler("cpu", vocab)
    for trial in range(50):
        row = torch.randn(1, vocab, generator=g)
        others = torch.randn(7, vocab, generator=g)
        seed, counter = 1000 + trial, trial
        alone = s.sample(row, _args(1, 1.0, 0.9, seeds=[seed], counter=counter))
        batch = torch.cat([others[:5], row, others[5:]])
        args = _args(8, 1.0, 0.9, seeds=[1, 2, 3, 4, 5, seed, 6, 7], counter=counter)
        assert s.sample(batch, args)[5] == alone[0]


def test_noise_is_uniform_and_keyed():
    seeds = torch.tensor([0, 1, 0], dtype=torch.int64)
    counters = torch.tensor([0, 0, 1], dtype=torch.int64)
    u = uniform_noise(seeds, counters, 4096)
    assert ((u > 0) & (u < 1)).all()
    assert abs(u.mean().item() - 0.5) < 0.02
    assert not torch.equal(u[0], u[1]) and not torch.equal(u[0], u[2])  # seed and position both matter
    assert torch.equal(u, uniform_noise(seeds, counters, 4096))  # a pure function


def test_prepare_counts_the_sampled_position():
    s = Sampler("cpu", 32)
    r = Request(0, [1, 2, 3], SamplingParams(8, temperature=1.0, seed=5))
    r.seed = 5
    r.output_ids = [9, 9]  # e.g. kept across a preemption
    args = s.prepare([r])
    assert args.counter.tolist() == [5] and args.seed.tolist() == [5] and not args.any_top_p


def test_sampling_params_validation():
    for bad in (dict(temperature=-0.1), dict(top_p=0.0), dict(top_p=1.1), dict(seed=-1), dict(seed=2**32)):
        with pytest.raises(ValueError):
            SamplingParams(4, **bad)
    assert SamplingParams(4).is_greedy and not SamplingParams(4, temperature=0.7).is_greedy


# --------------------------------------------------------------------------- GPU: no host synchronization


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["greedy", "sampled", "mixed"])
def test_no_host_sync(kind):
    vocab = 151_936
    s = Sampler("cuda", vocab)
    temps = {"greedy": [0.0] * 8, "sampled": [0.8] * 8, "mixed": [0.0, 0.8] * 4}[kind]
    reqs = [Request(i, [1] * (i + 3), SamplingParams(4, temperature=t, top_p=0.9)) for i, t in enumerate(temps)]
    logits = torch.randn(8, vocab, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = s.sample(logits, s.prepare(reqs))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert out.device.type == "cuda" and out.shape == (8,)
    greedy = [i for i, t in enumerate(temps) if t == 0]
    assert torch.equal(out[greedy], torch.argmax(logits, -1)[greedy])


def test_no_host_readback_in_source():
    """Backs up the sync-debug test above, which does not detect every synchronizing operation."""
    import inspect

    import miniserve.engine.sampler as mod

    src = inspect.getsource(mod)
    for call in (".item(", ".cpu(", ".tolist(", ".numpy(", "synchronize("):
        assert call not in src, call
