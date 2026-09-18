"""Engine loop: request state machine, scheduling, and the batched correctness anchor.

CPU tests cover the state machine and the scheduler. GPU tests check that the
engine reproduces single-request greedy decoding:

- one request alone in the engine runs the same shapes as the reference path,
  so its output must be token-exact;
- under concurrent load, batching changes GEMM shapes and BF16 logits can move
  by an ulp. A sequence may then diverge from the reference only where the
  reference top-1/top-2 logit gap is at most ``EPS``; token comparison of that
  sequence stops at the divergence. A divergence at a larger gap is a bug.
- because that check says nothing about tokens after the divergence, every
  engine token is also checked teacher-forced: running the reference path on
  the engine's own output, each chosen token's logit must be within ``EPS`` of
  the reference maximum. Injected batching bugs (wrong decode positions,
  positions not reset per prompt, swapped inputs) all fail this check, while
  some of them pass the first-divergence check alone.
"""

from __future__ import annotations

import pytest
import torch

from miniserve.engine.engine import Engine
from miniserve.engine.request import InvalidTransition, Request, RequestState, SamplingParams
from miniserve.engine.scheduler import Phase, Scheduler
from prompts import PROMPTS, encode

# Largest reference logit gap tolerated for a batched token. Measured on
# 64 sequences x 128 tokens: the largest gap at an actual divergence was 0.25
# (most were exact ties, gap 0.0), and teacher-forced margins never exceeded
# 0.25 either; EPS keeps a 2x margin.
EPS = 0.5
STOP_IDS = frozenset({151645, 151643})


def _req(rid: int, n_prompt: int, max_new: int = 4) -> Request:
    return Request(rid, list(range(1, n_prompt + 1)), SamplingParams(max_new))


# --------------------------------------------------------------------------- state machine


def test_request_lifecycle():
    r = _req(0, 3)
    assert r.state is RequestState.WAITING
    r.transition(RequestState.PREFILL)
    r.transition(RequestState.DECODE)
    r.transition(RequestState.FINISHED)
    assert r.is_done


@pytest.mark.parametrize(
    "path",
    [
        [RequestState.DECODE],  # must prefill first
        [RequestState.FINISHED],
        [RequestState.PREFILL, RequestState.WAITING],
        [RequestState.PREFILL, RequestState.DECODE, RequestState.PREFILL],
        [RequestState.ABORTED, RequestState.PREFILL],
        [RequestState.PREFILL, RequestState.FINISHED, RequestState.ABORTED],
    ],
)
def test_invalid_transitions(path):
    r = _req(0, 3)
    with pytest.raises(InvalidTransition):
        for s in path:
            r.transition(s)


@pytest.mark.parametrize("state", [RequestState.WAITING, RequestState.PREFILL, RequestState.DECODE])
def test_abort_from_any_live_state(state):
    r = _req(0, 3)
    for s in [RequestState.PREFILL, RequestState.DECODE]:
        if r.state is state:
            break
        r.transition(s)
    r.transition(RequestState.ABORTED)
    assert r.is_done


def test_stop_rule():
    r = Request(0, [1], SamplingParams(3, frozenset({9})))
    r.output_ids = [5]
    assert not r.should_stop()
    r.output_ids = [5, 9]
    assert r.should_stop()  # stop token, kept in the output
    r.output_ids = [5, 6, 7]
    assert r.should_stop()  # length


def test_request_validation():
    with pytest.raises(ValueError):
        Request(0, [], SamplingParams(1))
    with pytest.raises(ValueError):
        SamplingParams(0)


# --------------------------------------------------------------------------- scheduler


def _to_decode(batch):
    for r in batch.requests:
        r.transition(RequestState.DECODE)


def test_prefill_first_then_decode():
    s = Scheduler()
    a, b = _req(0, 3), _req(1, 5)
    s.add(a)
    s.add(b)
    batch = s.schedule()
    assert batch.phase is Phase.PREFILL and batch.requests == [a, b]
    assert batch.seq_lens == [3, 5]
    _to_decode(batch)
    batch = s.schedule()
    assert batch.phase is Phase.DECODE and batch.requests == [a, b] and batch.seq_lens == [1, 1]

    c = _req(2, 2)
    s.add(c)
    batch = s.schedule()  # a new arrival preempts the decode step
    assert batch.phase is Phase.PREFILL and batch.requests == [c]
    _to_decode(batch)
    assert s.schedule().requests == [a, b, c]  # admission order


def test_prefill_token_budget():
    s = Scheduler(max_prefill_tokens=10)
    reqs = [_req(i, 4) for i in range(4)]
    for r in reqs:
        s.add(r)
    assert s.schedule().requests == reqs[:2]  # 4 + 4 fits, a third would be 12
    assert s.schedule().requests == reqs[2:]


def test_oversized_prompt_admitted_alone():
    s = Scheduler(max_prefill_tokens=10)
    big, small = _req(0, 50), _req(1, 2)
    s.add(big)
    s.add(small)
    assert s.schedule().requests == [big]
    assert s.schedule().requests == [small]


def test_max_running():
    s = Scheduler(max_running=2)
    reqs = [_req(i, 1) for i in range(3)]
    for r in reqs:
        s.add(r)
    batch = s.schedule()
    assert batch.requests == reqs[:2]
    _to_decode(batch)
    assert s.schedule().phase is Phase.DECODE  # third waits: running is full
    reqs[0].transition(RequestState.FINISHED)
    s.retire(reqs[0])
    assert s.schedule().requests == [reqs[2]]


def test_retire_waiting_and_idle():
    s = Scheduler()
    r = _req(0, 3)
    s.add(r)
    with pytest.raises(ValueError):
        s.retire(r)  # not done
    r.transition(RequestState.ABORTED)
    s.retire(r)
    assert s.schedule() is None and not s.has_unfinished


# --------------------------------------------------------------------------- GPU: correctness anchor


@pytest.fixture(scope="module")
def tokenizer(qwen3_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen3_path)


@pytest.fixture(scope="module")
def model(qwen3_path):
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import load_config, load_weights

    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(qwen3_path)), load_weights(qwen3_path))


@pytest.fixture(scope="module")
def reference(tokenizer, model):
    """name -> (prompt ids, max_new_tokens, reference tokens, reference top-2 gaps per step)."""
    from miniserve.model.generate import greedy_generate

    out = {}
    for name, (prompt, n) in PROMPTS.items():
        ids = encode(tokenizer, prompt)
        gaps: list[float] = []
        toks = greedy_generate(model, ids, n, stop_ids=STOP_IDS, top2_gaps=gaps)
        out[name] = (ids, n, toks, gaps)
    return out


def _check_against_reference(name, ours, ref, gaps) -> int | None:
    """Return the divergence position (None if identical); fail if it is not at a near-tie."""
    if ours == ref:
        return None
    pos = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), None)
    assert pos is not None, f"[{name}] same prefix but different lengths: ours={len(ours)} ref={len(ref)}"
    assert gaps[pos] <= EPS, (
        f"[{name}] diverges at position {pos} where the reference top-2 gap is {gaps[pos]:.4f} > EPS={EPS}: "
        f"ours={ours[pos]} ref={ref[pos]}"
    )
    return pos


@torch.inference_mode()
def _forced_margins(model, prompt: list[int], tokens: list[int]) -> list[float]:
    """Reference path teacher-forced on ``tokens``: top-1 logit minus the chosen token's logit, per step."""
    cache = model.new_cache(len(prompt) + len(tokens))
    ids = torch.tensor(prompt, device=model.device)
    pos = torch.arange(len(prompt), device=model.device)
    out = []
    for t in tokens:
        logits = model.forward(ids, pos, cache).float()
        out.append(float(logits.max() - logits[t]))
        ids = torch.tensor([t], device=model.device)
        pos = torch.tensor([cache.length], device=model.device)
    return out


def _check_margins(name, model, prompt, tokens) -> float:
    m = _forced_margins(model, prompt, tokens)
    worst = max(range(len(m)), key=m.__getitem__)
    assert m[worst] <= EPS, (
        f"[{name}] token {tokens[worst]} at position {worst} is {m[worst]:.4f} below the reference maximum (EPS={EPS})"
    )
    return m[worst]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", list(PROMPTS))
def test_single_request_token_exact(name, model, reference):
    """Alone in the engine, a request runs the same shapes as the reference path."""
    ids, n, ref, _ = reference[name]
    eng = Engine(model)
    [ours] = eng.generate([ids], SamplingParams(n, STOP_IDS))
    assert ours == ref
    assert not eng.has_unfinished and not eng.requests


@pytest.mark.gpu
@pytest.mark.slow
def test_batched_prefill_logits_close(model, reference):
    """Model level: one prefill over all prompts vs. each prompt alone.

    Not bitwise (GEMM shapes differ). Differences must stay at the BF16 noise
    level, and argmax must agree wherever the top-2 gap is clear of it.
    """
    prompts = [reference[k][0] for k in PROMPTS]
    dev = model.device
    batched = model.forward_batch(
        torch.tensor([t for p in prompts for t in p], device=dev),
        torch.tensor([i for p in prompts for i in range(len(p))], device=dev),
        [model.new_cache(len(p)) for p in prompts],
        [len(p) for p in prompts],
    ).float()
    alone = torch.stack(
        [model.forward(torch.tensor(p, device=dev), torch.arange(len(p), device=dev), model.new_cache(len(p))) for p in prompts]
    ).float()
    max_diff = (batched - alone).abs().max().item()
    print(f"\nbatched vs single prefill over {len(prompts)} prompts: max|d| = {max_diff}")
    assert max_diff < 1.0, max_diff
    top2 = torch.topk(alone, 2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > EPS
    assert torch.equal(batched.argmax(-1)[decisive], alone.argmax(-1)[decisive])


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arrival", ["all_at_once", "staggered"])
def test_concurrent_matches_reference(arrival, model, reference):
    """Mixed concurrent load: prefills of several prompts and decode batches of varying size."""
    names = list(PROMPTS)
    # step -> names arriving before that step
    schedule = {0: names} if arrival == "all_at_once" else {0: names[:2], 3: names[2:4], 7: names[4:5], 20: names[5:]}
    eng = Engine(model)
    reqs, step = {}, 0
    while eng.has_unfinished or step <= max(schedule):
        for name in schedule.get(step, []):
            ids, n, _, _ = reference[name]
            reqs[name] = eng.add_request(ids, SamplingParams(n, STOP_IDS))
        eng.step()
        step += 1

    diverged, worst = {}, 0.0
    for name, r in reqs.items():
        assert r.state is RequestState.FINISHED and r.cache is None
        ids, _, ref, gaps = reference[name]
        pos = _check_against_reference(name, r.output_ids, ref, gaps)
        if pos is not None:
            diverged[name] = (pos, gaps[pos])
        worst = max(worst, _check_margins(name, model, ids, r.output_ids))
    print(
        f"\n[{arrival}] {step} steps; diverged {len(diverged)}/{len(reqs)} (position, reference gap): {diverged}; "
        f"max teacher-forced margin {worst}"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_abort_mid_decode(model, reference):
    names = ["short_en", "code", "zh"]
    eng = Engine(model)
    reqs = {k: eng.add_request(reference[k][0], SamplingParams(reference[k][1], STOP_IDS)) for k in names}
    for _ in range(10):
        eng.step()
    victim = reqs["code"]
    assert victim.state is RequestState.DECODE and victim.cache is not None
    eng.abort(victim.rid)
    assert victim.state is RequestState.ABORTED and victim.cache is None
    assert victim not in eng.scheduler.running and victim.rid not in eng.requests
    eng.abort(victim.rid)  # idempotent
    while eng.has_unfinished:
        eng.step()
    for k in ("short_en", "zh"):
        _check_against_reference(k, reqs[k].output_ids, reference[k][2], reference[k][3])
        _check_margins(k, model, reference[k][0], reqs[k].output_ids)
    assert len(victim.output_ids) == 10
