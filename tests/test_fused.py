"""The fused operator path (``miniserve/model/fused.py``) against the reference path.

The fused path runs different kernels, each rounding once where the reference
rounds between steps, so it is held to the tolerance rules of any
non-reference path (``anchor.py``), on every route the engine takes: a
request alone, mixed concurrent load, CUDA Graph replay, and speculative
decoding. Concatenating the projection weights, which the fused path needs,
must leave the reference path bitwise unchanged.
"""

from __future__ import annotations

import gc

import pytest
import torch

from anchor import EPS, STOP_IDS, check_against_reference, check_margins
from miniserve.engine.request import RequestState, SamplingParams
from miniserve.engine.scheduler import Phase
from miniserve.model.fused import FusedQwen3ForCausalLM, fuse_projections, with_fused_ops
from prompts import PROMPTS, encode
from test_engine import _assert_no_leak, _check_all, _engine, _run

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _load(path):
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import load_config, load_weights

    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))


@pytest.fixture(scope="module")
def tokenizer(qwen3_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen3_path)


@pytest.fixture(scope="module")
def model(qwen3_path):
    return _load(qwen3_path)


@pytest.fixture(scope="module")
def fused(model):
    return FusedQwen3ForCausalLM(model)


@pytest.fixture(scope="module")
def reference(tokenizer, model):
    """name -> (prompt ids, max_new_tokens, reference tokens, reference top-2 gaps per step)."""
    from miniserve.model.generate import greedy_generate

    out = {}
    for name, (prompt, n) in PROMPTS.items():
        ids = encode(tokenizer, prompt)
        gaps: list[float] = []
        out[name] = (ids, n, greedy_generate(model, ids, n, stop_ids=STOP_IDS, top2_gaps=gaps), gaps)
    return out


def _all_logits(model, ids: list[int]) -> torch.Tensor:
    dev = model.device
    return model.forward(torch.tensor(ids, device=dev), torch.arange(len(ids), device=dev), model.new_cache(len(ids)), all_logits=True)


@torch.inference_mode()
def test_concatenated_weights_leave_the_reference_bitwise(qwen3_path, tokenizer):
    """The reference path on row slices of the concatenated projections gives bitwise the same
    logits as on the separate tensors (prefill, every position, and incremental decode), and the
    concatenation holds no more device memory than the separate tensors did: counted as the
    memory the device has free, since tensors freed next to live ones leave holes the caching
    allocator keeps."""
    from miniserve.model.generate import greedy_generate

    m = _load(qwen3_path)
    ids = encode(tokenizer, PROMPTS["long_en"][0])
    before_logits = _all_logits(m, ids)
    gaps_before: list[float] = []
    before_tokens = greedy_generate(m, ids[:40], 16, top2_gaps=gaps_before)

    def device_memory() -> tuple[int, int]:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]

    alloc_before, free_before = device_memory()
    fuse_projections(m.w, m.cfg)
    alloc_after, free_after = device_memory()
    q = m.w["model.layers.5.self_attn.q_proj.weight"]
    assert q.untyped_storage().data_ptr() == m.w["model.layers.5.self_attn.qkv_proj.weight"].untyped_storage().data_ptr()
    print(f"\nallocated {alloc_before} -> {alloc_after} B; free {free_before} -> {free_after} B")
    assert alloc_after - alloc_before <= 2**20  # alignment padding only
    assert free_after >= free_before, (free_before, free_after)
    fuse_projections(m.w, m.cfg)  # idempotent
    assert m.w["model.layers.5.self_attn.q_proj.weight"] is q
    assert torch.equal(_all_logits(m, ids), before_logits)
    gaps_after: list[float] = []
    assert greedy_generate(m, ids[:40], 16, top2_gaps=gaps_after) == before_tokens
    assert gaps_after == gaps_before
    del m
    gc.collect()
    torch.cuda.empty_cache()


def test_the_switch_shares_weights(model, fused):
    assert with_fused_ops(model, False) is model
    assert with_fused_ops(fused, True) is fused
    assert fused.w is model.w and fused.reference is model
    assert isinstance(with_fused_ops(model, True), FusedQwen3ForCausalLM)


@torch.inference_mode()
def test_prefill_logits_close(model, fused, reference):
    """Each prompt alone, reference vs fused operators: the last position's logits within the
    bound the other kernel paths are held to, and at every position the same argmax wherever
    the reference top-2 gap exceeds the tolerance.

    The largest difference over every position is printed, not bounded: over 629 positions
    of the full vocabulary it reaches the size that batching alone produces on the reference
    operators (1.375 on the long prompt, against 1.22 for the fused ones)."""
    worst_last, worst, decisive_n, total = 0.0, 0.0, 0, 0
    for name in PROMPTS:
        ids = reference[name][0]
        ref, got = _all_logits(model, ids).float(), _all_logits(fused, ids).float()
        worst_last = max(worst_last, (ref[-1] - got[-1]).abs().max().item())
        worst = max(worst, (ref - got).abs().max().item())
        top2 = ref.topk(2, dim=-1).values
        decisive = (top2[:, 0] - top2[:, 1]) > EPS
        assert torch.equal(got.argmax(-1)[decisive], ref.argmax(-1)[decisive]), name
        decisive_n += int(decisive.sum())
        total += len(ids)
    print(
        f"\nfused vs reference prefill: last position max|d| {worst_last}; "
        f"all {total} positions max|d| {worst}, decisive {decisive_n}"
    )
    assert worst_last < 1.0, worst_last


def test_single_request_paged(model, reference):
    """Each anchor prompt alone in the engine on the fused path."""
    eng = _engine(model, attention="paged", fused_ops=True)
    assert eng.fused_ops and isinstance(eng.runner.model, FusedQwen3ForCausalLM)
    reqs = {}
    for name in PROMPTS:
        reqs.update(_run(eng, reference, {0: [name]})[0])
        _assert_no_leak(eng)
    _check_all("fused paged single", model, reference, reqs)


def test_single_request_contiguous(model, reference):
    """The fused operators on the reference attention path (contiguous KV, SDPA)."""
    eng = _engine(model, attention="contiguous", fused_ops=True)
    reqs = {}
    for name in PROMPTS:
        reqs.update(_run(eng, reference, {0: [name]})[0])
    _check_all("fused contiguous single", model, reference, reqs)


@pytest.mark.parametrize(
    "attention, chunk, graph, overlap",
    [
        ("paged", None, True, True),
        ("paged", None, False, True),
        ("paged", None, True, False),
        ("paged", 0, True, True),
        ("paged", 64, True, True),
    ],
    ids=["paged", "paged-eager", "paged-no-overlap", "paged-nochunk", "paged-chunk64"],
)
@pytest.mark.parametrize("arrival", ["all_at_once", "staggered"])
def test_concurrent_matches_reference(arrival, attention, chunk, graph, overlap, model, reference):
    """Mixed concurrent load on the fused path, the same schedules as the reference-path anchor."""
    names = list(PROMPTS)
    schedule = {0: names} if arrival == "all_at_once" else {0: names[:2], 3: names[2:4], 7: names[4:5], 20: names[5:]}
    eng = _engine(model, attention=attention, chunked_prefill_size=chunk, cuda_graph=graph, overlap=overlap, fused_ops=True)
    phases = []
    eng.logits_hook = lambda batch, logits: phases.append(batch.phase)
    reqs, step = _run(eng, reference, schedule)
    _check_all(f"fused {attention} chunk {chunk} {arrival}, {step} steps", model, reference, reqs)
    if chunk == 64 and arrival == "staggered":
        assert Phase.MIXED in phases
    _assert_no_leak(eng)


@pytest.mark.parametrize("lens", [[1, 40], [5, 900, 17, 300, 2500], list(range(3, 40, 3))])
def test_decode_graph_matches_eager(lens, model):
    """A fused decode batch replayed from a CUDA Graph against the same batch run eagerly, with
    the reference-path test's criterion; plus which KV slots the replay wrote."""
    import random

    rng = random.Random(len(lens))
    eng = _engine(model, attention="paged", chunked_prefill_size=0, overlap=False, fused_ops=True)
    runner = eng.runner
    reqs = [eng.add_request([rng.randrange(150_000) for _ in range(n)], SamplingParams(3)) for n in lens]
    while any(r.state is not RequestState.DECODE for r in reqs):
        eng.step()
    batch = eng.scheduler.schedule()
    assert batch.phase is Phase.DECODE and len(batch.requests) == len(lens)
    tables = [r.cache for r in batch.requests]
    bs, dummy = runner.pool.block_size, runner.graphs.dummy_block
    before = runner.pool.buf.cpu()
    graph_logits = runner.forward(batch).float().clone()
    after = runner.pool.buf.cpu()
    changed = (before != after).flatten(4).any(-1).any(0).any(0)
    del before, after
    assert {b * bs + o for b, o in changed.nonzero().tolist() if b != dummy} == {t.slot(t.num_tokens - 1) for t in tables}
    for t in tables:
        t.rewind(1)
    runner.use_cuda_graph = False
    eager_logits = runner.forward(batch).float()
    runner.use_cuda_graph = True
    diff = (graph_logits - eager_logits).abs().max().item()
    top2 = eager_logits.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > 2 * diff
    print(f"\n[{len(lens)} rows] fused graph vs eager max|d| {diff}")
    assert diff <= EPS
    assert torch.equal(graph_logits.argmax(-1)[decisive], eager_logits.argmax(-1)[decisive])
    for r in reqs:
        eng.abort(r.rid)
    _assert_no_leak(eng)


def test_switching_paths_at_run_time(model, reference):
    """One engine, switched between operator paths while idle: each path gives what an engine
    built on it gives, and the switch refuses to run while a request is in flight."""
    fuse_projections(model.w, model.cfg)  # before any graph is captured (check_switchable)
    names = ["short_en", "code", "zh"]
    prompts = [reference[k][0] for k in names]
    params = SamplingParams(24, STOP_IDS)
    fresh = {on: _engine(model, fused_ops=on).generate(prompts, params) for on in (False, True)}
    eng = _engine(model)
    assert not eng.fused_ops
    for on in (True, False, True):
        eng.set_fused_ops(on)
        assert eng.fused_ops is on and isinstance(eng.runner.model, FusedQwen3ForCausalLM) is on
        assert eng.generate(prompts, params) == fresh[on]
    eng.add_request(prompts[0], params)
    with pytest.raises(RuntimeError):
        eng.set_fused_ops(False)
    while eng.has_unfinished:
        eng.step()
    _assert_no_leak(eng)


def test_switching_refuses_to_move_weights_under_captured_graphs(qwen3_path):
    """Fusing moves the weights; graphs captured on the reference path would then read freed
    memory. With graphs the switch is refused until the weights are fused; without, it is fine."""
    from miniserve.spec.engine import SpecEngine

    m = _load(qwen3_path)
    eng = _engine(m, kv_pool_tokens=1024, max_running=4)
    with pytest.raises(RuntimeError, match="fuse_projections"):
        eng.set_fused_ops(True)
    spec = SpecEngine(m, m, gamma=2, max_running=4, kv_pool_tokens=1024)
    with pytest.raises(RuntimeError, match="fuse_projections"):
        spec.set_fused_ops(True)
    eager = _engine(m, kv_pool_tokens=1024, max_running=4, cuda_graph=False)
    eager.set_fused_ops(True)  # nothing captured: the weights may move
    assert eager.generate([[9707, 11, 847]], SamplingParams(4))
    del eng, spec, eager, m
    gc.collect()
    torch.cuda.empty_cache()


def test_the_description_records_the_path(model):
    from miniserve.engine.describe import engine_description

    eng = _engine(model, fused_ops=True, cuda_graph=False)
    assert engine_description(eng)["fused_ops"] is True
    eng.set_fused_ops(False)
    assert engine_description(eng)["fused_ops"] is False


# --------------------------------------------------------------------------- speculative decoding


def test_verify_graph_matches_eager_and_the_reference(model):
    """The fused verify pass (width gamma + 1), replayed and eager, against the reference path over
    each whole sequence: the same token wherever the reference is decisive."""
    import random

    from miniserve.spec.engine import SpecEngine

    rng = random.Random(7)
    gamma, lens = 4, [300, 7, 1200, 64, 33]
    eng = SpecEngine(model, model, gamma=gamma, max_running=8, kv_pool_tokens=8192, fused_ops=True)
    runner, graphs, width = eng.runner, eng._verify_graphs[gamma], gamma + 1
    assert isinstance(runner.model, FusedQwen3ForCausalLM) and graphs.model is runner.model
    tables = [runner.allocator.new_table() for _ in lens]
    runner.reserve(tables, lens)
    prompt_ids = [rng.randrange(150_000) for _ in range(sum(lens))]
    runner.forward_tokens(tables, prompt_ids, [p for n in lens for p in range(n)], lens)
    ids = [rng.randrange(150_000) for _ in range(len(lens) * width)]
    pos = [p for n in lens for p in range(n, n + width)]
    runner.reserve(tables, [width] * len(lens))
    slots = [s for t in tables for s in t.tail_slots(width)]
    runner.fence.wait()
    graph_logits = graphs.run(ids, pos, slots, tables).float().clone()
    for t in tables:
        t.rewind(width)
    runner.reserve(tables, [width] * len(lens))
    eager_logits = runner.forward_tokens(tables, ids, pos, [width] * len(lens)).float()
    ref, start = [], 0
    for i, n in enumerate(lens):
        seq = prompt_ids[start : start + n] + ids[i * width : (i + 1) * width]
        start += n
        ref.append(_all_logits(model, seq)[-width:].float())
    ref = torch.cat(ref)
    top2 = ref.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > EPS
    print(
        f"\n[fused verify, width {width}] max|d| graph-eager {(graph_logits - eager_logits).abs().max().item()}, "
        f"graph-ref {(graph_logits - ref).abs().max().item()}; decisive {int(decisive.sum())}/{len(decisive)}"
    )
    for name, logits in (("graph", graph_logits), ("eager", eager_logits)):
        assert torch.equal(logits.argmax(-1)[decisive], ref.argmax(-1)[decisive]), f"{name} path picked another token"
    for t in tables:
        t.release()


def test_the_drafts_graphs_propose_what_eager_proposes(model):
    """The fused draft's steps (first step width 2, then decode) replayed and eager: the same
    proposals and full acceptance when the draft is the target itself."""
    from miniserve.spec.engine import SpecEngine

    prompts = [[9707, 11, 847, 829, 374], [785, 6722, 315, 9625, 374]]
    runs = {}
    for graphs in (False, True):
        eng = SpecEngine(model, model, gamma=4, max_running=4, kv_pool_tokens=4096, cuda_graph=graphs, fused_ops=True)
        assert eng.draft.fused_ops
        runs[graphs] = (eng.generate(prompts, SamplingParams(24)), eng.acceptance)
    assert runs[True][0] == runs[False][0]
    assert runs[True][1] == runs[False][1] == 1.0, f"graphs {runs[True][1]}, eager {runs[False][1]}"


@pytest.fixture(scope="module")
def target():
    """The 1.7B target of the speculative anchor, loaded once for both of its cases."""
    from test_spec import _load as load_size
    from test_spec import _need_free_memory

    _need_free_memory(5.0)
    return load_size("1.7B")


@pytest.mark.parametrize("alone", [True, False], ids=["alone", "under_load"])
def test_speculation_matches_the_reference(tokenizer, model, target, alone):
    """The speculative anchor with both models on fused operators: 1.7B target, 0.6B draft,
    held to the tolerance rules against the target's reference path."""
    from test_spec import ANCHOR_PROMPTS, ANCHOR_TOKENS, _reference

    from miniserve.spec.engine import SpecEngine

    ids = {k: encode(tokenizer, PROMPTS[k][0]) for k in ANCHOR_PROMPTS}
    refs = {k: _reference(target, ids[k]) for k in ANCHOR_PROMPTS}
    eng = SpecEngine(target, model, gamma=4, max_running=4, kv_pool_tokens=2048, fused_ops=True)
    assert isinstance(eng.runner.model, FusedQwen3ForCausalLM) and eng.draft.fused_ops
    params = SamplingParams(ANCHOR_TOKENS, stop_token_ids=STOP_IDS)
    if alone:
        outs = {k: eng.generate([ids[k]], params)[0] for k in ANCHOR_PROMPTS}
    else:
        outs = dict(zip(ANCHOR_PROMPTS, eng.generate([ids[k] for k in ANCHOR_PROMPTS], params)))
    diverged, worst = {}, 0.0
    for k in ANCHOR_PROMPTS:
        ref, gaps = refs[k]
        pos = check_against_reference(k, outs[k], ref, gaps)
        if pos is not None:
            diverged[k] = (pos, gaps[pos])
        worst = max(worst, check_margins(k, target, ids[k], outs[k]))
    assert 0 < eng.acceptance < 1, f"the draft agreed {eng.acceptance:.3f} of the time: no rejection was exercised"
    print(
        f"\n[fused spec alone={alone}] alpha={eng.acceptance:.3f} tokens/round={eng.tokens_per_round:.2f} "
        f"diverged {len(diverged)}/{len(ANCHOR_PROMPTS)} {diverged}; max teacher-forced margin {worst}"
    )
    del eng
    gc.collect()
    torch.cuda.empty_cache()


def test_spec_switching_paths_at_run_time(model):
    """A speculative engine switched between paths: both models and the round graphs follow,
    and each path proposes and accepts what an engine built on it does."""
    from miniserve.spec.engine import SpecEngine

    fuse_projections(model.w, model.cfg)
    prompts = [[9707, 11, 847, 829, 374], [785, 6722, 315, 9625, 374]]
    params = SamplingParams(24)
    fresh = {
        on: SpecEngine(model, model, gamma=3, max_running=4, kv_pool_tokens=4096, fused_ops=on).generate(prompts, params)
        for on in (False, True)
    }
    eng = SpecEngine(model, model, gamma=3, max_running=4, kv_pool_tokens=4096)
    for on in (True, False):
        eng.set_fused_ops(on)
        assert eng.draft.fused_ops is on and isinstance(eng.runner.model, FusedQwen3ForCausalLM) is on
        assert eng._verify_graphs[3].model is eng.runner.model
        assert eng.generate(prompts, params) == fresh[on]
