"""Speculative decoding: the acceptance rule, the round on a toy model, and the anchor.

The toy tests run the whole loop on the CPU with a deterministic stand-in for
both models, so they can assert the property that matters: whatever the draft
proposes, the engine produces exactly the tokens plain decoding produces. The
stand-in reads every sequence back through its block table, so a token written
to the wrong slot, a position off by one, or KV left behind by a rejected
proposal shows up as a wrong token rather than as a slow run.
"""

from __future__ import annotations

import pytest
import torch

from miniserve.cache.block_allocator import BlockAllocator
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.engine.engine import Engine
from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.request import RequestState, SamplingParams
from miniserve.engine.sampler import Sampler
from miniserve.model.transfer import CopyFence
from miniserve.spec.draft import DraftRunner
from miniserve.spec.engine import SpecEngine, check_options
from miniserve.spec.verify import accept_prefix, accepted_tokens
from anchor import STOP_IDS, check_against_reference, check_margins

TOY_VOCAB = 64


# --------------------------------------------------------------------------- the acceptance rule


def test_accepts_the_agreeing_prefix():
    assert accept_prefix([5, 6, 7], [5, 6, 7, 8]) == 3
    assert accept_prefix([5, 6, 7], [5, 9, 7, 8]) == 1
    assert accept_prefix([5, 6, 7], [9, 6, 7, 8]) == 0
    assert accept_prefix([], [8]) == 0


def test_a_round_always_produces_one_token_and_at_most_gamma_plus_one():
    assert accepted_tokens([5, 6, 7], [5, 6, 7, 8]) == [5, 6, 7, 8]
    assert accepted_tokens([5, 6, 7], [5, 9, 1, 2]) == [5, 9]
    assert accepted_tokens([5, 6, 7], [9, 1, 2, 3]) == [9]


def test_verdict_must_cover_every_proposal_and_the_bonus():
    with pytest.raises(ValueError):
        accept_prefix([5, 6], [5, 6])


# --------------------------------------------------------------------------- CPU toy models


def _target_next(tokens: list[int]) -> int:
    """A deterministic 'model': the next token depends on every token and on the length."""
    return (sum((i + 1) * t for i, t in enumerate(tokens)) * 31 + len(tokens)) % TOY_VOCAB


def _draft_next(tokens: list[int], agree: str) -> int:
    """A draft that agrees with the target always, never, or on two positions out of three."""
    want = _target_next(tokens)
    if agree == "always":
        return want
    if agree == "never":
        return (want + 1) % TOY_VOCAB
    return want if (sum(tokens) + len(tokens)) % 3 else (want + 1) % TOY_VOCAB


class _ToyModel:
    """Stands in for a model and for the paged attention backend under it.

    Its store maps a pool slot to the (token, position) written there. Every pass
    writes the new tokens at the planned slots and then reads each sequence back
    through its block table, so stale or misplaced KV becomes a wrong answer.
    """

    def __init__(self, choose):
        self.choose = choose
        self.store: dict[int, tuple[int, int]] = {}
        self.passes = 0

    # the attention backend's half
    def plan(self, prefill, qo_lens, tables, slots):
        self.qo_lens, self.tables, self.slots = list(qo_lens), list(tables), list(slots)

    # the model's half
    def forward_with(self, input_ids, positions, attn, seq_lens):
        self._write(input_ids, positions)
        return torch.stack([self._logits(self._history(t)) for t in self.tables])

    def decode_logits(self, input_ids, positions, attn):
        """Logits of every new token, the shape a verify pass needs."""
        self._write(input_ids, positions)
        rows = []
        for table, n in zip(self.tables, self.qo_lens):
            hist = self._history(table)
            rows += [self._logits(hist[: len(hist) - n + j + 1]) for j in range(n)]
        return torch.stack(rows)

    def _write(self, input_ids, positions):
        self.passes += 1
        for slot, t, p in zip(self.slots, input_ids.tolist(), positions.tolist()):
            self.store[slot] = (t, p)

    def _history(self, table) -> list[int]:
        hist = [self.store[table.slot(i)] for i in range(table.num_tokens)]
        assert [p for _, p in hist] == list(range(table.num_tokens)), "KV positions out of order"
        return [t for t, _ in hist]

    def _logits(self, tokens: list[int]) -> torch.Tensor:
        out = torch.zeros(TOY_VOCAB)
        out[self.choose(tokens)] = 10.0
        return out


def _toy_runner(model, num_blocks: int, block_size: int) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)  # the real runner logic, without a GPU model
    runner.model = runner.flashinfer = model
    runner.device = torch.device("cpu")
    runner.attention = "paged"
    runner.block_backend = "python"
    runner.allocator = BlockAllocator(num_blocks, block_size)
    runner.kv = KVCacheManager(runner.allocator, False)
    runner.sampler = Sampler("cpu", TOY_VOCAB)
    runner.graphs, runner.use_cuda_graph = None, False
    runner.fence = CopyFence("cpu")
    return runner


def _toy_draft(model, num_blocks: int, block_size: int) -> DraftRunner:
    draft = DraftRunner.__new__(DraftRunner)
    draft.model = draft.attn = model
    draft.device = torch.device("cpu")
    draft.block_size = block_size
    draft.max_prefill_tokens = 4096
    draft.allocator = BlockAllocator(num_blocks, block_size)
    draft.pool = None
    draft.fence = CopyFence("cpu")
    draft.tables = {}
    draft.graphs = draft.first_graphs = None
    draft.use_first_graphs = False
    return draft


class _ToyGraphs:
    """Stands in for ``DecodeGraphs`` of some width: the same call, run eagerly on the toy
    model. What it checks is the host side of a graph pass: the inputs, slots and fills the
    round builds for a fixed width, and which rows of logits it takes back."""

    def __init__(self, model, width: int, max_batch: int = 8):
        self.model, self.width, self.max_batch = model, width, max_batch
        self.runs = 0

    def run(self, ids, pos, slots, tables, fill=None):
        assert len(ids) == len(pos) == len(slots) == len(tables) * self.width
        self.runs += 1
        self.model.plan(True, [self.width] * len(tables), tables, slots)
        ids_t = torch.tensor(ids)
        if fill is not None:
            ids_t.index_copy_(0, *fill)
        return self.model.decode_logits(ids_t, torch.tensor(pos), None)


def _toy_spec_engine(
    num_blocks: int, block_size: int, gamma: int, agree: str, graphs: bool = False, **kw
) -> SpecEngine:
    eng = SpecEngine.__new__(SpecEngine)
    Engine.__init__(
        eng,
        None,
        runner=_toy_runner(_ToyModel(_target_next), num_blocks, block_size),
        radix=False,
        overlap=False,
        **kw,
    )
    eng.draft = _toy_draft(_ToyModel(lambda t: _draft_next(t, agree)), num_blocks, block_size)
    eng._gamma, eng._fill_idx = 0, {}
    eng._cuda_graph, eng._round_graphs, eng._verify_graphs = False, graphs, {}
    if graphs:
        eng._verify_graphs[gamma] = _ToyGraphs(eng.runner.model, gamma + 1)
        eng.draft.first_graphs = _ToyGraphs(eng.draft.model, 2)
        eng.draft.use_first_graphs = True
    eng.gamma = gamma
    eng.spec_stats = dict(rounds=0, rows=0, proposed=0, accepted=0, tokens=0)
    return eng


def _toy_engine(num_blocks: int, block_size: int, **kw) -> Engine:
    return Engine(None, runner=_toy_runner(_ToyModel(_target_next), num_blocks, block_size), radix=False, **kw)


def _generate(eng, prompts, params) -> list[list[int]]:
    reqs = [eng.add_request(p, params) for p in prompts]
    while eng.has_unfinished:
        eng.step()
        if eng.runner.kv is not None:
            eng.runner.kv.check_invariants()
    return [r.output_ids for r in reqs]


PROMPTS = [[3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8], [1] * 20, [11, 22, 33, 44, 55, 6, 7]]


# --------------------------------------------------------------------------- the round, on the toy


@pytest.mark.parametrize("agree", ["always", "sometimes", "never"])
@pytest.mark.parametrize("gamma", [1, 2, 4])
@pytest.mark.parametrize("chunk", [0, 2048])
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_speculation_does_not_change_the_output(gamma, agree, chunk, graphs):
    """The property the whole design rests on, over every acceptance pattern:
    full acceptance (which leaves the draft a token behind), partial, and none; and over
    both ways a round runs (with graphs, the draft's first step feeds two tokens per
    request, recomputing one it already had)."""
    params = SamplingParams(24)
    plain = _generate(_toy_engine(64, 4, chunked_prefill_size=chunk), PROMPTS, params)
    eng = _toy_spec_engine(64, 4, gamma, agree, graphs=graphs, chunked_prefill_size=chunk)
    assert _generate(eng, PROMPTS, params) == plain
    assert eng.spec_stats["rounds"] > 0
    if agree == "always":
        assert eng.acceptance == 1.0
        # gamma + 1 per round, but for the last round of a request, which its token budget cuts short.
        assert gamma < eng.tokens_per_round <= gamma + 1
    if agree == "never":
        assert eng.acceptance == 0.0
        assert eng.tokens_per_round == 1.0


@pytest.mark.parametrize("gamma", [0, 1, 3])
def test_gamma_zero_is_plain_decoding(gamma):
    params = SamplingParams(16)
    plain = _generate(_toy_engine(64, 4), PROMPTS, params)
    assert _generate(_toy_spec_engine(64, 4, gamma, "sometimes"), PROMPTS, params) == plain


def test_gamma_can_change_mid_flight():
    """The ablation switches gamma between runs, and a request admitted while it was 0 has
    no draft cache at all: the first round has to fill it before it can propose."""
    params = SamplingParams(20)
    plain = _generate(_toy_engine(64, 4), PROMPTS, params)
    eng = _toy_spec_engine(64, 4, 0, "sometimes")
    reqs = [eng.add_request(p, params) for p in PROMPTS]
    for _ in range(4):
        eng.step()
    assert eng.draft.model.passes == 0, "the draft ran while gamma was 0"
    eng.gamma = 3
    while eng.has_unfinished:
        eng.step()
    assert [r.output_ids for r in reqs] == plain
    assert eng.spec_stats["rounds"] > 0


@pytest.mark.parametrize("agree", ["always", "sometimes"])
def test_a_round_stops_at_the_stop_token(agree):
    """Tokens past a stop token are dropped, KV and all, as if they had never been proposed."""
    stops = frozenset({_target_next(PROMPTS[0] + [_target_next(PROMPTS[0])])})
    params = SamplingParams(24, stop_token_ids=stops)
    plain = _generate(_toy_engine(64, 4), PROMPTS, params)
    eng = _toy_spec_engine(64, 4, 4, agree)
    ours = _generate(eng, PROMPTS, params)
    assert ours == plain
    assert any(o and o[-1] in stops for o in ours), "the stop token was never reached"
    assert all(s not in o[:-1] for o in ours for s in stops)


@pytest.mark.parametrize("agree", ["always", "sometimes"])
@pytest.mark.parametrize("num_blocks", [12, 20])
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_speculation_under_kv_pressure(num_blocks, agree, graphs):
    """A pool too small for every request at once: preemption, readmission, and a draft
    cache that has to be thrown away and recomputed with it."""
    params = SamplingParams(20)
    plain = _generate(_toy_engine(num_blocks, 4), PROMPTS, params)
    eng = _toy_spec_engine(num_blocks, 4, 4, agree, graphs=graphs)
    assert _generate(eng, PROMPTS, params) == plain
    assert eng.scheduler.num_preemptions > 0, "the pool was not tight enough to preempt"
    assert not eng.draft.tables, "a draft cache outlived its request"
    assert eng.draft.allocator.num_free == eng.draft.allocator.num_blocks


@pytest.mark.parametrize("agree", ["always", "sometimes", "never"])
def test_a_round_replays_its_graphs(agree):
    """Every round of a steady decode batch takes the graph paths: the verify pass, and the
    draft's first step with every request fed two tokens (whatever the last round accepted)."""
    eng = _toy_spec_engine(64, 4, 3, agree, graphs=True)
    _generate(eng, PROMPTS, SamplingParams(24))
    rounds = eng.spec_stats["rounds"]
    assert eng._verify_graphs[3].runs == rounds
    assert eng.draft.first_graphs.runs == rounds


def test_round_graphs_can_be_switched_off():
    params = SamplingParams(20)
    plain = _generate(_toy_engine(64, 4), PROMPTS, params)
    eng = _toy_spec_engine(64, 4, 3, "sometimes", graphs=True)
    eng.round_graphs = False
    assert _generate(eng, PROMPTS, params) == plain
    assert eng._verify_graphs[3].runs == eng.draft.first_graphs.runs == 0


def test_requests_reaching_decode_together_share_one_draft_prefill():
    """The draft's prefill packs requests into passes rather than running one per request."""
    eng = _toy_spec_engine(64, 4, 3, "sometimes")
    reqs = [eng.add_request(p, SamplingParams(8)) for p in PROMPTS]
    eng.step()  # the target's prefill of all four, then the draft's
    assert all(r.state is RequestState.DECODE for r in reqs)
    assert eng.draft.model.passes == 1
    assert all(eng.draft.covered(r.rid) == r.seq_len - 1 for r in reqs)


def test_a_draft_prefill_longer_than_a_pass_continues_in_order():
    eng = _toy_spec_engine(64, 4, 3, "sometimes")
    eng.draft.max_prefill_tokens = 5  # the prompts hold 8, 4, 20 and 7 tokens
    params = SamplingParams(12)
    plain = _generate(_toy_engine(64, 4), PROMPTS, params)
    assert _generate(eng, PROMPTS, params) == plain


def test_the_draft_never_holds_more_tokens_than_the_target():
    """The argument that sizing the draft pool like the target's is enough."""
    eng = _toy_spec_engine(64, 4, 4, "always")
    reqs = [eng.add_request(p, SamplingParams(20)) for p in PROMPTS]
    while eng.has_unfinished:
        eng.step()
        for r in reqs:
            if r.rid in eng.draft.tables and r.cache is not None:
                assert eng.draft.covered(r.rid) <= r.cache.num_tokens
        assert eng.draft.allocator.num_free >= eng.runner.allocator.num_free


def test_sampling_is_refused():
    eng = _toy_spec_engine(64, 4, 4, "always")
    with pytest.raises(ValueError, match="greedy-only"):
        eng.add_request([1, 2, 3], SamplingParams(4, temperature=0.7))


def test_the_prefix_cache_and_overlap_are_refused():
    check_options("paged", radix=False, overlap=False, gamma=4)
    with pytest.raises(ValueError, match="prefix cache"):
        check_options("paged", radix=True, overlap=False, gamma=4)
    with pytest.raises(ValueError, match="overlap"):
        check_options("paged", radix=False, overlap=True, gamma=4)
    with pytest.raises(ValueError, match="paged"):
        check_options("contiguous", radix=False, overlap=False, gamma=4)


# --------------------------------------------------------------------------- GPU: the anchor


@pytest.fixture(scope="module")
def tokenizer(qwen3_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen3_path)


def _load(size: str):
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import load_config, load_weights, model_path, spec_for

    path = model_path(spec_for(size), download=False)
    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))


def _need_free_memory(gib: float) -> None:
    """Skip unless the device can still hold what follows.

    The anchor below holds two models at once, which is several times what any other test
    needs. Run after the rest of the suite, the device is full of what earlier modules
    allocated, and no amount of collecting inside this process gives it back."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 1024**3
    if free < gib:
        pytest.skip(
            f"needs {gib:.1f} GiB of free device memory, {free:.1f} GiB left; "
            f"run this file on its own: make test PYTEST_ARGS='tests/test_spec.py'"
        )


@pytest.fixture(scope="module")
def qwen3(qwen3_path):
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import load_config, load_weights

    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(qwen3_path)), load_weights(qwen3_path))


ANCHOR_PROMPTS = ["short_en", "code", "zh"]
ANCHOR_TOKENS = 64


def _reference(model, ids: list[int]):
    """Reference path (one sequence, contiguous attention) and its top-2 gap per step."""
    from miniserve.model.generate import greedy_generate

    gaps: list[float] = []
    return greedy_generate(model, ids, ANCHOR_TOKENS, stop_ids=STOP_IDS, top2_gaps=gaps), gaps


@pytest.mark.gpu
@pytest.mark.slow
def test_the_drafts_graphs_propose_what_eager_proposes(qwen3):
    """The draft's decode steps follow each other with nothing in between, which is the one
    place where a replayed graph can read a staging buffer the host has already rewritten."""
    prompts = [[9707, 11, 847, 829, 374], [785, 6722, 315, 9625, 374]]
    params = SamplingParams(24)
    runs = {}
    for graphs in (False, True):
        eng = SpecEngine(qwen3, qwen3, gamma=4, max_running=4, kv_pool_tokens=4096, cuda_graph=graphs)
        runs[graphs] = (eng.generate(prompts, params), eng.acceptance)
    assert runs[True][0] == runs[False][0]
    assert runs[True][1] == runs[False][1] == 1.0, f"graphs {runs[True][1]}, eager {runs[False][1]}"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("lens", [[40], [5, 900, 2500], [300, 7, 1200, 64, 2000, 33]])
def test_a_verify_graph_matches_the_eager_pass(qwen3, gamma, lens):
    """A verify pass replayed from a graph of width gamma + 1 against the same pass run eagerly:
    every position's logits within the tolerance, and the padding rows write nothing but the
    dummy block. The graphs were captured over sequences as short as the width, so this also
    checks that planning long sequences updates what the replay reads."""
    import random

    from anchor import EPS

    rng = random.Random(len(lens) * 10 + gamma)
    eng = SpecEngine(qwen3, qwen3, gamma=gamma, max_running=8, kv_pool_tokens=8192)
    runner, graphs, width = eng.runner, eng._verify_graphs[gamma], gamma + 1
    tables = [runner.allocator.new_table() for _ in lens]
    runner.reserve(tables, lens)
    prompt_ids = [rng.randrange(150_000) for _ in range(sum(lens))]
    runner.forward_tokens(tables, prompt_ids, [p for n in lens for p in range(n)], lens)
    ids = [rng.randrange(150_000) for _ in range(len(lens) * width)]
    pos = [p for n in lens for p in range(n, n + width)]
    runner.reserve(tables, [width] * len(lens))
    slots = [s for t in tables for s in t.tail_slots(width)]
    before = runner.pool.buf.cpu()
    runner.fence.wait()
    graph_logits = graphs.run(ids, pos, slots, tables).float().clone()
    after = runner.pool.buf.cpu()
    bs, dummy = runner.pool.block_size, graphs.dummy_block
    changed = (before != after).flatten(4).any(-1).any(0).any(0)  # [blocks, block_size]
    del before, after
    assert {b * bs + o for b, o in changed.nonzero().tolist() if b != dummy} == set(slots)
    for t in tables:
        t.rewind(width)
    runner.reserve(tables, [width] * len(lens))
    eager_logits = runner.forward_tokens(tables, ids, pos, [width] * len(lens)).float()
    diff = (graph_logits - eager_logits).abs().max().item()
    top2 = eager_logits.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > 2 * diff
    print(f"\n[{len(lens)} rows, width {width}] verify graph vs eager max|d| {diff}")
    assert diff <= EPS
    assert torch.equal(graph_logits.argmax(-1)[decisive], eager_logits.argmax(-1)[decisive])
    for t in tables:
        t.release()


@pytest.mark.gpu
@pytest.mark.slow
def test_the_drafts_first_step_graph_matches_eager(qwen3):
    """The draft's first step, width 2, graph against eager: the logits of each request's last token."""
    import random

    from anchor import EPS

    rng = random.Random(3)
    lens = [9, 700, 31, 1500, 2]
    eng = SpecEngine(qwen3, qwen3, gamma=3, max_running=8, kv_pool_tokens=8192)
    draft = eng.draft
    rids = list(range(len(lens)))
    for rid, n in zip(rids, lens):
        draft.admit(rid)
    draft.prefill([(rid, [rng.randrange(150_000) for _ in range(n)]) for rid, n in zip(rids, lens)])
    feed = [[rng.randrange(150_000), rng.randrange(150_000)] for _ in lens]
    tables = [draft.tables[r] for r in rids]
    out = {}
    for graphs in (True, False):
        draft.use_first_graphs = graphs
        out[graphs] = draft._extend(tables, feed, lens).float().clone()
        for t in tables:
            t.rewind(2)
    assert draft.first_graphs is not None
    diff = (out[True] - out[False]).abs().max().item()
    print(f"\n[{len(lens)} rows, width 2] draft first-step graph vs eager max|d| {diff}")
    assert diff <= EPS
    assert torch.equal(out[True].argmax(-1), out[False].argmax(-1))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("gamma", [2, 5])
def test_self_speculation_accepts_everything(qwen3, gamma):
    """Proposing with the target itself: every proposal is its own, so a round produces
    gamma + 1 tokens and the loop's fast path (full acceptance, which leaves the draft one
    token behind) runs every round."""
    prompts = [[9707, 11, 847, 829, 374], [785, 6722, 315, 9625, 374], [16, 488, 220, 16, 284]]
    eng = SpecEngine(qwen3, qwen3, gamma=gamma, max_running=8, kv_pool_tokens=4096)
    eng.generate(prompts, SamplingParams(32))
    # Not exactly 1: the draft's batched decode and the verify pass run different kernels,
    # so an exact BF16 tie can break the other way and end the round there. The larger gamma
    # is, the more positions a round has to get through without meeting one.
    assert eng.acceptance > 0.9, f"self-speculation accepted {eng.acceptance:.3f}"
    assert eng.tokens_per_round > gamma * 0.9


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("alone", [True, False], ids=["alone", "under_load"])
def test_speculation_matches_the_reference(tokenizer, qwen3, alone):
    """The anchor, with a draft that is wrong often enough to exercise every path: a
    rejected proposal, the KV given back for it, and a round cut short by a stop token.

    Held to the tolerance rules (``anchor.py``), not to bitwise equality: a verify pass runs
    several tokens of several sequences through kernels the reference never uses.
    """
    from prompts import PROMPTS, encode

    _need_free_memory(5.0)  # the 1.7B target next to the 0.6B draft this module already holds
    target, draft = _load("1.7B"), qwen3  # a draft three times smaller: it is wrong often enough
    ids = {k: encode(tokenizer, PROMPTS[k][0]) for k in ANCHOR_PROMPTS}
    refs = {k: _reference(target, ids[k]) for k in ANCHOR_PROMPTS}
    eng = SpecEngine(target, draft, gamma=4, max_running=4, kv_pool_tokens=2048)
    params = SamplingParams(ANCHOR_TOKENS, stop_token_ids=STOP_IDS)
    if alone:
        outs = {k: eng.generate([ids[k]], params)[0] for k in ANCHOR_PROMPTS}
    else:
        got = eng.generate([ids[k] for k in ANCHOR_PROMPTS], params)
        outs = dict(zip(ANCHOR_PROMPTS, got))
    diverged = 0
    for k in ANCHOR_PROMPTS:
        ref, gaps = refs[k]
        diverged += check_against_reference(k, outs[k], ref, gaps) is not None
        check_margins(k, target, ids[k], outs[k])
    assert 0 < eng.acceptance < 1, f"the draft agreed {eng.acceptance:.3f} of the time: no rejection was exercised"
    print(f"alone={alone} alpha={eng.acceptance:.3f} tokens/round={eng.tokens_per_round:.2f} diverged={diverged}")
