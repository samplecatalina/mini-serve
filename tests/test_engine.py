"""Engine loop: request state machine, scheduling, and the batched correctness anchor.

CPU tests cover the state machine and the scheduler. GPU tests check that the
engine reproduces single-request greedy decoding:

- on the contiguous (reference) attention path, one request alone in the
  engine runs the same shapes as the reference, so its output must be
  token-exact;
- the paged path uses FlashInfer kernels, so even a single request is held to
  the tolerance rules below rather than bitwise equality;
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

from miniserve.cache.block_allocator import BlockAllocator, OutOfBlocks
from miniserve.cache.block_table import BlockTable
from miniserve.cache.kv_cache import KVCacheManager
from miniserve.engine.cuda_graph import bucket_for, graph_buckets
from miniserve.model.transfer import CopyFence
from miniserve.engine.engine import Engine
from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.request import InvalidTransition, Request, RequestState, SamplingParams
from miniserve.engine.sampler import Sampler
from miniserve.engine.scheduler import Batch, Phase, Scheduler
from prompts import PROMPTS, encode

# Largest reference logit gap tolerated for a batched token. Measured on
# 64 sequences x 128 tokens: the largest gap at an actual divergence was 0.25
# (most were exact ties, gap 0.0), and teacher-forced margins never exceeded
# 0.25 either; EPS keeps a 2x margin.
EPS = 0.5
STOP_IDS = frozenset({151645, 151643})
ATTENTION = ["contiguous", "paged"]
# GPU tests use a fixed pool instead of one sized to all free memory.
POOL_TOKENS = 16384


def _engine(model, **kw) -> Engine:
    kw.setdefault("kv_pool_tokens", POOL_TOKENS)
    return Engine(model, **kw)


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


def test_preempted_request_resumes_with_its_output():
    r = _req(0, 3)
    r.transition(RequestState.PREFILL)
    r.transition(RequestState.DECODE)
    r.output_ids = [7, 8]
    r.transition(RequestState.WAITING)  # preempted
    assert r.seq_len == 5  # the next prefill runs prompt + output
    r.transition(RequestState.PREFILL)
    assert Batch(Phase.PREFILL, [r]).seq_lens == [5]


def test_preempted_between_chunks():
    r = _req(0, 3)
    r.transition(RequestState.PREFILL)
    r.transition(RequestState.WAITING)  # preempted part-way through a chunked prefill
    assert r.state is RequestState.WAITING


@pytest.mark.parametrize(
    "path",
    [
        [RequestState.DECODE],  # must prefill first
        [RequestState.WAITING],
        [RequestState.PREFILL, RequestState.DECODE, RequestState.WAITING, RequestState.DECODE],  # re-prefill first
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


# --------------------------------------------------------------------------- scheduler: KV budget and preemption


def test_graph_buckets():
    assert graph_buckets(1) == [1]
    assert graph_buckets(64) == [1, 2, 4, 8, 16, 32, 64]
    assert graph_buckets(48) == [1, 2, 4, 8, 16, 32, 48]
    assert graph_buckets(256)[-2:] == [128, 256]
    b = graph_buckets(48)
    assert [bucket_for(b, n) for n in (1, 2, 3, 5, 17, 33, 48)] == [1, 2, 4, 8, 32, 48, 48]
    with pytest.raises(ValueError):
        bucket_for(b, 49)
    with pytest.raises(ValueError):
        graph_buckets(0)


def _paged_scheduler(num_blocks: int, block_size: int = 4, radix: bool = False, **kw) -> tuple[Scheduler, BlockAllocator]:
    a = BlockAllocator(num_blocks, block_size)
    return Scheduler(kv=KVCacheManager(a, radix), **kw), a


def _run_batch(batch, allocator):
    """Stand-in for the engine step: extend each block table by its rows; rows that reach the end
    of their sequence emit token 0 (a prefill completing moves to DECODE)."""
    done = batch.completes()
    for r, n in zip(batch.requests, batch.extend_lens):
        r.cache.append_tokens(n)
    for r, d in zip(batch.requests, done):
        if not d:
            continue
        r.output_ids.append(0)
        if r.state is RequestState.PREFILL:
            r.transition(RequestState.DECODE)


def test_admission_reserves_prefill_and_next_token():
    s, a = _paged_scheduler(3)  # 3 blocks of 4 tokens
    x, y = _req(0, 4, 8), _req(1, 4, 8)  # each needs ceil((4 + 1) / 4) = 2 blocks
    s.add(x)
    s.add(y)
    assert s.schedule().requests == [x]  # y would need 2 more, 1 is left


def test_admission_keeps_room_for_running_decodes():
    s, a = _paged_scheduler(4)
    x = _req(0, 8, 8)
    s.add(x)
    _run_batch(s.schedule(), a)  # x: 8 tokens in 2 full blocks; its next token needs a new block
    big, small = _req(1, 4, 8), _req(2, 2, 8)  # need 2 and 1 blocks
    s.add(big)
    s.add(small)
    batch = s.schedule()  # free 2 - 1 reserved for x = 1: big does not fit, small may not skip ahead
    assert batch.phase is Phase.DECODE and batch.requests == [x] and not batch.preempted
    _run_batch(batch, a)
    assert list(s.waiting) == [big, small]


def test_preemption_newest_first_back_to_front_of_queue():
    s, a = _paged_scheduler(6)
    x, y, z = (_req(i, 4, 20) for i in range(3))  # max_len 24 = the whole pool
    for r in (x, y, z):
        s.add(r)
    _run_batch(s.schedule(), a)  # all three admitted (2 blocks each reserved); 1 block used each
    late = _req(3, 4, 20)
    s.add(late)
    for _ in range(4):  # tokens 4..7 fill each request's second block; late never fits
        batch = s.schedule()
        assert batch.phase is Phase.DECODE and not batch.preempted
        _run_batch(batch, a)
    assert a.num_free == 0
    batch = s.schedule()  # each needs a third block, none free: z goes, freeing 2
    assert batch.requests == [x, y] and batch.preempted == [z]
    assert z.state is RequestState.WAITING and z.cache is None and z.num_preemptions == 1
    assert list(s.waiting) == [z, late] and s.running == [x, y] and s.num_preemptions == 1
    assert z.seq_len == 4 + 5  # resumes with its output
    _run_batch(batch, a)
    a.check_invariants()


def test_lone_request_is_never_preempted():
    s, a = _paged_scheduler(3)
    x = _req(0, 4, 8)  # max_len 12 = the whole pool
    s.add(x)
    while not x.should_stop():
        batch = s.schedule()
        assert not batch.preempted
        _run_batch(batch, a)
    assert a.num_free == 0


def test_request_larger_than_pool_rejected():
    s, _ = _paged_scheduler(3)
    with pytest.raises(ValueError):
        s.add(_req(0, 10, 3))  # 13 tokens > 12
    s.add(_req(1, 10, 2))  # 12 fits exactly


# --------------------------------------------------------------------------- scheduler: chunked prefill


def _rows(batch):
    return [(r.rid, st, n) for r, st, n in zip(batch.requests, batch.starts, batch.extend_lens)]


def test_chunked_decode_first_then_chunks():
    s, a = _paged_scheduler(64, chunked_prefill_size=8)
    x = _req(0, 3, 8)
    s.add(x)
    b = s.schedule()
    assert b.phase is Phase.PREFILL and _rows(b) == [(0, 0, 3)]
    _run_batch(b, a)
    long = _req(1, 20, 4)
    s.add(long)
    # x decodes (1 token), the long prompt gets the other 7 of the budget
    b = s.schedule()
    assert b.phase is Phase.MIXED and _rows(b) == [(0, 3, 1), (1, 0, 7)] and b.completes() == [True, False]
    assert b.num_decode == 1
    _run_batch(b, a)
    assert long.state is RequestState.PREFILL and long.output_ids == [] and long.cache.num_tokens == 7
    b = s.schedule()  # the next chunk continues where the last stopped
    assert _rows(b) == [(0, 4, 1), (1, 7, 7)]
    _run_batch(b, a)
    b = s.schedule()
    assert _rows(b) == [(0, 5, 1), (1, 14, 6)] and b.completes() == [True, True]
    _run_batch(b, a)
    assert long.state is RequestState.DECODE and long.output_ids == [0]
    b = s.schedule()
    assert b.phase is Phase.DECODE and _rows(b) == [(0, 6, 1), (1, 20, 1)]


def test_chunked_budget_counts_decodes_and_admits_in_order():
    s, a = _paged_scheduler(64, chunked_prefill_size=6)
    xs = [_req(i, 2, 8) for i in range(3)]
    for x in xs:
        s.add(x)
    b = s.schedule()  # three short prompts fit the budget together
    assert _rows(b) == [(0, 0, 2), (1, 0, 2), (2, 0, 2)]
    _run_batch(b, a)
    y, z = _req(3, 10, 4), _req(4, 1, 4)
    s.add(y)
    s.add(z)
    b = s.schedule()  # 3 decodes, 3 tokens left: a chunk of y; z waits behind it (head of line)
    assert _rows(b) == [(0, 2, 1), (1, 2, 1), (2, 2, 1), (3, 0, 3)] and list(s.waiting) == [z]
    assert sum(b.extend_lens) == 6


def test_chunked_admission_reserves_whole_prefill():
    s, a = _paged_scheduler(6, chunked_prefill_size=4)  # 6 blocks of 4 tokens
    x = _req(0, 16, 4)  # needs ceil(17 / 4) = 5 blocks for its whole prefill + 1
    y = _req(1, 4, 4)  # needs 2
    s.add(x)
    s.add(y)
    b = s.schedule()
    assert _rows(b) == [(0, 0, 4)]  # x admitted; y does not fit next to x's reservation
    assert s._outstanding(s.running) <= s.kv.num_available
    _run_batch(b, a)
    b = s.schedule()
    assert _rows(b) == [(0, 4, 4)] and list(s.waiting) == [y]


def test_chunked_preempts_newest_even_mid_prefill():
    """x decodes and grows while y, admitted later, is still being prefilled in chunks. When x's
    next token no longer fits next to y's reservation, y (the newest) is preempted mid-prefill; its
    computed full blocks stay in the prefix cache and are hit when it is admitted again."""
    s, a = _paged_scheduler(8, radix=True, chunked_prefill_size=4)  # 8 blocks of 4 tokens
    x = _req(0, 4, 28)  # max_len 32: the whole pool, alone
    s.add(x)
    _run_batch(s.schedule(), a)  # x: 4 tokens in 1 block, DECODE
    y = Request(1, list(range(50, 70)), SamplingParams(4))  # 20 tokens: reserves ceil(21 / 4) = 6 blocks
    s.add(y)
    for step in range(4):  # x decodes 1, y prefills 3 per step
        b = s.schedule()
        assert _rows(b) == [(0, 4 + step, 1), (1, 3 * step, 3)] and not b.preempted
        assert s._outstanding(s.running) <= s.kv.num_available
        _run_batch(b, a)
        s.kv.commit(y)  # as the engine does after a chunk
    assert x.cache.num_tokens == 8 and y.cache.num_tokens == 12 and y.state is RequestState.PREFILL
    # x's next token needs a 3rd block; with y's remaining reservation (6 - 3) nothing is left
    b = s.schedule()
    assert b.preempted == [y] and y.state is RequestState.WAITING and y.cache is None
    assert b.phase is Phase.DECODE and _rows(b) == [(0, 8, 1)]
    assert s.kv.tree.num_cached_blocks == 3  # y's computed chunks, now held by the tree alone
    s.kv.check_invariants()
    _run_batch(b, a)
    assert list(s.waiting) == [y] and y.num_preemptions == 1
    # When y is admitted again it resumes from the cached 12 tokens.
    y_cached = s.kv.acquire(y)
    assert y_cached == 12
    s.kv.abandon(y)


def test_abandoned_admission_leaves_no_trace():
    """A request refused after its prefix lookup gives the lookup back: no table, no lock, no stats."""
    s, a = _paged_scheduler(3, radix=True)
    x = _req(0, 4, 8)
    s.add(x)
    _run_batch(s.schedule(), a)  # x: 4 tokens in 1 block (2 reserved), DECODE
    s.kv.commit(x)  # x's full block enters the tree, locked by x
    y = Request(1, x.prompt_ids + [7, 7, 7, 7], SamplingParams(4))  # shares x's first block
    s.add(y)
    batch = s.schedule()
    # y hits 1 block, needs ceil(9 / 4) - 1 = 2 more; available 2 free - 1 for x's next token = 1
    assert batch.phase is Phase.DECODE and batch.requests == [x]
    assert y.cache is None and y.cache_node is None and y.num_cached_tokens == 0 and list(s.waiting) == [y]
    assert s.stats == dict(first_tokens=4, first_cached=0, re_tokens=0, re_cached=0)
    assert x.cache_node.lock == 1  # y's lock was given back
    s.kv.check_invariants()


def test_admission_counts_cached_prefix():
    """A hit shrinks both the block need and the prefill tokens: y fits only because of it."""
    s, a = _paged_scheduler(4, radix=True)
    x = _req(0, 8, 4)  # 8 tokens = 2 full blocks
    s.add(x)
    _run_batch(s.schedule(), a)
    s.kv.commit(x)  # x holds 2 full blocks, both now in the tree; 2 free
    y = Request(1, x.prompt_ids + [9], SamplingParams(2))  # 9 tokens: 2 blocks cached, 1 token to compute
    s.add(y)
    # y needs ceil(10 / 4) - 2 = 1 block; available: 2 free - 1 for x's next token = 1.
    # Without the hit it would need 3 and not fit.
    batch = s.schedule()
    assert batch.phase is Phase.PREFILL and batch.requests == [y]
    assert y.num_cached_tokens == 8 and batch.seq_lens == [1]
    assert y.cache.blocks[:2] == x.cache.blocks[:2]
    assert s.stats["first_cached"] == 8
    s.kv.check_invariants()


# --------------------------------------------------------------------------- CPU simulation with a toy model

TOY_VOCAB = 64
TOY_STOP = 0


def _toy_next(tokens: list[int]) -> int:
    """A deterministic 'model': the next token depends on every token of the sequence and its position."""
    return (sum((i + 1) * t for i, t in enumerate(tokens)) * 31 + len(tokens)) % TOY_VOCAB


def _toy_logits(tokens: list[int]) -> torch.Tensor:
    """Logits over the toy vocabulary: pseudo-random in the whole sequence, with ``_toy_next`` far ahead."""
    key = (sum((i + 1) * t for i, t in enumerate(tokens)) * 131 + len(tokens)) % 2**31
    logits = torch.randn(TOY_VOCAB, generator=torch.Generator().manual_seed(key))
    logits[_toy_next(tokens)] += 10.0  # greedy decoding follows _toy_next
    return logits


def _toy_generate(prompt: list[int], params: SamplingParams, seed: int = 0) -> list[int]:
    """The request alone: one row at a time through the same sampler."""
    sampler = Sampler("cpu", TOY_VOCAB)
    seq, out = list(prompt), []
    while len(out) < params.max_new_tokens:
        r = Request(0, list(prompt), params, output_ids=list(out))
        r.seed = seed
        out.append(int(sampler.sample(_toy_logits(seq)[None], sampler.prepare([r]))[0]))
        seq.append(out[-1])
        if out[-1] in params.stop_token_ids:
            break
    return out


class _ToyModel:
    """Stands in for the model and the paged attention backend under the real ``ModelRunner``.

    Its KV store maps pool slots to (token, position). Each pass writes the new
    tokens at the planned slots, then reads every sequence back through its
    block table: a missing, stale or misplaced entry shows up as a wrong
    position or a wrong next token.
    """

    def __init__(self):
        self.store: dict[int, tuple[int, int]] = {}

    def plan(self, prefill, qo_lens, tables, slots):
        self.tables, self.slots = list(tables), list(slots)

    def forward_with(self, input_ids, positions, attn, seq_lens):
        for slot, t, p in zip(self.slots, input_ids.tolist(), positions.tolist()):
            self.store[slot] = (t, p)
        rows = []
        for table in self.tables:
            hist = [self.store[table.slot(i)] for i in range(table.num_tokens)]
            assert [p for _, p in hist] == list(range(table.num_tokens)), "KV positions out of order"
            rows.append(_toy_logits([t for t, _ in hist]))
        return torch.stack(rows)


def _step_checked(eng: Engine, chunk: int):
    """One step plus the invariants every step must keep: cache and allocator consistent and the
    step within its token budget. (A budget error shows up as OutOfBlocks inside the step.)"""
    batch = eng.step()
    eng.runner.kv.check_invariants()
    launched = eng.launched
    if launched is not None and chunk:
        assert sum(launched.extend_lens) <= chunk, (launched.phase, launched.extend_lens)
    return batch


def _toy_engine(num_blocks: int, block_size: int, **kw) -> Engine:
    runner = ModelRunner.__new__(ModelRunner)  # the real runner logic, without a GPU model
    runner.model = runner.flashinfer = _ToyModel()
    runner.device = torch.device("cpu")
    runner.attention = "paged"
    runner.allocator = BlockAllocator(num_blocks, block_size)
    runner.kv = KVCacheManager(runner.allocator, kw.pop("radix", True))
    runner.sampler = Sampler("cpu", TOY_VOCAB)
    runner.graphs, runner.use_cuda_graph = None, False
    runner.fence = CopyFence("cpu")
    return Engine(None, runner=runner, **kw)


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("chunk", [0, 3, 2048])
@pytest.mark.parametrize("num_blocks", [8, 16, 64])
@pytest.mark.parametrize("seed", range(10))
def test_simulated_load_under_kv_pressure(seed, num_blocks, chunk, overlap):
    """Random arrivals, lengths and sampling settings in a small pool: no OutOfBlocks, no stall,
    and every output (greedy or sampled) equals the request run alone, preempted or not."""
    import random

    rng = random.Random(seed)
    block_size = 4
    eng = _toy_engine(
        num_blocks, block_size, max_running=16, max_prefill_tokens=48, chunked_prefill_size=chunk, overlap=overlap
    )
    cap = num_blocks * block_size
    arrivals: dict[int, list[tuple[list[int], SamplingParams]]] = {}
    for _ in range(40):
        n = rng.randint(1, min(30, cap - 1))
        prompt = [rng.randrange(1, TOY_VOCAB) for _ in range(n)]
        temperature, top_p = rng.choice([(0.0, 1.0), (4.0, 1.0), (4.0, 0.9)])
        req_seed = rng.choice([None, rng.randrange(2**32)])  # None: assigned by the engine
        params = SamplingParams(rng.randint(1, min(40, cap - n)), frozenset({TOY_STOP}), temperature, top_p, req_seed)
        arrivals.setdefault(rng.randrange(60), []).append((prompt, params))
    reqs, step = [], 0
    while eng.has_unfinished or step < 60:
        for prompt, params in arrivals.get(step, []):
            reqs.append(eng.add_request(prompt, params))
        _step_checked(eng, chunk)
        step += 1
        assert step < 20_000, "no progress"
    for r in reqs:
        assert r.state is RequestState.FINISHED
        assert r.output_ids == _toy_generate(r.prompt_ids, r.params, r.seed), f"request {r.rid} {r.params}"
    _assert_no_leak(eng)
    if num_blocks == 8:
        assert eng.scheduler.num_preemptions > 0, "the small pool should force preemption"
    print(f"\nseed {seed}, {num_blocks} blocks: {step} steps, {eng.scheduler.num_preemptions} preemptions")


def test_overlap_edge_cases_occur_in_simulation(monkeypatch):
    """The simulated loads above, with overlap, do reach the cases a one-step lag adds: a request
    sampling a stop token after its next step was already launched (that step's token is
    dropped), and a request preempted before its last token was read back, which then ends it
    (WAITING -> FINISHED). Otherwise their passing would say nothing about these paths."""
    import random

    seen = {"dropped": 0, "waiting_to_finished": 0}
    drop, transition = Request.drop_pending, Request.transition

    def counting_drop(self):
        if self.num_pending and self.state is not RequestState.ABORTED:
            seen["dropped"] += 1
        drop(self)

    def counting_transition(self, new):
        if self.state is RequestState.WAITING and new is RequestState.FINISHED:
            seen["waiting_to_finished"] += 1
        transition(self, new)

    monkeypatch.setattr(Request, "drop_pending", counting_drop)
    monkeypatch.setattr(Request, "transition", counting_transition)
    for seed in range(10):
        for chunk in (0, 3):
            rng = random.Random(seed)
            eng = _toy_engine(8, 4, max_running=16, max_prefill_tokens=48, chunked_prefill_size=chunk)
            reqs = []
            for step in range(2000):
                if step < 60:
                    for _ in range(rng.randrange(3)):
                        n = rng.randint(1, 20)
                        params = SamplingParams(rng.randint(1, 31 - n), frozenset({TOY_STOP}))
                        reqs.append(eng.add_request([rng.randrange(1, TOY_VOCAB) for _ in range(n)], params))
                elif not eng.has_unfinished:
                    break
                _step_checked(eng, chunk)
            for r in reqs:
                assert r.output_ids == _toy_generate(r.prompt_ids, r.params, r.seed)
            _assert_no_leak(eng)
    print(f"\n{seen}")
    assert seen["dropped"] > 0 and seen["waiting_to_finished"] > 0


def test_chunks_are_cached_as_they_complete():
    """Each chunk's full blocks enter the prefix cache as soon as it is computed. A takes the whole
    6-token budget per step; B (sharing A's first 30 tokens) is admitted in the step of A's last
    chunk, and already hits A's earlier chunks (36 tokens committed: B's 7 whole shared blocks)."""
    eng = _toy_engine(64, 4, max_running=8, chunked_prefill_size=6)
    prompt = list(range(1, 41))  # 40 tokens: 6 full chunks of 6, then 4
    a = eng.add_request(prompt, SamplingParams(4, frozenset({TOY_STOP})))
    b = eng.add_request(prompt[:30] + [9, 9], SamplingParams(4, frozenset({TOY_STOP})))
    for _ in range(6):
        eng.step()
        assert b.state is RequestState.WAITING  # A's chunks use the whole budget
    assert a.cache.num_tokens == 36
    eng.step()  # A's last 4 tokens + B admitted with the 2 left
    assert a.state is RequestState.DECODE and b.state is RequestState.PREFILL
    assert b.num_cached_tokens == 28
    while eng.has_unfinished:
        eng.step()
    for r in (a, b):
        assert r.output_ids == _toy_generate(r.prompt_ids, r.params, r.seed)
    _assert_no_leak(eng)


@pytest.mark.parametrize("chunk", [0, 5, 2048])
@pytest.mark.parametrize("radix", [True, False])
@pytest.mark.parametrize("num_blocks", [12, 24, 96])
@pytest.mark.parametrize("seed", range(8))
def test_simulated_shared_prefixes(seed, num_blocks, radix, chunk):
    """Requests built from a few shared prefixes, in a pool small enough to preempt and evict.

    The toy model reads every sequence's history back from the KV store through
    its block table and checks each (token, position), so a cached block holding
    the wrong tokens, or one evicted and overwritten while still in use, changes
    the output. Every output must equal the request run alone, and the tree and
    allocator invariants must hold after every step."""
    import random

    rng = random.Random(seed)
    block_size = 4
    eng = _toy_engine(num_blocks, block_size, max_running=8, max_prefill_tokens=40, radix=radix, chunked_prefill_size=chunk)
    cap = num_blocks * block_size
    prefixes = [[rng.randrange(1, TOY_VOCAB) for _ in range(rng.randint(4, 17))] for _ in range(3)]
    arrivals: dict[int, list[tuple[list[int], SamplingParams]]] = {}
    for _ in range(30):
        prompt = rng.choice(prefixes) + [rng.randrange(1, TOY_VOCAB) for _ in range(rng.randint(0, 6))]
        if rng.random() < 0.2:
            prompt = list(rng.choice(prefixes))  # an exact repeat
        temperature = rng.choice([0.0, 4.0])
        max_new = rng.randint(1, min(24, cap - len(prompt)))
        params = SamplingParams(max_new, frozenset({TOY_STOP}), temperature, 1.0, rng.randrange(2**32))
        arrivals.setdefault(rng.randrange(40), []).append((prompt, params))
    reqs, step = [], 0
    while eng.has_unfinished or step < 40:
        for prompt, params in arrivals.get(step, []):
            reqs.append(eng.add_request(prompt, params))
        _step_checked(eng, chunk)
        step += 1
        assert step < 20_000, "no progress"
    for r in reqs:
        assert r.state is RequestState.FINISHED
        assert r.output_ids == _toy_generate(r.prompt_ids, r.params, r.seed), f"request {r.rid} {r.params}"
    st = eng.scheduler.stats
    if radix:
        assert st["first_cached"] > 0, st
        if st["re_tokens"]:  # (a preempted request whose last token was in flight is never readmitted)
            assert st["re_cached"] > 0, st  # readmitted requests find their own blocks
    else:
        assert st["first_cached"] == st["re_cached"] == 0
    _assert_no_leak(eng)
    print(f"\nseed {seed}, {num_blocks} blocks, radix {radix}: {step} steps, "
          f"{eng.scheduler.num_preemptions} preemptions, stats {st}")


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


class _Positions:
    """Expected positions of each step, from a KV frontier the test keeps itself.

    A request's frontier is how many of its tokens have KV. A row continues from it: a decode
    row, or the next chunk of a chunked prefill. Only a row that (re)admits a request starts
    from the request's cached prefix; preemption forgets the frontier. Deriving positions this
    way, rather than from the batch's own ``starts``, catches a scheduler that resumes a chunk
    or a decode at the wrong position."""

    def __init__(self):
        self.frontier: dict[int, int] = {}

    def expected(self, batch) -> list[int]:
        for r in batch.preempted:
            self.frontier.pop(r.rid, None)
        out = []
        for r, n in zip(batch.requests, batch.extend_lens):
            start = self.frontier.get(r.rid, r.num_cached_tokens)
            out += range(start, start + n)
            self.frontier[r.rid] = start + n
        return out


def _assert_no_leak(eng):
    """Idle: no request holds a block; after clearing the prefix cache every block is free."""
    kv = eng.runner.kv
    if kv is not None:
        kv.check_invariants()
        assert kv.num_idle_blocks() == kv.allocator.num_blocks
        if kv.tree is not None:
            assert kv.tree.num_evictable == kv.tree.num_cached_blocks  # nothing locked
            kv.tree.clear()
        assert kv.allocator.num_free == kv.allocator.num_blocks


def _spy_positions(eng, model, monkeypatch) -> list[list[int]]:
    """Record the positions of every forward pass, eager (``forward_with``) or a decode graph
    replay (the positions staged into the graph's input buffer)."""
    seen: list[list[int]] = []
    inner = model.forward_with

    def spy(input_ids, positions, attn, seq_lens):
        seen.append(positions.tolist())
        return inner(input_ids, positions, attn, seq_lens)

    monkeypatch.setattr(model, "forward_with", spy)
    graphs = eng.runner.graphs
    if graphs is not None:
        run = graphs.run

        def graph_spy(ids, pos, slots, tables, fill=None):
            seen.append(list(pos))
            return run(ids, pos, slots, tables, fill)

        monkeypatch.setattr(graphs, "run", graph_spy)
    return seen


def _run(eng, reference, schedule):
    """Drive the engine; ``schedule`` maps step -> prompt names arriving before that step."""
    reqs, step = {}, 0
    while eng.has_unfinished or step <= max(schedule):
        for name in schedule.get(step, []):
            ids, n, _, _ = reference[name]
            reqs[name] = eng.add_request(ids, SamplingParams(n, STOP_IDS))
        eng.step()
        step += 1
    return reqs, step


def _check_all(label, model, reference, reqs) -> tuple[dict, float]:
    diverged, worst = {}, 0.0
    for name, r in reqs.items():
        assert r.state is RequestState.FINISHED and r.cache is None
        ids, _, ref, gaps = reference[name]
        pos = _check_against_reference(name, r.output_ids, ref, gaps)
        if pos is not None:
            diverged[name] = (pos, gaps[pos])
        worst = max(worst, _check_margins(name, model, ids, r.output_ids))
    print(
        f"\n[{label}] diverged {len(diverged)}/{len(reqs)} (position, reference gap): {diverged}; "
        f"max teacher-forced margin {worst}"
    )
    return diverged, worst


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("name", list(PROMPTS))
def test_single_request_contiguous_token_exact(name, model, reference):
    """Alone in the engine on the reference attention path, a request runs the same shapes as the reference."""
    ids, n, ref, _ = reference[name]
    eng = _engine(model, attention="contiguous")
    [ours] = eng.generate([ids], SamplingParams(n, STOP_IDS))
    assert ours == ref
    assert not eng.has_unfinished and not eng.requests


@pytest.mark.gpu
@pytest.mark.slow
def test_single_request_paged(model, reference):
    """Each anchor prompt alone in the engine on the paged path, within the tolerance rules."""
    eng = _engine(model, attention="paged")
    reqs = {}
    for name in PROMPTS:
        reqs.update(_run(eng, reference, {0: [name]})[0])
        _assert_no_leak(eng)
    _check_all("paged single", model, reference, reqs)


@pytest.mark.gpu
@pytest.mark.slow
def test_paged_prefill_logits_close(model, reference):
    """Model level: one paged (FlashInfer) prefill over all prompts vs each prompt alone on the reference path."""
    prompts = [reference[k][0] for k in PROMPTS]
    eng = _engine(model, attention="paged")
    reqs = [Request(i, p, SamplingParams(1)) for i, p in enumerate(prompts)]
    for r in reqs:
        r.transition(RequestState.PREFILL)
        eng.runner.kv.acquire(r)
    paged = eng.runner.forward(Batch(Phase.PREFILL, reqs)).float()
    dev = model.device
    alone = torch.stack(
        [model.forward(torch.tensor(p, device=dev), torch.arange(len(p), device=dev), model.new_cache(len(p))) for p in prompts]
    ).float()
    max_diff = (paged - alone).abs().max().item()
    print(f"\npaged batched vs reference single prefill over {len(prompts)} prompts: max|d| = {max_diff}")
    assert max_diff < 1.0, max_diff
    top2 = torch.topk(alone, 2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > EPS
    assert torch.equal(paged.argmax(-1)[decisive], alone.argmax(-1)[decisive])
    for r in reqs:
        eng.runner.release(r)
    _assert_no_leak(eng)


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
@pytest.mark.parametrize(
    "attention, chunk, graph, overlap",
    [
        ("contiguous", None, True, True),
        ("paged", None, True, True),
        ("paged", None, False, True),
        ("paged", None, True, False),
        ("paged", 0, True, True),
        ("paged", 64, True, True),
    ],
    ids=["contiguous", "paged", "paged-eager", "paged-no-overlap", "paged-nochunk", "paged-chunk64"],
)
@pytest.mark.parametrize("arrival", ["all_at_once", "staggered"])
def test_concurrent_matches_reference(arrival, attention, chunk, graph, overlap, model, reference):
    """Mixed concurrent load: prefills of several prompts and decode batches of varying size.
    Paged runs with the default chunk budget (2048: mixed prefill + decode steps), without chunking
    (prefill-only and decode-only steps), and with 64-token chunks (every long prompt cut); decode
    steps replay CUDA Graphs except in ``paged-eager``, and each step is launched before the previous
    one is read back except in ``paged-no-overlap``."""
    names = list(PROMPTS)
    schedule = {0: names} if arrival == "all_at_once" else {0: names[:2], 3: names[2:4], 7: names[4:5], 20: names[5:]}
    eng = _engine(model, attention=attention, chunked_prefill_size=chunk, cuda_graph=graph, overlap=overlap)
    phases = []
    eng.logits_hook = lambda batch, logits: phases.append(batch.phase)
    reqs, step = _run(eng, reference, schedule)
    _check_all(f"{attention} chunk {chunk} {arrival}, {step} steps, phases {sorted({p.name for p in phases})}", model, reference, reqs)
    if chunk == 64 and arrival == "staggered":
        assert Phase.MIXED in phases
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_chunk_boundaries_match_whole_prefill(model, reference):
    """The long prompt prefilled in 64-token chunks (9 chunks, boundaries mid-block and on block
    edges alike) against one prefill on the reference path: last-position logits close, the same
    argmax where the reference is decisive, and greedy output passing the anchor."""
    ids, n, ref, gaps = reference["long_en"]
    chunks, logits_at_end = [], []

    def hook(batch, logits):
        chunks.append(batch.extend_lens[0])
        if batch.completes()[0] and not logits_at_end:
            logits_at_end.append(logits[0].float().clone())

    for size in (64, 100):
        chunks.clear()
        logits_at_end.clear()
        eng = _engine(model, attention="paged", chunked_prefill_size=size)
        eng.logits_hook = hook
        r = eng.add_request(ids, SamplingParams(n, STOP_IDS))
        while eng.has_unfinished:
            eng.step()
        prefill_chunks = chunks[: -(len(r.output_ids) - 1) or None]
        assert prefill_chunks == [size] * (len(ids) // size) + ([len(ids) % size] if len(ids) % size else [])
        dev = model.device
        whole = model.forward(torch.tensor(ids, device=dev), torch.arange(len(ids), device=dev), model.new_cache(len(ids))).float()
        diff = (logits_at_end[0] - whole).abs().max().item()
        top2 = torch.topk(whole, 2).values
        print(f"\nchunk {size}: {len(prefill_chunks)} chunks, last-position max|d| vs whole prefill {diff:.4f}")
        assert diff < 1.0, diff
        if top2[0] - top2[1] > EPS:
            assert logits_at_end[0].argmax() == whole.argmax()
        pos = _check_against_reference(f"chunk {size}", r.output_ids, ref, gaps)
        _check_margins(f"chunk {size}", model, ids, r.output_ids)
        print(f"chunk {size}: diverged at {pos}")
        _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_paged_fragmented_pool(model, reference):
    """Requests get scattered, descending physical blocks; attention must follow the block tables."""
    eng = _engine(model, attention="paged", kv_pool_tokens=512 * 16)
    a = eng.runner.allocator
    held = a.allocate(a.num_blocks)
    a.free(held[::2])  # every other block free; LIFO hands them out in descending order
    reqs, _ = _run(eng, reference, {0: list(PROMPTS)})
    _check_all("paged fragmented", model, reference, reqs)
    a.free(held[1::2])
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_paged_pool_exhaustion_changes_nothing(model, reference):
    eng = _engine(model, attention="paged", kv_pool_tokens=128)
    ids = reference["long_en"][0]  # 563 tokens
    with pytest.raises(ValueError):  # could never fit: rejected on submission
        eng.add_request(ids, SamplingParams(4, STOP_IDS))
    assert not eng.has_unfinished
    # Below the scheduler, the runner still refuses a batch that does not fit, changing nothing.
    r = Request(0, ids, SamplingParams(4))
    r.transition(RequestState.PREFILL)
    eng.runner.kv.acquire(r)
    with pytest.raises(OutOfBlocks):
        eng.runner.forward(Batch(Phase.PREFILL, [r]))
    assert r.cache.blocks == [] and eng.runner.allocator.num_free == 8
    eng.runner.release(r)
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("attention", ATTENTION)
def test_abort_mid_decode(attention, model, reference):
    names = ["short_en", "code", "zh"]
    eng = _engine(model, attention=attention)
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
    # ten tokens launched; with overlap the tenth was not read back when it was aborted
    assert len(victim.output_ids) == (9 if eng.overlap else 10)
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("attention", ATTENTION)
def test_positions_contract(attention, model, reference, monkeypatch):
    """Exact positions fed to the model at every step.

    Token-level checks are nearly blind to a uniform position shift of all
    decode tokens: RoPE is relative, so the model sees what looks like one
    extra gap between prompt and output and keeps producing near-argmax tokens.
    """
    eng = _engine(model, attention=attention)  # before the spy: paged construction runs a profiling pass
    seen = _spy_positions(eng, model, monkeypatch)
    positions = _Positions()
    names = ["short_en", "code", "zh"]
    for k in names[:2]:
        eng.add_request(reference[k][0], SamplingParams(6, STOP_IDS))
    step = launches = 0
    while eng.has_unfinished:
        if step == 2:
            eng.add_request(reference[names[2]][0], SamplingParams(6, STOP_IDS))
        eng.step()
        launched = eng.launched  # with overlap, step() returns the previous batch
        if launched is not None:
            assert seen[-1] == positions.expected(launched), f"step {step} ({launched.phase.name})"
            launches += 1
        step += 1
    assert len(seen) == launches
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_shared_prefix_anchor(model, tokenizer, reference, monkeypatch):
    """Requests that reuse cached KV pass the anchor, with exact positions.

    A runs first. While A decodes, B (the same prompt) and C (A's first 512
    tokens plus a different ending) arrive and hit A's committed prompt blocks.
    After A finishes, E repeats the prompt and hits blocks no request holds."""
    from miniserve.model.generate import greedy_generate

    base, n, ref_a, gaps_a = reference["long_en"]
    tail = tokenizer(" In one word, the topic of the passage is").input_ids
    prompt_c = base[:512] + tail
    gaps_c: list[float] = []
    ref_c = greedy_generate(model, prompt_c, 64, stop_ids=STOP_IDS, top2_gaps=gaps_c)
    refs = {"A": (base, n, ref_a, gaps_a), "B": (base, n, ref_a, gaps_a), "C": (prompt_c, 64, ref_c, gaps_c)}
    refs["E"] = refs["A"]

    eng = _engine(model, attention="paged")
    seen = _spy_positions(eng, model, monkeypatch)
    positions = _Positions()
    reqs, step = {}, 0
    reqs["A"] = eng.add_request(base, SamplingParams(n, STOP_IDS))
    while eng.has_unfinished or "E" not in reqs:
        if step == 3:
            for k in "BC":
                reqs[k] = eng.add_request(refs[k][0], SamplingParams(refs[k][1], STOP_IDS))
        if not eng.has_unfinished and "E" not in reqs:
            reqs["E"] = eng.add_request(base, SamplingParams(n, STOP_IDS))
        eng.step()
        batch = eng.launched  # with overlap, step() returns the previous batch
        if batch is None:
            step += 1
            continue
        assert seen[-1] == positions.expected(batch), f"step {step} ({batch.phase.name})"
        if batch.phase is Phase.PREFILL:
            cached = {k: r.num_cached_tokens for k, r in reqs.items() if r in batch.requests}
            print(f"step {step}: prefill cached tokens {cached}")
        eng.runner.kv.check_invariants()
        step += 1
    full = (len(base) - 1) // 16 * 16  # every whole block but the one holding the last prompt token
    assert [reqs[k].num_cached_tokens for k in "ABCE"] == [0, full, 512, full]
    assert eng.scheduler.stats["first_cached"] == 2 * full + 512
    _check_all("shared prefix", model, refs, reqs)
    _assert_no_leak(eng)


# --------------------------------------------------------------------------- GPU: KV pool size and pressure


@pytest.mark.gpu
@pytest.mark.slow
def test_kv_pool_sizing(model):
    import gc

    eng = _engine(model, attention="paged", kv_pool_tokens=1000)  # rounded down to whole blocks
    prof = eng.runner.kv_profile
    # the pool has one more block than the allocator: the decode graphs' padding block
    assert eng.runner.allocator.num_blocks == prof["num_blocks"] == 62 and eng.runner.pool.num_blocks == 63
    assert prof["peak_activation_bytes"] > 0 and prof["max_blocks"] > 62
    too_many = 2 * prof["max_blocks"] * 16
    del eng
    with pytest.raises(ValueError):  # an exact experiment size, never silently clipped
        _engine(model, attention="paged", kv_pool_tokens=too_many)
    gc.collect()

    eng = Engine(model, attention="paged")  # as large as memory allows
    prof = eng.runner.kv_profile
    print(f"\nKV pool sized to memory: {prof}")
    assert eng.runner.allocator.num_blocks == prof["num_blocks"] == prof["max_blocks"]
    assert eng.runner.pool.num_bytes == (prof["num_blocks"] + 1) * prof["block_bytes"]
    assert prof["block_bytes"] == 2 * 28 * 16 * 8 * 128 * 2  # Qwen3-0.6B, BF16: 112 KiB per token
    assert prof["num_blocks"] * prof["block_bytes"] <= 0.9 * (prof["free_bytes"] - prof["peak_activation_bytes"])
    # The profiled peak covers the largest sampling step (64 rows, every row sampled with a nucleus).
    from miniserve.engine.sampler import SamplingArgs

    logits = torch.randn(64, model.cfg.vocab_size, device=model.device, dtype=model.dtype)
    ones, rows = torch.ones(64, device=model.device), torch.arange(64, device=model.device)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    eng.runner.sampler.sample(logits, SamplingArgs(ones, ones * 0.9, rows, rows, any_top_p=True))
    torch.cuda.synchronize()
    sampling_peak = torch.cuda.max_memory_allocated() - base
    print(f"sampling peak over 64 rows: {sampling_peak} B")
    assert 0 < sampling_peak <= prof["peak_activation_bytes"]
    del eng, logits
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("lens", [[40], [5, 900, 2500], [17] * 5, [300, 7, 1200, 64, 64, 2000, 33, 900, 450, 12, 1500, 80, 3, 700, 260, 1024, 1025]])
def test_decode_graph_matches_eager(lens, model):
    """A decode batch replayed from a CUDA Graph against the same batch run eagerly: logits within
    the tolerance (the graph's bucket pads the batch, which can change GEMM shapes), and the padding
    rows write nothing but the dummy block. The graphs were captured with every sequence of length
    1, so this also checks that planning a batch of long sequences updates what the replay reads."""
    import random

    rng = random.Random(len(lens))
    # without overlap: the batch's input tokens must be known on the host to run it twice by hand
    eng = _engine(model, attention="paged", chunked_prefill_size=0, overlap=False)
    runner = eng.runner
    reqs = [eng.add_request([rng.randrange(150_000) for _ in range(n)], SamplingParams(3)) for n in lens]
    while any(r.state is not RequestState.DECODE for r in reqs):
        eng.step()  # prefill-only steps
    batch = eng.scheduler.schedule()
    assert batch.phase is Phase.DECODE and len(batch.requests) == len(lens)
    tables = [r.cache for r in batch.requests]
    bs, dummy = runner.pool.block_size, runner.graphs.dummy_block
    before = runner.pool.buf.cpu()
    graph_logits = runner.forward(batch).float().clone()
    after = runner.pool.buf.cpu()
    # Slots whose K or V changed in any layer, outside the dummy block: exactly the new token of each request.
    changed = (before != after).flatten(4).any(-1).any(0).any(0)  # [blocks, block_size]
    del before, after
    got = {b * bs + o for b, o in changed.nonzero().tolist() if b != dummy}
    assert got == {t.slot(t.num_tokens - 1) for t in tables}
    for t in tables:  # rewind the new token and run the same batch eagerly
        t.num_tokens -= 1
    runner.use_cuda_graph = False
    eager_logits = runner.forward(batch).float()
    runner.use_cuda_graph = True
    diff = (graph_logits - eager_logits).abs().max().item()
    top2 = eager_logits.topk(2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > 2 * diff
    print(f"\n[{len(lens)} rows, bucket {min(g for g in runner.graphs.buckets if g >= len(lens))}] graph vs eager max|d| {diff}")
    assert diff <= EPS
    assert torch.equal(graph_logits.argmax(-1)[decisive], eager_logits.argmax(-1)[decisive])
    for r in reqs:
        eng.abort(r.rid)
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_launch_does_not_wait_for_the_device(model, monkeypatch):
    """With overlap, a step launches the next batch and then reads back the previous one: it must
    not wait for the device before that read-back. A long sleep kernel is queued between two steps.

    - A graph decode step launches a handful of operations: the whole step (launch, then the
      read-back of the previous batch, which ran before the sleep) returns long before the
      sleep ends. Without overlap (control) it waits for its own batch, behind the sleep.
    - An eager step launches about 1500 kernels, more than the driver queues ahead of the
      device (about 1000 on the development machine), so launching the forward pass itself
      blocks behind any long sleep; that is queue depth, not a synchronization, and in steady
      state only bounds how far the CPU runs ahead. For eager decode and mixed steps the check
      is that everything before the forward pass (scheduling, inputs, planning) does not wait.
    """
    import time

    eng = _engine(model, attention="paged", chunked_prefill_size=256)
    reqs = [eng.add_request(list(range(1, 200 + 7 * k)), SamplingParams(200)) for k in range(4)]
    while any(r.state is not RequestState.DECODE for r in reqs) or eng.launched.phase is not Phase.DECODE:
        eng.step()
    reached_forward = []
    inner = model.forward_with

    def spy(*a, **k):
        reached_forward.append(time.perf_counter())
        return inner(*a, **k)

    monkeypatch.setattr(model, "forward_with", spy)
    sleep_cycles = 1_500_000_000  # about 0.6 s at the 4060's 2.5 GHz
    for case in ("graph decode", "eager decode", "mixed", "no overlap"):
        eng.runner.use_cuda_graph = case != "eager decode"
        eng.overlap = case != "no overlap"
        if case == "mixed":
            reqs.append(eng.add_request(list(range(1, 600)), SamplingParams(4)))
        eng.step()
        torch.cuda._sleep(sleep_cycles)
        reached_forward.clear()
        t = time.perf_counter()
        eng.step()
        elapsed = time.perf_counter() - t
        to_forward = reached_forward[0] - t if reached_forward else None
        torch.cuda.synchronize()
        t = time.perf_counter()
        torch.cuda._sleep(sleep_cycles)
        torch.cuda.synchronize()
        sleep_s = time.perf_counter() - t
        print(
            f"\n{case} ({eng.launched.phase.name}): step {elapsed * 1e3:.1f} ms, "
            f"to the forward pass {'-' if to_forward is None else f'{to_forward * 1e3:.1f}'} ms, with a {sleep_s * 1e3:.0f} ms sleep queued"
        )
        if case == "graph decode":
            assert elapsed < 0.25 * sleep_s
        elif case == "no overlap":
            assert elapsed > 0.8 * sleep_s
        else:
            assert to_forward is not None and to_forward < 0.25 * sleep_s
            if case == "mixed":
                assert eng.launched.phase is Phase.MIXED
    eng.overlap = True
    while eng.has_unfinished:
        eng.step()
    _assert_no_leak(eng)


@pytest.fixture(scope="module")
def workload64(tokenizer, model):
    """64 distinct requests mixing the anchor prompts (about 10 to 570 tokens) with output limits of 8 to 64.

    name -> (prompt ids, max_new_tokens, reference tokens, reference top-2 gaps), like ``reference``.
    """
    from miniserve.model.generate import greedy_generate

    names, limits = list(PROMPTS), [8, 16, 32, 64]
    out = {}
    for i in range(64):
        ids = encode(tokenizer, f"Request {i}.") + encode(tokenizer, PROMPTS[names[i % len(names)]][0])
        n = limits[i % len(limits)]
        gaps: list[float] = []
        toks = greedy_generate(model, ids, n, stop_ids=STOP_IDS, top2_gaps=gaps)
        out[f"r{i:02d}"] = (ids, n, toks, gaps)
    return out


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arrival,pool_tokens", [("all_at_once", 2048), ("staggered", 1536)])
def test_64_concurrent_under_kv_pressure(arrival, pool_tokens, model, workload64, monkeypatch):
    """64 concurrent requests of mixed length in a pool far smaller than their
    total demand. Admission control and preemption must keep every step within the pool (no
    OutOfBlocks, no OOM), the run must finish (no deadlock), preemption must actually happen, and
    every output must pass the correctness anchor, with exact positions at every step."""
    eng = _engine(model, attention="paged", kv_pool_tokens=pool_tokens)
    seen = _spy_positions(eng, model, monkeypatch)
    positions = _Positions()
    names = sorted(workload64)
    # staggered: bursts of 8 every 5 steps, so requests join while others are mid-decode
    arrive = {0: names} if arrival == "all_at_once" else {5 * i: names[8 * i : 8 * i + 8] for i in range(8)}
    reqs, step = {}, 0
    while eng.has_unfinished or step <= max(arrive):
        for k in arrive.get(step, []):
            ids, n, _, _ = workload64[k]
            reqs[k] = eng.add_request(ids, SamplingParams(n, STOP_IDS))
        eng.step()
        batch = eng.launched  # with overlap, step() returns the previous batch
        if batch is not None:
            assert seen[-1] == positions.expected(batch), f"step {step} ({batch.phase.name})"
        eng.runner.kv.check_invariants()
        step += 1
        assert step < 20_000, "no progress"
    resumed = sum(r.num_preemptions > 0 for r in reqs.values())
    print(
        f"\n[{arrival}, pool {pool_tokens} tokens] {step} steps, "
        f"{eng.scheduler.num_preemptions} preemptions of {resumed} requests; prefix cache {eng.scheduler.stats}"
    )
    assert eng.scheduler.num_preemptions > 0, "the pool is too large to exercise preemption"
    assert eng.scheduler.stats["re_cached"] > 0, "readmitted requests should find their own blocks cached"
    _check_all(f"64 concurrent {arrival}, pool {pool_tokens}", model, workload64, reqs)
    _assert_no_leak(eng)


@pytest.mark.gpu
@pytest.mark.slow
def test_sampled_requests_alongside_greedy(model, reference):
    """Sampled requests share batches with greedy ones: greedy output still passes the anchor, and
    the sampled output is reproducible (same seeds, same schedule) and actually sampled."""
    sampled = {
        f"s_{k}": (reference[k][0], SamplingParams(32, STOP_IDS, temperature=0.8, top_p=0.9, seed=i))
        for i, k in enumerate(["short_en", "code", "chat"])
    }

    def run():
        eng = _engine(model, attention="paged")
        greedy = {k: eng.add_request(reference[k][0], SamplingParams(reference[k][1], STOP_IDS)) for k in PROMPTS}
        samp = {k: eng.add_request(ids, p) for k, (ids, p) in sampled.items()}
        while eng.has_unfinished:
            eng.step()
        _assert_no_leak(eng)
        return greedy, samp

    greedy, first = run()
    _check_all("greedy next to sampled", model, reference, greedy)
    _, second = run()
    for k in sampled:
        assert first[k].output_ids == second[k].output_ids, k
    differs = [k for k in sampled if first[k].output_ids != reference[k[2:]][2][: len(first[k].output_ids)]]
    print(f"\nsampled outputs differing from greedy: {differs}")
    assert differs, "temperature 0.8 never left the greedy path"
