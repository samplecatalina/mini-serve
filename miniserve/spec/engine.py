"""The engine loop with speculation: the draft proposes, one target pass verifies.

A decode step becomes a *round*:

1. the draft model produces ``gamma`` tokens for every request in the batch,
   one forward pass each, all of them queued without waiting for the device;
2. the target runs a single pass over ``gamma + 1`` positions per request
   (the token it sampled last, then the proposals), which is the same kind of
   pass as a prefill chunk: several new tokens per sequence, causal, paged KV;
3. the host reads the proposals and the target's choices back -- the one
   synchronization of a round -- keeps the longest agreeing prefix plus the
   target's own next token (``verify.py``), and gives back the KV of the
   rejected positions with ``rewind``.

Greedy output is therefore exactly the output of plain decoding, whatever the
draft proposes; a bad draft costs time, never correctness.

Three things this path gives up, all of them deliberate:

- **overlap scheduling**. How many tokens a round produces is only known once
  the host has read the verdict, and the next round's inputs, block tables and
  stop decisions all depend on it. So a round synchronizes, and the CPU-GPU
  overlap of the plain loop is gone.
- **the prefix cache**. The draft keeps its own pool, sized like the target's
  in tokens. That is only provably enough while the two tables grow in step,
  which prefix sharing breaks (the target would share a long prefix between
  requests and the draft would not, or the other way round). See
  ``spec/draft.py``.
- **the target's decode graphs**. A verify pass is an extend pass, so the
  captured decode graphs would never run; they are not captured at all, and
  their memory goes to the KV pool instead.

What a round does capture: the verify pass, as a graph of width ``gamma + 1``
per batch-size bucket (captured when ``gamma`` is set, so never while a run
is being measured), and every draft step (``spec/draft.py``). Without them an
eager pass of a small model is bound by the host issuing its kernels, not by
the device reading its weights. ``round_graphs`` switches all of them off,
which is the eager arm of the ablation.

A batch that is not a plain decode batch (a prefill, or a mixed batch under
chunked prefill) runs through the ordinary loop, one token per decoding
request. The draft then simply falls behind, and the next round feeds it
everything it missed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from miniserve.engine.cuda_graph import DecodeGraphs, graph_buckets
from miniserve.engine.engine import Engine
from miniserve.engine.model_runner import ModelRunner
from miniserve.engine.request import Request, RequestState, SamplingParams
from miniserve.engine.scheduler import Batch, Phase
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.spec.draft import FIRST_WIDTH, DraftRunner
from miniserve.spec.verify import accept_prefix

# Of the free memory, the share the two KV pools may take. Lower than the plain
# engine's, because the draft's graphs and its block tables come out of the rest.
KV_MEM_FRACTION = 0.85


def block_bytes(cfg: Qwen3Config, block_size: int, dtype: torch.dtype) -> int:
    """Bytes one KV block of a model takes: K and V, every layer, every KV head."""
    return 2 * cfg.num_layers * block_size * cfg.num_kv_heads * cfg.head_dim * dtype.itemsize


def check_options(attention: str, radix: bool, overlap: bool, gamma: int) -> None:
    """Raise for the settings a round cannot support, with the reason it cannot."""
    if attention != "paged":
        raise ValueError(f"speculative decoding needs paged attention, got {attention!r}")
    if radix:
        raise ValueError(
            "speculative decoding runs with the prefix cache off: the draft pool is sized "
            "like the target's, which only holds while the two block tables grow in step"
        )
    if overlap:
        raise ValueError("speculative decoding reads each round's verdict back, so it cannot overlap steps")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")


class SpecEngine(Engine):
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        draft_model: Qwen3ForCausalLM,
        gamma: int = 4,
        max_running: int = 64,
        max_prefill_tokens: int = 8192,
        attention: str = "paged",
        kv_pool_tokens: int | None = None,
        seed: int = 0,
        radix: bool = False,
        chunked_prefill_size: int | None = None,
        cuda_graph: bool = True,
        cuda_graph_max_bs: int | None = None,
        overlap: bool = False,
        schedule_policy: str = "fcfs",
        block_backend: str | None = None,
        block_size: int = 16,
    ):
        """``draft_model``: the proposing model, same tokenizer, same dtype (it may be the
        target itself, which makes every proposal a correct one and is how the loop is tested).
        ``gamma``: proposals per round; 0 runs plain decode steps through the same object,
        which is the off arm of the ablation. ``cuda_graph``: capture the graphs of a round
        (the draft's steps and the target's verify pass; the target's decode graphs are
        never captured here). Every other argument means what it
        means for ``Engine``, except that ``radix`` and ``overlap`` must be off."""
        check_options(attention, radix, overlap, gamma)
        if model.dtype is not draft_model.dtype:
            raise ValueError(f"draft dtype {draft_model.dtype} differs from the target's {model.dtype}")
        if model.cfg.vocab_size != draft_model.cfg.vocab_size:
            raise ValueError(
                f"draft vocabulary {draft_model.cfg.vocab_size} differs from the target's {model.cfg.vocab_size}"
            )
        target_bytes = block_bytes(model.cfg, block_size, model.dtype)
        draft_bytes = block_bytes(draft_model.cfg, block_size, draft_model.dtype)
        runner = ModelRunner(
            model,
            attention="paged",
            kv_pool_tokens=kv_pool_tokens,
            block_size=block_size,
            max_prefill_tokens=max_prefill_tokens,
            # The pools hold the same number of tokens, so they split the memory in the
            # ratio of one block's size.
            kv_mem_fraction=KV_MEM_FRACTION * target_bytes / (target_bytes + draft_bytes),
            max_running=max_running,
            radix=False,
            cuda_graph=False,
            block_backend=block_backend,
            # A verify pass holds the logits of every verified position, not one row per request.
            sample_rows=max_running * (gamma + 1),
        )
        super().__init__(
            None,
            runner=runner,
            max_running=max_running,
            max_prefill_tokens=max_prefill_tokens,
            kv_pool_tokens=kv_pool_tokens,
            seed=seed,
            radix=False,
            chunked_prefill_size=chunked_prefill_size,
            overlap=False,
            schedule_policy=schedule_policy,
        )
        self.draft = DraftRunner(
            draft_model,
            num_blocks=runner.allocator.num_blocks,
            block_size=block_size,
            max_prefill_tokens=max_prefill_tokens,
            cuda_graph=cuda_graph,
            cuda_graph_max_bs=cuda_graph_max_bs or max_running,
            block_backend=block_backend,
            workspace=runner.flashinfer.workspace,
        )
        self._cuda_graph = cuda_graph
        self._graph_max_bs = cuda_graph_max_bs or max_running
        self._verify_graphs: dict[int, DecodeGraphs] = {}
        self._round_graphs = cuda_graph
        self._gamma = 0
        self.gamma = gamma
        self._fill_idx: dict[tuple[int, int], torch.Tensor] = {}
        self.spec_stats = dict(rounds=0, rows=0, proposed=0, accepted=0, tokens=0)

    # ------------------------------------------------------------------ configuration

    @property
    def gamma(self) -> int:
        """Proposals per round; 0 decodes one token at a time. Settable between steps."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"gamma must be >= 0, got {value}")
        self._gamma = value
        # A round appends up to gamma + 1 tokens to a request, and the scheduler has to
        # hold blocks for all of them before it starts.
        self.scheduler.tokens_per_step = value + 1
        runner = self.runner
        if self._cuda_graph and 0 < value < runner.pool.block_size and value not in self._verify_graphs:
            cfg = runner.model.cfg
            self._verify_graphs[value] = DecodeGraphs(
                runner.model,
                runner.pool,
                dummy_block=runner.allocator.num_blocks,
                buckets=graph_buckets(self._graph_max_bs),
                workspace=runner.flashinfer.workspace,
                num_heads=cfg.num_heads,
                scale=runner.model.attn_scale,
                fence=runner.fence,
                width=value + 1,
            )

    @property
    def round_graphs(self) -> bool:
        """Whether a round replays captured graphs (verify pass, draft's first step); settable
        between steps. The draft's single-token steps keep their decode graphs either way."""
        return self._round_graphs

    @round_graphs.setter
    def round_graphs(self, value: bool) -> None:
        if value and not self._cuda_graph:
            raise ValueError("round graphs were not captured (cuda_graph=False)")
        self._round_graphs = value
        self.draft.use_first_graphs = value

    @property
    def acceptance(self) -> float:
        """Accepted proposals over proposals made (alpha); 0 if nothing was proposed."""
        s = self.spec_stats
        return s["accepted"] / s["proposed"] if s["proposed"] else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Mean tokens a request gets out of one round (1 + accepted); 0 if there was none."""
        s = self.spec_stats
        return s["tokens"] / s["rows"] if s["rows"] else 0.0

    def add_request(self, prompt_ids: Sequence[int], params: SamplingParams) -> Request:
        if not params.is_greedy:
            raise ValueError("speculative decoding is greedy-only for now; use the plain engine to sample")
        return super().add_request(prompt_ids, params)

    # ------------------------------------------------------------------ the loop

    def step(self) -> Batch | None:
        batch = self.scheduler.schedule()
        if batch is None:
            self.launched = None
            return None
        for r in batch.preempted:
            self.draft.release(r.rid)
        self.launched = batch
        if self._gamma and batch.phase is Phase.DECODE:
            self._round(batch)
        else:
            self._process(self._launch(batch, None))
            if self._gamma:
                self._draft_prefill([r for r in batch.requests if r.state is RequestState.DECODE])
        return batch

    def _retire(self, req: Request) -> None:
        self.draft.release(req.rid)
        super()._retire(req)

    def _draft_prefill(self, reqs: Sequence[Request]) -> None:
        """Give the draft the KV of every request in ``reqs`` that has none.

        It covers the target's tokens but the last one: a round feeds the draft that token,
        and the draft's answer to it is the round's first proposal. Called when a request
        reaches the decode phase, and again at the start of a round, which is where requests
        that got there while gamma was 0 are picked up. With gamma 0 the draft therefore
        does no work at all, which is what makes that arm of the ablation meaningful."""
        todo = []
        for r in reqs:
            if r.rid not in self.draft.tables:
                self.draft.admit(r.rid)
            if self.draft.covered(r.rid) == 0 and r.seq_len > 1:
                todo.append((r.rid, r.token_slice(0, r.seq_len - 1)))
        self.draft.prefill(todo)

    def _round(self, batch: Batch) -> None:
        # The NVTX ranges below let a profiler split a round into its parts (host and device).
        nvtx = torch.cuda.nvtx
        reqs = batch.requests
        g, width, b = self._gamma, self._gamma + 1, len(batch.requests)
        with nvtx.range("draft_prefill"):
            self._draft_prefill(reqs)
        for r in reqs:
            if r.cache.num_tokens != r.seq_len - 1:
                raise RuntimeError(f"request {r.rid}: {r.cache.num_tokens} cached tokens, sequence is {r.seq_len}")

        # 1. propose. The draft is fed whatever the target has and it has not: normally the
        # last sampled token, two tokens after a round every proposal of which was accepted,
        # more after steps that ran without speculation.
        starts = self._first_step_starts(reqs)
        pending = [r.token_slice(s, r.seq_len - s) for r, s in zip(reqs, starts)]
        with nvtx.range("propose"):
            proposals = self.draft.propose([r.rid for r in reqs], pending, starts, g)

        # 2. verify, in one pass over gamma + 1 positions per request. The proposals stay on
        # the device; only their positions in the token ids are known here.
        with nvtx.range("verify"):
            tables = [r.cache for r in reqs]
            self.runner.reserve(tables, [width] * b)
            ids: list[int] = []
            pos: list[int] = []
            for r in reqs:
                ids += [r.output_ids[-1], *([0] * g)]
                pos += range(r.seq_len - 1, r.seq_len - 1 + width)
            fill = (self._proposal_rows(b, g), proposals.reshape(-1))
            graphs = self._verify_graphs.get(g) if self._round_graphs else None
            if graphs is not None and b <= graphs.max_batch:
                slots = [s for t in tables for s in t.tail_slots(width)]
                self.runner.fence.wait()  # the staging buffer may still feed the last copy
                logits = graphs.run(ids, pos, slots, tables, fill=fill)
            else:
                logits = self.runner.forward_tokens(tables, ids, pos, [width] * b, fill=fill)
            chosen = logits.argmax(dim=-1).view(b, width)

        # 3. read both back in one copy and settle each request on the host.
        with nvtx.range("readback"):
            verdict = torch.cat((proposals, chosen), dim=1).tolist()
        with nvtx.range("settle"):
            self.spec_stats["rounds"] += 1
            for r, row in zip(reqs, verdict):
                self._settle(r, row[:g], row[g:])

    def _first_step_starts(self, reqs: Sequence[Request]) -> list[int]:
        """Where the draft's feed starts for each request: at what it covers, or, when the first
        step can replay its graph, two tokens before the end for every request. Those missing a
        single token then give back the KV of the one before it, and recompute it."""
        starts = [self.draft.covered(r.rid) for r in reqs]
        g = self.draft.first_graphs
        if not self._round_graphs or g is None or len(reqs) > g.max_batch:
            return starts
        if not all(FIRST_WIDTH <= r.seq_len and r.seq_len - FIRST_WIDTH <= s for r, s in zip(reqs, starts)):
            return starts  # someone misses more (steps ran without speculation): eager
        starts = [r.seq_len - FIRST_WIDTH for r in reqs]
        for r, s in zip(reqs, starts):
            self.draft.rewind_to(r.rid, s)
        return starts

    def _settle(self, req: Request, proposals: list[int], chosen: list[int]) -> None:
        """Append what this round produced for one request, and give back the rest of its KV."""
        k = accept_prefix(proposals, chosen)
        new = [*proposals[:k], chosen[k]]
        # Everything past the request's own end is dropped: its token budget, or a stop token,
        # which is kept but ends the request (the same rule as one token at a time).
        room = req.params.max_new_tokens - len(req.output_ids)
        keep = min(len(new), room)
        stop = next((i for i, t in enumerate(new[:keep]) if t in req.params.stop_token_ids), None)
        if stop is not None:
            keep = stop + 1
        req.output_ids += new[:keep]
        s = self.spec_stats
        s["rows"] += 1
        s["proposed"] += len(proposals)
        s["accepted"] += k
        s["tokens"] += keep
        if stop is not None or len(req.output_ids) >= req.params.max_new_tokens:
            req.transition(RequestState.FINISHED)
            self._retire(req)
            return
        req.cache.rewind(len(proposals) + 1 - keep)  # the rejected positions' KV
        self.draft.rewind_to(req.rid, req.seq_len - 1)

    def _proposal_rows(self, b: int, g: int) -> torch.Tensor:
        """Where a round's proposals go in the verify pass's token ids: every position of a
        request but its first. Fixed for a batch size, so it is built once."""
        key = (b, g)
        if key not in self._fill_idx:
            idx = [row * (g + 1) + 1 + j for row in range(b) for j in range(g)]
            self._fill_idx[key] = torch.tensor(idx, dtype=torch.long, device=self.runner.device)
        return self._fill_idx[key]
