"""Server: incremental detokenization, OpenAI parameter mapping, the engine thread, and HTTP end to end.

CPU tests use the real Qwen3 tokenizer and the toy model runner of
``test_engine``. The end-to-end tests start a real uvicorn server on a
background thread and talk to it over HTTP.

What must hold:

- the text increments of a request concatenate to ``decode`` of its tokens
  (minus a final stop token), and no increment carries half a character;
- cancelling a request returns its KV blocks whether it is queued, decoding or
  requeued after a preemption, over the engine-thread API and over HTTP when
  the client disconnects;
- serving does not change output: a greedy request over HTTP returns exactly
  the text of the same request run offline on the same engine, both starting
  from an empty prefix cache.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time

import pytest

from miniserve.engine.request import RequestState, SamplingParams
from miniserve.server.async_engine import AsyncEngine, EngineDead
from miniserve.server.protocol import ChatCompletionRequest, CompletionRequest, ProtocolError, sampling_params
from miniserve.server.tokenizer import REPLACEMENT_CHAR, DetokenizeItem, IncrementalDetokenizer, ThreadPoolTokenizer
from test_engine import TOY_STOP, TOY_VOCAB, _toy_engine, _toy_generate

QWEN_STOP = frozenset({151645, 151643})


@pytest.fixture(scope="module")
def tokenizer(qwen3_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen3_path)


# --------------------------------------------------------------------------- incremental detokenization

TEXTS = {
    "zh": "长城是中国古代的军事防御工程，始建于春秋战国时期。",
    "emoji": "Deploy done 🚀🔥 — 테스트 통과 ✅, ставка 👍🏽!",
    "code": "def f(x):\n    return {'a': x ** 2}  # 平方\n",
    "en": "The capital of France is Paris. It is known for the Eiffel Tower.",
}


def _stream_text(tokenizer, ids, chunk=1) -> list[str]:
    d = IncrementalDetokenizer()
    out = []
    for i in range(0, len(ids), chunk):
        out.append(d.step(tokenizer, ids[i : i + chunk], finished=i + chunk >= len(ids)))
    return out


@pytest.mark.parametrize("name", list(TEXTS))
@pytest.mark.parametrize("chunk", [1, 3])
def test_detokenizer_concatenates_to_decode(tokenizer, name, chunk):
    ids = tokenizer(TEXTS[name]).input_ids
    parts = _stream_text(tokenizer, ids, chunk)
    assert "".join(parts) == tokenizer.decode(ids) == TEXTS[name]
    assert not any(REPLACEMENT_CHAR in p for p in parts)


def test_detokenizer_holds_back_partial_characters(tokenizer):
    # Characters split over several byte-level tokens, e.g. an emoji and a rare CJK character.
    ids = _split_char(tokenizer)
    text = tokenizer.decode(ids)
    d = IncrementalDetokenizer()
    emitted = [d.step(tokenizer, [t], finished=False) for t in ids]
    assert "" in emitted  # something was held back
    assert "".join(emitted) == text


def _split_char(tokenizer) -> list[int]:
    """Ids of a character that byte-level BPE splits over several tokens."""
    for ch in "龘𠀀🦩🫠鱻":
        ids = tokenizer(ch).input_ids
        if len(ids) > 1:
            return ids
    raise AssertionError("no split character found")


def test_detokenizer_flushes_truncated_character(tokenizer):
    cut = _split_char(tokenizer)[:-1]  # generation stopped mid-character
    parts = _stream_text(tokenizer, cut)
    assert "".join(parts) == tokenizer.decode(cut)
    assert parts[-1].endswith(REPLACEMENT_CHAR)


@pytest.mark.parametrize("seed", range(20))
def test_detokenizer_random_tokens(tokenizer, seed):
    rng = random.Random(seed)
    ids = [rng.randrange(151_643) for _ in range(rng.randint(1, 80))]
    assert "".join(_stream_text(tokenizer, ids, rng.choice([1, 2, 5]))) == tokenizer.decode(ids)


def test_thread_pool_backend_keys_and_cleanup(tokenizer):
    a, b = tokenizer("你好，世界").input_ids, tokenizer("hello world").input_ids

    async def run():
        backend = ThreadPoolTokenizer(tokenizer, num_workers=2)
        outs = {"a": [], "b": []}
        for i in range(max(len(a), len(b))):
            items = [
                DetokenizeItem(k, ids[i : i + 1], i + 1 >= len(ids)) for k, ids in (("a", a), ("b", b)) if i < len(ids)
            ]
            for it, text in zip(items, await backend.detokenize(items)):
                outs[it.key].append(text)
        assert backend.num_states == 0
        backend.close()
        return "".join(outs["a"]), "".join(outs["b"])

    assert asyncio.run(run()) == ("你好，世界", "hello world")


# --------------------------------------------------------------------------- OpenAI parameters


def _chat(**kw):
    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **kw)


def test_openai_defaults():
    p = sampling_params(_chat(), 10, 100, QWEN_STOP)
    assert p.temperature == 1.0 and p.top_p == 1.0 and p.seed is None
    assert p.max_new_tokens == 90  # chat: the rest of the context
    assert p.stop_token_ids == QWEN_STOP
    assert sampling_params(CompletionRequest(prompt="x"), 10, 100, QWEN_STOP).max_new_tokens == 16


def test_openai_mapping():
    p = sampling_params(_chat(temperature=0, max_tokens=5, seed=-1, top_p=0.5), 10, 100, QWEN_STOP)
    assert p.is_greedy and p.max_new_tokens == 5 and p.seed == 2**32 - 1 and p.top_p == 0.5
    assert sampling_params(_chat(max_completion_tokens=7), 10, 100, QWEN_STOP).max_new_tokens == 7
    assert sampling_params(_chat(ignore_eos=True), 10, 100, QWEN_STOP).stop_token_ids == frozenset()
    # no-op values of unsupported parameters are accepted
    sampling_params(_chat(n=1, stop=[], presence_penalty=0, frequency_penalty=0, logprobs=False), 10, 100, QWEN_STOP)


@pytest.mark.parametrize(
    "kw, prompt_len",
    [
        (dict(n=2), 10),
        (dict(stop=["\n"]), 10),
        (dict(presence_penalty=0.5), 10),
        (dict(frequency_penalty=0.1), 10),
        (dict(logprobs=True), 10),
        (dict(top_logprobs=2), 10),
        (dict(max_tokens=91), 10),  # prompt + max_tokens > context
        (dict(max_tokens=0), 10),
        (dict(max_tokens=5, max_completion_tokens=5), 10),
        (dict(temperature=-1), 10),
        (dict(top_p=0), 10),
        (dict(), 0),  # empty prompt
        (dict(), 100),  # no room for a single token
    ],
)
def test_openai_rejections(kw, prompt_len):
    with pytest.raises(ProtocolError):
        sampling_params(_chat(**kw), prompt_len, 100, QWEN_STOP)


def test_chat_content_parts():
    req = ChatCompletionRequest(messages=[{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert req.chat() == [{"role": "user", "content": "ab"}]
    with pytest.raises(ProtocolError):
        ChatCompletionRequest(messages=[{"role": "user", "content": [{"type": "image_url"}]}]).chat()


# --------------------------------------------------------------------------- engine thread (toy model)


class _ToyTokenizer:
    """A backend whose 'text' is one bracketed number per token; it keeps no state."""

    async def encode(self, prompt, chat_template_kwargs=None):
        return list(prompt)

    async def detokenize(self, items):
        return ["".join(f"<{t}>" for t in it.new_ids) for it in items]

    def close(self):
        pass


def _toy_async(num_blocks=64, block_size=4, **kw) -> AsyncEngine:
    kw.setdefault("max_running", 16)
    kw.setdefault("max_prefill_tokens", 48)
    return AsyncEngine(_toy_engine(num_blocks, block_size, **kw), _ToyTokenizer())


async def _drain(stream):
    ids, text, out = [], [], None
    while out is None or out.finish_reason is None:
        out = await stream.get()
        ids += out.token_ids
        text.append(out.text)
    return ids, "".join(text), out


async def _assert_idle_and_clean(ae: AsyncEngine):
    def state(eng):
        kv = eng.runner.kv
        kv.check_invariants()  # includes: nothing in the prefix cache is locked unless held
        locked = kv.tree.num_cached_blocks - kv.tree.num_evictable if kv.tree else 0
        return eng.has_unfinished, len(eng.requests) + locked, kv.num_idle_blocks(), kv.allocator.num_blocks

    busy, live, free, total = await ae.call(state)
    assert not busy and live == 0 and free == total, (busy, live, free, total)
    assert not ae._live and not ae._live_by_rid


def _toy_requests(rng, n, cap):
    out = []
    for _ in range(n):
        plen = rng.randint(1, 24)
        prompt = [rng.randrange(1, TOY_VOCAB) for _ in range(plen)]
        temperature = rng.choice([0.0, 4.0])
        out.append((prompt, SamplingParams(rng.randint(1, min(40, cap - plen)), frozenset({TOY_STOP}), temperature, 1.0, rng.randrange(2**32))))
    return out


@pytest.mark.parametrize("seed", range(3))
def test_async_engine_outputs_match_request_alone(seed):
    rng = random.Random(seed)

    async def run():
        ae = _toy_async(num_blocks=16)
        await ae.start()
        reqs = _toy_requests(rng, 24, 16 * 4)
        streams = []
        for prompt, params in reqs:
            streams.append(ae.submit(prompt, params))
            await asyncio.sleep(rng.choice([0, 0, 0.001]))
        results = await asyncio.gather(*(_drain(s) for s in streams))
        for (prompt, params), (ids, text, last) in zip(reqs, results):
            assert ids == _toy_generate(prompt, params, params.seed)
            text_ids = ids[:-1] if last.finish_reason == "stop" else ids
            assert text == "".join(f"<{t}>" for t in text_ids)
            assert last.finish_reason == ("stop" if ids[-1] == TOY_STOP else "length")
            assert last.num_output_tokens == len(ids) and last.num_prompt_tokens == len(prompt)
        preemptions = await ae.call(lambda eng: eng.scheduler.num_preemptions)
        await _assert_idle_and_clean(ae)
        await ae.stop()
        return preemptions

    assert asyncio.run(run()) > 0, "the small pool should force preemption"


def test_abort_waiting_request():
    async def run():
        ae = _toy_async(max_running=1)
        await ae.start()
        long = ae.submit([1, 2, 3], SamplingParams(40, frozenset(), 0.0))
        queued = ae.submit([4, 5, 6], SamplingParams(40, frozenset(), 0.0))
        ae.abort(queued)  # FIFO: handled right after its add, while it waits behind `long`
        ids, _, last = await _drain(queued)
        assert (ids, last.finish_reason, last.num_output_tokens) == ([], "abort", 0)
        await _drain(long)
        await _assert_idle_and_clean(ae)
        await ae.stop()

    asyncio.run(run())


def test_abort_decoding_and_preempted_requests():
    """Aborts issued from the step hook reach the engine before its next ``schedule()``, so the
    request is aborted in exactly the state the hook saw: decoding, or waiting after a preemption."""
    seen: dict[str, int] = {"decode": 0, "preempted_waiting": 0}

    def hook(ae, batch):
        for live in list(ae._live.values()):
            r = live.req
            if r.state is RequestState.WAITING and r.num_preemptions > 0 and not getattr(r, "_aborting", False):
                seen["preempted_waiting"] += 1
            elif r.state is RequestState.DECODE and len(r.output_ids) >= 3 and r.rid % 3 == 0 and not getattr(r, "_aborting", False):
                seen["decode"] += 1
            else:
                continue
            r._aborting = True
            ae.abort(live.stream)

    async def run():
        ae = _toy_async(num_blocks=8)
        ae.step_hook = hook
        await ae.start()
        rng = random.Random(0)
        reqs = _toy_requests(rng, 40, 8 * 4)
        streams = [ae.submit(p, sp) for p, sp in reqs]
        results = await asyncio.gather(*(_drain(s) for s in streams))
        for (prompt, params), (ids, _, last) in zip(reqs, results):
            if last.finish_reason == "abort":  # a prefix of what the request would have produced
                assert ids == _toy_generate(prompt, params, params.seed)[: len(ids)]
            else:
                assert ids == _toy_generate(prompt, params, params.seed)
        await _assert_idle_and_clean(ae)
        await ae.stop()
        return sum(last.finish_reason == "abort" for *_, last in results)

    aborted = asyncio.run(run())
    assert seen["decode"] > 0 and seen["preempted_waiting"] > 0, seen
    assert aborted == seen["decode"] + seen["preempted_waiting"]


def test_abort_is_idempotent_and_late_abort_is_noop():
    async def run():
        ae = _toy_async()
        await ae.start()
        s = ae.submit([1, 2], SamplingParams(5, frozenset(), 0.0))
        ids, _, last = await _drain(s)
        ae.abort(s)  # already finished
        ae.abort(s)
        assert last.finish_reason == "length" and len(ids) == 5
        await _assert_idle_and_clean(ae)
        await ae.stop()

    asyncio.run(run())


def test_call_propagates_exceptions():
    async def run():
        ae = _toy_async()
        await ae.start()
        assert await ae.call(lambda eng: eng.runner.allocator.num_blocks) == 64
        with pytest.raises(ZeroDivisionError):
            await ae.call(lambda eng: 1 / 0)
        await ae.stop()

    asyncio.run(run())


def test_engine_failure_ends_streams_and_refuses_new_requests():
    async def run():
        ae = _toy_async()
        forward = ae.engine.runner.forward
        calls = []

        def failing_forward(batch, fill=None):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("device lost")
            return forward(batch, fill)

        ae.engine.runner.forward = failing_forward
        await ae.start()
        s = ae.submit([1, 2, 3], SamplingParams(20, frozenset(), 0.0))
        _, _, last = await _drain(s)
        assert last.finish_reason == "error" and "device lost" in last.error
        with pytest.raises(EngineDead):
            ae.submit([1], SamplingParams(1))
        with pytest.raises(EngineDead):
            await ae.call(lambda eng: None)
        await ae.stop()

    asyncio.run(run())


# --------------------------------------------------------------------------- HTTP end to end (GPU)

POOL_TOKENS = 1536  # small enough that 32 concurrent requests preempt


@pytest.fixture(scope="module")
def server(qwen3_path):
    """A real uvicorn server on a free port, on a background thread. Yields (base_url, async_engine, tokenizer)."""
    import torch
    import uvicorn

    from miniserve.engine.engine import Engine
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import load_config, load_weights
    from miniserve.server.__main__ import build_server

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(qwen3_path)), load_weights(qwen3_path))
    engine = Engine(model, max_running=16, kv_pool_tokens=POOL_TOKENS)
    app = build_server(engine, qwen3_path, "test-model")
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while not srv.started:
        assert thread.is_alive() and time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    port = srv.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", app.state.engine, app.state.engine.tokenizer.tokenizer, srv
    srv.should_exit = True
    thread.join(timeout=60)


def _on_engine(server, fn):
    """Run ``fn(engine)`` on the engine thread from a test (the server runs its own event loop)."""
    _, ae, _, srv = server
    return asyncio.run_coroutine_threadsafe(ae.call(fn), ae._loop).result(timeout=300)


def _clear_prefix_cache(server):
    """Start from an empty prefix cache. With KV reuse, a request's computation (and so, at BF16
    near-ties, its output) depends on what is cached; comparisons need the same cache state."""
    _on_engine(server, lambda eng: eng.runner.kv.set_radix(eng.runner.kv.radix))


def _offline(server, prompt_ids, max_tokens):
    """The same request run alone on the same engine from an empty prefix cache, submitted directly
    (the server must be idle)."""
    params = SamplingParams(max_tokens, QWEN_STOP)
    _clear_prefix_cache(server)
    return _on_engine(server, lambda eng: eng.generate([prompt_ids], params)[0])


def _text_of(tokenizer, ids):
    return tokenizer.decode(ids[:-1] if ids and ids[-1] in QWEN_STOP else ids)


def _sse_events(lines):
    for line in lines:
        if line.startswith("data: "):
            data = line[len("data: ") :]
            yield data if data == "[DONE]" else json.loads(data)


@pytest.mark.gpu
@pytest.mark.slow
def test_http_completion_matches_offline(server):
    import httpx

    url, _, tok, _ = server
    prompt = "The capital of France is"
    _clear_prefix_cache(server)
    r = httpx.post(f"{url}/v1/completions", json=dict(prompt=prompt, max_tokens=64, temperature=0), timeout=120)
    assert r.status_code == 200, r.text
    body = r.json()
    ids = _offline(server, tok(prompt).input_ids, 64)
    assert body["choices"][0]["text"] == _text_of(tok, ids)
    assert body["choices"][0]["finish_reason"] == ("stop" if ids[-1] in QWEN_STOP else "length")
    assert body["usage"] == dict(prompt_tokens=len(tok(prompt).input_ids), completion_tokens=len(ids), total_tokens=len(tok(prompt).input_ids) + len(ids))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("content", ["请用三句话介绍一下长城的历史。", "Explain what a KV cache is in one paragraph."])
def test_http_chat_stream_matches_offline(server, content):
    import httpx

    url, _, tok, _ = server
    messages = [{"role": "user", "content": content}]
    req = dict(
        messages=messages,
        max_tokens=128,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    _clear_prefix_cache(server)
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=req, timeout=120) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = list(_sse_events(r.iter_lines()))
    assert events[-1] == "[DONE]"
    chunks, usage = events[:-2], events[-2]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "test-model" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert not any(REPLACEMENT_CHAR in c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert len(chunks) > 10  # streamed, not one block
    prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
    ids = _offline(server, prompt_ids, 128)
    assert text == _text_of(tok, ids)
    assert chunks[-1]["choices"][0]["finish_reason"] == ("stop" if ids[-1] in QWEN_STOP else "length")
    assert usage["choices"] == [] and usage["usage"]["completion_tokens"] == len(ids)
    assert usage["usage"]["prompt_tokens"] == len(prompt_ids)


@pytest.mark.gpu
@pytest.mark.slow
def test_http_seeded_sampling_reproducible(server):
    import httpx

    url, _, _, _ = server
    req = dict(prompt="Once upon a time", max_tokens=48, temperature=0.8, top_p=0.9, seed=1234)
    a, b = (httpx.post(f"{url}/v1/completions", json=req, timeout=120).json() for _ in range(2))
    greedy = httpx.post(f"{url}/v1/completions", json=req | dict(temperature=0), timeout=120).json()
    assert a["choices"][0]["text"] == b["choices"][0]["text"] != greedy["choices"][0]["text"]


@pytest.mark.gpu
@pytest.mark.slow
def test_http_errors_and_metadata(server):
    import httpx

    url, _, _, _ = server
    assert httpx.get(f"{url}/health").json() == {"status": "ok"}
    models = httpx.get(f"{url}/v1/models").json()
    assert models["data"][0]["id"] == "test-model" and models["data"][0]["max_model_len"] == POOL_TOKENS
    for bad in [
        dict(prompt="x", n=2),
        dict(prompt="x", stop=["\n"]),
        dict(prompt="x", max_tokens=POOL_TOKENS),  # prompt + output exceed the pool
        dict(prompt="x", temperature="hot"),
        dict(max_tokens=3),  # no prompt
    ]:
        r = httpx.post(f"{url}/v1/completions", json=bad, timeout=30)
        assert r.status_code == 400 and "error" in r.json(), (bad, r.status_code, r.text)


@pytest.mark.gpu
@pytest.mark.slow
def test_http_disconnects_return_all_blocks(server):
    """32 concurrent requests under a small pool; clients disconnect at random points (some while
    still queued, some mid-stream, some non-streaming). Afterwards the engine is empty and every
    block is free; requests that were not cancelled completed normally."""
    import httpx

    url, _, _, _ = server
    aborted_states: list[tuple[str, int]] = []

    def record_aborts(eng):
        original = eng.abort

        def abort(rid):
            req = eng.requests.get(rid)
            if req is not None:
                aborted_states.append((req.state.name, req.num_preemptions))
            original(rid)

        eng.abort = abort
        return eng.scheduler.num_preemptions

    preemptions_before = _on_engine(server, record_aborts)
    rng = random.Random(0)

    async def client(i, http):
        stream = rng.random() < 0.75
        cut = rng.choice(["queued", "after_chunks", "never", "never"])
        req = dict(prompt=f"Request {i}: write a long story about the number {i}.", max_tokens=rng.randint(192, 512), temperature=0, ignore_eos=True, stream=stream)
        if not stream:
            timeout = 0.05 if cut == "queued" else (1.0 if cut == "after_chunks" else 300)
            try:
                r = await http.post(f"{url}/v1/completions", json=req, timeout=timeout)
                return "done" if r.status_code == 200 else f"status {r.status_code}"
            except httpx.TimeoutException:
                return "cut"
        async with http.stream("POST", f"{url}/v1/completions", json=req, timeout=300) as r:
            if cut == "queued":
                return "cut"  # headers arrive before the first token: close while the request may still wait
            n = 0
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    n += 1
                    if cut == "after_chunks" and n >= rng.randint(2, 30):
                        return "cut"
                    if line == "data: [DONE]":
                        return "done"
        return "ended"

    async def run():
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as http:
            return await asyncio.gather(*(client(i, http) for i in range(32)))

    outcomes = asyncio.run(run())
    assert "ended" not in outcomes and all(o in ("cut", "done") for o in outcomes), outcomes

    def idle_state(eng):
        kv = eng.runner.kv
        kv.check_invariants()
        locked = kv.tree.num_cached_blocks - kv.tree.num_evictable if kv.tree else 0
        return eng.has_unfinished, len(eng.requests) + locked, kv.num_idle_blocks(), kv.allocator.num_blocks, eng.scheduler.num_preemptions

    deadline = time.monotonic() + 120
    while (state := _on_engine(server, idle_state))[0] and time.monotonic() < deadline:
        time.sleep(0.2)
    busy, live, free, total, preemptions = state
    assert not busy and live == 0 and free == total, state
    print(f"\noutcomes: {outcomes.count('cut')} cut, {outcomes.count('done')} done; "
          f"preemptions {preemptions - preemptions_before}; aborted in states {aborted_states}")
    states = {s for s, _ in aborted_states}
    assert {"WAITING", "DECODE"} <= states, aborted_states
    assert preemptions > preemptions_before, "the small pool should force preemption"
