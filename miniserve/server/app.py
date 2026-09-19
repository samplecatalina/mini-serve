"""HTTP front end: OpenAI-compatible routes, server-sent events, and cancellation on disconnect.

Every generation request gets a watcher task that waits for the client's
``http.disconnect``. When it arrives, the watcher aborts the request in the
engine, which returns its KV blocks, and wakes the request's stream so the
handler stops. This works the same whether the request is still queued
(nothing sent yet), decoding, or requeued after a preemption, and for
streaming and non-streaming responses alike. A handler that ends early for
any other reason (a failed send, server shutdown) aborts the request in its
``finally``; aborting twice is harmless.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from miniserve.server.async_engine import AsyncEngine, ClientDisconnected, EngineDead, RequestStream, StreamOutput
from miniserve.server.protocol import ChatCompletionRequest, CompletionRequest, ProtocolError, sampling_params

# Status for a response nobody reads because the client went away (nginx convention).
CLIENT_CLOSED = 499


def _error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)


def build_app(
    engine: AsyncEngine,
    *,
    model_name: str,
    context_len: int,
    stop_token_ids: frozenset[int],
) -> FastAPI:
    """``context_len``: bound on prompt + output tokens of one request (model context and KV pool)."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="miniserve", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return _error(400, str(exc.errors()))

    @app.exception_handler(ProtocolError)
    async def _protocol_error(_: Request, exc: ProtocolError):
        return _error(400, str(exc))

    @app.exception_handler(EngineDead)
    async def _engine_dead(_: Request, exc: EngineDead):
        return _error(503, str(exc), "server_error")

    @app.get("/health")
    async def health():
        if engine.dead is not None:
            return _error(503, f"engine stopped: {engine.dead!r}", "server_error")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        card = dict(id=model_name, object="model", created=0, owned_by="miniserve", max_model_len=context_len)
        return {"object": "list", "data": [card]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        prompt_ids = await engine.tokenizer.encode(req.chat(), req.chat_template_kwargs)
        return await _generate(req, request, prompt_ids, chat=True)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        prompt_ids = await engine.tokenizer.encode(req.prompt)
        return await _generate(req, request, prompt_ids, chat=False)

    async def _generate(req, request: Request, prompt_ids: list[int], chat: bool):
        params = sampling_params(req, len(prompt_ids), context_len, stop_token_ids)
        stream = engine.submit(prompt_ids, params)
        watcher = asyncio.create_task(_watch_disconnect(request, stream, engine))
        fmt = _Format(chat, req.model or model_name)
        if req.stream:
            include_usage = req.stream_options is not None and req.stream_options.include_usage
            return StreamingResponse(_sse(stream, watcher, fmt, include_usage), media_type="text/event-stream")
        try:
            return await _collect(stream, fmt)
        finally:
            _end(stream, watcher)

    async def _sse(stream: RequestStream, watcher: asyncio.Task, fmt: _Format, include_usage: bool) -> AsyncIterator[str]:
        try:
            first = True
            while True:
                out = await stream.get()
                if out.finish_reason in ("abort", "error"):
                    yield _event({"error": {"message": out.error or out.finish_reason, "type": "server_error"}})
                    break
                if out.text or first and out.finish_reason is None:
                    yield _event(fmt.chunk(out.text, None, first))
                    first = False
                if out.finish_reason is not None:
                    yield _event(fmt.chunk("", out.finish_reason, first))
                    if include_usage:
                        yield _event(fmt.usage_chunk(out))
                    break
            yield "data: [DONE]\n\n"
        except ClientDisconnected:
            return
        finally:
            _end(stream, watcher)

    async def _collect(stream: RequestStream, fmt: _Format) -> Response:
        parts = []
        try:
            while True:
                out = await stream.get()
                parts.append(out.text)
                if out.finish_reason is not None:
                    break
        except ClientDisconnected:
            return Response(status_code=CLIENT_CLOSED)
        if out.finish_reason in ("abort", "error"):
            return _error(503 if out.finish_reason == "abort" else 500, out.error or out.finish_reason, "server_error")
        return JSONResponse(fmt.complete("".join(parts), out))

    def _end(stream: RequestStream, watcher: asyncio.Task) -> None:
        watcher.cancel()
        if not stream.finished:
            engine.abort(stream)

    return app


async def _watch_disconnect(request: Request, stream: RequestStream, engine: AsyncEngine) -> None:
    """Wait for the client to disconnect, then abort the request and wake the stream's consumer.

    The body has been read by then, so the next ASGI message is the disconnect
    (or nothing, until the response is complete and this task is cancelled).
    Aborting here, not only in the consumer, frees the blocks even if the
    response machinery stops iterating the stream without closing it.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            if not stream.finished:
                engine.abort(stream)
            stream.disconnect()
            return


def _event(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


class _Format:
    """Response objects of the chat and the legacy completions API."""

    def __init__(self, chat: bool, model: str):
        self.chat = chat
        self.model = model
        self.id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
        self.created = int(time.time())

    def _envelope(self, choices: list[dict], streaming: bool) -> dict:
        obj = ("chat.completion.chunk" if streaming else "chat.completion") if self.chat else "text_completion"
        return dict(id=self.id, object=obj, created=self.created, model=self.model, choices=choices)

    def chunk(self, text: str, finish_reason: str | None, first: bool) -> dict:
        if self.chat:
            delta = {"role": "assistant"} if first else {}
            if text or first:
                delta["content"] = text
            choice = dict(index=0, delta=delta, finish_reason=finish_reason)
        else:
            choice = dict(index=0, text=text, finish_reason=finish_reason)
        return self._envelope([choice], streaming=True)

    def usage_chunk(self, last: StreamOutput) -> dict:
        return self._envelope([], streaming=True) | {"usage": _usage(last)}

    def complete(self, text: str, last: StreamOutput) -> dict:
        if self.chat:
            choice = dict(index=0, message={"role": "assistant", "content": text}, finish_reason=last.finish_reason)
        else:
            choice = dict(index=0, text=text, finish_reason=last.finish_reason)
        return self._envelope([choice], streaming=False) | {"usage": _usage(last)}


def _usage(last: StreamOutput) -> dict:
    return dict(
        prompt_tokens=last.num_prompt_tokens,
        completion_tokens=last.num_output_tokens,
        total_tokens=last.num_prompt_tokens + last.num_output_tokens,
    )
