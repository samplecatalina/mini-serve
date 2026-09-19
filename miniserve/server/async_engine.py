"""The engine behind the server: one thread owns the ``Engine``, the event loop talks to it through queues.

``Engine`` is not thread-safe, and ``step()`` blocks for as long as the
forward pass takes, so it runs in a dedicated thread and nothing else touches
it. The event loop side sends commands (add, abort, call) through a
thread-safe FIFO; the engine thread drains it between steps, and blocks on it
when there is nothing to run. Because the queue is FIFO, an abort always
reaches the engine after the add it refers to.

After each step, the engine thread collects the new tokens of every request
in the batch and hands them to the event loop in one ``call_soon_threadsafe``.
A single coroutine on the loop (the "pump") detokenizes them in one backend
call per batch and pushes the text to each request's stream. Tokens kept
across a preemption are not sent twice: the engine thread remembers how many
tokens of each request it has already handed over.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miniserve.engine.engine import Engine
from miniserve.engine.request import Request, RequestState, SamplingParams
from miniserve.server.tokenizer import DetokenizeItem, TokenizerBackend

log = logging.getLogger(__name__)


class EngineDead(RuntimeError):
    """The engine thread has stopped after an error; no new requests are accepted."""


@dataclass
class StreamOutput:
    text: str  # text increment
    token_ids: list[int]  # new tokens (a stop token included)
    finish_reason: str | None = None  # "stop" | "length" | "abort" | "error" on the last output
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0  # tokens generated so far
    error: str | None = None


class RequestStream:
    """The event loop side of one request: outputs arrive in order, the last one has a ``finish_reason``."""

    _DISCONNECTED = object()

    def __init__(self, prompt_ids: list[int], params: SamplingParams):
        self.prompt_ids = prompt_ids
        self.params = params
        self.finished = False
        self._queue: asyncio.Queue = asyncio.Queue()

    def _push(self, out: StreamOutput) -> None:
        self._queue.put_nowait(out)

    def disconnect(self) -> None:
        """Wake the consumer: the client is gone. ``get`` then raises ``ClientDisconnected``."""
        self._queue.put_nowait(self._DISCONNECTED)

    async def get(self) -> StreamOutput:
        item = await self._queue.get()
        if item is self._DISCONNECTED:
            raise ClientDisconnected
        if item.finish_reason is not None:
            self.finished = True
        return item


class ClientDisconnected(Exception):
    pass


# Commands from the event loop to the engine thread.
@dataclass
class _Add:
    stream: RequestStream


@dataclass
class _Abort:
    stream: RequestStream


@dataclass
class _Call:
    fn: Callable[[Engine], Any]
    future: asyncio.Future


_STOP = object()


@dataclass
class _Update:
    stream: RequestStream
    new_ids: list[int]
    finish_reason: str | None
    num_output_tokens: int
    error: str | None = None


class _Live:
    """Engine thread bookkeeping of one request."""

    __slots__ = ("req", "stream", "sent")

    def __init__(self, req: Request, stream: RequestStream):
        self.req = req
        self.stream = stream
        self.sent = 0  # output tokens already handed to the event loop


class AsyncEngine:
    """Owns the engine thread and the detokenization pump. Create and use it on one event loop."""

    def __init__(self, engine: Engine, tokenizer: TokenizerBackend):
        self.engine = engine  # touched only by the engine thread once started
        self.tokenizer = tokenizer
        self.dead: BaseException | None = None
        # Test hook, called on the engine thread after every step with (async_engine, batch).
        self.step_hook: Callable | None = None
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        # Engine thread only.
        self._live: dict[RequestStream, _Live] = {}
        self._live_by_rid: dict[int, _Live] = {}
        # Makes "check dead, then enqueue" and "set dead, then drain the queue" atomic,
        # so no command is enqueued after the drain and left unanswered.
        self._dead_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._updates: asyncio.Queue | None = None
        self._pump_task: asyncio.Task | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ event loop side

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._updates = asyncio.Queue()
        self._pump_task = asyncio.create_task(self._pump(), name="detokenize-pump")
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        if self._thread is not None:
            self._commands.put(_STOP)
            await asyncio.to_thread(self._thread.join)
            self._thread = None
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        self.tokenizer.close()

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> RequestStream:
        stream = RequestStream(prompt_ids, params)
        self._send(_Add(stream))
        return stream

    def _send(self, cmd) -> None:
        with self._dead_lock:
            if self.dead is not None:
                raise EngineDead(f"engine stopped: {self.dead!r}")
            self._commands.put(cmd)

    def abort(self, stream: RequestStream) -> None:
        """Abort a request if it is still running; a no-op for a finished one. Callable from any thread."""
        self._commands.put(_Abort(stream))

    async def call(self, fn: Callable[[Engine], Any]) -> Any:
        """Run ``fn(engine)`` on the engine thread between two steps and return its result."""
        future = self._loop.create_future()
        self._send(_Call(fn, future))
        return await future

    async def _pump(self) -> None:
        while True:
            updates = await self._updates.get()
            while not self._updates.empty():  # catch up on a backlog in one backend call
                updates.extend(self._updates.get_nowait())
            try:
                texts = await self.tokenizer.detokenize(
                    [DetokenizeItem(u.stream, self._text_ids(u), u.finish_reason is not None) for u in updates]
                )
            except Exception as e:  # a tokenizer failure ends the affected requests, not the server
                log.exception("detokenization failed")
                texts = [""] * len(updates)
                for u in updates:
                    if u.finish_reason is None:
                        self.abort(u.stream)  # stop generating for a request that has just ended
                    u.finish_reason, u.error = "error", f"detokenization failed: {e!r}"
            for u, text in zip(updates, texts):
                u.stream._push(
                    StreamOutput(
                        text=text,
                        token_ids=u.new_ids,
                        finish_reason=u.finish_reason,
                        num_prompt_tokens=len(u.stream.prompt_ids),
                        num_output_tokens=u.num_output_tokens,
                        error=u.error,
                    )
                )

    @staticmethod
    def _text_ids(u: _Update) -> list[int]:
        """Tokens that become text: a final stop token does not."""
        if u.finish_reason == "stop" and u.new_ids and u.new_ids[-1] in u.stream.params.stop_token_ids:
            return u.new_ids[:-1]
        return u.new_ids

    # ------------------------------------------------------------------ engine thread

    def _run(self) -> None:
        eng = self.engine
        try:
            while True:
                # Block for a command only when idle; otherwise take what is there and step.
                cmd = self._commands.get() if not eng.has_unfinished else self._get_nowait()
                while cmd is not None:
                    if cmd is _STOP:
                        self._fail_all("server shutting down", "abort")
                        return
                    self._handle(cmd)
                    cmd = self._get_nowait()
                if not eng.has_unfinished:
                    continue
                batch = eng.step()
                if batch is None:
                    continue
                updates = []
                for r in batch.requests:
                    live = self._live_by_rid.get(r.rid)
                    if live is None:
                        continue
                    new = r.output_ids[live.sent :]
                    live.sent = len(r.output_ids)
                    finish = None
                    if r.state is RequestState.FINISHED:
                        finish = "stop" if r.output_ids[-1] in r.params.stop_token_ids else "length"
                        self._forget(live)
                    updates.append(_Update(live.stream, new, finish, live.sent))
                self._publish(updates)
                if self.step_hook is not None:
                    self.step_hook(self, batch)
        except BaseException as e:
            log.exception("engine thread failed")
            with self._dead_lock:
                self.dead = e
                self._fail_all(f"engine error: {e!r}", "error")
                self._drain_after_death(e)

    def _get_nowait(self):
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _handle(self, cmd) -> None:
        eng = self.engine
        if isinstance(cmd, _Add):
            s = cmd.stream
            try:
                req = eng.add_request(s.prompt_ids, s.params)
            except ValueError as e:  # the server validates first; this is a backstop
                self._publish([_Update(s, [], "error", 0, error=str(e))])
                return
            live = _Live(req, s)
            self._live[s] = live
            self._live_by_rid[req.rid] = live
        elif isinstance(cmd, _Abort):
            live = self._live.get(cmd.stream)
            if live is None:
                return  # finished already, or never admitted
            eng.abort(live.req.rid)
            self._forget(live)
            self._publish([_Update(cmd.stream, [], "abort", live.sent)])
        elif isinstance(cmd, _Call):
            try:
                result = cmd.fn(eng)
            except Exception as e:
                self._loop.call_soon_threadsafe(_set_exception, cmd.future, e)
            else:
                self._loop.call_soon_threadsafe(_set_result, cmd.future, result)
        else:
            raise TypeError(f"unknown command {cmd!r}")

    def _forget(self, live: _Live) -> None:
        del self._live[live.stream]
        del self._live_by_rid[live.req.rid]

    def _publish(self, updates: list[_Update]) -> None:
        if updates:
            self._loop.call_soon_threadsafe(self._updates.put_nowait, updates)

    def _fail_all(self, message: str, reason: str) -> None:
        updates = [_Update(lv.stream, [], reason, lv.sent, error=message) for lv in self._live.values()]
        self._live.clear()
        self._live_by_rid.clear()
        self._publish(updates)

    def _drain_after_death(self, e: BaseException) -> None:
        """Answer commands that raced with the failure, so no caller waits forever."""
        while (cmd := self._get_nowait()) is not None:
            if isinstance(cmd, _Add):
                self._publish([_Update(cmd.stream, [], "error", 0, error=f"engine error: {e!r}")])
            elif isinstance(cmd, _Call):
                self._loop.call_soon_threadsafe(_set_exception, cmd.future, EngineDead(repr(e)))


def _set_result(future: asyncio.Future, value) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(future: asyncio.Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)
