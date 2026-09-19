"""Tokenization for the server: a replaceable backend, and incremental detokenization.

The server talks to a ``TokenizerBackend`` only through two coroutines:
``encode`` (prompt -> ids) and ``detokenize`` (a batch of new tokens per
request -> a batch of text increments). The per-request decoding state lives
inside the backend, keyed by an opaque request key and dropped when the request
finishes, so a backend that runs the tokenizer in a subprocess can keep that
state on its side without changing the interface. The only backend so far,
``ThreadPoolTokenizer``, runs a Hugging Face tokenizer in a thread pool so that
tokenization does not block the event loop.

Byte-level BPE splits many characters (most CJK characters, emoji) over two or
more tokens. Decoding each token on its own would emit U+FFFD for the partial
bytes. ``IncrementalDetokenizer`` holds text back until it decodes to complete
characters, and only then emits it.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

REPLACEMENT_CHAR = "�"

# A prompt is raw text, token ids, or a chat message list (rendered with the chat template).
Prompt = str | list[int] | list[dict]


@dataclass
class DetokenizeItem:
    key: Hashable  # identifies the request across calls
    new_ids: list[int]  # tokens produced since the previous call for this key
    finished: bool  # last call for this key: flush all remaining text and drop the state


class TokenizerBackend(ABC):
    @abstractmethod
    async def encode(self, prompt: Prompt, chat_template_kwargs: dict | None = None) -> list[int]:
        """Token ids of a prompt. Chat message lists get the chat template with a generation prompt."""

    @abstractmethod
    async def detokenize(self, items: Sequence[DetokenizeItem]) -> list[str]:
        """Text increment of each item, in order. Keys are independent of each other."""

    def close(self) -> None:
        pass


@dataclass
class IncrementalDetokenizer:
    """Text of a growing token sequence, emitted as soon as it is made of complete characters.

    Two token offsets bound the decode window: ``prefix`` is where the window
    starts and ``read`` how far text has been emitted. After new tokens arrive,
    ``decode(ids[prefix:])`` minus ``decode(ids[prefix:read])`` is the new text;
    it is emitted unless it ends in U+FFFD (an incomplete character), in which
    case it waits for more tokens. Decoding a short window instead of the whole
    sequence keeps each step O(new tokens), and starting the window a little
    before ``read`` gives the tokenizer the context it needs to decode the
    boundary the same way the full sequence would.
    """

    ids: list[int] = field(default_factory=list)
    prefix: int = 0
    read: int = 0

    def step(self, tokenizer, new_ids: Sequence[int], finished: bool) -> str:
        self.ids.extend(new_ids)
        if self.read == len(self.ids):
            return ""
        before = tokenizer.decode(self.ids[self.prefix : self.read], skip_special_tokens=False)
        text = tokenizer.decode(self.ids[self.prefix :], skip_special_tokens=False)
        new = text[len(before) :]
        if not finished and (not new or new.endswith(REPLACEMENT_CHAR)):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return new


class ThreadPoolTokenizer(TokenizerBackend):
    """A Hugging Face tokenizer run in a thread pool. Decoding states live in this object."""

    def __init__(self, tokenizer, num_workers: int = 1):
        self.tokenizer = tokenizer
        self._pool = ThreadPoolExecutor(num_workers, thread_name_prefix="tokenizer")
        self._states: dict[Hashable, IncrementalDetokenizer] = {}

    async def encode(self, prompt: Prompt, chat_template_kwargs: dict | None = None) -> list[int]:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._encode, prompt, chat_template_kwargs)

    def _encode(self, prompt: Prompt, chat_template_kwargs: dict | None) -> list[int]:
        if isinstance(prompt, str):
            return self.tokenizer(prompt).input_ids
        if prompt and isinstance(prompt[0], dict):
            return list(
                self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **(chat_template_kwargs or {}),
                )
            )
        return list(prompt)

    async def detokenize(self, items: Sequence[DetokenizeItem]) -> list[str]:
        # One executor call per batch. The server calls this from a single coroutine,
        # so no two calls touch the same state concurrently.
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._detokenize, list(items))

    def _detokenize(self, items: list[DetokenizeItem]) -> list[str]:
        out = []
        for it in items:
            state = self._states.setdefault(it.key, IncrementalDetokenizer())
            out.append(state.step(self.tokenizer, it.new_ids, it.finished))
            if it.finished:
                del self._states[it.key]
        return out

    @property
    def num_states(self) -> int:
        return len(self._states)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
