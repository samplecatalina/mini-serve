"""The subset of the OpenAI API the server accepts, and its mapping to engine sampling parameters.

Defaults follow OpenAI, not the engine: ``temperature`` defaults to 1.0 (the
engine's ``SamplingParams`` defaults to 0, greedy), ``max_tokens`` defaults to
16 for completions and to the rest of the context for chat completions.
Parameters the engine does not implement are rejected rather than silently
ignored, since ignoring them would return output the client did not ask for.

Two extensions shared with vLLM and SGLang: ``ignore_eos`` (generate exactly
``max_tokens`` tokens, as benchmarks with fixed output lengths need) and
``chat_template_kwargs`` (e.g. ``{"enable_thinking": false}`` for Qwen3).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from miniserve.engine.request import SamplingParams


class ProtocolError(ValueError):
    """A request the server cannot serve as asked; answered with HTTP 400."""


class StreamOptions(BaseModel):
    include_usage: bool = False


class _GenerationRequest(BaseModel):
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False
    # Accepted only at their no-op values.
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: bool | int | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict] | None = None

    def text(self) -> str:
        if self.content is None or isinstance(self.content, str):
            return self.content or ""
        parts = []
        for part in self.content:
            if part.get("type") != "text":
                raise ProtocolError(f"unsupported content part type {part.get('type')!r}")
            parts.append(part.get("text", ""))
        return "".join(parts)


class ChatCompletionRequest(_GenerationRequest):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = None  # newer name of max_tokens
    chat_template_kwargs: dict | None = None
    top_logprobs: int | None = None

    def chat(self) -> list[dict]:
        return [{"role": m.role, "content": m.text()} for m in self.messages]

    def requested_max_tokens(self) -> int | None:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ProtocolError("give max_tokens or max_completion_tokens, not both")
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens


class CompletionRequest(_GenerationRequest):
    prompt: str | list[int]

    def requested_max_tokens(self) -> int | None:
        return 16 if self.max_tokens is None else self.max_tokens  # the OpenAI default for completions


def check_supported(req: _GenerationRequest) -> None:
    unsupported = []
    if req.n not in (None, 1):
        unsupported.append("n > 1")
    if req.stop not in (None, "", []):
        unsupported.append("stop")
    if req.presence_penalty not in (None, 0):
        unsupported.append("presence_penalty")
    if req.frequency_penalty not in (None, 0):
        unsupported.append("frequency_penalty")
    if req.logprobs not in (None, False, 0) or getattr(req, "top_logprobs", None) not in (None, 0):
        unsupported.append("logprobs")
    if unsupported:
        raise ProtocolError(f"not supported: {', '.join(unsupported)}")


def sampling_params(
    req: ChatCompletionRequest | CompletionRequest,
    num_prompt_tokens: int,
    context_len: int,
    stop_token_ids: frozenset[int],
) -> SamplingParams:
    """Engine parameters of a request whose prompt is ``num_prompt_tokens`` long.

    ``context_len`` bounds prompt + output: the smaller of the model's context
    and the KV pool, so an accepted request always fits the pool on its own.
    """
    check_supported(req)
    if num_prompt_tokens == 0:
        raise ProtocolError("empty prompt")
    room = context_len - num_prompt_tokens
    if room < 1:
        raise ProtocolError(f"prompt of {num_prompt_tokens} tokens leaves no room in a context of {context_len}")
    max_tokens = req.requested_max_tokens()
    if max_tokens is None:
        max_tokens = room
    elif max_tokens < 1:
        raise ProtocolError(f"max_tokens must be >= 1, got {max_tokens}")
    elif max_tokens > room:
        raise ProtocolError(
            f"prompt ({num_prompt_tokens} tokens) + max_tokens ({max_tokens}) exceeds the context of {context_len}"
        )
    try:
        return SamplingParams(
            max_new_tokens=max_tokens,
            stop_token_ids=frozenset() if req.ignore_eos else stop_token_ids,
            temperature=1.0 if req.temperature is None else req.temperature,
            top_p=1.0 if req.top_p is None else req.top_p,
            seed=None if req.seed is None else req.seed & 0xFFFFFFFF,  # OpenAI seeds are any integer
        )
    except ValueError as e:
        raise ProtocolError(str(e)) from None
