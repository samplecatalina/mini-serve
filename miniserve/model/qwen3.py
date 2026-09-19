"""Qwen3 dense decoder.

Plain PyTorch, with the operation order and precision of every step mirroring
Hugging Face transformers (``modeling_qwen3.py`` with the SDPA attention
backend). Attention is pluggable (``miniserve.model.attention``); with
``ContiguousAttention`` this is the reference path, whose logits are
bitwise-identical to HF. Faster backends are compared against it.

Tokens are flattened: activations are ``[num_tokens, hidden]`` with no batch
dimension, and ``positions`` gives each token's absolute position.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miniserve.model.attention import AttentionBackend, ContiguousAttention
from miniserve.model.transfer import to_device


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, cfg: dict) -> Qwen3Config:
        if cfg.get("rope_scaling"):
            raise NotImplementedError(f"rope_scaling={cfg['rope_scaling']!r} is not supported")
        if cfg.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act={cfg['hidden_act']!r} is not supported")
        return cls(
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            vocab_size=cfg["vocab_size"],
            num_layers=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=float(cfg["rope_theta"]),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


class ContiguousKVCache:
    """Per-layer K/V buffers of shape ``[max_len, num_kv_heads, head_dim]`` for one sequence."""

    def __init__(self, cfg: Qwen3Config, max_len: int, device: torch.device | str, dtype: torch.dtype):
        shape = (max_len, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.max_len = max_len
        self.length = 0  # tokens committed; advanced by the model after all layers ran

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store ``k``/``v`` for the new tokens; return K/V for the whole sequence so far."""
        start, end = self.length, self.length + k.shape[0]
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} > max_len={self.max_len}")
        self.k[layer][start:end] = k
        self.v[layer][start:end] = v
        return self.k[layer][:end], self.v[layer][:end]


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Normalize in fp32, cast back, then scale in the input dtype (same as HF).
    xf = x.to(torch.float32)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Qwen3ForCausalLM:
    def __init__(self, cfg: Qwen3Config, weights: dict[str, torch.Tensor]):
        self.cfg = cfg
        self.w = weights
        self.dtype = weights["model.embed_tokens.weight"].dtype
        self.device = weights["model.embed_tokens.weight"].device
        # Computed on the CPU in fp32 like the HF module constructor, then moved.
        hd = cfg.head_dim
        self.inv_freq = (
            1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).to(torch.float) / hd))
        ).to(self.device)

    def new_cache(self, max_len: int) -> ContiguousKVCache:
        return ContiguousKVCache(self.cfg, max_len, self.device, self.dtype)

    def _rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]  # [T, hd/2], fp32
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def _attention(
        self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn: AttentionBackend
    ) -> torch.Tensor:
        cfg, w, p = self.cfg, self.w, f"model.layers.{i}.self_attn."
        t = x.shape[0]
        q = F.linear(x, w[p + "q_proj.weight"]).view(t, cfg.num_heads, cfg.head_dim)
        k = F.linear(x, w[p + "k_proj.weight"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        v = F.linear(x, w[p + "v_proj.weight"]).view(t, cfg.num_kv_heads, cfg.head_dim)
        q = _rms_norm(q, w[p + "q_norm.weight"], cfg.rms_norm_eps)
        k = _rms_norm(k, w[p + "k_norm.weight"], cfg.rms_norm_eps)

        c, s = cos[:, None, :], sin[:, None, :]  # broadcast over heads
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s

        out = attn.attend(i, q, k, v)
        return F.linear(out.reshape(t, cfg.num_heads * cfg.head_dim), w[p + "o_proj.weight"])

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.w, f"model.layers.{i}.mlp."
        gate = F.silu(F.linear(x, w[p + "gate_proj.weight"]))
        return F.linear(gate * F.linear(x, w[p + "up_proj.weight"]), w[p + "down_proj.weight"])

    @property
    def attn_scale(self) -> float:
        return self.cfg.head_dim**-0.5

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: ContiguousKVCache,
        all_logits: bool = False,
    ) -> torch.Tensor:
        """Run ``input_ids`` (the next tokens of the sequence in ``cache``) on the reference path.

        Returns logits ``[vocab]`` for the last token, or ``[T, vocab]``.
        """
        n = input_ids.shape[0]
        h = self._hidden(input_ids, positions, ContiguousAttention([cache], [n], self.attn_scale), [n])
        h = _rms_norm(h if all_logits else h[-1:], self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        logits = F.linear(h, self.w["lm_head.weight"])
        return logits if all_logits else logits[0]

    @torch.inference_mode()
    def forward_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        caches: list[ContiguousKVCache],
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Several sequences in one pass on the reference path (one contiguous cache each).

        With a single sequence this runs exactly the same operations as :meth:`forward`.
        """
        return self.forward_with(input_ids, positions, ContiguousAttention(caches, seq_lens, self.attn_scale), seq_lens)

    @torch.inference_mode()
    def forward_with(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttentionBackend,
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Run several sequences in one pass with the given attention backend.

        ``input_ids`` and ``positions`` hold the new tokens of all sequences
        concatenated; sequence ``b`` contributes ``seq_lens[b]`` tokens.
        Returns logits ``[len(seq_lens), vocab]`` for the last token of each sequence.
        """
        h = self._hidden(input_ids, positions, attn, seq_lens)
        last = to_device([n - 1 for n in itertools.accumulate(seq_lens)], torch.long, h.device)
        h = _rms_norm(h[last], self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        return F.linear(h, self.w["lm_head.weight"])

    def decode_logits(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        """One new token per sequence: logits ``[T, vocab]``, the same operations as
        :meth:`forward_with` with every ``seq_lens`` entry 1, minus its last-token gather
        (an identity here) and the host-to-device copy that gather needs. Capturable in a
        CUDA Graph: no host synchronization."""
        h = self._hidden(input_ids, positions, attn, None)
        h = _rms_norm(h, self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        return F.linear(h, self.w["lm_head.weight"])

    def _hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttentionBackend,
        seq_lens: list[int] | None,
    ) -> torch.Tensor:
        """Decoder stack; returns final hidden states ``[num_tokens, hidden]`` before the last norm.
        ``seq_lens`` None: one token per sequence."""
        cfg, w = self.cfg, self.w
        if seq_lens is not None and sum(seq_lens) != input_ids.shape[0]:
            raise ValueError(f"seq_lens {seq_lens} do not match {input_ids.shape[0]} tokens")
        cos, sin = self._rope(positions)

        h = F.embedding(input_ids, w["model.embed_tokens.weight"])
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            x = _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_norm_eps)
            h = h + self._attention(i, x, cos, sin, attn)
            h = h + self._mlp(i, _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps))
        attn.finish()
        return h
