"""Qwen3 dense decoder, reference path.

Plain PyTorch, with the operation order and precision of every step mirroring
Hugging Face transformers (``modeling_qwen3.py`` with the SDPA attention
backend). The goal is bitwise-identical logits, so that greedy decoding matches
the HF reference token for token. Faster kernels are compared against this path.

Tokens are flattened: activations are ``[num_tokens, hidden]`` with no batch
dimension, and ``positions`` gives each token's absolute position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
        self,
        i: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: ContiguousKVCache,
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

        k_all, v_all = cache.write(i, k, v)
        # SDPA expects [batch, heads, seq, head_dim].
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_all.transpose(0, 1).unsqueeze(0),
            v_all.transpose(0, 1).unsqueeze(0),
            is_causal=t > 1,
            scale=cfg.head_dim**-0.5,
            enable_gqa=True,
        )
        out = out.squeeze(0).transpose(0, 1).reshape(t, cfg.num_heads * cfg.head_dim)
        return F.linear(out, w[p + "o_proj.weight"])

    def _mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.w, f"model.layers.{i}.mlp."
        gate = F.silu(F.linear(x, w[p + "gate_proj.weight"]))
        return F.linear(gate * F.linear(x, w[p + "up_proj.weight"]), w[p + "down_proj.weight"])

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: ContiguousKVCache,
        all_logits: bool = False,
    ) -> torch.Tensor:
        """Run ``input_ids`` (the next tokens of the sequence in ``cache``).

        ``is_causal`` for multi-token calls assumes the cache was empty
        (a prefill); extending a non-empty cache by several tokens needs a
        proper offset mask and is not supported by this reference path.
        Returns logits ``[vocab]`` for the last token, or ``[T, vocab]``.
        """
        cfg, w = self.cfg, self.w
        t = input_ids.shape[0]
        if t > 1 and cache.length != 0:
            raise NotImplementedError("multi-token forward on a non-empty cache")
        cos, sin = self._rope(positions)

        h = F.embedding(input_ids, w["model.embed_tokens.weight"])
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            h = h + self._attention(i, _rms_norm(h, w[p + "input_layernorm.weight"], cfg.rms_norm_eps), cos, sin, cache)
            h = h + self._mlp(i, _rms_norm(h, w[p + "post_attention_layernorm.weight"], cfg.rms_norm_eps))
        cache.length += t

        h = _rms_norm(h if all_logits else h[-1:], w["model.norm.weight"], cfg.rms_norm_eps)
        logits = F.linear(h, w["lm_head.weight"])
        return logits if all_logits else logits[0]
