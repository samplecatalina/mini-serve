"""Qwen3 forward on fused FlashInfer operators: the engine's optional fast path.

The reference path (``qwen3.py``) runs every step as separate PyTorch
operators so that its logits are bitwise-identical to Hugging Face
transformers. That costs about 49 kernels per layer, most of them tiny
elementwise ones whose time is launch latency, not arithmetic. This path runs
the same model on the operators mini-sglang and sglang use:

- RMSNorm, and residual add + RMSNorm in one kernel (``fused_add_rmsnorm``);
- q/k norm in place on views of the fused QKV output;
- rotary embedding in place from a cos/sin table (``apply_rope_with_cos_sin_cache_inplace``);
- SiLU(gate) * up in one kernel;
- Q/K/V and gate/up each as one GEMM over concatenated weights.

About 13 kernels per layer. Each fused operator keeps its intermediates in
fp32 and rounds once, where the reference rounds to BF16 between steps, so the
two paths differ by an ulp or two; this path is held to the tolerance rules of
any non-reference path (``tests/anchor.py``), and the reference stays the anchor.

The fused model shares the reference model's weights: the concatenated
projections replace the separate ones, which become row slices (views) of
them. A row slice of a row-major matrix is contiguous, and a GEMM over it
gives bitwise the same result as over the original tensor, so the reference
path is unchanged and nothing is held twice.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from miniserve.model.attention import AttentionBackend, ContiguousAttention
from miniserve.model.qwen3 import ContiguousKVCache, Qwen3Config, Qwen3ForCausalLM
from miniserve.model.transfer import to_device


# Per layer, in the order they are laid out in the layer's buffer; the two groups of
# projections are concatenated along their output rows.
_GROUPS = (("self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")), ("mlp.gate_up_proj", ("gate_proj", "up_proj")))
_OTHERS = (
    "self_attn.o_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
_ALIGN = 128  # elements between tensors in a layer's buffer (256 bytes in BF16)


def fuse_projections(weights: dict[str, torch.Tensor], cfg: Qwen3Config) -> None:
    """Concatenate each layer's Q/K/V and gate/up projection weights, in place.

    Adds ``self_attn.qkv_proj.weight`` and ``mlp.gate_up_proj.weight`` per layer and
    points the separate entries at row slices of them. Every weight of the layer moves
    into one new buffer, so the allocator segments that held the originals empty out
    completely and go back to the device: freeing only the projections would leave
    holes between the tensors still living next to them, which the caching allocator
    cannot return. One layer at a time: the peak is about one layer above the steady
    state. Idempotent."""
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        if p + "self_attn.qkv_proj.weight" in weights:
            continue
        parts = [[p + f"{g.split('.')[0]}.{n}.weight" for n in names] for g, names in _GROUPS]
        src = [weights[k] for group in parts for k in group] + [weights[p + k] for k in _OTHERS]
        # The members of a group sit back to back (they form one matrix); everything
        # else starts on an aligned offset.
        last_of_group = {sum(len(g) for g in parts[: j + 1]) - 1 for j in range(len(parts))}
        grouped = sum(len(g) for g in parts)
        offsets, total = [], 0
        for j, t in enumerate(src):
            offsets.append(total)
            total += t.numel()
            if j >= grouped or j in last_of_group:
                total = -(-total // _ALIGN) * _ALIGN
        buf = torch.empty(total, dtype=src[0].dtype, device=src[0].device)
        views = [buf[o : o + t.numel()].view(t.shape).copy_(t) for o, t in zip(offsets, src)]
        k = 0
        for (group, _), keys in zip(_GROUPS, parts):
            rows = sum(views[k + j].shape[0] for j in range(len(keys)))
            first = views[k]
            weights[p + group + ".weight"] = buf[offsets[k] : offsets[k] + rows * first.shape[1]].view(rows, first.shape[1])
            for key in keys:
                weights[key] = views[k]
                k += 1
        for key in _OTHERS:
            weights[p + key] = views[k]
            k += 1
        del src, views
        if i % 4 == 3:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()


class FusedQwen3ForCausalLM(Qwen3ForCausalLM):
    """The same model and interface as :class:`Qwen3ForCausalLM`, on fused operators.

    Built from a reference model, whose weights it shares (see the module docstring).
    """

    def __init__(self, base: Qwen3ForCausalLM):
        import flashinfer

        fuse_projections(base.w, base.cfg)
        self.cfg = base.cfg
        self.w = base.w
        self.dtype = base.dtype
        self.device = base.device
        self.inv_freq = base.inv_freq
        self.reference = base  # the same weights on the reference operators
        self._fi = flashinfer

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: ContiguousKVCache,
        all_logits: bool = False,
    ) -> torch.Tensor:
        n = input_ids.shape[0]
        x, residual = self._layers(input_ids, positions, ContiguousAttention([cache], [n], self.attn_scale))
        if not all_logits:
            x, residual = x[-1:], residual[-1:]
        logits = self._logits(x, residual)
        return logits if all_logits else logits[0]

    @torch.inference_mode()
    def forward_with(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn: AttentionBackend,
        seq_lens: list[int],
    ) -> torch.Tensor:
        if sum(seq_lens) != input_ids.shape[0]:
            raise ValueError(f"seq_lens {seq_lens} do not match {input_ids.shape[0]} tokens")
        x, residual = self._layers(input_ids, positions, attn)
        last = to_device([n - 1 for n in itertools.accumulate(seq_lens)], torch.long, x.device)
        return self._logits(x[last], residual[last])

    def decode_logits(self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend) -> torch.Tensor:
        """Every row's logits, no host synchronization (capturable in a CUDA Graph)."""
        return self._logits(*self._layers(input_ids, positions, attn))

    def _logits(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """The last layer's residual add, the final norm and the LM head, for the rows given.
        ``x`` and ``residual`` must be tensors of their own (they are overwritten)."""
        self._fi.fused_add_rmsnorm(x, residual, self.w["model.norm.weight"], self.cfg.rms_norm_eps)
        return F.linear(x, self.w["lm_head.weight"])

    def _layers(
        self, input_ids: torch.Tensor, positions: torch.Tensor, attn: AttentionBackend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder stack. Returns the last MLP output and the residual stream before it is
        added: the final hidden state is their sum, which :meth:`_logits` forms fused with
        the final norm."""
        cfg, w, fi = self.cfg, self.w, self._fi
        eps = cfg.rms_norm_eps
        t = input_ids.shape[0]
        # The same fp32 angles as the reference's ``_rope``, as a table of one row per token.
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        rows = torch.arange(t, device=input_ids.device)

        residual = F.embedding(input_ids, w["model.embed_tokens.weight"])
        x = fi.rmsnorm(residual, w["model.layers.0.input_layernorm.weight"], eps)
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            if i > 0:
                fi.fused_add_rmsnorm(x, residual, w[p + "input_layernorm.weight"], eps)
            x = self._attention_fused(i, x, cos_sin, rows, attn)
            fi.fused_add_rmsnorm(x, residual, w[p + "post_attention_layernorm.weight"], eps)
            x = F.linear(fi.silu_and_mul(F.linear(x, w[p + "mlp.gate_up_proj.weight"])), w[p + "mlp.down_proj.weight"])
        attn.finish()
        return x, residual

    def _attention_fused(
        self, i: int, x: torch.Tensor, cos_sin: torch.Tensor, rows: torch.Tensor, attn: AttentionBackend
    ) -> torch.Tensor:
        cfg, w, fi, p = self.cfg, self.w, self._fi, f"model.layers.{i}.self_attn."
        t, d = x.shape[0], cfg.head_dim
        qkv = F.linear(x, w[p + "qkv_proj.weight"])
        q, k, v = qkv.split([cfg.num_heads * d, cfg.num_kv_heads * d, cfg.num_kv_heads * d], dim=-1)
        q3, k3 = q.view(t, cfg.num_heads, d), k.view(t, cfg.num_kv_heads, d)
        fi.rmsnorm(q3, w[p + "q_norm.weight"], cfg.rms_norm_eps, out=q3)
        fi.rmsnorm(k3, w[p + "k_norm.weight"], cfg.rms_norm_eps, out=k3)
        fi.apply_rope_with_cos_sin_cache_inplace(rows, q, k, d, cos_sin, is_neox=True)
        out = attn.attend(i, q3, k3, v.view(t, cfg.num_kv_heads, d))
        return F.linear(out.reshape(t, cfg.num_heads * d), w[p + "o_proj.weight"])


def with_fused_ops(model: Qwen3ForCausalLM, on: bool) -> Qwen3ForCausalLM:
    """``model`` on fused operators if ``on`` (sharing its weights), else ``model`` itself."""
    if not on or isinstance(model, FusedQwen3ForCausalLM):
        return model
    return FusedQwen3ForCausalLM(model)

