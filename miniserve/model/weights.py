"""Checkpoint location and safetensors loading for Qwen3 models.

Weights are loaded straight into device tensors of the requested dtype and
validated against the shapes implied by ``config.json``, so a wrong or partial
checkpoint fails at load time rather than as a shape error deep in a forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str  # pinned commit so that reference outputs stay reproducible


QWEN3_0_6B = ModelSpec("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")

_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt"]


def model_path(spec: ModelSpec = QWEN3_0_6B, download: bool = False) -> Path:
    """Local snapshot directory of ``spec``; downloads it only if ``download``."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            spec.repo_id,
            revision=spec.revision,
            allow_patterns=_ALLOW_PATTERNS,
            local_files_only=not download,
        )
    )


def load_config(path: Path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())


def expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Parameter name -> shape for a Qwen3 dense decoder, derived from its config."""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or h // n_heads
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, h),
        "model.norm.weight": (h,),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        shapes.update(
            {
                p + "input_layernorm.weight": (h,),
                p + "post_attention_layernorm.weight": (h,),
                p + "self_attn.q_proj.weight": (n_heads * hd, h),
                p + "self_attn.k_proj.weight": (n_kv * hd, h),
                p + "self_attn.v_proj.weight": (n_kv * hd, h),
                p + "self_attn.o_proj.weight": (h, n_heads * hd),
                p + "self_attn.q_norm.weight": (hd,),
                p + "self_attn.k_norm.weight": (hd,),
                p + "mlp.gate_proj.weight": (inter, h),
                p + "mlp.up_proj.weight": (inter, h),
                p + "mlp.down_proj.weight": (h, inter),
            }
        )
    if not cfg.get("tie_word_embeddings", False):
        shapes["lm_head.weight"] = (vocab, h)
    return shapes


def load_weights(
    path: Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Load every tensor of the checkpoint at ``path`` and validate it.

    With tied embeddings, ``lm_head.weight`` is returned as the same tensor
    object as ``model.embed_tokens.weight`` (no copy).
    """
    path = Path(path)
    cfg = load_config(path)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")

    weights: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device=str(device)) as sf:
            for name in sf.keys():
                weights[name] = sf.get_tensor(name).to(dtype)

    tied = cfg.get("tie_word_embeddings", False)
    if tied:
        # Some tied checkpoints still ship a duplicate lm_head; the embedding wins.
        weights.pop("lm_head.weight", None)

    want = expected_shapes(cfg)
    missing = sorted(want.keys() - weights.keys())
    unexpected = sorted(weights.keys() - want.keys())
    bad = [
        f"{k}: {tuple(weights[k].shape)} != {want[k]}"
        for k in sorted(want.keys() & weights.keys())
        if tuple(weights[k].shape) != want[k]
    ]
    if missing or unexpected or bad:
        raise ValueError(
            f"checkpoint {path} does not match its config: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} shape_mismatch={bad[:5]}"
        )

    if tied:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights
