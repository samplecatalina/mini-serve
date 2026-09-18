import json

import pytest
import torch
from safetensors.torch import save_file

from miniserve.model.weights import expected_shapes, load_config, load_weights

TINY_CFG = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "vocab_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "num_hidden_layers": 2,
    "tie_word_embeddings": True,
}


def _write_ckpt(path, cfg, tensors):
    (path / "config.json").write_text(json.dumps(cfg))
    save_file(tensors, str(path / "model.safetensors"))


def test_tiny_checkpoint_roundtrip(tmp_path):
    tensors = {k: torch.randn(s) for k, s in expected_shapes(TINY_CFG).items()}
    _write_ckpt(tmp_path, TINY_CFG, tensors)
    w = load_weights(tmp_path, device="cpu")
    assert w["lm_head.weight"] is w["model.embed_tokens.weight"]
    assert all(t.dtype == torch.bfloat16 for t in w.values())
    torch.testing.assert_close(w["model.norm.weight"], tensors["model.norm.weight"].bfloat16())


def test_rejects_shape_mismatch(tmp_path):
    tensors = {k: torch.randn(s) for k, s in expected_shapes(TINY_CFG).items()}
    tensors["model.layers.1.self_attn.k_proj.weight"] = torch.randn(8, 8)
    _write_ckpt(tmp_path, TINY_CFG, tensors)
    with pytest.raises(ValueError, match="k_proj"):
        load_weights(tmp_path, device="cpu")


def test_rejects_missing_tensor(tmp_path):
    tensors = {k: torch.randn(s) for k, s in expected_shapes(TINY_CFG).items()}
    del tensors["model.layers.0.mlp.up_proj.weight"]
    _write_ckpt(tmp_path, TINY_CFG, tensors)
    with pytest.raises(ValueError, match="up_proj"):
        load_weights(tmp_path, device="cpu")


@pytest.fixture(scope="module")
def hf_model(qwen3_path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(qwen3_path, dtype=torch.bfloat16).cuda().eval()


@pytest.mark.gpu
@pytest.mark.slow
def test_loader_matches_hf_state_dict(qwen3_path, hf_model):
    ours = load_weights(qwen3_path)
    theirs = hf_model.state_dict()
    assert ours.keys() == theirs.keys()
    for k, t in theirs.items():
        assert torch.equal(ours[k], t), k
    assert load_config(qwen3_path)["tie_word_embeddings"] is True
    assert ours["lm_head.weight"] is ours["model.embed_tokens.weight"]


@pytest.mark.gpu
@pytest.mark.slow
def test_hf_forward_once(qwen3_path, hf_model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen3_path)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        logits = hf_model(ids).logits
    assert logits.shape == (1, ids.shape[1], hf_model.config.vocab_size)
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
