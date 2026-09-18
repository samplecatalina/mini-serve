"""Correctness anchor: greedy output must match Hugging Face transformers token for token.

Same checkpoint, BF16 on both sides, same seed. On a mismatch the failure
message reports the first diverging position, both tokens, and the HF top-1 vs
top-2 logit gap at that position (a tiny gap points to numerics rather than a
logic bug).
"""

from __future__ import annotations

import pytest
import torch

from miniserve.model.generate import greedy_generate
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.model.weights import load_config, load_weights

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

SEED = 0

_LONG_TEXT = (
    "The history of computing is often told as a sequence of machines, but it is equally a history of ideas "
    "about representation: how numbers, text, images and eventually programs themselves can be encoded as "
    "patterns of symbols and manipulated by rules. "
) * 12

PROMPTS = {
    "short_en": ("The capital of France is", 128),
    "long_en": (_LONG_TEXT + "\nSummarize the passage above in three sentences:", 128),
    "code": ("def quicksort(arr):\n    \"\"\"Sort a list of integers.\"\"\"\n", 128),
    "zh": ("请用三句话介绍一下长城的历史。", 128),
    "chat": ([{"role": "user", "content": "Explain what a KV cache is in one paragraph."}], 128),
    "long_gen": ("Write a short story about a lighthouse keeper who finds a message in a bottle.", 512),
}


@pytest.fixture(scope="module")
def tokenizer(qwen3_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen3_path)


@pytest.fixture(scope="module")
def hf_model(qwen3_path):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(SEED)
    return AutoModelForCausalLM.from_pretrained(qwen3_path, dtype=torch.bfloat16).cuda().eval()


@pytest.fixture(scope="module")
def stop_ids(hf_model) -> list[int]:
    eos = hf_model.generation_config.eos_token_id
    return eos if isinstance(eos, list) else [eos]


@pytest.fixture(scope="module")
def model(qwen3_path):
    torch.manual_seed(SEED)
    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(qwen3_path)), load_weights(qwen3_path))


def _encode(tokenizer, prompt) -> list[int]:
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False
        )
    return tokenizer(prompt).input_ids


@torch.inference_mode()
def _hf_greedy(hf_model, prompt_ids: list[int], n: int, stop_ids: list[int]) -> list[int]:
    from transformers import GenerationConfig

    ids = torch.tensor([prompt_ids], device="cuda")
    cfg = GenerationConfig(do_sample=False, max_new_tokens=n, eos_token_id=stop_ids, pad_token_id=stop_ids[0])
    out = hf_model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
    return out[0, len(prompt_ids) :].tolist()


@torch.inference_mode()
def _hf_top2_gap(hf_model, ids: list[int]) -> float:
    logits = hf_model(torch.tensor([ids], device="cuda")).logits[0, -1].float()
    top = torch.topk(logits, 2).values
    return float(top[0] - top[1])


@pytest.mark.parametrize("name", list(PROMPTS))
def test_greedy_matches_hf(name, tokenizer, hf_model, model, stop_ids):
    prompt, n = PROMPTS[name]
    prompt_ids = _encode(tokenizer, prompt)
    torch.manual_seed(SEED)
    ref = _hf_greedy(hf_model, prompt_ids, n, stop_ids)
    torch.manual_seed(SEED)
    ours = greedy_generate(model, prompt_ids, n, stop_ids=stop_ids)
    print(f"\n[{name}] prompt {len(prompt_ids)} tokens, generated {len(ours)} (hf {len(ref)})")

    if ours != ref:
        pos = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), min(len(ours), len(ref)))
        if pos == min(len(ours), len(ref)):
            pytest.fail(f"[{name}] same prefix but different lengths: ours={len(ours)} hf={len(ref)}")
        gap = _hf_top2_gap(hf_model, prompt_ids + ref[:pos])
        pytest.fail(
            f"[{name}] diverges at generated position {pos}/{n} (prompt {len(prompt_ids)} tokens): "
            f"ours={ours[pos]} {tokenizer.decode([ours[pos]])!r} vs "
            f"hf={ref[pos]} {tokenizer.decode([ref[pos]])!r}; HF top1-top2 logit gap={gap:.4f}"
        )


@torch.inference_mode()
def test_prefill_logits_match_hf(tokenizer, hf_model, model):
    """Teacher-forced: logits at every prompt position must be bitwise equal to HF.

    The reference path mirrors the HF op order exactly, so any difference is a
    bug. This is stricter than token equality on purpose: small errors (e.g. a
    wrong norm weight in one layer) can leave greedy tokens unchanged for a
    long time while still shifting the logits.
    """
    ids = _encode(tokenizer, PROMPTS["long_en"][0])
    ref = hf_model(torch.tensor([ids], device="cuda")).logits[0]
    t = torch.tensor(ids, device="cuda")
    ours = model.forward(t, torch.arange(len(ids), device="cuda"), model.new_cache(len(ids)), all_logits=True)
    diff = (ours.float() - ref.float()).abs().max().item()
    print(f"\nprefill logits over {len(ids)} positions: max|ours - hf| = {diff}")
    assert torch.equal(ours, ref), f"max|ours - hf| = {diff}"


@torch.inference_mode()
def test_incremental_decode_matches_full_forward(tokenizer, model):
    """KV cache correctness: logits from prefill + single-token decode steps agree
    with one forward pass over the whole sequence.

    Not bitwise: the two paths run GEMMs of different shapes, so BF16 results
    can differ by an ulp (HF shows the same behaviour). A broken cache produces
    differences orders of magnitude larger. Argmax must agree wherever the top-2
    gap exceeds a few BF16 ulps.
    """
    prompt_ids = _encode(tokenizer, PROMPTS["short_en"][0])
    steps = 32
    cache = model.new_cache(len(prompt_ids) + steps)
    ids = torch.tensor(prompt_ids, device="cuda")
    pos = torch.arange(len(prompt_ids), device="cuda")
    incr, gen = [], []
    for _ in range(steps):
        logits = model.forward(ids, pos, cache).float()
        incr.append(logits)
        gen.append(int(logits.argmax()))
        ids = torch.tensor([gen[-1]], device="cuda")
        pos = torch.tensor([cache.length], device="cuda")

    full_ids = prompt_ids + gen
    full = model.forward(
        torch.tensor(full_ids, device="cuda"),
        torch.arange(len(full_ids), device="cuda"),
        model.new_cache(len(full_ids)),
        all_logits=True,
    ).float()[len(prompt_ids) - 1 : -1]  # logits at position p predict token p+1
    incr = torch.stack(incr)

    max_diff = (incr - full).abs().max().item()
    print(f"\nincremental vs full forward over {steps} steps: max|d| = {max_diff}")
    assert max_diff < 1.0, max_diff
    top2 = torch.topk(full, 2, dim=-1).values
    decisive = (top2[:, 0] - top2[:, 1]) > 0.25
    assert torch.equal(incr.argmax(-1)[decisive], full.argmax(-1)[decisive])


def test_config_rejects_unsupported(qwen3_path):
    cfg = load_config(qwen3_path)
    with pytest.raises(NotImplementedError):
        Qwen3Config.from_dict({**cfg, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}})
