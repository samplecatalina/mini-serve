"""Greedy outputs of the anchor prompts on this machine, for comparison across devices.

Writes one JSON file: the device, the reference path's tokens and top-1 minus top-2 logit gap
at every step (the reference path is bitwise identical to Hugging Face transformers on a given
device), and the engine's tokens with each prompt alone and with all prompts at once (paged
attention, default settings: prefix cache, chunked prefill, CUDA Graphs, overlap).

Usage: ``python -m miniserve.tools.golden --out golden.json``; compare files with
``python -m miniserve.tools.compare_golden a.json b.json ...``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

STOP_IDS = frozenset({151645, 151643})


def load_prompts():
    """The anchor prompt set of the test suite (``tests/prompts.py``)."""
    path = Path(__file__).resolve().parents[2] / "tests" / "prompts.py"
    spec = importlib.util.spec_from_file_location("anchor_prompts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPTS, mod.encode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kv-pool-tokens", type=int, default=16384)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from miniserve.engine.engine import Engine
    from miniserve.engine.request import SamplingParams
    from miniserve.model.generate import greedy_generate
    from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from miniserve.model.weights import QWEN3_0_6B, load_config, load_weights, model_path

    path = model_path(QWEN3_0_6B, download=False)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))
    prompts, encode = load_prompts()
    out = dict(
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        model=f"{QWEN3_0_6B.repo_id}@{QWEN3_0_6B.revision}",
        prompts={},
    )
    ids = {name: encode(tokenizer, p) for name, (p, _) in prompts.items()}
    for name, (_, n) in prompts.items():
        gaps: list[float] = []
        ref = greedy_generate(model, ids[name], n, stop_ids=STOP_IDS, top2_gaps=gaps)
        out["prompts"][name] = dict(prompt_ids=ids[name], reference=ref, reference_gaps=gaps)
    eng = Engine(model, kv_pool_tokens=args.kv_pool_tokens)
    for name, (_, n) in prompts.items():
        [alone] = eng.generate([ids[name]], SamplingParams(n, STOP_IDS))
        out["prompts"][name]["engine_alone"] = alone
    eng.runner.kv.set_radix(eng.runner.kv.radix)  # the concurrent run starts from an empty prefix cache
    reqs = {name: eng.add_request(ids[name], SamplingParams(n, STOP_IDS)) for name, (_, n) in prompts.items()}
    while eng.has_unfinished:
        eng.step()
    for name, r in reqs.items():
        out["prompts"][name]["engine_together"] = r.output_ids
    Path(args.out).write_text(json.dumps(out))
    print(f"{out['device']}: {len(prompts)} prompts -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
