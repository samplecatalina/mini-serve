"""Acceptance rate of speculative decoding, on natural text and on random tokens.

The throughput benchmark (``bench/offline.py``) feeds random token ids, which
is neutral for everything whose cost depends only on how many tokens there
are. Acceptance is the one quantity that depends on what the tokens are: a
continuation of random ids degenerates into repetition, which a small draft
guesses more easily. So the acceptance rate that describes the draft is
measured here, on a fixed set of natural prompts (English, code, Chinese, a
chat message, a long generation), and on random prompts for comparison with
the throughput rows.

Every configuration is also run with the target proposing for itself. Its
acceptance would be exactly 1 if proposals and verification ran the same
kernels; they do not (one token per sequence against several), so exact BF16
ties break differently and a round ends there. That self-proposal rate is the
ceiling a draft can be read against, and it falls as gamma grows.

Greedy decoding, stop tokens honoured for natural prompts, none for random
ones. One CSV row per (draft, prompt set, gamma).

    python -m bench.spec_alpha --model 8B --draft 0.6B --gammas 2 4 6 --out m5_5_alpha
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import os
import random
import sys

import torch

from bench import sidecar
from miniserve.engine.request import SamplingParams
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.model.weights import load_config, load_weights, model_path, spec_for
from miniserve.spec.engine import SpecEngine

STOP_IDS = frozenset({151645, 151643})  # <|im_end|>, <|endoftext|>
RANDOM_VOCAB = 150_000  # below the special-token range


def load(size: str) -> tuple[Qwen3ForCausalLM, str]:
    path = model_path(spec_for(size), download=False)
    return Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path)), path


def text_prompts(tokenizer_path: str) -> tuple[list[list[int]], list[int]]:
    from transformers import AutoTokenizer

    from tests.prompts import PROMPTS, encode

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    ids = [encode(tok, p) for p, _ in PROMPTS.values()]
    return ids, [n for _, n in PROMPTS.values()]


def random_prompts(n: int, length: int, output_len: int, seed: int = 0) -> tuple[list[list[int]], list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, RANDOM_VOCAB) for _ in range(length)] for _ in range(n)], [output_len] * n


def measure(target, draft, prompts, output_lens, gammas, stop, kv_pool_tokens) -> list[dict]:
    eng = SpecEngine(target, draft, gamma=gammas[0], max_running=len(prompts), kv_pool_tokens=kv_pool_tokens)
    out = []
    for g in gammas:
        eng.gamma = g
        eng.spec_stats = dict.fromkeys(eng.spec_stats, 0)
        reqs = [eng.add_request(p, SamplingParams(n, stop_token_ids=stop)) for p, n in zip(prompts, output_lens)]
        while eng.has_unfinished:
            eng.step()
        s = eng.spec_stats
        out.append(
            dict(
                gamma=g,
                alpha=round(eng.acceptance, 4),
                tokens_per_round=round(eng.tokens_per_round, 3),
                rounds=s["rounds"],
                output_tokens=sum(len(r.output_ids) for r in reqs),
            )
        )
        print(f"  gamma={g} alpha={out[-1]['alpha']} tokens/round={out[-1]['tokens_per_round']}", flush=True)
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="8B", help="target model size")
    ap.add_argument("--draft", default="0.6B", help="draft model size")
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--random-prompts", type=int, default=8, help="number of random prompts (0: none)")
    ap.add_argument("--random-len", type=int, default=512)
    ap.add_argument("--random-output-len", type=int, default=128)
    ap.add_argument("--kv-pool-tokens", type=int, default=16384)
    ap.add_argument("--no-self", action="store_true", help="skip the target proposing for itself")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    args.results_dir = args.results_dir or sidecar.device_profile()[1].results_dir
    env = sidecar.environment()
    if env["git_dirty"] and not args.allow_dirty:
        print("refusing to measure uncommitted code (tracked files modified)", file=sys.stderr)
        return 2
    device = torch.cuda.get_device_name()
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    target, path = load(args.model)
    draft = load(args.draft)[0]
    sets = {"text": (*text_prompts(path), STOP_IDS)}
    if args.random_prompts:
        sets["random"] = (*random_prompts(args.random_prompts, args.random_len, args.random_output_len), frozenset())
    drafts = {args.draft: draft}
    if not args.no_self:
        drafts["same"] = target

    rows = []
    for dname, d in drafts.items():
        for sname, (prompts, lens, stop) in sets.items():
            print(f"draft {dname}, {sname} prompts ({len(prompts)})", flush=True)
            for r in measure(target, d, prompts, lens, args.gammas, stop, args.kv_pool_tokens):
                rows.append(
                    dict(
                        run_id=run_id,
                        device=device,
                        target_model=args.model,
                        draft_model=dname,
                        prompts=sname,
                        num_prompts=len(prompts),
                        **r,
                        git_commit=env["git_commit"][:12],
                    )
                )

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = f"{args.results_dir}/{args.out}.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {len(rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
