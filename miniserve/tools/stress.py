"""Offline stress run: submit many requests of random length at once and run them to completion.

Checks that the engine neither runs out of memory nor stalls under KV pool
pressure, and reports how the pool was sized and how often requests were
preempted. It measures no latency or throughput.

Usage: ``python -m miniserve.tools.stress --num-requests 64 --kv-pool-tokens 2048``
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from miniserve.engine.cli import add_engine_args, engine_kwargs
from miniserve.engine.engine import Engine
from miniserve.engine.request import RequestState, SamplingParams
from miniserve.engine.scheduler import Phase
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.model.weights import QWEN3_0_6B, load_config, load_weights, model_path

STOP_IDS = frozenset({151645, 151643})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_engine_args(ap)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, nargs=2, default=[4, 1024], metavar=("MIN", "MAX"))
    ap.add_argument("--max-new-tokens", type=int, nargs=2, default=[1, 256], metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1_000_000, help="fail if not done after this many steps")
    args = ap.parse_args()

    path = model_path(QWEN3_0_6B, download=False)
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))
    eng = Engine(model, **engine_kwargs(args))

    rng = random.Random(args.seed)
    reqs = []
    for _ in range(args.num_requests):
        # Random ids below the special-token range; prompts are not meant to be meaningful.
        prompt = [rng.randrange(150_000) for _ in range(rng.randint(*args.prompt_len))]
        reqs.append(eng.add_request(prompt, SamplingParams(rng.randint(*args.max_new_tokens), STOP_IDS)))

    steps = prefill_steps = max_decode_batch = max_used = 0
    alloc = eng.runner.allocator
    while eng.has_unfinished:
        if steps >= args.max_steps:
            print(f"not finished after {steps} steps", file=sys.stderr)
            return 1
        batch = eng.step()
        steps += 1
        if batch.phase is Phase.PREFILL:
            prefill_steps += 1
        else:
            max_decode_batch = max(max_decode_batch, len(batch.requests))
        if alloc is not None:
            max_used = max(max_used, alloc.num_used)

    report = dict(
        engine=engine_kwargs(args),
        kv_profile=getattr(eng.runner, "kv_profile", None),
        num_requests=len(reqs),
        prompt_tokens=sum(len(r.prompt_ids) for r in reqs),
        output_tokens=sum(len(r.output_ids) for r in reqs),
        steps=steps,
        prefill_steps=prefill_steps,
        max_decode_batch=max_decode_batch,
        max_blocks_used=max_used,
        preemptions=eng.scheduler.num_preemptions,
        preempted_requests=sum(r.num_preemptions > 0 for r in reqs),
        all_finished=all(r.state is RequestState.FINISHED for r in reqs),
        blocks_returned=alloc is None or alloc.num_free == alloc.num_blocks,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_finished"] and report["blocks_returned"] else 1


if __name__ == "__main__":
    sys.exit(main())
