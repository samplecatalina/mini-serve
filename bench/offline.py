"""Engine-level benchmark: open-loop arrivals straight into ``Engine``, no HTTP.

Measures the engine alone (scheduling, KV cache, forward, sampling), without
tokenization, HTTP or server-sent events, so that an engine change can be
measured without the front end's noise. End-to-end serving numbers come from
a separate harness.

Timing. Requests arrive at scheduled times (Poisson at ``--arrival-rate``
requests/s, or all at t=0 with rate 0) and are submitted between steps once
their time has come. A token counts as produced when the step that sampled it
returns (the step waits for the device once, when it reads the sampled tokens
back). TTFT is first token minus scheduled arrival; ITL is the gap between
consecutive tokens of a request; output throughput is output tokens over the
span from the first arrival to the last token.

Protocol. One process, one engine, one model load. Warm-up runs the same kind
of workload (different seed) until the SM clock settles (at least
``--warmup-min-s``), then the arms alternate A B B A A B ... for ``--rounds``
rounds per arm, the prefix cache cleared before every run. Each run is one
CSV row; the sidecar JSON records configuration, environment, preflight and
GPU clocks, power and throttle reasons per run.

Usage (through make, which supplies the host power state):
    make bench-offline BENCH_ARGS="--workload shared --arrival-rate 0 --ablate radix --out m2_1_radix"
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import torch

from bench import sidecar
from miniserve.engine.cli import add_engine_args, engine_kwargs
from miniserve.engine.engine import Engine
from miniserve.engine.request import SamplingParams
from miniserve.engine.scheduler import Phase
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.model.weights import QWEN3_0_6B, load_config, load_weights, model_path

VOCAB_LIMIT = 150_000  # random prompt ids stay below the special-token range
RESULTS_DIR = "results/rtx4060-laptop"


# --------------------------------------------------------------------------- workloads


@dataclass
class Workload:
    prompts: list[list[int]]
    output_len: int
    arrivals: list[float]  # seconds after start, sorted


def make_workload(args, seed: int) -> Workload:
    """``shared``: groups of requests sharing a long random prefix, each with its own suffix, in
    random order. ``unique``: the same lengths with nothing shared."""
    rng = random.Random(seed)
    n = args.groups * args.per_group

    def rand(k):
        return [rng.randrange(VOCAB_LIMIT) for _ in range(k)]

    if args.workload == "shared":
        prefixes = [rand(args.prefix_len) for _ in range(args.groups)]
        prompts = [prefixes[g] + rand(args.suffix_len) for g in range(args.groups) for _ in range(args.per_group)]
        rng.shuffle(prompts)
    elif args.workload == "unique":
        prompts = [rand(args.prefix_len + args.suffix_len) for _ in range(n)]
    else:
        raise ValueError(args.workload)
    if args.arrival_rate > 0:
        t, arrivals = 0.0, []
        for _ in range(n):
            arrivals.append(t)
            t += rng.expovariate(args.arrival_rate)
    else:
        arrivals = [0.0] * n
    return Workload(prompts, args.output_len, arrivals)


# --------------------------------------------------------------------------- one run


@dataclass
class RunResult:
    wall_s: float
    ttft: list[float] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prefill_tokens_computed: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    decode_batch_sum: int = 0
    span_s: float = 0.0


def run(eng: Engine, w: Workload) -> RunResult:
    params = SamplingParams(w.output_len)  # no stop tokens: exactly output_len tokens each
    n = len(w.prompts)
    token_times: list[list[float]] = [[] for _ in range(n)]
    req_index: dict[int, int] = {}
    res = RunResult(0.0)
    t0 = time.perf_counter()
    i = 0
    while i < n or eng.has_unfinished:
        now = time.perf_counter() - t0
        while i < n and w.arrivals[i] <= now:
            req_index[eng.add_request(w.prompts[i], params).rid] = i
            i += 1
        if not eng.has_unfinished:
            time.sleep(max(0.0, w.arrivals[i] - (time.perf_counter() - t0)))
            continue
        torch.cuda.nvtx.range_push("step")
        batch = eng.step()
        torch.cuda.nvtx.range_pop()
        t = time.perf_counter() - t0
        if batch is None:
            continue
        if batch.phase is Phase.PREFILL:
            res.prefill_steps += 1
            # After the step each request holds one more token than the prefill ran.
            res.prefill_tokens_computed += sum(r.seq_len - 1 - r.num_cached_tokens for r in batch.requests)
        else:
            res.decode_steps += 1
            res.decode_batch_sum += len(batch.requests)
        for r in batch.requests:
            token_times[req_index[r.rid]].append(t)
    for k, times in enumerate(token_times):
        assert len(times) == w.output_len, f"request {k}: {len(times)} tokens"
        res.ttft.append(times[0] - w.arrivals[k])
        res.itl += [b - a for a, b in zip(times, times[1:])]
    res.output_tokens = n * w.output_len
    res.span_s = max(t[-1] for t in token_times) - min(w.arrivals)
    res.wall_s = time.perf_counter() - t0
    return res


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# --------------------------------------------------------------------------- arms

ABLATIONS = {"radix": ("radix_on", "radix_off"), "none": ("default",)}


def set_arm(eng: Engine, arm: str) -> None:
    kv = eng.runner.kv
    if arm == "radix_on":
        kv.set_radix(True)
    elif arm == "radix_off":
        kv.set_radix(False)
    elif arm == "default":
        kv.set_radix(kv.radix)  # clears the prefix cache
    else:
        raise ValueError(arm)


def abba(arms: tuple[str, ...], rounds: int) -> list[str]:
    """A B B A A B ... : each arm ``rounds`` times, neither always first."""
    if len(arms) == 1:
        return list(arms) * rounds
    a, b = arms
    order = []
    for k in range(rounds):
        order += [a, b] if k % 2 == 0 else [b, a]
    return order


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_engine_args(ap)
    g = ap.add_argument_group("benchmark")
    g.add_argument("--workload", choices=["shared", "unique"], required=True)
    g.add_argument("--groups", type=int, default=8)
    g.add_argument("--per-group", type=int, default=8)
    g.add_argument("--prefix-len", type=int, default=1024)
    g.add_argument("--suffix-len", type=int, default=64)
    g.add_argument("--output-len", type=int, default=64)
    g.add_argument("--arrival-rate", type=float, default=0.0, help="requests/s (Poisson); 0: all at once")
    g.add_argument("--workload-seed", type=int, default=0)
    g.add_argument("--ablate", choices=sorted(ABLATIONS), default="none")
    g.add_argument("--rounds", type=int, default=3, help="runs per arm")
    g.add_argument("--warmup-min-s", type=float, default=10.0)
    g.add_argument("--warmup-max-s", type=float, default=90.0)
    g.add_argument("--out", required=True, help=f"results name: {RESULTS_DIR}/<out>.csv and a sidecar per run id")
    g.add_argument("--allow-dirty", action="store_true", help="development only: results from uncommitted code")
    g.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    pre = sidecar.preflight()  # before loading anything onto the GPU
    env = sidecar.environment()
    if env["git_dirty"] and not args.allow_dirty:
        print("refusing to measure uncommitted code (tracked files modified)", file=sys.stderr)
        return 2

    path = model_path(QWEN3_0_6B, download=False)
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))
    eng = Engine(model, **engine_kwargs(args))
    arms = ABLATIONS[args.ablate]
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(args.results_dir, exist_ok=True)

    with sidecar.GpuSampler() as gpu:
        # Warm-up by time: same workload shape, another seed, until the SM clock settles.
        warm = make_workload(args, seed=args.workload_seed + 1_000_003)
        t_warm = time.monotonic()
        warm_runs = 0
        while True:
            set_arm(eng, arms[warm_runs % len(arms)])
            run(eng, warm)
            warm_runs += 1
            elapsed = time.monotonic() - t_warm
            if elapsed >= args.warmup_min_s and gpu.settled():
                break
            if elapsed >= args.warmup_max_s:
                print(f"SM clock not settled after {elapsed:.0f} s of warm-up", file=sys.stderr)
                return 3
        warmup = dict(seconds=round(elapsed, 1), runs=warm_runs, gpu=gpu.summary(since=t_warm))

        w = make_workload(args, seed=args.workload_seed)
        rows, per_run_gpu = [], []
        for k, arm in enumerate(abba(arms, args.rounds)):
            set_arm(eng, arm)
            eng.scheduler.stats = dict.fromkeys(eng.scheduler.stats, 0)
            preempt0 = eng.scheduler.num_preemptions
            t_start = gpu.sample_now()
            r = run(eng, w)
            gpu_run = gpu.summary(since=t_start, until=gpu.sample_now())
            per_run_gpu.append(dict(run=k, arm=arm, **gpu_run))
            st = eng.scheduler.stats
            rows.append(
                dict(
                    run_id=run_id,
                    run=k,
                    arm=arm,
                    workload=args.workload,
                    arrival_rate=args.arrival_rate,
                    requests=len(w.prompts),
                    prompt_len=args.prefix_len + args.suffix_len,
                    output_len=args.output_len,
                    kv_pool_tokens=eng.runner.allocator.num_blocks * eng.runner.allocator.block_size,
                    ttft_p50_ms=round(pct(r.ttft, 50) * 1e3, 2),
                    ttft_p95_ms=round(pct(r.ttft, 95) * 1e3, 2),
                    itl_p50_ms=round(pct(r.itl, 50) * 1e3, 2),
                    itl_p99_ms=round(pct(r.itl, 99) * 1e3, 2),
                    output_tok_s=round(r.output_tokens / r.span_s, 1),
                    span_s=round(r.span_s, 3),
                    prefill_tokens_computed=r.prefill_tokens_computed,
                    hit_rate=round(st["first_cached"] / st["first_tokens"], 4) if st["first_tokens"] else 0.0,
                    cached_tokens=st["first_cached"] + st["re_cached"],
                    preemptions=eng.scheduler.num_preemptions - preempt0,
                    prefill_steps=r.prefill_steps,
                    decode_steps=r.decode_steps,
                    mean_decode_batch=round(r.decode_batch_sum / max(1, r.decode_steps), 2),
                    sm_mhz_mean=gpu_run["sm_mhz"]["mean"],
                    git_commit=env["git_commit"][:12],
                )
            )
            print(" ".join(f"{k}={v}" for k, v in rows[-1].items() if k not in ("run_id", "git_commit")), flush=True)

    csv_path = f"{args.results_dir}/{args.out}.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    sidecar.write_sidecar(
        f"{args.results_dir}/{args.out}.{run_id}.sidecar.json",
        dict(
            run_id=run_id,
            config=dict(vars(args), engine=engine_kwargs(args), kv_profile=getattr(eng.runner, "kv_profile", None)),
            environment=env,
            preflight=pre,
            warmup=warmup,
            gpu_during_runs=per_run_gpu,
        ),
    )
    print_summary(rows, arms)
    return 0


def print_summary(rows: list[dict], arms: tuple[str, ...]) -> None:
    keys = ["ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p99_ms", "output_tok_s", "prefill_tokens_computed", "hit_rate", "preemptions"]
    med = {a: {k: statistics.median(r[k] for r in rows if r["arm"] == a) for k in keys} for a in arms}
    print("\nmedian per arm:")
    for a in arms:
        print(f"  {a}: " + ", ".join(f"{k}={med[a][k]}" for k in keys))
    if len(arms) == 2:
        a, b = arms
        print(f"  {a} vs {b}: " + ", ".join(
            f"{k} {100 * (med[a][k] / med[b][k] - 1):+.1f}%" for k in keys if med[b][k]
        ))


if __name__ == "__main__":
    sys.exit(main())
