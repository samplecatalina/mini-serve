"""The blueprint as a second baseline: mini-sglang, on the same GPU and the same requests.

The project is a reproduction with two additions, so the number that says most
about it is not how it compares to a production server but how it compares to
the thing it was written from. That comparison only means something if the two
engines are given the identical work, so the requests are generated once, by
the same code that generates them for every other measurement here, written to
a file as token ids, and replayed on both sides. Neither engine tokenizes
anything, so a tokenizer difference cannot show up as an engine difference.

mini-sglang ships a Dockerfile on the pinned commit, from the same CUDA base
image as this project's, and an in-process offline entry point (``minisgl.llm.LLM``)
that takes token ids and returns token ids. So the comparison runs engine to
engine, with no HTTP, no load generator and no server front end in either path:
one process per side, one workload, the same preflight and the same
clock-settling warm-up rule from ``bench/sidecar.py``, which both sides import.

Three modes, because two of them run in different images:

    python -m bench.blueprint dump --out W.json --workload unique ...   # this image
    python -m bench.blueprint run  --workload W.json --out NAME         # the blueprint image
    python -m bench.blueprint table --out NAME                          # this image

``dump`` writes the requests, ``run`` replays them against mini-sglang and
writes rows in the same shape as ``bench/offline.py`` writes its own, and
``table`` puts the two arms side by side with the record of what was aligned
between them and what could not be.

What cannot be aligned is recorded rather than hidden: the two engines page the
KV cache differently (16 tokens per block here, 1 per page there), so the pool
is matched on tokens, not on blocks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import sys
import time

from bench import sidecar

# The blueprint, pinned. Every comparison and every module-by-module note in
# this project refers to this commit.
BLUEPRINT_COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"

# Settings that decide the result and that both engines expose. The value is
# what is asked of each side; the table records what each side reported back.
ALIGNED = ("kv_pool_tokens", "max_running", "cuda_graph_max_bs", "max_prefill_tokens",
           "prefix_cache", "attention_backend", "dtype", "model", "sampling")

# Differences that exist and cannot be removed by a flag.
UNALIGNABLE = {
    "kv_paging": "16 tokens per block here, 1 token per page in the blueprint: "
                 "the pool is matched on tokens, not on blocks",
    "scheduler_process_model": "one process with an engine thread here, "
                               "separate scheduler/tokenizer/detokenizer processes over ZMQ there "
                               "(bypassed in both by using each engine's in-process offline path)",
}


# --------------------------------------------------------------------------- dump


def dump(argv: list[str]) -> int:
    from bench.offline import make_workload
    from miniserve.model.weights import model_path, spec_for

    ap = argparse.ArgumentParser(prog="bench.blueprint dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--workload", choices=["shared", "unique", "mixed", "policy"], required=True)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--per-group", type=int, default=16)
    ap.add_argument("--prefix-len", type=int, default=96)
    ap.add_argument("--suffix-len", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=96)
    ap.add_argument("--num-long", type=int, default=8)
    ap.add_argument("--long-len", type=int, default=3072)
    ap.add_argument("--long-output-len", type=int, default=16)
    ap.add_argument("--short-len", type=int, default=256)
    ap.add_argument("--long-fraction", type=float, default=0.25)
    ap.add_argument("--short-output-len", type=int, default=32)
    ap.add_argument("--arrival-rate", type=float, default=0.0)
    ap.add_argument("--workload-seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.arrival_rate:
        raise SystemExit("the blueprint's offline entry point takes a whole batch: "
                         "an arrival process cannot be replayed through it")

    # The same seeds bench/offline.py uses, so the measured workload and the
    # warm-up workload are the ones it would have run.
    w = make_workload(a, seed=a.workload_seed)
    warm = make_workload(a, seed=a.workload_seed + 1_000_003)
    rec = {
        "caliber": vars(a),
        "model_path": str(model_path(spec_for(a.model), download=False)),
        "prompts": w.prompts, "output_lens": w.output_lens,
        "warm_prompts": warm.prompts, "warm_output_lens": warm.output_lens,
    }
    with open(a.out, "w") as f:
        json.dump(rec, f)
    print(f"{len(w.prompts)} requests, {sum(map(len, w.prompts))} prompt tokens, "
          f"{sum(w.output_lens)} output tokens -> {a.out}")
    return 0


# --------------------------------------------------------------------------- run


def run(argv: list[str]) -> int:
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    ap = argparse.ArgumentParser(prog="bench.blueprint run")
    ap.add_argument("--workload", required=True, help="the file written by `dump`")
    ap.add_argument("--out", required=True, help="results/<device>/<out>_blueprint.csv")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kv-pool-tokens", type=int, default=32768)
    ap.add_argument("--max-running", type=int, default=256)
    ap.add_argument("--cuda-graph-max-bs", type=int, default=256)
    ap.add_argument("--max-prefill-tokens", type=int, default=8192)
    ap.add_argument("--attention-backend", default="fi")
    ap.add_argument("--warmup-min-s", type=float, default=10.0)
    ap.add_argument("--warmup-max-s", type=float, default=90.0)
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)

    pre = sidecar.preflight()  # before anything reaches the GPU
    results_dir = a.results_dir or sidecar.device_profile()[1].results_dir
    with open(a.workload) as f:
        wl = json.load(f)

    class Timed(LLM):
        """The blueprint's offline loop, with a timestamp on every token it hands back.

        ``offline_send_result`` is the one place every produced token passes
        through, which is the same place this project's own harness reads them:
        both sides time the token, not the request.
        """

        def start(self, n: int) -> None:
            self.token_times = [[] for _ in range(n)]
            self.t0 = time.perf_counter()

        def offline_send_result(self, reply) -> None:
            t = time.perf_counter() - self.t0
            for msg in reply:
                self.token_times[msg.uid].append(t)
            super().offline_send_result(reply)

    # The blueprint's engine asserts that nothing has touched CUDA yet, so the
    # software description -- which imports flashinfer, and so initialises it --
    # is collected after the engine exists rather than before. The preflight
    # above reads nvidia-smi only, and still runs first.
    llm = Timed(
        wl["model_path"],
        dtype=torch.bfloat16,
        max_running_req=a.max_running,
        cuda_graph_max_bs=a.cuda_graph_max_bs,
        max_extend_tokens=a.max_prefill_tokens,
        attention_backend=a.attention_backend,
        cache_type="radix",
        page_size=1,
        num_page_override=a.kv_pool_tokens,
    )

    def once(prompts, output_lens) -> dict:
        llm.start(len(prompts))
        params = [SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=n) for n in output_lens]
        llm.generate(prompts, params)
        times = llm.token_times
        for k, ts in enumerate(times):
            if len(ts) != output_lens[k]:
                raise SystemExit(f"request {k}: {len(ts)} tokens, asked for {output_lens[k]}")
        ttft = [ts[0] for ts in times]
        e2e = [ts[-1] for ts in times]
        itl = [b - a_ for ts in times for a_, b in zip(ts, ts[1:])]
        span = max(e2e)
        return dict(
            ttft_p50_ms=round(pct(ttft, 50) * 1e3, 2), ttft_p95_ms=round(pct(ttft, 95) * 1e3, 2),
            e2e_mean_ms=round(statistics.mean(e2e) * 1e3, 1), e2e_p99_ms=round(pct(e2e, 99) * 1e3, 1),
            itl_p50_ms=round(pct(itl, 50) * 1e3, 2), itl_p99_ms=round(pct(itl, 99) * 1e3, 2),
            itl_max_ms=round(max(itl) * 1e3, 2),
            output_tok_s=round(sum(output_lens) / span, 1), span_s=round(span, 3),
        )

    env = sidecar.environment()
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    with sidecar.GpuSampler() as gpu:
        t_warm = time.monotonic()
        warm_runs = 0
        while True:
            once(wl["warm_prompts"], wl["warm_output_lens"])
            warm_runs += 1
            elapsed = time.monotonic() - t_warm
            if elapsed >= a.warmup_min_s and gpu.settled():
                break
            if elapsed >= a.warmup_max_s:
                print(f"SM clock not settled after {elapsed:.0f} s of warm-up", file=sys.stderr)
                return 3
        warmup = dict(seconds=round(elapsed, 1), runs=warm_runs, gpu=gpu.summary(since=t_warm))

        rows, per_run_gpu = [], []
        for k in range(a.rounds):
            t_start = gpu.sample_now()
            r = once(wl["prompts"], wl["output_lens"])
            gpu_run = gpu.summary(since=t_start, until=gpu.sample_now())
            per_run_gpu.append(dict(run=k, arm="minisgl", **gpu_run))
            rows.append(dict(
                run_id=run_id, run=k, arm="minisgl",
                workload=wl["caliber"]["workload"], requests=len(wl["prompts"]),
                prompt_tokens=sum(map(len, wl["prompts"])), output_tokens=sum(wl["output_lens"]),
                kv_pool_tokens=a.kv_pool_tokens, **r,
                blueprint_commit=BLUEPRINT_COMMIT[:12], git_commit=env["git_commit"][:12],
                sm_mhz_mean=gpu_run["sm_mhz"]["mean"],
            ))
            print(" ".join(f"{key}={val}" for key, val in rows[-1].items()), flush=True)

    os.makedirs(results_dir, exist_ok=True)
    csv_path = f"{results_dir}/{a.out}_blueprint.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    sidecar.write_sidecar(f"{results_dir}/{a.out}_blueprint.{run_id}.sidecar.json", dict(
        run_id=run_id,
        config=dict(vars(a), blueprint_commit=BLUEPRINT_COMMIT, caliber=wl["caliber"],
                    engine=engine_readback(llm, a)),
        environment=dict(env, minisgl=BLUEPRINT_COMMIT[:12]),
        preflight=pre, warmup=warmup, gpu_during_runs=per_run_gpu,
    ))
    print(f"{csv_path}")
    return 0


def engine_readback(llm, a) -> dict:
    """What the blueprint actually ended up running, read off the engine objects.

    Several of these are decided inside the engine, and one of them is decided
    against the free memory it finds: the blueprint adjusts its own config
    before building anything, so the flags are an intention and only the
    objects say what happened. A comparison that quotes the flags is quoting
    the intention.
    """
    eng = llm.engine
    cache = llm.cache_manager
    return {
        "kv_pool_tokens": eng.num_pages * cache.page_size,
        "num_pages": eng.num_pages,
        "page_size": cache.page_size,
        "max_running": llm.table_manager.page_table.shape[0] - 1,  # one row is the dummy request
        "max_prefill_tokens": llm.prefill_budget,
        "cuda_graph_bs": eng.graph_runner.graph_bs_list,
        "max_seq_len": eng.max_seq_len,
        "attention_backend": type(eng.attn_backend).__name__,
        "prefix_cache": type(cache.prefix_cache).__name__,
        "dtype": str(eng.dtype),
        "model_path": a.__dict__.get("model_path") or llm.tokenizer.name_or_path,
        "asked": {k: v for k, v in vars(a).items() if k != "results_dir"},
    }


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# --------------------------------------------------------------------------- table


def table(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="bench.blueprint table")
    ap.add_argument("--out", required=True, help="the name both arms were written under")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    results_dir = a.results_dir or sidecar.device_profile()[1].results_dir

    def med(path: str, key: str) -> float:
        with open(path, newline="") as f:
            vals = [float(r[key]) for r in csv.DictReader(f) if r[key] not in ("", None)]
        if not vals:
            raise SystemExit(f"{path}: no {key}")
        return statistics.median(vals)

    ours = f"{results_dir}/{a.out}.csv"
    theirs = f"{results_dir}/{a.out}_blueprint.csv"
    keys = ("output_tok_s", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "itl_p99_ms", "e2e_mean_ms")
    rows = []
    for key in keys:
        mine, other = med(ours, key), med(theirs, key)
        rows.append(dict(metric=key, miniserve=round(mine, 2), minisgl=round(other, 2),
                         ratio=round(mine / other, 4) if other else ""))
    out = f"{results_dir}/{a.out}_compare.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(f"{r['metric']:<16} {r['miniserve']:>10} {r['minisgl']:>10}  {r['ratio']}")
    print(out)
    return 0


MODES = {"dump": dump, "run": run, "table": table}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"usage: python -m bench.blueprint {{{'|'.join(MODES)}}} ...", file=sys.stderr)
        return 2
    return MODES[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
