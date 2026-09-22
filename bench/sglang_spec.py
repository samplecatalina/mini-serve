"""Speculative decoding in the reference server (sglang), on the same requests as bench/offline.py.

The question this answers is how far a two-model draft (a small model of the
same family proposing, the target verifying) is from a draft head trained for
the target (EAGLE-3), and how much of any gap is the algorithm rather than the
engine. So the comparison has three sglang arms beside this project's own:

    plain          no speculation
    standalone     sglang's two-model speculation: Qwen3-0.6B proposes a chain
    eagle3         a public EAGLE-3 head for Qwen3-8B, the same chain length
    eagle3_auto    the same head with the draft shape sglang picks for itself

``standalone`` against this project's own speculative engine is engine against
engine on one algorithm; ``eagle3`` against ``standalone`` is algorithm against
algorithm inside one engine.

As in bench/blueprint.py, the requests are generated once by this project's
workload code and written as token ids; sglang's offline engine takes the ids
and returns ids, so no tokenizer and no HTTP front end is in either path. One
engine per (arm, running-request cap): the cap and the captured graph sizes are
engine settings.

    python -m bench.sglang_spec dump --out W.json --max-running N ...   # this image
    python -m bench.sglang_spec run --workload W.json --arm eagle3 ...  # the sglang image
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

from bench import sidecar

# Public EAGLE-3 head for Qwen3-8B, pinned (the one sglang's documentation uses for Qwen3).
EAGLE3_HEAD = ("Tengyunw/qwen3_8b_eagle3", "2a1059d51f622b8cad7d7d72840153ffea5488a0")

ARMS = ("plain", "standalone", "eagle3", "eagle3_auto")


def dump(argv: list[str]) -> int:
    from bench.offline import make_workload
    from miniserve.model.weights import model_path, spec_for

    ap = argparse.ArgumentParser(prog="bench.sglang_spec dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--draft", default="0.6B")
    ap.add_argument("--max-running", type=int, required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, default=480)
    ap.add_argument("--suffix-len", type=int, default=32)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--workload-seed", type=int, default=0)
    a = ap.parse_args(argv)
    # bench/offline.py's `unique` workload with --groups <requests> --per-group 1: the caliber of
    # its speculative-decoding runs.
    cal = argparse.Namespace(
        workload="unique", groups=a.requests, per_group=1, prefix_len=a.prefix_len, suffix_len=a.suffix_len,
        output_len=a.output_len, num_long=0, long_len=0, long_output_len=0, short_len=0, long_fraction=0.0,
        short_output_len=0, arrival_rate=0.0,
    )
    w = make_workload(cal, seed=a.workload_seed)
    warm = make_workload(cal, seed=a.workload_seed + 1_000_003)
    # The natural prompts bench/spec_alpha.py measures acceptance on, as ids: acceptance is the
    # one quantity that depends on what the tokens are, and a random continuation is not what a
    # draft head was trained on.
    from bench.spec_alpha import STOP_IDS, text_prompts

    text_ids, text_lens = text_prompts(str(model_path(spec_for(a.model), download=False)))
    rec = {
        "caliber": dict(vars(a), workload="unique"),
        "model_path": str(model_path(spec_for(a.model), download=False)),
        "draft_path": str(model_path(spec_for(a.draft), download=False)),
        "prompts": w.prompts, "output_lens": w.output_lens,
        "warm_prompts": warm.prompts, "warm_output_lens": warm.output_lens,
        "text_prompts": text_ids, "text_output_lens": text_lens, "stop_ids": sorted(STOP_IDS),
    }
    with open(a.out, "w") as f:
        json.dump(rec, f)
    print(f"{len(w.prompts)} requests, {sum(map(len, w.prompts))} prompt tokens -> {a.out}")
    return 0


def eagle3_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(EAGLE3_HEAD[0], revision=EAGLE3_HEAD[1], local_files_only=True)


def engine_args(arm: str, wl: dict, a) -> dict:
    kw = dict(
        model_path=wl["model_path"], dtype="bfloat16", attention_backend="flashinfer",
        max_running_requests=a.max_running, cuda_graph_max_bs=a.max_running,
        max_total_tokens=a.kv_pool_tokens, chunked_prefill_size=a.chunked_prefill_size,
        max_prefill_tokens=a.max_prefill_tokens, skip_tokenizer_init=True, log_level="warning",
        # Prefill in piecewise CUDA graphs is on by default in 0.5.10 and, on this caliber, fails
        # at a cap of 2 (FlashInfer's prefill plan gets a kv_indptr one entry too long). This
        # engine does not capture prefills either, so it is off on every arm.
        disable_piecewise_cuda_graph=True,
    )
    chain = dict(speculative_num_steps=a.gamma, speculative_eagle_topk=1, speculative_num_draft_tokens=a.gamma + 1)
    if arm == "standalone":
        kw.update(speculative_algorithm="STANDALONE", speculative_draft_model_path=wl["draft_path"], **chain)
    elif arm == "eagle3":
        kw.update(speculative_algorithm="EAGLE3", speculative_draft_model_path=eagle3_path(), **chain)
    elif arm == "eagle3_auto":
        kw.update(speculative_algorithm="EAGLE3", speculative_draft_model_path=eagle3_path())
    elif arm != "plain":
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
    return kw


def run(argv: list[str]) -> int:
    import sglang

    ap = argparse.ArgumentParser(prog="bench.sglang_spec run")
    ap.add_argument("--workload", required=True, help="the file written by `dump`")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True, help="results/<device>/<out>.csv")
    ap.add_argument("--max-running", type=int, required=True)
    ap.add_argument("--gamma", type=int, default=4, help="chain length of the aligned speculative arms")
    ap.add_argument("--kv-pool-tokens", type=int, default=65536)
    ap.add_argument("--chunked-prefill-size", type=int, default=2048)
    ap.add_argument("--max-prefill-tokens", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup-min-s", type=float, default=10.0)
    ap.add_argument("--warmup-max-s", type=float, default=300.0)
    ap.add_argument("--text-acceptance", action="store_true",
                    help="also decode the natural prompts once (stop tokens honoured) and record tokens per round")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)

    pre = sidecar.preflight()
    results_dir = a.results_dir or sidecar.device_profile()[1].results_dir
    with open(a.workload) as f:
        wl = json.load(f)
    kw = engine_args(a.arm, wl, a)
    llm = sglang.Engine(**kw)
    info = llm.get_server_info()
    # What the engine runs with, read back: a flag it overrode would make this a different
    # measurement, not a slower or faster engine.
    eff = {k: info.get(k) for k in ("max_running_requests", "cuda_graph_max_bs", "max_total_tokens",
                                     "chunked_prefill_size", "speculative_algorithm", "speculative_num_steps",
                                     "speculative_eagle_topk", "speculative_num_draft_tokens", "attention_backend")}
    if eff["max_running_requests"] != a.max_running:
        raise SystemExit(f"sglang runs with max_running_requests={eff['max_running_requests']}, asked {a.max_running}")

    def once(prompts, output_lens, stop=None) -> dict:
        if stop is None:
            params = [dict(temperature=0.0, max_new_tokens=n, ignore_eos=True) for n in output_lens]
        else:
            params = [dict(temperature=0.0, max_new_tokens=n, stop_token_ids=stop) for n in output_lens]
        t0 = time.perf_counter()
        outs = llm.generate(input_ids=prompts, sampling_params=params)
        span = time.perf_counter() - t0
        got = [o["meta_info"]["completion_tokens"] for o in outs]
        if stop is None and got != list(output_lens):
            raise SystemExit(f"token counts differ from what was asked: {got[:4]}...")
        verify = sum(o["meta_info"].get("spec_verify_ct", 0) for o in outs)
        return dict(output_tok_s=round(sum(got) / span, 1), span_s=round(span, 3), output_tokens=sum(got),
                    spec_tokens_per_round=round(sum(got) / verify, 4) if verify else "")

    env = dict(sidecar.environment(), sglang=sglang.__version__)
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    with sidecar.GpuSampler() as gpu:
        # The same two settling rules as bench/offline.py: the last window of samples against
        # the one before, or a whole warm-up run's mean clock against the previous run's (an
        # 8B model's prefills hit the power cap and its decodes do not, so short windows of
        # its samples never agree).
        t_warm, warm_runs, prev_mean, settled_by = time.monotonic(), 0, None, None
        while True:
            t_run = gpu.sample_now()
            once(wl["warm_prompts"], wl["warm_output_lens"])
            mean = gpu.summary(since=t_run, until=gpu.sample_now())["sm_mhz"]["mean"]
            warm_runs += 1
            elapsed = time.monotonic() - t_warm
            if elapsed >= a.warmup_min_s:
                if gpu.settled():
                    settled_by = "window"
                elif prev_mean is not None and abs(mean - prev_mean) <= 0.02 * mean:
                    settled_by = "run"
            if settled_by:
                break
            prev_mean = mean
            if elapsed >= a.warmup_max_s:
                print(f"SM clock not settled after {elapsed:.0f} s of warm-up", file=sys.stderr)
                llm.shutdown()
                return 3
        warmup = dict(seconds=round(elapsed, 1), runs=warm_runs, settled_by=settled_by, gpu=gpu.summary(since=t_warm))
        rows, per_run_gpu = [], []
        for k in range(a.rounds):
            t_start = gpu.sample_now()
            r = once(wl["prompts"], wl["output_lens"])
            g = gpu.summary(since=t_start, until=gpu.sample_now())
            per_run_gpu.append(dict(run=k, arm=a.arm, **g))
            r.pop("output_tokens")
            rows.append(dict(
                run_id=run_id, run=k, arm=a.arm, engine="sglang", workload="unique", requests=len(wl["prompts"]),
                prompt_tokens=sum(map(len, wl["prompts"])), output_tokens=sum(wl["output_lens"]),
                kv_pool_tokens=a.kv_pool_tokens, max_running=a.max_running, **r,
                spec_algorithm=eff["speculative_algorithm"] or "", spec_steps=eff["speculative_num_steps"] or "",
                spec_topk=eff["speculative_eagle_topk"] or "", spec_draft_tokens=eff["speculative_num_draft_tokens"] or "",
                sm_mhz_mean=g["sm_mhz"]["mean"], sglang_version=sglang.__version__,
                git_commit=env["git_commit"][:12],
            ))
            print(" ".join(f"{key}={val}" for key, val in rows[-1].items()), flush=True)
        accept, t = None, None
        if a.text_acceptance and a.arm != "plain":
            # Measured after the throughput rows and kept apart from them: a failure here must
            # not take those rows with it.
            try:
                t = once(wl["text_prompts"], wl["text_output_lens"], stop=wl["stop_ids"])
            except Exception as exc:  # noqa: BLE001 - reported, and the rows above are still written
                print(f"text acceptance failed: {exc!r}", file=sys.stderr)
                t = None
        if t is not None:
            accept = dict(run_id=run_id, arm=a.arm, engine="sglang", prompts="text", num_prompts=len(wl["text_prompts"]),
                          spec_algorithm=eff["speculative_algorithm"], spec_steps=eff["speculative_num_steps"],
                          spec_topk=eff["speculative_eagle_topk"], spec_draft_tokens=eff["speculative_num_draft_tokens"],
                          tokens_per_round=t["spec_tokens_per_round"], output_tokens=t["output_tokens"],
                          git_commit=env["git_commit"][:12])
            print(" ".join(f"{key}={val}" for key, val in accept.items()), flush=True)
    llm.shutdown()

    os.makedirs(results_dir, exist_ok=True)
    path = f"{results_dir}/{a.out}.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    if accept is not None:
        apath = f"{results_dir}/{a.out}_accept.csv"
        new = not os.path.exists(apath)
        with open(apath, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(accept))
            if new:
                wr.writeheader()
            wr.writerow(accept)
    sidecar.write_sidecar(f"{results_dir}/{a.out}.{run_id}.sidecar.json", dict(
        run_id=run_id,
        config=dict(vars(a), caliber=wl["caliber"], engine_args={k: str(v) for k, v in kw.items()}, engine=eff,
                    eagle3_head=EAGLE3_HEAD if a.arm.startswith("eagle3") else None),
        environment=env, preflight=pre, warmup=warmup, gpu_during_runs=per_run_gpu,
    ))
    print(path)
    return 0


MODES = {"dump": dump, "run": run}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(f"usage: python -m bench.sglang_spec {{{'|'.join(MODES)}}} ...", file=sys.stderr)
        return 2
    return MODES[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
