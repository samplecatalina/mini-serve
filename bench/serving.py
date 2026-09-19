"""The main caliber: genai-bench against a running server, with the record that makes it comparable.

This is the number the project is judged on, so the run is arranged so that
nothing about it has to be taken on trust:

- **the load generator is genai-bench**, unmodified and pinned, in an image that
  holds nothing else (``Dockerfile.bench``). The same client measures whatever
  server it is pointed at, so a comparison is not a comparison of two harnesses;
- **the settings come from the server, not from the command line**. Both this
  engine (``/server_info``) and sglang (``/get_server_info``) report what they
  actually ended up running, and that is what is recorded. Several settings that
  decide the result — the KV pool size, the chunked-prefill budget, the captured
  graph batch sizes — are chosen inside the engine and cannot be read off a
  command line;
- **the fairness record is collected, not written by hand**, and a run refuses to
  produce results when a field of it is missing. The list is in ``FAIRNESS``;
- **the warmup is a length of time, not a number of requests**. Clocks cannot be
  locked on either device and take tens of seconds to settle, so a fixed count
  discards a different amount of unsettled time on each machine. genai-bench
  takes a fraction of the run, so the fraction is computed from the seconds
  asked for, and the seconds are recorded so both sides of a comparison discard
  the same window.

The engine's own throughput, measured without any of this by ``bench.offline``,
stays as the cross-check: the difference between the two is the serving front
end, and a sudden change in that difference means one of them is wrong.

    python -m bench.serving --url http://127.0.0.1:8000 --out NAME \\
        --concurrency 1 4 32 256 --scenario "D(1024,256)" --warmup-s 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import subprocess
import time
import urllib.parse
import urllib.request

from bench import sidecar

# Every field a comparison could turn on. A run that cannot fill one of these
# does not write results: a number whose conditions are unknown is worse than
# no number, because it will be quoted anyway.
FAIRNESS = (
    "client",              # the load generator and its version
    "client_host",         # it must be the same host as the server: no network in the path
    "server",              # which engine answered
    "server_host",
    "server_engine",       # its effective configuration, read back from the server
    "server_runtime",      # torch / CUDA / attention backend versions, and the GPU
    "model",               # repository and pinned revision
    "sampling",            # what was asked of the model, identically on both sides
    "scenarios",
    "concurrency",
    "warmup_s",            # the same discarded window on both sides
    "warmup_ratio",
    "image",               # the container this client ran in
    "cpu",                 # it decides the host-side cost, which is what overlap addresses
    "blueprint_commit",    # the reference implementation these numbers are read against
)

BLUEPRINT_COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"  # sgl-project/mini-sglang, see docs/design.md

INFO_ROUTE = {"miniserve": "/server_info", "sglang": "/get_server_info"}


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(url: str, kind: str, seconds: float) -> dict:
    """Block until the server answers, then return what it says it is running."""
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            return get_json(url + INFO_ROUTE[kind])
        except Exception as e:  # not up yet
            last = e
            time.sleep(1.0)
    raise SystemExit(f"{url} did not answer {INFO_ROUTE[kind]} within {seconds:.0f}s ({last})")


def normalize_server_info(kind: str, info: dict) -> tuple[dict, dict, dict]:
    """(engine, runtime, model) from whichever server answered."""
    if kind == "miniserve":
        return info["engine"], info["runtime"], info["model"]
    # sglang reports one flat object; keep it whole rather than guess a mapping,
    # and pull out the few fields that have to be compared field by field.
    engine = {k: v for k, v in info.items() if not k.startswith("_")}
    runtime = dict(version=info.get("version"), attention_backend=info.get("attention_backend"))
    model = dict(repo_id=info.get("model_path"), revision=info.get("revision"))
    return engine, runtime, model


def client_version() -> str | None:
    """The load generator's version, or None when it cannot be established.

    None is not "unknown": it makes the fairness record incomplete and the run
    refuses. A benchmark whose client cannot be identified cannot be repeated.
    """
    if os.path.exists("/opt/bench-version"):  # written when the client image was built
        return open("/opt/bench-version").read().strip() or None
    try:
        out = subprocess.run(["genai-bench", "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def host_name() -> str:
    """The machine, not the container. Under docker the container has a name of its
    own, so the launcher passes the host's in; under apptainer they are the same."""
    return os.environ.get("MINISERVE_HOST") or socket.gethostname()


def prune_raw(out_dir: str) -> list[str]:
    """Keep what a number can be rebuilt from; drop what can be redrawn from it.

    genai-bench writes a plot set and a spreadsheet beside its per-run JSON, about
    half a megabyte per run. Both are regenerable (``genai-bench plot`` / ``excel``)
    and results are committed, so only the JSON and the log are kept.
    """
    dropped = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith((".png", ".xlsx")):
            os.remove(os.path.join(out_dir, f))
            dropped.append(f)
    return dropped


def loaded_gpu(samples, util_min: int = 50) -> dict:
    """The clocks while the GPU was actually working.

    The sampler covers the whole client process, and genai-bench spends its first
    seconds loading a tokenizer with the GPU idle; averaging over that window
    understates the clock the measurement ran at.
    """
    busy = [x for x in samples if x.util >= util_min]
    if not busy:
        return {}

    def stats(vals):
        return dict(min=min(vals), max=max(vals), mean=round(sum(vals) / len(vals), 2))

    return dict(
        num_samples=len(busy), util_min=util_min,
        sm_mhz=stats([x.sm_mhz for x in busy]), temp_c=stats([x.temp_c for x in busy]),
        power_w=stats([x.power_w for x in busy]), power_limit_w=stats([x.power_limit_w for x in busy]),
        clocks_event_reasons=sorted({x.reasons for x in busy}),
    )


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def fairness_record(args, info: dict, engine: dict, runtime: dict, model: dict, ratio: float) -> dict:
    """Everything a comparison could turn on, collected from the run rather than written down."""
    return dict(
        client=(lambda v: f"genai-bench {v}" if v else None)(client_version()),
        client_host=host_name(),
        server=info.get("server", args.server_kind),
        server_host=urllib.parse.urlparse(args.url).hostname,
        server_engine=engine,
        server_runtime=runtime,
        model=model,
        sampling=args.sampling,
        scenarios=list(args.scenario),
        concurrency=list(args.concurrency),
        warmup_s=args.warmup_s,
        warmup_ratio=round(ratio, 4),
        image=args.image,
        cpu=cpu_model(),
        blueprint_commit=BLUEPRINT_COMMIT,
    )


def missing_fairness(record: dict) -> list[str]:
    """Which required fields are absent or empty. A run with any of these writes nothing."""
    return [k for k in FAIRNESS if not record.get(k) and record.get(k) != 0]


def warmup_ratio(warmup_s: float, max_time_per_run_min: int) -> float:
    """genai-bench discards a fraction of each run; turn the seconds asked for into that fraction."""
    total = max_time_per_run_min * 60
    ratio = warmup_s / total
    if ratio >= 0.5:
        raise SystemExit(
            f"a {warmup_s:.0f}s warmup is {100 * ratio:.0f}% of a {total:.0f}s run; "
            f"raise --max-time-per-run to at least {math.ceil(warmup_s / 0.4 / 60)} minutes"
        )
    return ratio


def genai_bench_cmd(args, ratio: float, exp_dir: str, base: str) -> list[str]:
    cmd = [
        "genai-bench", "benchmark",
        "--api-backend", "openai",
        "--api-base", args.url,
        "--api-key", "none",
        "--task", "text-to-text",
        "--api-model-name", args.model_name,
        "--model-tokenizer", args.tokenizer,
        "--max-time-per-run", str(args.max_time_per_run),
        "--max-requests-per-run", str(args.max_requests_per_run),
        "--warmup-ratio", f"{ratio:.4f}",
        "--metrics-time-unit", "ms",
        "--experiment-base-dir", base,
        "--experiment-folder-name", exp_dir,
        "--log-dir", os.path.join(base, exp_dir),
        "--additional-request-params", json.dumps(args.sampling),
    ]
    for c in args.concurrency:
        cmd += ["--num-concurrency", str(c)]
    for s in args.scenario:
        cmd += ["--traffic-scenario", s]
    if args.server_engine:
        cmd += ["--server-engine", args.server_engine]
    if args.server_version:
        cmd += ["--server-version", args.server_version]
    return cmd


def flatten(path: str) -> list[dict]:
    """One row per (scenario, concurrency) from genai-bench's per-run JSON."""
    d = json.load(open(path))
    a, s = d["aggregated_metrics"], d["aggregated_metrics"]["stats"]

    def q(metric: str, key: str):
        v = s.get(metric, {}).get(key)
        return None if v is None else round(v, 3)

    return [dict(
        scenario=a["scenario"],
        concurrency=a["num_concurrency"],
        run_duration_s=round(a["run_duration"], 3),
        num_requests=a["num_requests"],
        num_completed=a["num_completed_requests"],
        num_errors=a["num_error_requests"],
        error_rate=a["error_rate"],
        requests_per_s=round(a["requests_per_second"], 3),
        output_tok_s=round(a["mean_output_throughput_tokens_per_s"], 1),
        total_tok_s=round(a["mean_total_tokens_throughput_tokens_per_s"], 1),
        ttft_p50_ms=q("ttft", "p50"), ttft_p95_ms=q("ttft", "p95"), ttft_p99_ms=q("ttft", "p99"),
        tpot_p50_ms=q("tpot", "p50"), tpot_p99_ms=q("tpot", "p99"), tpot_max_ms=q("tpot", "max"),
        e2e_p50_ms=q("e2e_latency", "p50"), e2e_p99_ms=q("e2e_latency", "p99"),
        input_tokens_mean=q("num_input_tokens", "mean"),
        output_tokens_mean=q("num_output_tokens", "mean"),
        time_unit=d["_time_unit"],
        source=os.path.basename(path),
    )]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="a server that is already running")
    ap.add_argument("--server-kind", choices=sorted(INFO_ROUTE), default="miniserve")
    ap.add_argument("--label", default=None, help="name of this arm in the results (default: --server-kind)")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B", help="the model name in the request body")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path; default: the model path the server reports")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 32, 256])
    ap.add_argument("--scenario", nargs="+", default=["D(1024,256)"], help='genai-bench traffic scenarios, e.g. "D(1024,256)"')
    ap.add_argument("--max-time-per-run", type=int, default=3, help="minutes per (scenario, concurrency); genai-bench takes whole minutes")
    ap.add_argument("--max-requests-per-run", type=int, default=2048)
    ap.add_argument("--warmup-s", type=float, default=30.0,
                    help="seconds discarded at the start of each run; the L40S takes about 30 s to settle")
    ap.add_argument("--sampling", type=json.loads, default={"temperature": 0, "ignore_eos": True},
                    help="request parameters, identical on both sides of a comparison")
    ap.add_argument("--server-engine", default=None, help="genai-bench metadata: vLLM | SGLang | ...")
    ap.add_argument("--server-version", default=None)
    ap.add_argument("--wait-s", type=float, default=600.0, help="how long to wait for the server")
    ap.add_argument("--out", required=True, help="results/<device>/<out>.csv, a sidecar and the raw output")
    ap.add_argument("--results-dir", default=None, help="default: the device profile's (bench/sidecar.py)")
    ap.add_argument("--preflight-json", default=None,
                    help="conditions recorded by `python -m bench.sidecar` before the server was started; "
                         "without it the check runs here, which only works against a server that is not up yet")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development only: accept a server running from an uncommitted tree")
    ap.add_argument("--image", default=os.environ.get("MINISERVE_BENCH_IMAGE", ""),
                    help="what identifies the client container (digest or SIF hash); recorded, and required")
    args = ap.parse_args()
    args.label = args.label or args.server_kind

    # The GPU has to be idle *before* the server loads the model, so when a
    # server is already running the check must have happened earlier, elsewhere.
    if args.preflight_json:
        pre = json.load(open(args.preflight_json))
    else:
        pre = sidecar.preflight()
    _, profile = sidecar.device_profile()
    results_dir = args.results_dir or profile.results_dir
    if not args.image:
        raise SystemExit("--image (or MINISERVE_BENCH_IMAGE) is required: results name the client they came from")

    info = wait_healthy(args.url, args.server_kind, args.wait_s)
    engine, runtime, model = normalize_server_info(args.server_kind, info)
    tokenizer = args.tokenizer or model.get("path") or model.get("repo_id")
    args.tokenizer = tokenizer

    # The client is pinned by its image digest; what can drift is the server's code,
    # so that is what is checked — and the server is the one being measured.
    if runtime.get("git_dirty") and not args.allow_dirty:
        raise SystemExit(f"the server at {args.url} is running from an uncommitted tree; results would not be reproducible")

    ratio = warmup_ratio(args.warmup_s, args.max_time_per_run)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    raw_dir = f"{args.out}.{run_id}.genai-bench"
    os.makedirs(results_dir, exist_ok=True)

    fairness = fairness_record(args, info, engine, runtime, model, ratio)
    missing = missing_fairness(fairness)
    if missing:
        raise SystemExit(f"the fairness record is missing {missing}; refusing to run")

    cmd = genai_bench_cmd(args, ratio, raw_dir, results_dir)
    print("$ " + " ".join(cmd), flush=True)
    with sidecar.GpuSampler() as gpu:
        t_start = gpu.sample_now()
        rc = subprocess.run(cmd).returncode
        t_end = gpu.sample_now()
        gpu_during = gpu.summary(since=t_start, until=t_end)
        gpu_loaded = loaded_gpu([x for x in gpu.snapshot() if t_start <= x.t <= t_end])
    if rc != 0:
        raise SystemExit(f"genai-bench exited {rc}; no results written")

    out_dir = os.path.join(results_dir, raw_dir)
    runs = sorted(f for f in os.listdir(out_dir) if f.endswith(".json") and f != "experiment_metadata.json")
    if not runs:
        raise SystemExit(f"genai-bench wrote no per-run JSON into {out_dir}")

    dropped = prune_raw(out_dir)
    rows: list[dict] = []
    for f in runs:
        for row in flatten(os.path.join(out_dir, f)):
            rows.append(dict(run_id=run_id, arm=args.label, device=pre["idle_gpu"]["name"], **row))
    rows.sort(key=lambda r: (r["scenario"], r["concurrency"]))

    path = f"{results_dir}/{args.out}.csv"
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)

    sidecar.write_sidecar(
        f"{results_dir}/{args.out}.{run_id}.sidecar.json",
        dict(
            run_id=run_id,
            config=vars(args),
            environment=dict(client=fairness["client"], cpu=fairness["cpu"], **runtime),
            preflight=pre,
            warmup=dict(seconds=args.warmup_s, ratio=round(ratio, 4),
                        note="genai-bench discards this fraction of every run; both sides of a comparison use the same seconds"),
            gpu_during_runs=gpu_during,
            gpu_while_loaded=gpu_loaded,
            fairness=fairness,
            raw=dict(dir=raw_dir, kept=runs + ["experiment_metadata.json"], dropped=dropped),
        ),
    )

    print(f"\n{len(rows)} rows -> {path}   (raw genai-bench output in {out_dir})")
    hdr = f"{'arm':<12}{'scenario':<16}{'conc':>6}{'out tok/s':>11}{'TTFT p50':>10}{'TTFT p95':>10}{'TPOT p50':>10}{'TPOT p99':>10}{'err':>6}"
    print(hdr)
    for r in rows:
        print(f"{r['arm']:<12}{r['scenario']:<16}{r['concurrency']:6d}{r['output_tok_s']:11.1f}"
              f"{r['ttft_p50_ms']:10.1f}{r['ttft_p95_ms']:10.1f}{r['tpot_p50_ms']:10.2f}{r['tpot_p99_ms']:10.2f}"
              f"{r['num_errors']:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
