"""HTTP smoke test and front-end overhead check against a running server.

Sends the requests of an offline-bench workload (``bench.offline.make_workload``, same
arguments, same random draws) to ``/v1/completions`` as token-id prompts, streaming, greedy,
``ignore_eos``, all at once with at most ``--concurrency`` in flight. Every request must finish
with exactly its output length. Output throughput here against the offline bench on the same
workload is the cost of the serving front end (HTTP, SSE, detokenization, one process).

Usage: ``python -m bench.http_smoke --url http://127.0.0.1:8000 --workload unique --groups 4
--per-group 8 --concurrency 32 --out results/<device>/NAME.csv``
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time

import httpx

from bench.offline import make_workload, pct


async def one(client: httpx.AsyncClient, url: str, prompt: list[int], n: int, t0: float) -> dict:
    body = dict(model="default", prompt=prompt, max_tokens=n, temperature=0, stream=True, ignore_eos=True,
                stream_options=dict(include_usage=True))
    start = time.perf_counter()
    stamps, usage, finish = [], None, None
    async with client.stream("POST", f"{url}/v1/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            msg = json.loads(line[6:])
            if msg.get("usage"):
                usage = msg["usage"]
            for c in msg.get("choices", []):
                if c.get("text"):
                    stamps.append(time.perf_counter())
                finish = c.get("finish_reason") or finish
    tokens = usage["completion_tokens"] if usage else None
    return dict(start=start - t0, ttft=stamps[0] - start if stamps else None,
                gaps=[b - a for a, b in zip(stamps, stamps[1:])], end=(stamps[-1] if stamps else time.perf_counter()) - t0,
                tokens=tokens, expected=n, finish=finish)


async def run(args) -> dict:
    w = make_workload(args, seed=args.workload_seed)
    sem = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        (await client.get(f"{args.url}/health")).raise_for_status()
        t0 = time.perf_counter()

        async def guarded(p, n):
            async with sem:
                return await one(client, args.url, p, n, t0)

        res = await asyncio.gather(*(guarded(p, n) for p, n in zip(w.prompts, w.output_lens)))
    ok = [r for r in res if r["tokens"] == r["expected"] and r["finish"] == "length"]
    span = max(r["end"] for r in res)
    gaps = [g for r in res for g in r["gaps"]]
    ttft = [r["ttft"] for r in res if r["ttft"] is not None]
    return dict(
        requests=len(res),
        succeeded=len(ok),
        concurrency=args.concurrency,
        output_tokens=sum(r["tokens"] or 0 for r in res),
        output_tok_s=round(sum(r["tokens"] or 0 for r in res) / span, 1),
        ttft_p50_ms=round(pct(ttft, 50) * 1e3, 2),
        ttft_p95_ms=round(pct(ttft, 95) * 1e3, 2),
        chunk_gap_p50_ms=round(pct(gaps, 50) * 1e3, 2) if gaps else "",
        chunk_gap_p99_ms=round(pct(gaps, 99) * 1e3, 2) if gaps else "",
        span_s=round(span, 3),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=None, help="append one CSV row here")
    ap.add_argument("--label", default="")
    g = ap.add_argument_group("workload (as bench.offline)")
    g.add_argument("--workload", choices=["shared", "unique", "mixed", "policy"], default="unique")
    for name, default in (("groups", 4), ("per-group", 8), ("prefix-len", 1024), ("suffix-len", 64), ("output-len", 64),
                          ("num-long", 0), ("long-len", 3072), ("long-output-len", 16), ("short-len", 256),
                          ("short-output-len", 32), ("workload-seed", 0)):
        g.add_argument(f"--{name}", type=int, default=default)
    g.add_argument("--long-fraction", type=float, default=0.25)
    g.add_argument("--arrival-rate", type=float, default=0.0)
    args = ap.parse_args()
    row = asyncio.run(run(args))
    row = dict(label=args.label, workload=args.workload, **row)
    print(json.dumps(row))
    if args.out:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    return 0 if row["succeeded"] == row["requests"] else 1


if __name__ == "__main__":
    sys.exit(main())
