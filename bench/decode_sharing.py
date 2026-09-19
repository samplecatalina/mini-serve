"""Decode attention with and without shared KV blocks: one layer, one FlashInfer decode kernel.

A batch of sequences of equal length is decoded against a one-layer KV pool
in three layouts: every sequence on its own blocks (``distinct``), groups of
sequences sharing their leading blocks (``shared``, as a radix cache produces
for requests with a common prefix), and all sequences sharing one prefix
(``shared-all``). The kernel reads the same logical amount of KV in every
case; only how many distinct bytes sit behind it changes. If shared blocks
are served from L2 instead of DRAM, the kernel gets faster as sharing grows.

Shapes are Qwen3-0.6B's (16 query heads, 8 KV heads, head_dim 128, BF16) and
the engine's decode wrapper settings. Timing: CUDA events around ``--iters``
back-to-back launches, median of ``--repeats``.

Usage: make bench-decode-sharing BENCH_ARGS="--out m2_1_decode_sharing"
Under Nsight Compute, ``--case`` and small ``--iters`` select one layout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import statistics
import sys

import torch

from bench import sidecar

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, BLOCK = 16, 8, 128, 16
KV_BYTES_PER_TOKEN = 2 * NUM_KV_HEADS * HEAD_DIM * 2  # K and V, BF16, one layer


def layouts(num_seqs: int, seq_blocks: int, prefix_blocks: int, groups: int) -> dict[str, list[list[int]]]:
    """Block ids per sequence for each case."""
    own = seq_blocks - prefix_blocks
    out = {}
    out["distinct"] = [list(range(s * seq_blocks, (s + 1) * seq_blocks)) for s in range(num_seqs)]

    def grouped(g):
        tables, nxt = [], g * prefix_blocks
        for s in range(num_seqs):
            prefix = list(range((s % g) * prefix_blocks, (s % g + 1) * prefix_blocks))
            tables.append(prefix + list(range(nxt, nxt + own)))
            nxt += own
        return tables

    out["shared"] = grouped(groups)
    out["shared-all"] = grouped(1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=1152, help="tokens per sequence (a multiple of 16)")
    ap.add_argument("--prefix-len", type=int, default=1024, help="shared leading tokens (a multiple of 16)")
    ap.add_argument("--groups", type=int, default=8, help="sequences share a prefix in this many groups")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--case", choices=["distinct", "shared", "shared-all"], default=None, help="run one layout only")
    ap.add_argument("--out", default=None, help="results name (omit under a profiler)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    import flashinfer

    pre = sidecar.preflight() if args.out else None
    env = sidecar.environment()
    if args.out and env["git_dirty"] and not args.allow_dirty:
        print("refusing to measure uncommitted code (tracked files modified)", file=sys.stderr)
        return 2

    seq_blocks, prefix_blocks = args.seq_len // BLOCK, args.prefix_len // BLOCK
    cases = layouts(args.num_seqs, seq_blocks, prefix_blocks, args.groups)
    num_blocks = args.num_seqs * seq_blocks
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    k = torch.randn(num_blocks, BLOCK, NUM_KV_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn_like(k)
    q = torch.randn(args.num_seqs, NUM_QO_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16, generator=g)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=NUM_QO_HEADS // NUM_KV_HEADS >= 4, backend="fa2"
    )

    rows = []
    run_id = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    with sidecar.GpuSampler() as gpu:
        for name, tables in cases.items():
            if args.case and name != args.case:
                continue
            i32 = dict(dtype=torch.int32, device=dev)
            indptr = torch.tensor([0] + [seq_blocks * (s + 1) for s in range(args.num_seqs)], **i32)
            indices = torch.tensor([b for t in tables for b in t], **i32)
            last = torch.full((args.num_seqs,), BLOCK, **i32)
            wrapper.plan(indptr, indices, last, num_qo_heads=NUM_QO_HEADS, num_kv_heads=NUM_KV_HEADS,
                         head_dim=HEAD_DIM, page_size=BLOCK, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            for _ in range(10):
                wrapper.run(q, (k, v))
            times = []
            t_case = gpu.sample_now()
            for _ in range(args.repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    wrapper.run(q, (k, v))
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e3 / args.iters)  # microseconds per launch
            us = statistics.median(times)
            logical = args.num_seqs * args.seq_len * KV_BYTES_PER_TOKEN
            unique = len({b for t in tables for b in t}) * BLOCK * KV_BYTES_PER_TOKEN
            rows.append(dict(
                run_id=run_id, case=name, num_seqs=args.num_seqs, seq_len=args.seq_len, prefix_len=args.prefix_len,
                groups=args.groups if name == "shared" else (1 if name == "shared-all" else args.num_seqs),
                logical_kv_mib=round(logical / 2**20, 1), unique_kv_mib=round(unique / 2**20, 1),
                kernel_us=round(us, 2), kernel_us_min=round(min(times), 2), kernel_us_max=round(max(times), 2),
                logical_gb_s=round(logical / us / 1e3, 1), unique_gb_s=round(unique / us / 1e3, 1),
                sm_mhz_mean=gpu.summary(since=t_case, until=gpu.sample_now())["sm_mhz"]["mean"],
                git_commit=env["git_commit"][:12],
            ))
            print(" ".join(f"{a}={b}" for a, b in rows[-1].items() if a not in ("run_id", "git_commit")), flush=True)
        gpu_summary = gpu.summary()

    if args.out:
        d = "results/rtx4060-laptop"
        os.makedirs(d, exist_ok=True)
        path = f"{d}/{args.out}.csv"
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        sidecar.write_sidecar(f"{d}/{args.out}.{run_id}.sidecar.json", dict(
            run_id=run_id, config=vars(args), environment=env, preflight=pre,
            warmup=dict(note="10 untimed launches per case"), gpu_during_runs=[gpu_summary],
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
