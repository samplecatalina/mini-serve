"""Command-line options shared by everything that builds an ``Engine``."""

from __future__ import annotations

import argparse

from miniserve.engine.model_runner import ATTENTION_MODES
from miniserve.engine.policy import POLICIES


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("engine")
    g.add_argument("--max-running", type=int, default=64, help="max requests in the running batch")
    g.add_argument("--max-prefill-tokens", type=int, default=8192, help="token budget of one prefill batch")
    g.add_argument("--attention", choices=ATTENTION_MODES, default="paged")
    g.add_argument(
        "--kv-pool-tokens",
        type=int,
        default=None,
        help="exact KV pool size in tokens (rounded down to whole blocks); "
        "default: as large as GPU memory allows. An experiment dimension, not only a limit.",
    )
    g.add_argument("--seed", type=int, default=0, help="seeds sampling for requests that bring no seed of their own")
    g.add_argument("--disable-radix", action="store_true", help="no prefix cache: every prefill computes all its tokens")
    g.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
        help="tokens per step, decodes included, with prefills cut into chunks batched with decodes; "
        "0: whole prefills in prefill-only steps (default: 2048 with paged attention)",
    )
    g.add_argument(
        "--schedule-policy",
        choices=sorted(POLICIES),
        default="fcfs",
        help="admission order and preemption choice: fcfs, sjf (least remaining work first), "
        "cache (longest cached prefix first)",
    )
    g.add_argument(
        "--disable-overlap",
        action="store_true",
        help="read back each step's tokens before scheduling the next (no CPU-GPU overlap)",
    )
    g.add_argument("--disable-cuda-graph", action="store_true", help="run decode steps eagerly (no CUDA Graphs)")
    g.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help="largest decode batch captured in a CUDA Graph (default: --max-running); "
        "graphs are captured for powers of two below it and for it",
    )


def engine_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        max_running=args.max_running,
        max_prefill_tokens=args.max_prefill_tokens,
        attention=args.attention,
        kv_pool_tokens=args.kv_pool_tokens,
        seed=args.seed,
        radix=not args.disable_radix,
        chunked_prefill_size=args.chunked_prefill_size,
        cuda_graph=not args.disable_cuda_graph,
        overlap=not args.disable_overlap,
        schedule_policy=args.schedule_policy,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
    )
