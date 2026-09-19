"""Command-line options shared by everything that builds an ``Engine``."""

from __future__ import annotations

import argparse

from miniserve.engine.model_runner import ATTENTION_MODES


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


def engine_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        max_running=args.max_running,
        max_prefill_tokens=args.max_prefill_tokens,
        attention=args.attention,
        kv_pool_tokens=args.kv_pool_tokens,
    )
