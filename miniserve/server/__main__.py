"""Run the OpenAI-compatible server.

Usage: ``python -m miniserve.server --port 8000 [--kv-pool-tokens N] ...``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from miniserve.engine.cli import add_engine_args, engine_kwargs
from miniserve.engine.describe import engine_description, runtime_description
from miniserve.engine.engine import Engine
from miniserve.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
from miniserve.model.weights import QWEN3_0_6B, ModelSpec, load_config, load_weights, model_path
from miniserve.server.app import build_app
from miniserve.server.async_engine import AsyncEngine
from miniserve.server.tokenizer import ThreadPoolTokenizer


def stop_token_ids(path: Path) -> frozenset[int]:
    """End-of-sequence ids from ``generation_config.json`` (Qwen3: <|im_end|> and <|endoftext|>)."""
    eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
    return frozenset(eos if isinstance(eos, list) else [eos])


def context_len(engine: Engine, cfg: dict) -> int:
    """Longest prompt + output one request may have: the model context, and the KV pool if there is one."""
    n = cfg["max_position_embeddings"]
    a = engine.runner.allocator
    return n if a is None else min(n, a.num_blocks * a.block_size)


def build_server(engine: Engine, path: Path, served_model_name: str, tokenizer_workers: int = 1,
                 spec: ModelSpec = QWEN3_0_6B):
    """The FastAPI app for an engine over the model at ``path``."""
    from transformers import AutoTokenizer

    tokenizer = ThreadPoolTokenizer(AutoTokenizer.from_pretrained(path), tokenizer_workers)
    # Read the configuration before the engine thread starts: afterwards the engine
    # belongs to that thread, and these values do not change while the server runs.
    info = dict(
        server="miniserve",
        model=dict(repo_id=spec.repo_id, revision=spec.revision, path=str(path)),
        engine=engine_description(engine),
        runtime=runtime_description(),
    )
    return build_app(
        AsyncEngine(engine, tokenizer),
        model_name=served_model_name,
        context_len=context_len(engine, load_config(path)),
        stop_token_ids=stop_token_ids(path),
        server_info=info,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_engine_args(ap)
    g = ap.add_argument_group("server")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--served-model-name", default=QWEN3_0_6B.repo_id)
    g.add_argument("--tokenizer-workers", type=int, default=1, help="threads of the tokenizer pool")
    g.add_argument("--log-level", default="info")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    path = model_path(QWEN3_0_6B, download=False)
    model = Qwen3ForCausalLM(Qwen3Config.from_dict(load_config(path)), load_weights(path))
    engine = Engine(model, **engine_kwargs(args))
    app = build_server(engine, path, args.served_model_name, args.tokenizer_workers)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
