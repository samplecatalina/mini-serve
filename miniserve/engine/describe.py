"""What this engine is actually running: the effective configuration, not the command line.

A benchmark comparison is only as good as the record of what each side was
configured with, and several of the settings that matter most are decided
inside the engine rather than passed in. The KV pool defaults to whatever GPU
memory allows; the chunked-prefill budget defaults differently per attention
mode; the set of captured CUDA Graph batch sizes is derived from a maximum.
Reading these back from the running engine is the only way to know them, and
it is what the fairness record for a benchmark is built from.

The comparable engines expose the same thing (sglang answers ``/get_server_info``),
so the two sides of a comparison can be diffed rather than transcribed.
"""

from __future__ import annotations

import subprocess

from miniserve.engine.engine import Engine


def engine_description(engine: Engine) -> dict:
    """The engine's effective configuration, as it ended up, for the record."""
    runner, sched = engine.runner, engine.scheduler
    alloc, graphs = runner.allocator, runner.graphs
    return dict(
        attention=runner.attention,
        max_running=sched.max_running,
        max_prefill_tokens=sched.max_prefill_tokens,
        chunked_prefill_size=sched.chunked_prefill_size,
        schedule_policy=sched.policy.name,
        overlap=engine.overlap,
        radix=bool(runner.kv is not None and runner.kv.radix),
        kv_pool_tokens=None if alloc is None else alloc.num_blocks * alloc.block_size,
        kv_pool_blocks=None if alloc is None else alloc.num_blocks,
        block_size=None if alloc is None else alloc.block_size,
        block_backend=runner.block_backend,
        cuda_graph=runner.use_cuda_graph,
        cuda_graph_buckets=None if graphs is None else list(graphs.buckets),
        cuda_graph_bytes=None if graphs is None else graphs.graph_bytes,
        cuda_graph_capture_s=None if graphs is None else round(graphs.capture_s, 2),
        dtype=str(runner.model.dtype).removeprefix("torch."),
    )


def runtime_description() -> dict:
    """The software and the device underneath, for the same record."""
    import flashinfer
    import torch

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # a checkout is not always available (a container run from a tarball)
            return None

    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        gpu=None if props is None else props.name,
        gpu_memory_bytes=None if props is None else props.total_memory,
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
    )
