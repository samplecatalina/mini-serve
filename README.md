# mini-serve

A single-GPU LLM inference engine: continuous batching, paged KV cache, radix prefix cache, chunked prefill, CPU-GPU overlap scheduling, CUDA Graph decode, a C++ scheduling core, and two-model speculative decoding.

Work in progress: continuous batching, the paged KV cache, the radix prefix cache and chunked prefill are done; see `docs/design.md` for the design and `docs/optimization-log.md` for every measurement with the prediction made before it.

## Results so far

Engine-level measurements on an RTX 4060 Laptop GPU (Qwen3-0.6B, BF16, 32k-token KV pool; median of 3 runs; rows in `results/rtx4060-laptop/`):

| Change | Workload | Result |
|---|---|---|
| Radix prefix cache | 64 requests at once, 8 groups sharing 1024-token prefixes | 1487 vs 409 output tok/s; TTFT p50 1.24 vs 4.63 s |
| Radix prefix cache | same, 4 req/s | TTFT p50 31 vs 74 ms |
| Chunked prefill, 512-token chunks | 4 req/s, 8 long (3072-token) prompts among 64 | ITL p99 45 vs 61 ms, ITL max 48 vs 466 ms |
| Chunked prefill, 512-token chunks | 64 × 1088-token prompts at once | ITL p99 47 vs 615 ms; 535 vs 420 output tok/s |

Why the numbers look the way they do (shared blocks served from L2 in decode; chunks filling GPU time that decode steps leave idle; larger chunks worsening the ITL tail) is in the optimization log.

## Development environment

Everything runs inside one Docker image (`Dockerfile`), which is also the image converted to a SIF for cluster runs. Dependency versions are pinned exactly in `pyproject.toml` / `uv.lock` (torch 2.9.1+cu128, triton 3.5.1, flashinfer-python 0.6.7.post3, transformers 5.3.0, Python 3.12), matching the reference sglang 0.5.10 environment used for comparisons.

```bash
make image       # build the development image
make env-check   # toolchain, pinned versions, GPU, Triton/FlashInfer JIT, cache locations
make weights     # download the pinned Qwen3-0.6B snapshot
make test        # pytest
make bench-offline BENCH_ARGS="--workload shared --ablate radix --out NAME"   # engine-level benchmark
```

Engine-level benchmarks (`bench/offline.py`) feed requests straight into the engine on an open-loop arrival schedule, alternate the compared settings in one process (A B B A ...), and write one CSV row per run to `results/<device>/` together with a sidecar JSON of the measurement conditions (GPU clocks, power limit and throttle reasons, temperature, versions, commit). A run refuses to start if the GPU is busy, the power configuration is wrong, or the working tree has uncommitted changes.

Requirements on the host: an NVIDIA driver supporting CUDA 12.8 or newer and Docker with the NVIDIA container runtime. JIT and model caches live in the `miniserve-cache` Docker volume, mounted at `/cache`.

Two build settings are fixed on purpose:

- **Compiler baselines**: C/C++ code is built with `-march=x86-64-v2` and CUDA code for `sm_89` only. `-march=native` on a development CPU with AVX-512 produces binaries that fault with illegal instructions on AVX2-only cluster nodes.
- **Cache locations**: FlashInfer (`FLASHINFER_WORKSPACE_BASE`), Triton (`TRITON_CACHE_DIR`), tvm-ffi (`TVM_FFI_CACHE_DIR`) and Hugging Face (`HF_HOME`) caches are redirected out of `$HOME`, which is small on shared clusters. `make env-check` fails if any of them resolves under `$HOME`.

## Correctness

`tests/test_consistency.py` is the correctness anchor: greedy decoding must match Hugging Face transformers token for token (same checkpoint, BF16, same seed). The model in `miniserve/model/qwen3.py` is a plain-PyTorch reference path that mirrors the transformers op order and precision, so its prefill logits are required to be bitwise equal to HF, which is stricter than token equality: a wrong norm weight in one layer can leave 64 greedy tokens unchanged while shifting logits by ~1.9. Optimized paths are checked against the same tests.
