# mini-serve

A single-GPU LLM inference engine: continuous batching, paged KV cache, radix prefix cache, chunked prefill, CPU-GPU overlap scheduling, CUDA Graph decode, a C++ scheduling core, and two-model speculative decoding.

Written from [mini-sglang](https://github.com/sgl-project/mini-sglang) (pinned at `9a91cfa`), re-implementing its scheduling features and measuring each with an ablation; the C++ core and speculative decoding have no counterpart in it. `docs/design.md` has the design and where it differs from the blueprint; `docs/optimization-log.md` has every measurement next to the prediction written before it, including the ones that were wrong.

## Status

All milestones are complete (M0–M5, tag `v1.0`).

- **Implemented and ablated**: continuous batching over a paged KV cache (16-token blocks), admission against a block budget with preemption, radix prefix cache, chunked prefill mixed with decode, decode CUDA Graphs, one-step-lag overlap scheduling, three scheduling policies, GPU-side sampling, an OpenAI-compatible server with streaming; a C++ core for block accounting, block-table packing and the prefix tree; two-model speculative decoding with greedy verification.
- **Implemented, not the default**: the C++ core is selected with `--block-backend cpp`. It shortens the host path but measures no end-to-end change on either GPU tested (below), so the Python reference stays the default.
- **Designed, not implemented**: rejection sampling for speculative decoding at temperature > 0 (only greedy verification exists); speculative decoding in the HTTP server (it runs through the engine-level harness).
- **Open measurements**: listed at the end of `docs/design.md`.

## Results

Every number below links to a row under `results/`; Qwen3-0.6B in BF16 unless stated.

**Primary benchmark, L40S** (genai-bench over HTTP, 1024-token prompts, 256 output tokens; `results/l40s/m4_2_main.csv`):

| Concurrency | Output tok/s | TTFT p50 | TPOT p50 | Batch the engine formed |
|---|---|---|---|---|
| 1 | 172.5 | 30.1 ms | 4.22 ms | 1.0 |
| 32 | 1,992.8 | 113.2 ms | 11.51 ms | 31.47 |
| 256 | 2,855.9 | 267.2 ms | 61.85 ms | 243.31 |

**Against sglang 0.5.10**, same job, same node, ABBA over three rounds (`results/l40s/m4_2_compare.csv`): 61.6% of its throughput at concurrency 1, 89.9% at 32, **101.7% at 256**. At 256 its TTFT p50 is 1,994.1 ms against 275.7 ms here; its TPOT is lower. The per-step cost of this engine is higher; its scheduler emits first tokens sooner.

**Against the blueprint**, engine to engine on the same token ids, development GPU, 32k-token pool (`results/rtx4060-laptop/m4_4_*`, `m5x_blueprint_*`):

| Workload | This engine ÷ blueprint | Why |
|---|---|---|
| 256 requests, pool too small to hold them all | 0.656 | Admission against a block budget preempts; 42% more prefill tokens recomputed |
| Same, a quarter of requests 8× longer | 0.759 | Fewer preemptions |
| Shared prefixes, cache emptied before each run | 0.898 | — |
| Short chat answers that stop at end of turn, declared maximum 512 | **1.40** | The blueprint reserves the declared maximum for every request |
| Same requests, declared maximum 128 | 0.815 | Reservation now close to the real length |

The last two rows differ only in the declared maximum: the blueprint's throughput moves 1.72×, this engine's does not.

**Ablations on the L40S** (`results/l40s/m4_2_*.csv`):

| Feature | Effect |
|---|---|
| Decode CUDA Graphs | +129.4% throughput at concurrency 32, +0.9% at 256 (host-bound small batches vs a 50 ms GPU step) |
| Overlap scheduling | +14.7% at 32, +26.7% at 256 over HTTP; +5.2% at a batch of 32 offline (the server's own threads add host time to hide) |
| Chunked prefill | TTFT p50 −74.9% at 32 and −88.1% at 256; sustained batch 142.63 → 243.39 |
| Cache-aware admission (shared prefixes, 512 requests) | +43.9% / +30.9% / +30.7% throughput in 41k / 102k / 354k-token pools |
| Shortest-job-first | TTFT p50 −99.4%, p95 +201.8% (40k pool): the median bought with the tail |

**C++ core** (`results/rtx4060-laptop/m5_1_5_*`, `m5_2_*`, `results/l40s/m5_7_*`): host path at batch 248 **1.541 → 0.920 ms** on the development host, 2.894 → 1.890 ms on the L40S node; evicting one block from a 19,172-node prefix tree 5,553 µs (Python) → 804 µs (same algorithm in C++) → 1.0 µs (ordered index). End to end: 4,019.2 vs 4,019.2 tok/s on the L40S — with CUDA Graphs and overlap the saved host time is off the critical path.

**Speculative decoding, Qwen3-8B target / 0.6B draft, L40S** (`results/l40s/m5_5_*`, `m5_6_*`): against the default engine 1.97× with one request running, 1.37× at 8, 1.03× at 32, 0.95× at 64 (best γ at each cap; random prompts, which understate it — acceptance at γ=4 is 0.431 on them and 0.561 on natural text). In sglang on the same token ids, the same algorithm is 13–29% faster, and a public EAGLE-3 head emits 1.8–2.2 tokens per round against 3.1–3.7 for the 0.6B draft.

**Overlap across three platforms**, one offline harness, pinned pool (`results/*/m5_7_overlap.csv`):

| 0.6B, batch cap | 8 | 32 | 128 |
|---|---|---|---|
| RTX 4060 Laptop (Ryzen 9 7945HX) | +0.9% | +1.5% | — |
| L40S (Xeon 6740E) | +5.4% | +5.2% | +6.8% |
| H100 (Xeon Platinum 8470) | +3.2% | +5.2% | +10.9% |

The gain is roughly host time hidden per step ÷ step time, set by the CPU and the GPU respectively; the cluster nodes pair them differently, which is why the H100 is lower than the L40S at small batches and higher at large ones.

### Feature ablations on the development GPU

Engine-level measurements on an RTX 4060 Laptop GPU (Qwen3-0.6B, BF16, 32k-token KV pool; median of 3 runs; rows in `results/rtx4060-laptop/`):

| Change | Workload | Result |
|---|---|---|
| Radix prefix cache | 64 requests at once, 8 groups sharing 1024-token prefixes | 1487 vs 409 output tok/s; TTFT p50 1.24 vs 4.63 s |
| Radix prefix cache | same, 4 req/s | TTFT p50 31 vs 74 ms |
| Chunked prefill, 512-token chunks | 4 req/s, 8 long (3072-token) prompts among 64 | ITL p99 45 vs 61 ms, ITL max 48 vs 466 ms |
| Chunked prefill, 512-token chunks | 64 × 1088-token prompts at once | ITL p99 47 vs 615 ms; 535 vs 420 output tok/s |
| CUDA Graph decode | 4 × 1088-token prompts at once, 256 output tokens | ITL p50 10.1 vs 19.6 ms; 362 vs 194 output tok/s |
| CUDA Graph decode | 4 req/s, 8 long prompts among 64 | ITL p50 8.6 vs 19.8 ms, ITL p99 30 vs 124 ms |
| CUDA Graph decode | 64 × 1088-token prompts at once | 538 vs 475 output tok/s |
| Overlap scheduling (with CUDA Graphs) | 64 × 1088-token prompts at once | 543 vs 528 output tok/s |
| Overlap scheduling (with CUDA Graphs) | 4 × 1088-token prompts at once, 256 output tokens | 364 vs 358 output tok/s; TTFT p50 238 vs 229 ms |
| Overlap scheduling (without CUDA Graphs) | same | 203 vs 190 output tok/s |
| Cache-aware admission vs FCFS | 128 requests at once, 16 groups sharing 512-token prefixes, 25% with 512-token outputs; 8k-token KV pool | 1136 vs 894 output tok/s; mean latency 6.8 vs 8.9 s |
| Shortest-job-first vs FCFS | same | mean latency 5.2 vs 8.9 s, short requests 1.9 vs 7.2 s; long-request p99 21.3 vs 22.3 s |
| Any policy | same, 16k-token KV pool | throughput within 4% of FCFS: the pool holds the queue, order stops mattering |
| Overlap scheduling (with CUDA Graphs) | 4 req/s, 8 long prompts among 64 | ITL p50 8.55 vs 8.73 ms, ITL p99 27 vs 29 ms; TTFT p50 39 vs 31 ms |

Ported to a cluster L40S with the same container image (byte-identical, built locally and copied over): 32 requests of 1088 tokens with 256 output tokens each run at 2569 output tok/s through the engine and 2832 tok/s through 32 concurrent HTTP streams, against 744 / 738 tok/s on the 4060. Greedy output was checked on each device against that device's reference path; across devices the reference path itself differs at BF16 near-ties (see `docs/design.md`).

CUDA Graph memory: 7 graphs (batch sizes 1, 2, 4, …, 64) share one memory pool of about 0.1 GB (107,355,648 bytes in `m3_1_cuda_graph` runs, captured in 0.45 s), plus one 1.75 MiB KV block for padding rows; the KV pool is sized before capture and the graphs use the memory it leaves free.

Why the numbers look the way they do (shared blocks served from L2 in decode; larger chunks worsening the ITL tail; kernel submission time that a decode graph removes; why overlap adds little once decode steps are graphs) is in the optimization log.

## Development environment

Everything runs inside one Docker image (`Dockerfile`), which is also the image converted to a SIF for cluster runs. Dependency versions are pinned exactly in `pyproject.toml` / `uv.lock` (torch 2.9.1+cu128, triton 3.5.1, flashinfer-python 0.6.7.post3, transformers 5.3.0, Python 3.12), matching the reference sglang 0.5.10 environment used for comparisons.

```bash
make image       # build the development image
make env-check   # toolchain, pinned versions, GPU, Triton/FlashInfer JIT, cache locations
make weights     # download the pinned Qwen3-0.6B snapshot
make test        # pytest
make bench-offline BENCH_ARGS="--workload shared --ablate radix --out NAME"   # engine-level benchmark
make bench          # primary benchmark: genai-bench over HTTP, with fairness checklist and sidecar
make bench-roofline # the device's measured copy / read bandwidth and BF16 GEMM rate, the denominators
make gate           # pre-merge regression check against results/<device>/gate_baseline.json (about 90 s)
```

Engine-level benchmarks (`bench/offline.py`) feed requests straight into the engine on an open-loop arrival schedule, alternate the compared settings in one process (A B B A ...), and write one CSV row per run to `results/<device>/` together with a sidecar JSON of the measurement conditions (GPU clocks, power limit and throttle reasons, temperature, versions, commit). A run refuses to start if the GPU is busy, the power configuration is wrong, or the working tree has uncommitted changes.

Requirements on the host: an NVIDIA driver supporting CUDA 12.8 or newer and Docker with the NVIDIA container runtime. JIT and model caches live in the `miniserve-cache` Docker volume, mounted at `/cache`.

Two build settings are fixed on purpose:

- **Compiler baselines**: C/C++ code is built with `-march=x86-64-v2` and CUDA code for `sm_89` only. `-march=native` on a development CPU with AVX-512 produces binaries that fault with illegal instructions on AVX2-only cluster nodes.
- **Cache locations**: FlashInfer (`FLASHINFER_WORKSPACE_BASE`), Triton (`TRITON_CACHE_DIR`), tvm-ffi (`TVM_FFI_CACHE_DIR`) and Hugging Face (`HF_HOME`) caches are redirected out of `$HOME`, which is small on shared clusters. `make env-check` fails if any of them resolves under `$HOME`.

## Correctness

`tests/test_consistency.py` is the correctness anchor: greedy decoding must match Hugging Face transformers token for token (same checkpoint, BF16, same seed). The model in `miniserve/model/qwen3.py` is a plain-PyTorch reference path that mirrors the transformers op order and precision, so its prefill logits are required to be bitwise equal to HF, which is stricter than token equality: a wrong norm weight in one layer can leave 64 greedy tokens unchanged while shifting logits by ~1.9. Optimized paths are checked against the same tests: batched, paged, cached, chunked, graph-replayed and overlapped decoding may diverge from the reference only where its top two logits are within a stated tolerance, and every token they emit is re-checked teacher-forced (details in `docs/design.md`). The whole suite runs against both block backends, and speculative decoding is held to the target model's own greedy output under the same rule.
