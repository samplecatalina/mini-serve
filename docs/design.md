# mini-serve — Design

A single-GPU LLM inference engine built as a reproduction, with additions, of
[mini-sglang](https://github.com/sgl-project/mini-sglang). The features that
mini-sglang already ships (overlap scheduling, chunked prefill, radix cache) are
re-implemented here and then measured with ablations. Two components are new
relative to the blueprint: a C++ scheduling core and two-model speculative
decoding.

## Goals

- OpenAI-compatible API subset (`/v1/completions`, `/v1/chat/completions`, streaming SSE)
- Continuous batching over a paged KV cache
- Radix prefix cache with reference counting and LRU eviction
- Chunked prefill
- CPU-GPU overlap scheduling (one-step-lagged metadata assembly on a separate CUDA stream)
- CUDA Graph capture for decode steps, bucketed by batch size
- Sampling: greedy, temperature, top-p, executed on the GPU without host synchronization
- C++ core (`minicore/`): block allocator and radix tree behind pybind11, validated against the Python reference implementation with the same test suite
- Two-model speculative decoding (draft: Qwen3-0.6B, target: Qwen3-8B), greedy exact-match first, rejection sampling second

## Non-goals

- Tensor parallelism. Single-GPU is the premise: dropping TP complexity is what buys room for scheduling depth.
- Quantization, multi-model or multi-tenant serving, production-grade fault tolerance.
- Custom attention kernels. FlashInfer is used for paged attention, the same backend as the blueprint, so comparisons stay fair.
- Volta (sm_70). No BF16 and no FlashInfer / FlashAttention-2 support; porting would change both the dtype and the attention backend, so the comparison would no longer be of the same system.
- Full EAGLE. Public EAGLE-3 draft heads exist for Qwen3-8B, but wiring feature-level drafting and tree verification is close to a second model runner. Two-model drafting is the deliverable; an EAGLE-3 comparison is optional follow-up work.

## Correctness anchor

Greedy output must match HF `transformers` token for token (same BF16 weights, same seed). Every feature that touches the forward path or the cache (batching, paged KV, radix cache, chunked prefill, CUDA Graphs, overlap, the C++ core) is required to leave greedy output unchanged, both for a single request and under a mixed concurrent load. The anchor is re-run on each platform the engine runs on.

Two levels of strictness apply. The plain-PyTorch reference model mirrors the transformers op order, so its prefill logits must be bitwise equal to HF. Paths that change GEMM shapes or kernels (batching, FlashInfer attention, fused norms) cannot be bitwise equal in BF16: even incremental decoding versus a full forward pass of the same sequence differs by up to 0.34 in logits, enough to flip argmax where the top two logits are one BF16 ulp apart. For those paths a sequence may diverge from the reference only at a position where the reference top-1/top-2 logit gap is at most ε; comparison of that sequence stops there. Divergence at a larger gap is a bug. Reports state ε and the divergence rate. Because that rule says nothing about tokens after the first divergence, every token a batched path produces is also checked teacher-forced: running the reference model on the path's own output, each chosen token's logit must be within ε of the reference maximum. With batching alone, ε = 0.5 (measured on 64 sequences of 128 tokens: largest gap at a divergence 0.25, most divergences at exact BF16 ties). In mutation tests, a decode-position off-by-one passed the first-divergence rule on every sequence and failed the teacher-forced check. With FlashInfer paged attention (16-token pages) ε stays 0.5: a single request is no longer bitwise equal to the reference, divergences occur at reference gaps of at most 0.125, and teacher-forced margins stay within 0.25. One bug class is nearly invisible at the token level: shifting every decode position by one looks to a RoPE model like an extra gap between prompt and output, so positions fed to the model are also checked exactly.

## Blueprint

Pinned at [`sgl-project/mini-sglang@9a91cfa`](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2) (2026-05-17). All comparisons and module-by-module notes refer to this commit.

Where this engine deliberately differs:

| Area | mini-sglang | mini-serve | Why |
|---|---|---|---|
| Process model | API server, tokenizer, detokenizer and per-rank scheduler processes over ZMQ | One process: asyncio server, engine thread, tokenizer thread pool | No tensor parallelism, so multiple processes only add IPC |
| KV page size | 1 (its FlashInfer path asserts `page_size == 1`) | 16 by default, configurable | Smaller page tables and radix trees; requires real paged indices for FlashInfer |
| Admission | Reserves `prompt + max_tokens` KV slots up front; no preemption | Admission against a KV budget with preemption and pluggable policies (FCFS, shortest-first, cache-aware) | Lets 256 concurrent requests share a small pool, and makes scheduling policy an ablation axis |
| KV pool sizing | 90% of free memory minus weights | Peak activation profiled, plus an explicit `--kv-pool-tokens` override | Pool size is an experiment variable |
| Allocator / radix tree | Python, GPU-tensor free list, full-tree scan on each eviction | Python reference first, then a C++ core with the same tests | Data-structure hot paths moved to C++ |
| Sampling | FlashInfer sampling kernels; random numbers drawn by row position in the batch; greedy rows in a mixed batch approximated by temperature 1e-6 | Gumbel-max with noise hashed from (request seed, token position, vocabulary index); greedy rows always take `argmax` | A request samples the same tokens wherever it sits in a batch and after preemption; greedy output never depends on its batch neighbours |
| Reference model | Fused FlashInfer ops from the start | Bitwise-HF reference path first; fused ops compared against it | Correctness anchor above |

CUDA Graph batch-size buckets differ by default between mini-sglang (up to 160), sglang 0.5.10 (up to 32) and this engine; comparisons set them explicitly and record them in the fairness checklist.

## Architecture

**Single process.** mini-sglang splits API server, tokenizer, detokenizer and one scheduler per GPU rank into separate processes over ZMQ, which exists to serve tensor-parallel ranks. Without TP that structure leaves only IPC overhead, so this engine collapses to one process: a uvicorn/FastAPI event loop for HTTP and SSE, an engine thread running the `step()` loop (schedule → forward → sample → postprocess), and a tokenizer thread pool. The known cost is GIL contention between HTTP, tokenization and Python scheduling. The tokenizer interface is replaceable by a subprocess if profiling shows it starving the engine loop; this is expected to matter most on hosts with weak single-thread CPUs.

```
miniserve/
├── server/        FastAPI, OpenAI protocol adapter, SSE
├── engine/        scheduler.py, model_runner.py, sampler.py
├── cache/         Python reference BlockAllocator / RadixTree
├── minicore/      C++ core (pybind11) + gtest
├── spec/          speculative decoding
├── bench/         benchmark wrappers, ablation scripts, fairness checklist, environment sidecar
└── tests/         unit / consistency / stress
```

Core types: `Request` (prompt ids, sampling params, state machine WAITING → PREFILL → DECODE → FINISHED/ABORTED, plus DECODE → WAITING on preemption), `Batch` (requests in this step plus block tables and positions), `BlockTable` (request → physical blocks).

## Key decisions

- **KV block size** defaults to 16 (configurable to 1). The radix tree is built at block granularity; a little prefix hit rate is traded for bounded metadata.
- **KV pool size is a first-class experiment variable.** The pool is auto-sized at startup (after loading weights, run one prefill of `--max-prefill-tokens` tokens to measure peak activations, give 90% of the remaining free memory to KV, derive block count from the model config), but `--kv-pool-tokens N` sets it exactly. A size that does not fit in memory is an error rather than silently clipped, since the value is reported as an experiment coordinate. On a large GPU the pool is big enough that scheduling policies converge; sweeping the pool size turns that hardware limit into a controlled axis.
- **Admission against a block budget, preemption by recomputation.** A waiting request of `L` tokens is admitted only if the pool can hold its prefill plus one decode token, after setting aside the blocks every running request needs for its next token; admission stops at the first request that does not fit. So right after any admission the next decode step fits. When a later decode step does not, the most recently admitted requests are preempted until it does: their blocks are freed, their generated tokens kept, and they return to the front of the waiting queue. On readmission the prefill runs prompt + generated tokens, whose last-position logits are exactly those of the interrupted decode step. This cannot deadlock: requests that could not fit in the whole pool are rejected on submission, and the oldest running request is never preempted, so it always makes progress. mini-sglang instead reserves `prompt + max_tokens` for every request and never preempts, which wastes the reservation of every request that stops early.
- **Sampling stays on the device, and is reproducible per request.** Greedy, temperature and top-p sampling run without host synchronization; the sampled tokens are copied back once per step with a non-blocking copy and an event, the only point where the host waits for the device after sampling. Sampled rows use the Gumbel-max trick (`argmax(logits / T + g)`, with tokens outside the nucleus masked out first); the noise `g` is not drawn from a stateful generator but hashed from the request's seed, the position of the token being sampled, and the vocabulary index. Batch composition, a request's row in the batch, and preemption with recomputation therefore do not change which noise a request sees. The price is a sort for top-p and `[batch, vocab]` temporaries, which the KV pool sizing accounts for.
- **Scheduling policy is a plugin** (`SchedulePolicy`): FCFS, shortest-job-first, cache-aware. Policies stay in Python; data structures and hot paths move to C++.
- **Overlap scheduling** follows the one-step-lag scheme: while the GPU runs step *t*, the CPU assembles step *t+1*; the dependency on step *t*'s sampled tokens is resolved with placeholder tensors filled in on the GPU. `--disable-overlap` exists from day one.
- **CUDA Graphs** are captured for decode only, one graph per batch-size bucket, with padding up to the bucket. No `.item()` / `.cpu()` in the model runner. When comparing against another engine, graph bucket configurations are aligned and recorded.
- **Environment is a build artifact.** The Dockerfile is the development environment; the same image is converted to an Apptainer SIF for cluster runs. Host compilers on shared clusters vary by node type, and this project depends on three layers of JIT (Triton, FlashInfer, and the baseline's sgl-kernel), so the toolchain travels with the project. JIT caches are redirected out of `$HOME` and pre-warmed. Every cluster job starts with a preflight assertion.
- **No `-march=native`, no `-arch=native`.** The C++ core and any nvcc invocation pin an explicit CPU baseline and `sm_89`, because a `.so` built on a node with AVX-512 crashes with SIGILL on a node without it.

## Platforms

| Platform | Role |
|---|---|
| RTX 4060 Laptop (Ada, 8 GB) | Development, unit tests, correctness, small-scale profiling. Secondary benchmark numbers, reported with the 8 GB caveat. |
| L40S 48 GB (Ada) | Primary benchmark: 256 concurrent sequences fit in the KV pool, ablations, comparison against mini-sglang and sglang. |
| H100 80 GB (Hopper) | Second data point: larger model, different CPU:GPU ratio, hardware counters. |

L40S and the 4060 share the same SM architecture constants, so code written locally runs on the cluster without architectural adaptation. Their CPU:GPU balance is very different, which is exactly what makes the overlap scheduling ablation informative across platforms.

## Measurement

- Benchmark harness: genai-bench. Reported: TTFT p50/p95, TPOT / inter-token latency p50/p99, output throughput. Both a low-concurrency and a high-concurrency operating point are reported.
- Every number is given three ways: absolute, relative to the baseline, and as a fraction of the hardware ceiling.
- Warmup is by time, not by request count: the first 30 s of samples are discarded so that clocks have settled. Both engines under comparison discard the same window.
- Fairness checklist for any comparison: pinned baseline commit and container image digest, dtype, attention backend and version, KV pool size, `max_num_batched_tokens`, CUDA Graph configuration, sampling parameters, CPU model, warmup window, ABBA alternation within the same job, median of ≥3 rounds.
- Clock, memory clock, temperature, power draw, power limit, GPU utilization and memory in use are recorded before and after each run in a sidecar file; the harness refuses to write results without it.
- Chunked prefill is evaluated on decode inter-token latency p99, not on peak memory, because with a pre-allocated KV pool peak memory does not change.
- Negative results are reported as measured.

## Roadmap

| Stage | Deliverable | Done when |
|---|---|---|
| M0 | Skeleton: environment, weight loading, single-request greedy generation | Token-exact match with HF transformers |
| M1 | Continuous batching, paged KV, FlashInfer attention, API | 64 mixed concurrent requests, no OOM, output unchanged |
| M2 | Radix cache, chunked prefill | Hit rate > 0, output unchanged |
| M3 | CUDA Graphs, overlap scheduling, scheduling policies | Ablation switches work; CPU gaps visibly narrow in the nsys timeline |
| M3.5 | Cluster port: container, preflight, end-to-end service on L40S | Greedy output identical across platforms |
| M4 | Benchmarks and three ablations on L40S; local regression gate | Every plotted line traces to a row in `results/` |
| M5 | C++ core; two-model speculative decoding | C++ backend passes the same tests as the Python reference; spec decoding reported with acceptance rate and large-batch regression |

Status: M0 not started. Numbers will appear in `results/` and be discussed in `optimization-log.md` as they are measured.
