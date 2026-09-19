"""The denominators: this GPU's achievable copy bandwidth and BF16 GEMM rate.

Every "percent of peak" in this repository divides by a number measured here,
not by a vendor specification. Two quantities are enough for the engine:

- **copy bandwidth**, because decoding is bandwidth bound. A decode step reads
  the weights once and the KV cache of every running sequence once, so the step
  time has a floor of (bytes per step) / (bandwidth this GPU actually reaches).
  A device-to-device copy is the closest cheap proxy: it streams two large
  buffers with no reuse, which is what the weight and KV reads do.
- **BF16 GEMM rate**, because prefill is compute bound. It sets the floor under
  time to first token.

Both are measured the way the engine is measured: the same preflight, the same
warmup-until-the-clocks-settle rule, CUDA events for timing, the median over
repeats, and a sidecar recording the conditions. Run:

    make bench-roofline                        # results/<device>/roofline.csv

The sizes are chosen to fit an 8 GB card with a desktop on it, so the same
command produces comparable rows on every device in bench/sidecar.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from typing import Callable

import torch

from bench import sidecar

# Vendor figures, recorded only so a row can state how much of the nominal
# bandwidth was reached. They are never a denominator; the measured value is.
SPEC_MEM_GB_S = {
    "NVIDIA GeForce RTX 4060 Laptop GPU": 256.0,  # 128-bit at 16 Gbps
    "NVIDIA L40S": 864.0,
    "NVIDIA H100 80GB HBM3": 3352.0,
}

COPY_MIB = (64, 256, 1024)
GEMM_N = (2048, 4096, 8192)


def time_median(fn, iters: int) -> float:
    """Median of ``iters`` timings of ``fn``, in milliseconds, by CUDA events."""
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for k in range(iters):
        start[k].record()
        fn()
        end[k].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(start, end))


def warmup(gpu: sidecar.GpuSampler, fn, min_s: float, max_s: float) -> dict:
    """Load the GPU with ``fn`` until the SM clock reading settles, once, before
    any case is timed. Clocks cannot be locked on either device, so the rule is
    to discard a window of the same length everywhere rather than to fix one.

    Returns what the window looked like, for the sidecar.
    """
    t0 = time.monotonic()
    while True:
        fn()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        if elapsed >= max_s or (elapsed >= min_s and gpu.settled()):
            break
    return dict(seconds=round(elapsed, 1), settled=gpu.settled(), min_s=min_s, max_s=max_s)


def copy_case(mib: int, iters: int) -> tuple[dict, Callable[[], None]]:
    """A device-to-device copy of ``mib`` MiB: it reads that much and writes it."""
    n = mib * 1024 * 1024 // 2  # bfloat16 elements
    src = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    moved = 2 * src.numel() * src.element_size()  # read + write

    def run():
        dst.copy_(src, non_blocking=True)

    return dict(kernel="copy", dtype="bfloat16", size=f"{mib}MiB", bytes=moved, flops=0, iters=iters), run


def gemm_case(n: int, iters: int) -> tuple[dict, Callable[[], None]]:
    """A square BF16 matmul, large enough to be tensor-core bound."""
    a = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(n, n, dtype=torch.bfloat16, device="cuda")
    flops = 2 * n**3

    def run():
        torch.matmul(a, b, out=out)

    return dict(kernel="gemm", dtype="bfloat16", size=str(n), bytes=0, flops=flops, iters=iters), run


def measure(gpu: sidecar.GpuSampler, case, settle_iters: int) -> dict:
    """Time one case. The clocks are already settled; this only fills the caches
    and lets the allocator hand out the buffers before the timed repeats."""
    row, run = case
    for _ in range(settle_iters):
        run()
    torch.cuda.synchronize()
    t_start = gpu.sample_now()
    ms = time_median(run, row["iters"])
    t_end = gpu.sample_now()
    row = dict(row, ms_median=round(ms, 4))
    row["gb_s"] = round(row["bytes"] / (ms * 1e-3) / 1e9, 1) if row["bytes"] else ""
    row["tflop_s"] = round(row["flops"] / (ms * 1e-3) / 1e12, 2) if row["flops"] else ""
    row["_window"] = gpu.summary(since=t_start, until=t_end)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=50, help="timed repeats per case; the median is reported")
    ap.add_argument("--warmup-min-s", type=float, default=30.0, help="the L40S needs about 30 s to settle")
    ap.add_argument("--warmup-max-s", type=float, default=90.0)
    ap.add_argument("--settle-iters", type=int, default=10, help="untimed repeats before each case")
    ap.add_argument("--out", default="roofline", help="results/<device>/<out>.csv and a sidecar per run id")
    ap.add_argument("--results-dir", default=None, help="default: the device profile's (bench/sidecar.py)")
    ap.add_argument("--allow-dirty", action="store_true", help="development only: results from uncommitted code")
    args = ap.parse_args()

    pre = sidecar.preflight()  # before anything is on the GPU
    name, profile = sidecar.device_profile()
    results_dir = args.results_dir or profile.results_dir
    env = sidecar.environment()
    if env["git_dirty"] and not args.allow_dirty:
        raise SystemExit("the working tree has uncommitted changes; results would not be reproducible")

    run_id = time.strftime("%Y%m%dT%H%M%S")
    rows: list[dict] = []
    with sidecar.GpuSampler() as gpu:
        # One warmup window for the whole run, on the heaviest case, so every
        # case below is timed at the same settled clock.
        _, heat = gemm_case(max(GEMM_N), args.iters)
        window = warmup(gpu, heat, args.warmup_min_s, args.warmup_max_s)
        del heat
        for mib in COPY_MIB:
            rows.append(measure(gpu, copy_case(mib, args.iters), args.settle_iters))
        for n in GEMM_N:
            rows.append(measure(gpu, gemm_case(n, args.iters), args.settle_iters))
        gpu_during = gpu.summary()

    spec = SPEC_MEM_GB_S.get(name)
    for row in rows:
        row["spec_mem_gb_s"] = spec if row["kernel"] == "copy" else ""
        row["pct_of_spec"] = round(100 * row["gb_s"] / spec, 1) if row["kernel"] == "copy" and spec else ""
        row["run_id"] = run_id
        row["device"] = name
        row["git_commit"] = env["git_commit"][:12]

    # The denominator is the largest copy, not the fastest one. A buffer near the
    # L2 cache's size is partly served by L2 and reads high; a decode step streams
    # over a gigabyte of weights and KV per step, so the large-buffer figure is the
    # one that bounds it. The smaller sizes are kept as rows: the gap between them
    # is the L2's contribution, and it is worth seeing.
    copies = [r for r in rows if r["kernel"] == "copy"]
    best_copy = max(copies, key=lambda r: r["bytes"])
    # The same rule for the GEMM rate, for the same reason and one more: on a
    # power-limited card the largest problem is the one that runs long enough to
    # hit the cap. The fastest size is reported beside it, because the gap between
    # them is how much of the quoted rate is a burst.
    gemms = [r for r in rows if r["kernel"] == "gemm"]
    best_gemm = max(gemms, key=lambda r: r["flops"])
    burst_gemm = max(gemms, key=lambda r: r["tflop_s"])

    os.makedirs(results_dir, exist_ok=True)
    path = f"{results_dir}/{args.out}.csv"
    fields = [
        "run_id", "device", "kernel", "dtype", "size", "iters", "ms_median",
        "bytes", "flops", "gb_s", "tflop_s", "spec_mem_gb_s", "pct_of_spec", "git_commit",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})

    sidecar.write_sidecar(
        f"{results_dir}/{args.out}.{run_id}.sidecar.json",
        dict(
            run_id=run_id,
            config=vars(args) | dict(copy_mib=list(COPY_MIB), gemm_n=list(GEMM_N)),
            environment=env,
            preflight=pre,
            warmup=window,
            gpu_during_runs=gpu_during,
            per_case_gpu={r["kernel"] + "/" + r["size"]: r["_window"] for r in rows},
            best=dict(
                copy_gb_s=best_copy["gb_s"], copy_size=best_copy["size"],
                gemm_tflop_s=best_gemm["tflop_s"], gemm_n=best_gemm["size"],
                gemm_burst_tflop_s=burst_gemm["tflop_s"], gemm_burst_n=burst_gemm["size"],
            ),
        ),
    )

    print(f"{name}  (rows appended to {path})")
    for row in rows:
        rate = f"{row['gb_s']} GB/s" if row["kernel"] == "copy" else f"{row['tflop_s']} TFLOP/s"
        pct = f"  {row['pct_of_spec']}% of {spec} GB/s spec" if row["pct_of_spec"] != "" else ""
        print(f"  {row['kernel']:5s} {row['size']:>7s}  {row['ms_median']:9.4f} ms  {rate:>16s}{pct}")
    print(f"\ndenominators (largest problem of each, the one that runs long enough to be bound by the hardware):"
          f"\n  copy      {best_copy['gb_s']} GB/s at {best_copy['size']}"
          f"\n  BF16 GEMM {best_gemm['tflop_s']} TFLOP/s at n={best_gemm['size']}"
          f"  (fastest size: {burst_gemm['tflop_s']} TFLOP/s at n={burst_gemm['size']})")
    print(json.dumps(gpu_during["sm_mhz"] | {"reasons": gpu_during["clocks_event_reasons"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
