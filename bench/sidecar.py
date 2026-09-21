"""Measurement conditions: a preflight check before a run, GPU sampling during it, and the sidecar record.

Every results row is written together with a sidecar JSON that records the
conditions it was measured under. A run refuses to start when a condition is
not met or a field cannot be read, so no number exists without its context.

Preflight (idle GPU, before any work):

- a known GPU model, which selects the device profile below (results directory
  and the checks that apply to it);
- the power limit at its configured maximum: ``power.max_limit`` 140 W on
  the RTX 4060 Laptop GPU in its high-performance power mode (a misconfigured
  mode shows 80 W). The laptop's dynamic power sharing between CPU and GPU
  moves ``enforced.power.limit`` below the maximum whenever the CPU is busy
  (starting the benchmark process alone pulls it down to about 135 W), so the
  enforced value is recorded, idle and as a trajectory during the run,
  together with the throttle reasons that show whether it ever bound;
- ``utilization.gpu`` and ``memory.used`` low (no other GPU work);
- on the development laptop, the host on AC power and in its high-performance
  power scheme. These live on the Windows side of WSL2 and are passed in by
  the launcher as ``MINISERVE_HOST_POWER`` (JSON); a missing value fails the
  preflight. Cluster GPUs (a whole GPU allocated to the job) have no host
  power state to check and their power limit is only recorded.

Clocks cannot be locked on this GPU, so they are recorded instead: SM and
memory clock, temperature, power draw, power limit and throttle reasons,
sampled in a background thread while the benchmark runs.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import threading
import time

FIELDS = (
    "name,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,enforced.power.limit,"
    "clocks_event_reasons.active,utilization.gpu,memory.used"
)

MAX_IDLE_UTIL = 10  # percent


@dataclasses.dataclass(frozen=True)
class DeviceProfile:
    results_dir: str
    power_max_limit_w: float | None  # asserted when set
    host_power: bool  # a laptop under WSL2: check the Windows power state
    max_idle_mem_mib: int


DEVICES = {
    # the Windows desktop holds some GPU memory
    "NVIDIA GeForce RTX 4060 Laptop GPU": DeviceProfile("results/rtx4060-laptop", 140.0, True, 2048),
    "NVIDIA L40S": DeviceProfile("results/l40s", None, False, 1024),
    "NVIDIA H100 80GB HBM3": DeviceProfile("results/h100", None, False, 1024),
}


def device_profile() -> tuple[str, DeviceProfile]:
    name = sample_gpu().name
    if name not in DEVICES:
        raise PreflightError(f"GPU {name!r} has no device profile (known: {sorted(DEVICES)})")
    return name, DEVICES[name]


class PreflightError(RuntimeError):
    pass


@dataclasses.dataclass
class GpuSample:
    t: float
    name: str
    sm_mhz: int
    mem_mhz: int
    temp_c: int
    power_w: float
    power_limit_w: float
    reasons: str
    util: int
    mem_used_mib: int


def sample_gpu() -> GpuSample:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout.splitlines()[0]
    f = [x.strip() for x in out.split(",")]
    if len(f) != 9:
        raise PreflightError(f"unexpected nvidia-smi output: {out!r}")
    return GpuSample(
        time.monotonic(), f[0], int(f[1]), int(f[2]), int(f[3]), float(f[4]), float(f[5]), f[6], int(f[7]), int(f[8])
    )


def host_power() -> dict:
    raw = os.environ.get("MINISERVE_HOST_POWER")
    if not raw:
        raise PreflightError("MINISERVE_HOST_POWER is not set (run through `make bench-offline`)")
    return json.loads(raw)


def _max_power_limit() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.max_limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    return float(out.split()[0])


def preflight(num_samples: int = 6, interval_s: float = 0.5) -> dict:
    """Check the conditions a measurement needs; return them for the sidecar. Raises PreflightError."""
    idle = []
    for k in range(num_samples):
        if k:
            time.sleep(interval_s)
        idle.append(sample_gpu())
    s = idle[-1]
    _, prof = device_profile()
    max_limit = _max_power_limit()
    host = host_power() if prof.host_power else None
    problems = []
    if prof.power_max_limit_w is not None and abs(max_limit - prof.power_max_limit_w) > 0.5:
        problems.append(f"power.max_limit is {max_limit} W, expected {prof.power_max_limit_w} W")
    if max(x.util for x in idle) > MAX_IDLE_UTIL:
        problems.append(f"GPU utilization up to {max(x.util for x in idle)}% before the run (another GPU workload?)")
    if s.mem_used_mib > prof.max_idle_mem_mib:
        problems.append(f"{s.mem_used_mib} MiB of GPU memory in use before the run")
    if host is not None and not host.get("ac_power"):
        problems.append("host is not on AC power")
    if host is not None and not host.get("high_performance"):
        problems.append(f"host power scheme is {host.get('power_scheme')!r}, not the high-performance one")
    if problems:
        raise PreflightError("; ".join(problems))
    return dict(
        idle_gpu=dataclasses.asdict(s),
        idle_power_limit_w=[x.power_limit_w for x in idle],
        power_max_limit_w=max_limit,
        host=host,
    )


class GpuSampler:
    """Samples nvidia-smi in a background thread while the GPU is loaded."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            try:
                s = sample_gpu()
            except Exception:  # a failed sample is skipped, not fatal
                s = None
            if s is not None:
                with self._lock:
                    self.samples.append(s)
            self._stop.wait(self.interval_s)

    def sample_now(self) -> float:
        """Take one sample right away (a window's boundaries); returns its time."""
        s = sample_gpu()
        with self._lock:
            self.samples.append(s)
        return s.t

    def snapshot(self) -> list[GpuSample]:
        with self._lock:
            return sorted(self.samples, key=lambda x: x.t)

    def settled(self, window: int = 5, tol: float = 0.02) -> bool:
        """Mean SM clock of the last ``window`` samples within ``tol`` of the ``window`` before them."""
        s = self.snapshot()
        if len(s) < 2 * window:
            return False
        earlier = sum(x.sm_mhz for x in s[-2 * window : -window]) / window
        recent = sum(x.sm_mhz for x in s[-window:]) / window
        return recent > 0 and abs(recent - earlier) <= tol * recent

    def summary(self, since: float | None = None, until: float | None = None) -> dict:
        s = [x for x in self.snapshot() if (since is None or x.t >= since) and (until is None or x.t <= until)]
        if not s:
            raise PreflightError("no GPU samples in the measured window")

        def stats(vals):
            return dict(min=min(vals), max=max(vals), mean=round(sum(vals) / len(vals), 2))

        return dict(
            num_samples=len(s),
            sm_mhz=stats([x.sm_mhz for x in s]),
            mem_mhz=stats([x.mem_mhz for x in s]),
            temp_c=stats([x.temp_c for x in s]),
            power_w=stats([x.power_w for x in s]),
            power_limit_w=stats([x.power_limit_w for x in s]),
            clocks_event_reasons=sorted({x.reasons for x in s}),
        )


def harness_commit() -> tuple[str, bool]:
    """Which commit of this harness is running, and whether it was modified.

    Normally git answers. The blueprint's own image, which runs this harness
    for the second baseline, does not carry git, so its launcher passes the
    answer in as ``MINISERVE_GIT``. An image with neither would write results
    that cannot be traced back to code, so it does not get to start.
    """

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except FileNotFoundError:
        raw = os.environ.get("MINISERVE_GIT")
        if not raw:
            raise PreflightError("no git in this image and no MINISERVE_GIT: "
                                 "the commit that produced the run cannot be recorded")
        g = json.loads(raw)
        return g["commit"], bool(g["dirty"])


def environment() -> dict:
    """Software and host description for the sidecar."""
    import flashinfer
    import torch

    commit, dirty = harness_commit()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "unknown")
    return dict(
        git_commit=commit,
        git_dirty=dirty,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        driver=subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip(),
        cpu=cpu,
    )


REQUIRED = ("run_id", "config", "environment", "preflight", "warmup", "gpu_during_runs")


def write_sidecar(path: str, record: dict) -> None:
    missing = [k for k in REQUIRED if record.get(k) in (None, {}, [])]
    if missing:
        raise PreflightError(f"sidecar is missing {missing}; refusing to write results")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def main() -> int:
    """``python -m bench.sidecar [path]``: check the conditions and record them.

    A benchmark that starts its own server cannot run this itself: by the time
    the server answers, the model is on the GPU and "the GPU is idle" is no
    longer checkable. So it runs first, on its own, and the run that follows is
    handed what it found.
    """
    import argparse

    ap = argparse.ArgumentParser(description="preflight the GPU and print what it found as JSON")
    ap.add_argument("out", nargs="?", help="write here as well as to stdout")
    args = ap.parse_args()
    record = preflight()
    text = json.dumps(record)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
