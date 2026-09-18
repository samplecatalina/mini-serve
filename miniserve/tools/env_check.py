"""Environment preflight.

Verifies everything the engine needs before any real work starts: pinned package
versions, CUDA toolchain consistency, compiler baselines, cache locations, and
that both runtime JIT paths (Triton and FlashInfer) actually compile and run.

Run with ``python -m miniserve.tools.env_check``. Exits non-zero on the first
report containing a failure, so it can serve as the first step of a batch job.
Each check is also exposed to pytest via ``tests/test_env.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cache locations that must be redirected away from $HOME. JIT artifacts
# written to a small, shared home directory have filled it up before.
CACHE_VARS = (
    "FLASHINFER_WORKSPACE_BASE",
    "TRITON_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "HF_HOME",
)
MARCH_BASELINE = "-march=x86-64-v2"


class CheckFailed(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def pinned_versions() -> dict[str, str]:
    """Exact ``==`` pins from pyproject.toml, the single source of truth."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    pins = {}
    for dep in deps:
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^;\s]+)", dep.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def check_python() -> str:
    v = sys.version_info
    _require((v.major, v.minor) == (3, 12), f"python {v.major}.{v.minor}, expected 3.12")
    return sys.version.split()[0]


def check_pinned_versions() -> str:
    pins = pinned_versions()
    _require(len(pins) > 0, "no exact pins found in pyproject.toml")
    wrong = []
    for name, want in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            wrong.append(f"{name}: missing (want {want})")
            continue
        # Local version labels (e.g. torch 2.9.1+cu128) are checked separately.
        if have.split("+")[0] != want:
            wrong.append(f"{name}: {have} (want {want})")
    _require(not wrong, "; ".join(wrong))
    return ", ".join(f"{k}=={metadata.version(k)}" for k in pins)


def check_torch_cuda() -> str:
    import torch

    _require(torch.version.cuda is not None, "torch is a CPU-only build")
    _require(torch.cuda.is_available(), "torch.cuda.is_available() is False")
    return f"torch CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)"


def check_gpu_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}.{minor}"
    wanted = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    _require(wanted != "", "TORCH_CUDA_ARCH_LIST is not set")
    _require(
        arch in wanted.replace(";", " ").split(),
        f"device is sm_{major}{minor} but TORCH_CUDA_ARCH_LIST={wanted!r}",
    )
    return f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})"


def check_nvcc() -> str:
    import torch

    cuda_home = os.environ.get("CUDA_HOME", "")
    _require(cuda_home != "", "CUDA_HOME is not set")
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    _require(nvcc.is_file(), f"{nvcc} does not exist")
    out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    _require(m is not None, f"cannot parse nvcc version from: {out!r}")
    nvcc_ver = m.group(1)
    _require(
        nvcc_ver == torch.version.cuda,
        f"nvcc {nvcc_ver} != torch CUDA {torch.version.cuda}; JIT kernels would target a different runtime",
    )
    return f"{nvcc} (CUDA {nvcc_ver})"


def check_compile_flags() -> str:
    flags = {k: os.environ.get(k, "") for k in ("CFLAGS", "CXXFLAGS")}
    archs = {k: os.environ.get(k, "") for k in ("TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST")}
    for k, v in flags.items():
        _require(MARCH_BASELINE in v.split(), f"{k}={v!r} does not contain {MARCH_BASELINE}")
    for k, v in {**flags, **archs}.items():
        _require(v != "", f"{k} is not set")
        _require("native" not in v, f"{k}={v!r} targets the build machine (native)")
    return ", ".join(f"{k}={v}" for k, v in {**flags, **archs}.items())


_CXX20_SRC = r"""
#include <version>
#include <concepts>
#include <span>
#include <cstdio>
template <std::integral T> T twice(T x) { return 2 * x; }
int main() {
    int a[3] = {1, 2, 3};
    std::span<int> s(a);
    std::printf("%d\n", twice(s[2]));
    return 0;
}
"""


def check_cxx20() -> str:
    cxx = os.environ.get("CXX", "g++")
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "t.cpp", Path(d) / "t"
        src.write_text(_CXX20_SRC)
        cmd = [cxx, "-std=c++20", *os.environ.get("CXXFLAGS", "").split(), str(src), "-o", str(exe)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        _require(r.returncode == 0, f"{' '.join(cmd)} failed: {r.stderr.strip()[-400:]}")
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.strip()
        _require(out == "6", f"C++20 test program printed {out!r}, expected '6'")
    ver = subprocess.run([cxx, "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    return f"{cxx} {ver}: -std=c++20 compiles and runs"


def check_cache_dirs() -> str:
    home = Path(os.path.expanduser("~")).resolve()
    parts = []
    for var in CACHE_VARS:
        val = os.environ.get(var, "")
        _require(val != "", f"{var} is not set")
        p = Path(val)
        _require(p.is_absolute(), f"{var}={val!r} is not an absolute path")
        p.mkdir(parents=True, exist_ok=True)
        _require(not p.resolve().is_relative_to(home), f"{var}={val} is under $HOME ({home})")
        _require(os.access(p, os.W_OK), f"{var}={val} is not writable")
        parts.append(f"{var}={val}")
    return ", ".join(parts)


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0


def check_triton_jit() -> str:
    import torch

    from miniserve.tools._triton_smoke import add

    x = torch.arange(1000, device="cuda", dtype=torch.float32)
    y = torch.ones_like(x)
    torch.testing.assert_close(add(x, y), x + y)
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    n = _count_files(cache)
    _require(n > 0, f"kernel ran but nothing was written to TRITON_CACHE_DIR={cache}")
    return f"vector add matches torch; {n} files in {cache}"


def check_flashinfer_jit() -> str:
    """Decode attention through FlashInfer's nvcc + host-compiler JIT path.

    Some FlashInfer ops (e.g. rmsnorm) default to CuTe DSL kernels compiled
    in-process, which never invoke nvcc or the host compiler. Decode attention
    is built by nvcc/ninja, which is the path that depends on the toolchain.
    """
    import flashinfer
    import torch

    # Qwen3-0.6B attention shape: 16 query heads sharing 8 KV heads, head_dim 128.
    n_qo, n_kv, hd, kv_len = 16, 8, 128, 257
    q = torch.randn(n_qo, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_len, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_len, n_kv, hd, device="cuda", dtype=torch.bfloat16)
    out = flashinfer.single_decode_with_kv_cache(q, k, v)

    kf = k.float().repeat_interleave(n_qo // n_kv, dim=1)  # [kv_len, n_qo, hd]
    vf = v.float().repeat_interleave(n_qo // n_kv, dim=1)
    scores = torch.einsum("hd,lhd->hl", q.float(), kf) / hd**0.5
    ref = torch.einsum("hl,lhd->hd", scores.softmax(-1), vf)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    cache = Path(os.environ["FLASHINFER_WORKSPACE_BASE"])
    libs = list(cache.rglob("*.so"))
    _require(len(libs) > 0, f"no JIT-built .so under FLASHINFER_WORKSPACE_BASE={cache}")
    return f"GQA decode attention matches reference; {len(libs)} JIT .so under {cache}"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("python", check_python),
    ("pinned versions", check_pinned_versions),
    ("torch cuda", check_torch_cuda),
    ("gpu arch", check_gpu_arch),
    ("nvcc", check_nvcc),
    ("compile flags", check_compile_flags),
    ("c++20", check_cxx20),
    ("cache dirs", check_cache_dirs),
    ("triton jit", check_triton_jit),
    ("flashinfer jit", check_flashinfer_jit),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
            status = "PASS"
        except Exception as e:  # report every check, including unexpected errors
            detail = f"{type(e).__name__}: {e}"
            status = "FAIL"
            failed += 1
        print(f"[{status}] {name:<16} {detail}", flush=True)
    print(f"env-check: {len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
