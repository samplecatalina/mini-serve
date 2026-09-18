"""A minimal Triton kernel for the environment check.

Triton kernels must live in a real source file (Triton reads the function's
source through ``inspect``), so this cannot be defined inline in a script piped
through stdin.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    _add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    return out
