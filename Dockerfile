# Development image: the same image is used for local development (docker, with
# the repository bind-mounted at /workspace) and, converted to a SIF, for runs on
# the cluster (apptainer). Keeping the host toolchain inside the image matters
# because FlashInfer and Triton compile code at runtime with the host gcc/nvcc.
#
# Layers go from most stable to most volatile.

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

# 1. System toolchain. python3.12-dev provides Python.h, which the Triton
#    launcher needs when it JIT-compiles its C stub.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. uv, same version as used to produce uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv

# 3. Python dependencies from the lockfile. The environment lives outside the
#    bind-mounted source tree so host and container environments never mix.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /opt/miniserve-deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group dev \
    && rm -rf /root/.cache/uv

# 4. Runtime configuration.
#    - All JIT and model caches live under /cache, never under $HOME. Locally
#      /cache is a named volume; on the cluster it is bound to project storage.
#    - Compiler baselines are pinned explicitly. -march=native would emit
#      AVX-512 on the development CPU, which faults on AVX2-only cluster nodes.
RUN mkdir -p /cache/flashinfer /cache/triton /cache/tvm-ffi /cache/huggingface /tmp/home \
    && chmod -R 1777 /cache /tmp/home
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/home \
    MINISERVE_CACHE=/cache \
    FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    TRITON_CACHE_DIR=/cache/triton \
    TVM_FFI_CACHE_DIR=/cache/tvm-ffi \
    HF_HOME=/cache/huggingface \
    CFLAGS="-march=x86-64-v2" \
    CXXFLAGS="-march=x86-64-v2" \
    TORCH_CUDA_ARCH_LIST=8.9 \
    FLASHINFER_CUDA_ARCH_LIST=8.9 \
    MINISERVE_IN_CONTAINER=1

WORKDIR /workspace
