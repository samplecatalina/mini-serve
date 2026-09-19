# Every target runs inside the development image. On the host, commands are
# wrapped in `docker run`; inside the container (MINISERVE_IN_CONTAINER=1) they
# run directly. Override RUN to use another runtime, e.g.
#   make test RUN="apptainer exec --nv --bind \$$PWD:/workspace --bind /path/to/cache:/cache miniserve.sif"

IMAGE       ?= miniserve-dev:latest
CACHE_VOL   ?= miniserve-cache
DOCKER_RUN  := docker run --rm --gpus all --ipc=host \
               --user $(shell id -u):$(shell id -g) \
               -v $(CURDIR):/workspace -v $(CACHE_VOL):/cache -w /workspace

ifeq ($(MINISERVE_IN_CONTAINER),1)
RUN ?=
else
RUN ?= $(DOCKER_RUN) $(IMAGE)
endif

PYTEST_ARGS ?=
BENCH_ARGS  ?=
# Power scheme that counts as high performance for the benchmark preflight.
# The development laptop uses a vendor scheme; elsewhere Windows' own
# "High performance" GUID is the default of bench/host_power.sh.
HIGH_PERF_SCHEME ?= 52521609-efc9-4268-b9ba-67dea73f18b2
# Apptainer image of the development image, for the cluster: built by the official
# apptainer container from a `docker save` of $(IMAGE), so both run the same bytes.
SIF_DIR     ?= $(HOME)/sif
APPTAINER   ?= ghcr.io/apptainer/apptainer@sha256:cfc99015d4af5e4f3f52ce5d5e569ed1e214b4d49a56a2f77b140d5e9cbee595
# Nsight Systems on the host, mounted read-only into the container for profiling.
NSYS_HOST   ?= /opt/nvidia/nsight-systems/2026.1.3
NSYS_OUT    ?= profiling/rtx4060-laptop/offline
NCU_HOST    ?= /opt/nvidia/nsight-compute/2026.2.1
NCU_OUT     ?= profiling/rtx4060-laptop/decode_sharing
NCU_ARGS    ?= --metrics gpu__time_duration.sum,dram__bytes_read.sum,lts__t_sector_hit_rate.pct

.PHONY: help image lock shell env-check weights test bench profile bench-offline profile-offline bench-roofline bench-decode-sharing ncu-decode-sharing sif

help:
	@echo "image      build the development image ($(IMAGE))"
	@echo "lock       re-resolve uv.lock after editing pyproject.toml"
	@echo "shell      interactive shell in the development container"
	@echo "env-check  verify toolchain, pinned versions, GPU, JIT paths and caches"
	@echo "weights    download the pinned Qwen3-0.6B snapshot into the cache"
	@echo "test       run pytest (PYTEST_ARGS='-m \"not slow\"' to skip slow tests)"
	@echo "bench-offline    engine-level benchmark (BENCH_ARGS=...; see bench/offline.py)"
	@echo "profile-offline  the same under Nsight Systems, report in NSYS_OUT"
	@echo "bench-roofline   this GPU's copy bandwidth and BF16 GEMM rate: the denominators"
	@echo "bench-decode-sharing  decode attention with and without shared KV blocks"
	@echo "ncu-decode-sharing    one layout of it under Nsight Compute (BENCH_ARGS='--case shared --iters 3')"
	@echo "sif        convert the development image to an Apptainer image (SIF_DIR) for the cluster"
	@echo "bench      not implemented yet"
	@echo "profile    not implemented yet"

image:
	docker build -t $(IMAGE) .

lock:
	uv lock --python 3.12

shell:
	$(DOCKER_RUN) -it $(IMAGE) bash

env-check:
	$(RUN) python -m miniserve.tools.env_check

weights:
	$(RUN) python -m miniserve.tools.fetch_weights

test:
	$(RUN) python -m pytest $(PYTEST_ARGS)

bench-offline:
	$(DOCKER_RUN) -e PYTHONPATH=/workspace \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		$(IMAGE) python -m bench.offline $(BENCH_ARGS)

profile-offline:
	mkdir -p $(dir $(NSYS_OUT))
	$(DOCKER_RUN) -e PYTHONPATH=/workspace -v $(NSYS_HOST):/opt/nsys:ro -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		$(IMAGE) /opt/nsys/target-linux-x64/nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false \
		--capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
		-o $(NSYS_OUT) -f true python -m bench.offline $(BENCH_ARGS)

bench-roofline:
	$(DOCKER_RUN) -e PYTHONPATH=/workspace \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		$(IMAGE) python -m bench.roofline $(BENCH_ARGS)

bench-decode-sharing:
	$(DOCKER_RUN) -e PYTHONPATH=/workspace \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		$(IMAGE) python -m bench.decode_sharing $(BENCH_ARGS)

ncu-decode-sharing:
	mkdir -p $(dir $(NCU_OUT))
	$(DOCKER_RUN) -e PYTHONPATH=/workspace -v $(NCU_HOST):/opt/ncu:ro $(IMAGE) \
		/opt/ncu/ncu -k regex:BatchDecode $(NCU_ARGS) -o $(NCU_OUT) -f python -m bench.decode_sharing $(BENCH_ARGS)

sif:
	mkdir -p $(SIF_DIR)
	docker save $(IMAGE) -o $(SIF_DIR)/image.tar
	docker run --rm --privileged -v $(SIF_DIR):/work $(APPTAINER) \
		apptainer build -F /work/miniserve.sif docker-archive:///work/image.tar
	rm -f $(SIF_DIR)/image.tar
	docker image inspect $(IMAGE) --format '{{.Id}}' > $(SIF_DIR)/miniserve.sif.docker-id
	sha256sum $(SIF_DIR)/miniserve.sif | tee $(SIF_DIR)/miniserve.sif.sha256

bench profile:
	@echo "make $@: not implemented yet" >&2; exit 1
