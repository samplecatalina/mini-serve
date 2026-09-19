# Every target runs inside the development image. On the host, commands are
# wrapped in `docker run`; inside the container (MINISERVE_IN_CONTAINER=1) they
# run directly. Override RUN to use another runtime, e.g.
#   make test RUN="apptainer exec --nv --bind \$$PWD:/workspace --bind /path/to/cache:/cache miniserve.sif"

IMAGE       ?= miniserve-dev:latest
# The load generator lives in its own image: the same client must measure every
# engine being compared, and it has no business inside the engine's pinned env.
BENCH_IMAGE ?= miniserve-bench:latest
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

.PHONY: help image bench-image lock shell env-check weights test bench serve profile bench-offline profile-offline bench-roofline bench-decode-sharing ncu-decode-sharing sif

help:
	@echo "image      build the development image ($(IMAGE))"
	@echo "bench-image  build the load generator image ($(BENCH_IMAGE)): genai-bench, no torch"
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
	@echo "bench      the main caliber: start the server, run genai-bench against it (BENCH_ARGS=...)"
	@echo "serve      run the server in the foreground (ENGINE_ARGS=...)"
	@echo "profile    not implemented yet"

image:
	docker build -t $(IMAGE) .

bench-image:
	docker build -f Dockerfile.bench -t $(BENCH_IMAGE) .

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

# The main caliber, in one command: the server in one container, genai-bench in
# another, talking over the loopback of this host. ENGINE_ARGS configures the
# server (an ablation arm); BENCH_ARGS configures the load (see bench/serving.py).
ENGINE_ARGS ?= 
SERVER_NAME ?= miniserve-server
SERVER_PORT ?= 8000

bench:
	@docker rm -f $(SERVER_NAME) >/dev/null 2>&1 || true
	# First, while nothing is loaded: the GPU must be idle and the host in its
	# benchmark state. Once the server is up, "idle" is no longer checkable.
	docker run --rm --gpus all --user $(shell id -u):$(shell id -g) \
		-v $(CURDIR):/workspace -w /workspace -e PYTHONPATH=/workspace \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		$(BENCH_IMAGE) python -m bench.sidecar .preflight.json > /dev/null
	docker run -d --rm --name $(SERVER_NAME) --gpus all --ipc=host --network host \
		--user $(shell id -u):$(shell id -g) -v $(CURDIR):/workspace -v $(CACHE_VOL):/cache -w /workspace \
		-e PYTHONPATH=/workspace $(IMAGE) \
		python -m miniserve.server --host 127.0.0.1 --port $(SERVER_PORT) $(ENGINE_ARGS)
	@trap 'docker rm -f $(SERVER_NAME) >/dev/null 2>&1 || true' EXIT; \
	docker run --rm --gpus all --network host --user $(shell id -u):$(shell id -g) \
		-v $(CURDIR):/workspace -v $(CACHE_VOL):/cache -w /workspace \
		-e MINISERVE_HOST_POWER='$(shell HIGH_PERF_SCHEME=$(HIGH_PERF_SCHEME) bench/host_power.sh)' \
		-e MINISERVE_BENCH_IMAGE="$$(docker image inspect $(BENCH_IMAGE) --format '{{.Id}}')" \
		-e MINISERVE_HOST="$(shell hostname)" \
		$(BENCH_IMAGE) python -m bench.serving --url http://127.0.0.1:$(SERVER_PORT) \
		--preflight-json .preflight.json $(BENCH_ARGS)

serve:
	$(DOCKER_RUN) -p $(SERVER_PORT):$(SERVER_PORT) $(IMAGE) \
		python -m miniserve.server --host 0.0.0.0 --port $(SERVER_PORT) $(ENGINE_ARGS)

profile:
	@echo "make $@: not implemented yet" >&2; exit 1
