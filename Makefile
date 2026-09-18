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

.PHONY: help image lock shell env-check weights test bench profile

help:
	@echo "image      build the development image ($(IMAGE))"
	@echo "lock       re-resolve uv.lock after editing pyproject.toml"
	@echo "shell      interactive shell in the development container"
	@echo "env-check  verify toolchain, pinned versions, GPU, JIT paths and caches"
	@echo "weights    download the pinned Qwen3-0.6B snapshot into the cache"
	@echo "test       run pytest (PYTEST_ARGS='-m \"not slow\"' to skip slow tests)"
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

bench profile:
	@echo "make $@: not implemented yet" >&2; exit 1
