import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def qwen3_path():
    from miniserve.model.weights import QWEN3_0_6B, model_path

    try:
        return model_path(QWEN3_0_6B, download=False)
    except Exception as e:  # weights are part of the environment: fail, do not skip
        pytest.fail(f"pinned Qwen3-0.6B snapshot not in cache ({e}); run `make weights`")
