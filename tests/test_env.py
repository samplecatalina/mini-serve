import pytest

from miniserve.tools.env_check import CHECKS

_SLOW = {"triton jit", "flashinfer jit"}


@pytest.mark.gpu
@pytest.mark.parametrize(
    "check",
    [pytest.param(fn, id=name, marks=[pytest.mark.slow] if name in _SLOW else []) for name, fn in CHECKS],
)
def test_env(check):
    check()
