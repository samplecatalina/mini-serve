"""The warm-up rule every benchmark harness uses (``bench.sidecar.warm_up``).

A benchmark that stops warming up too early measures a GPU on its way to its
steady clock; one that never stops fails a job. Both have happened: a copy of
this rule that had lost its second criterion refused seven cells of one
cluster job. These cover the two criteria, the time limit, and that there is
one copy.
"""

from __future__ import annotations

import inspect

from bench import blueprint, offline, sglang_spec, sidecar


class FakeGpu:
    """Scripted per-pass mean clocks; ``settled`` answers from a script too."""

    def __init__(self, pass_means, window_settles_at=None):
        self.pass_means = list(pass_means)
        self.window_settles_at = window_settles_at
        self.passes = 0

    def sample_now(self):
        return 0.0

    def settled(self):
        return self.window_settles_at is not None and self.passes >= self.window_settles_at

    def summary(self, since=None, until=None):
        if until is None:
            return {"sm_mhz": {"mean": 0.0}}
        return {"sm_mhz": {"mean": self.pass_means[self.passes - 1]}}


class Clock:
    """Each pass takes ``step`` seconds."""

    def __init__(self, gpu, step):
        self.gpu, self.step = gpu, step

    def __call__(self):
        return self.gpu.passes * self.step


def warm(gpu, min_s=10.0, max_s=100.0, step=4.0, arms=(None,), log=None):
    def run_once(arm):
        gpu.passes += 1
        if log is not None:
            log.append(arm)

    return sidecar.warm_up(gpu, run_once, min_s, max_s, arms=arms, clock=Clock(gpu, step))


def test_the_window_rule_stops_once_the_minimum_time_has_passed():
    gpu = FakeGpu([2000.0] * 10, window_settles_at=1)
    w = warm(gpu)
    # Settled from the first pass, but not before 10 s: passes at 4, 8, 12 s.
    assert w["settled_by"] == "window" and w["runs"] == 3


def test_the_run_rule_settles_a_load_whose_windows_never_agree():
    # Window never settles; pass means 2400, 2300, 2250, 2240. At 12 s the
    # third pass is 2.2% off the second (not settled); the fourth is 0.4% off
    # the third.
    gpu = FakeGpu([2400.0, 2300.0, 2250.0, 2240.0, 2240.0])
    w = warm(gpu)
    assert w["settled_by"] == "run" and w["runs"] == 4


def test_the_run_rule_compares_a_pass_with_the_previous_pass_of_the_same_arm():
    # Two arms alternate; each arm's own means are steady, but the arms differ
    # by 10% from each other, so comparing neighbouring passes would never
    # settle.
    gpu = FakeGpu([2000.0, 2200.0] * 5)
    log = []
    w = warm(gpu, arms=("a", "b"), log=log)
    assert w["settled_by"] == "run"
    assert log == ["a", "b", "a"]  # settled on the second pass of arm "a", at 12 s


def test_the_time_limit_ends_warm_up_without_a_verdict():
    gpu = FakeGpu([2000.0, 2200.0] * 20)
    w = warm(gpu, max_s=20.0)
    assert w["settled_by"] is None and w["seconds"] >= 20.0


def test_every_harness_uses_the_one_rule():
    for module in (offline, sglang_spec, blueprint):
        src = inspect.getsource(module)
        assert "sidecar.warm_up(" in src, module.__name__
        assert "gpu.settled(" not in src, f"{module.__name__} has its own copy of the rule"
