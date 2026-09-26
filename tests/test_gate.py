"""The pre-merge gate: the rules that decide pass from fail.

The gate's value is that it says no. These cover the cases where it has to,
without a GPU: a drop past the threshold, a gain large enough that the
reference is the stale side, a session too unsteady to judge, and the order
and checkout that make the comparison fair in the first place.
"""

from __future__ import annotations

import subprocess

import pytest

from bench import gate


def process(tok_s: float, runs=None) -> dict:
    runs = runs or [tok_s, tok_s, tok_s]
    return {"tok_s": tok_s, "runs": runs, "spread": 0.0, "commit": "0123456789ab",
            "args": gate.ARMS["unique"]}


def sides(ref: float, cur: float, ref_second: float | None = None) -> dict:
    """Two reference processes around two current ones, as ORDER runs them."""
    second = ref if ref_second is None else ref_second
    return {"reference": [process(ref), process(second)], "current": [process(cur), process(cur)]}


def test_a_small_drop_passes():
    verdict, line = gate.judge("unique", sides(3500.0, 3400.0))
    assert verdict == "pass" and "ok" in line


def test_a_drop_past_the_threshold_fails():
    # 5% is the threshold, so just past it must fail.
    verdict, line = gate.judge("unique", sides(3500.0, 3500.0 * 0.949))
    assert verdict == "fail" and "REGRESSED" in line


def test_the_threshold_itself_is_not_a_failure():
    verdict, _ = gate.judge("unique", sides(3500.0, 3500.0 * 0.951))
    assert verdict == "pass"


def test_a_large_gain_fails_so_the_reference_gets_refreshed():
    verdict, line = gate.judge("unique", sides(3500.0, 3500.0 * 1.2))
    assert verdict == "fail" and "--update" in line


def test_a_session_that_moves_more_than_half_the_threshold_is_inconclusive():
    # The two reference processes disagree by 3%: a 5% verdict has no margin
    # left, whichever way the current commit lands.
    verdict, line = gate.judge("unique", sides(3500.0, 3300.0, ref_second=3500.0 * 0.97))
    assert verdict == "inconclusive" and "INCONCLUSIVE" in line


def test_the_verdict_uses_every_run_of_each_side():
    # Medians over all runs, not over the per-process medians: one process
    # with an outlier run does not decide it.
    s = {"reference": [process(3500.0, [3490.0, 3500.0, 3510.0]), process(3500.0, [3495.0, 3500.0, 3505.0])],
         "current": [process(3300.0, [3300.0, 3310.0, 3320.0]), process(3310.0, [3290.0, 3310.0, 3330.0])]}
    verdict, line = gate.judge("unique", s)
    assert verdict == "fail" and "3310.0 vs 3500.0" in line


def test_drift_against_the_recorded_number_is_reported_not_judged():
    # The reference reads 4% below what was recorded on another day; the
    # current commit matches the reference, so the verdict is pass.
    verdict, line = gate.judge("unique", sides(3360.0, 3360.0), recorded={"tok_s": 3500.0})
    assert verdict == "pass" and "recorded 3500.0: -4.0%" in line


def test_the_reference_brackets_the_current_commit():
    calls = []

    def fake(arm, out_dir, extra, tree=None):
        calls.append("reference" if tree else "current")
        return process(3500.0)

    got = gate.run_abba("unique", "/tmp/x", [], "/tmp/ref", run=fake)
    assert calls == ["reference", "current", "current", "reference"]
    assert len(got["reference"]) == 2 and len(got["current"]) == 2


def test_the_reference_runs_from_its_own_checkout(monkeypatch, tmp_path):
    seen = {}

    def fake_run(argv, cwd=None, env=None, **kw):
        seen.update(cwd=cwd, pythonpath=(env or {}).get("PYTHONPATH"))
        out = argv[argv.index("--results-dir") + 1]
        (tmp_path / "gate_unique.csv").write_text("output_tok_s,git_commit\n3500.0,abc\n")
        assert out == str(tmp_path)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    m = gate.measure("unique", str(tmp_path), [], tree="/tmp/ref")
    assert seen == {"cwd": "/tmp/ref", "pythonpath": "/tmp/ref"}
    assert m["tok_s"] == 3500.0 and m["commit"] == "abc"


def test_the_checkout_is_removed_even_when_the_measurement_fails(monkeypatch):
    commands = []

    def fake_run(argv, **kw):
        commands.append(argv[:3])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        with gate.checkout("abc123", "/tmp/ref"):
            raise SystemExit("the arm did not complete")
    assert commands == [["git", "worktree", "add"], ["git", "worktree", "remove"]]


def test_both_arms_are_measured():
    # One arm cannot see a prefix-cache regression, the other cannot see a
    # decode regression as clearly; the gate is only as wide as this dict.
    assert set(gate.ARMS) == {"unique", "shared"}
    assert "--workload" in gate.ARMS["shared"]
    assert gate.ARMS["shared"][gate.ARMS["shared"].index("--workload") + 1] == "shared"


def test_the_caliber_is_fixed_not_taken_from_the_command_line():
    import inspect

    src = inspect.getsource(gate.main)
    for flag in ("--workload", "--rounds", "--groups", "--output-len", "--prefix-len"):
        assert f'"{flag}"' not in src, f"{flag} must not be settable on the gate"


def test_uncommitted_changes_are_refused_before_anything_runs(monkeypatch, tmp_path):
    import json

    base = {"device": "dev", "tolerance": gate.TOLERANCE, "reference_commit": "abc",
            "arms": {arm: process(3500.0) | {"args": gate.ARMS[arm]} for arm in gate.ARMS}}
    path = tmp_path / "gate_baseline.json"
    path.write_text(json.dumps(base))

    class Profile:
        results_dir = str(tmp_path)

    monkeypatch.setattr(gate.sidecar, "device_profile", lambda: ("dev", Profile()))
    monkeypatch.setattr(gate, "uncommitted", lambda: True)
    monkeypatch.setattr(gate, "checkout", lambda *a: pytest.fail("nothing may run on dirty code"))
    monkeypatch.setattr(gate.sys, "argv", ["gate"])
    assert gate.main() == 2
