"""The pre-merge gate: the rules that decide pass from fail.

The gate's value is that it says no. These cover the cases where it has to,
without a GPU: a drop past the threshold, a caliber that no longer matches the
baseline's, and a gain large enough that the baseline is the stale side.
"""

from __future__ import annotations

from bench import gate


def baseline(tok_s: float, args=None) -> dict:
    return {"tok_s": tok_s, "commit": "abcdef123456", "args": args or gate.ARMS["unique"]}


def measured(tok_s: float, args=None) -> dict:
    return {"tok_s": tok_s, "runs": [tok_s], "spread": 0.0,
            "commit": "0123456789ab", "args": args or gate.ARMS["unique"]}


def test_a_small_drop_passes():
    ok, line = gate.compare("unique", measured(3400.0), baseline(3500.0))
    assert ok and "ok" in line


def test_a_drop_past_the_threshold_fails():
    # 5% is the threshold, so just past it must fail.
    ok, line = gate.compare("unique", measured(3500.0 * 0.949), baseline(3500.0))
    assert not ok and "REGRESSED" in line


def test_the_threshold_itself_is_not_a_failure():
    ok, _ = gate.compare("unique", measured(3500.0 * 0.951), baseline(3500.0))
    assert ok


def test_a_large_gain_fails_so_the_baseline_gets_refreshed():
    ok, line = gate.compare("unique", measured(3500.0 * 1.2), baseline(3500.0))
    assert not ok and "--update" in line


def test_a_different_caliber_is_refused_rather_than_subtracted():
    stale = baseline(3500.0, args=["--workload", "unique", "--rounds", "1"])
    ok, line = gate.compare("unique", measured(3500.0), stale)
    assert not ok and "different arguments" in line


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
