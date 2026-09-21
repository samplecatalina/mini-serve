"""A short benchmark that a change has to clear before it is merged.

The point of a gate is that it measures the same thing every time, so the
workload is written down here rather than taken from the command line: a
caliber that can be adjusted per run cannot detect anything. The baseline it
compares against lives in ``results/<device>/gate_baseline.json`` next to every
other number this project quotes, and records the commit it came from and the
exact arguments that produced it. If those arguments no longer match, the gate
refuses to run instead of subtracting two numbers that were never comparable.

Two arms, because one does not cover the engine:

- ``unique`` -- 256 requests with no shared prefix. Decode-bound, and the arm
  whose run-to-run spread is known: four separate processes of three runs each
  on one commit stayed inside 0.81% (``results/rtx4060-laptop/m5_1_block_backend.csv``),
  so the 5% threshold sits about 18 standard deviations out. An alarm here is
  almost certainly real, which is what a gate a person runs by hand needs.
- ``shared`` -- the same requests behind a shared prefix, so that the prefix
  cache is on the measured path. Without it a regression in the radix tree
  costs nothing the gate can see.

The gate fails on a gain above the threshold as well. A baseline that is only
ever allowed to be met drifts downwards in usefulness: once throughput has
improved and the baseline has not, the next real regression falls from a
higher place and still clears the line.

    python -m bench.gate            # check the working tree against the baseline
    python -m bench.gate --update   # record the current numbers as the baseline

Runs are written to a temporary directory, not to ``results/``: only the
baseline belongs in the record.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

from bench import sidecar

# The caliber. Shared by both arms; 256 requests, a run of about seven seconds.
# Taken from the block-backend comparison, whose repeated runs are what the
# threshold below is justified against.
COMMON = [
    "--groups", "16",
    "--per-group", "16",
    "--suffix-len", "32",
    "--output-len", "96",
    "--rounds", "3",
]

ARMS = {
    # No prefix is shared, so every request pays its own prefill.
    "unique": COMMON + ["--workload", "unique", "--prefix-len", "96"],
    # Sixteen groups of sixteen requests behind a 512-token shared prefix: the
    # second request of each group onwards should find it in the prefix cache.
    "shared": COMMON + ["--workload", "shared", "--prefix-len", "512"],
}

# Five percent: far enough outside the measured run-to-run spread that an alarm
# is worth acting on, close enough that a real regression cannot hide under it.
TOLERANCE = 0.05

BASELINE_NAME = "gate_baseline.json"


def measure(arm: str, out_dir: str, extra: list[str]) -> dict:
    """Run one arm and return its rows' throughput, or raise on a failed run."""
    argv = [sys.executable, "-m", "bench.offline", *ARMS[arm], *extra,
            "--results-dir", out_dir, "--out", f"gate_{arm}"]
    print(f"--- {arm}: {' '.join(ARMS[arm])}", flush=True)
    r = subprocess.run(argv)
    if r.returncode != 0:
        raise SystemExit(f"gate: the {arm} arm did not complete (exit {r.returncode})")
    with open(f"{out_dir}/gate_{arm}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    tok_s = sorted(float(row["output_tok_s"]) for row in rows)
    if not tok_s:
        raise SystemExit(f"gate: the {arm} arm produced no rows")
    return {
        "tok_s": round(statistics.median(tok_s), 1),
        "runs": tok_s,
        "spread": round((tok_s[-1] - tok_s[0]) / statistics.median(tok_s), 4),
        "commit": rows[0]["git_commit"],
        "args": ARMS[arm],
    }


def compare(arm: str, now: dict, was: dict) -> tuple[bool, str]:
    if now["args"] != was["args"]:
        return False, (f"{arm}: the baseline was measured with different arguments\n"
                       f"      baseline: {' '.join(was['args'])}\n"
                       f"      now:      {' '.join(now['args'])}")
    ratio = now["tok_s"] / was["tok_s"]
    delta = f"{now['tok_s']} vs {was['tok_s']} tok/s ({ratio - 1:+.1%}, baseline {was['commit']})"
    if ratio < 1 - TOLERANCE:
        return False, f"{arm}: REGRESSED  {delta}"
    if ratio > 1 + TOLERANCE:
        return False, (f"{arm}: FASTER than the baseline by more than {TOLERANCE:.0%}  {delta}\n"
                       f"      rerun with --update so the next change is measured against this.")
    return True, f"{arm}: ok  {delta}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update", action="store_true",
                    help="write the measured numbers as the new baseline")
    ap.add_argument("--repeat", type=int, default=1,
                    help="with --update: measure this many times and take the median of the medians")
    ap.add_argument("--baseline", default=None, help="default: the device profile's results directory")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development only: measure uncommitted code")
    args = ap.parse_args()

    device, profile = sidecar.device_profile()
    path = args.baseline or f"{profile.results_dir}/{BASELINE_NAME}"
    extra = ["--allow-dirty"] if args.allow_dirty else []

    if not args.update and not os.path.exists(path):
        print(f"gate: no baseline at {path}; record one with --update", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="miniserve-gate-") as tmp:
        if args.update:
            takes = [{arm: measure(arm, f"{tmp}/{k}", extra) for arm in ARMS}
                     for k in range(args.repeat)]
            measured = {}
            for arm in ARMS:
                per_take = sorted(t[arm]["tok_s"] for t in takes)
                pick = min(takes, key=lambda t: abs(t[arm]["tok_s"] - statistics.median(per_take)))
                measured[arm] = dict(pick[arm], takes=per_take)
        else:
            measured = {arm: measure(arm, f"{tmp}/{arm}", extra) for arm in ARMS}
    elapsed = time.monotonic() - t0

    if args.update:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"device": device, "tolerance": TOLERANCE, "arms": measured}, f, indent=2)
            f.write("\n")
        for arm, m in measured.items():
            print(f"{arm}: {m['tok_s']} tok/s  spread {m['spread']:.2%}  takes {m['takes']}")
        print(f"gate: baseline written to {path} ({elapsed:.0f} s)")
        return 0

    with open(path) as f:
        base = json.load(f)
    if base["device"] != device:
        print(f"gate: the baseline is for {base['device']}, this is {device}", file=sys.stderr)
        return 2

    ok = True
    for arm in ARMS:
        was = base["arms"].get(arm)
        if was is None:
            print(f"{arm}: no baseline for this arm; rerun with --update", file=sys.stderr)
            ok = False
            continue
        passed, line = compare(arm, measured[arm], was)
        print(line, file=sys.stdout if passed else sys.stderr)
        ok &= passed
    print(f"gate: {'pass' if ok else 'FAIL'} ({elapsed:.0f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
