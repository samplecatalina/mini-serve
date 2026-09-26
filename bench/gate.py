"""A short benchmark that a change has to clear before it is merged.

The point of a gate is that it measures the same thing every time, so the
workload is written down here rather than taken from the command line: a
caliber that can be adjusted per run cannot detect anything. The baseline file
``results/<device>/gate_baseline.json`` records the reference commit and the
exact arguments; if the arguments no longer match, the gate refuses to run
instead of subtracting two numbers that were never comparable.

The verdict is relative and taken within one session. On this machine the same
code has read +2.6-3.1% on one day and -3.6-3.7% on another against a baseline
recorded earlier (``results/rtx4060-laptop/m5_2_gate_drift_*.csv``), with the
GPU clock at its ceiling both times -- more than half the threshold, from
nothing the repository did. So the gate checks the reference commit out
into a temporary worktree, runs it and the working commit alternately in four processes
per arm (reference, current, current, reference), and compares the medians of
all their runs. A drift that is linear in time lands on both sides equally, and
the two reference processes say how much the session itself moved: when they
disagree by more than half the threshold, the verdict is "inconclusive" rather
than pass or fail. The number recorded in the baseline file is still printed
next to the reference's number, as a running record of drift across sessions.

The reference is the commit the baseline file names, not the parent commit:
comparing each change only with the one before lets five 2% regressions in a
row each pass.

Two arms, because one does not cover the engine:

- ``unique`` -- 256 requests with no shared prefix. Decode-bound; four separate
  processes of three runs each on one commit stayed inside 0.81%
  (``results/rtx4060-laptop/m5_1_block_backend.csv``).
- ``shared`` -- the same requests behind a shared prefix, so that the prefix
  cache is on the measured path. Without it a regression in the radix tree
  costs nothing the gate can see.

The gate fails on a gain above the threshold as well. A reference that is only
ever allowed to be met drifts downwards in usefulness: once throughput has
improved and the reference has not, the next real regression falls from a
higher place and still clears the line.

    python -m bench.gate            # check the working commit against the reference
    python -m bench.gate --update   # make the working commit the reference

Runs are written to a temporary directory, not to ``results/``: only the
baseline file belongs in the record.
"""

from __future__ import annotations

import argparse
import contextlib
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
    # Pinned, not left to the engine's own sizing. Without this the pool is
    # derived from whatever GPU memory happens to be free, which on a desktop
    # machine moves by hundreds of megabytes between one day and the next; a
    # larger pool preempts less and reads as a throughput gain that no change
    # to this repository caused. A run that cannot get these tokens fails
    # rather than quietly taking fewer.
    "--kv-pool-tokens", "32768",
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
# When the reference moves this much between its two processes within one
# session, the session is too unsteady for a 5% verdict to mean anything.
UNSTEADY = TOLERANCE / 2
# Process order per arm: the reference brackets the working commit, so a drift
# that is linear in time costs both sides the same.
ORDER = ("reference", "current", "current", "reference")

BASELINE_NAME = "gate_baseline.json"


def measure(arm: str, out_dir: str, extra: list[str], tree: str | None = None) -> dict:
    """Run one arm in one process and return its rows' throughput, or raise on a failed run.

    ``tree``: a checkout of another commit to run instead of the working tree.
    It is a git worktree, so the run records that commit through git like any
    other; an exported copy without git would leave commits older than the
    ``MINISERVE_GIT`` override unable to say what they are.
    """
    argv = [sys.executable, "-m", "bench.offline", *ARMS[arm], *extra,
            "--results-dir", out_dir, "--out", f"gate_{arm}"]
    env = None if tree is None else dict(os.environ, PYTHONPATH=tree)
    r = subprocess.run(argv, cwd=tree, env=env)
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


@contextlib.contextmanager
def checkout(commit: str, dest: str):
    """A detached worktree of ``commit`` at ``dest`` for the duration of the block.

    Removed afterwards, including on failure; a worktree left behind by a
    killed process is cleared by the next ``git worktree prune``.
    """
    r = subprocess.run(["git", "worktree", "add", "--detach", "--quiet", dest, commit])
    if r.returncode != 0:
        raise SystemExit(f"gate: could not check out reference commit {commit}")
    try:
        yield dest
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", dest])


def run_abba(arm: str, tmp: str, extra: list[str], ref_tree: str, run=measure) -> dict:
    """Four processes of one arm in ORDER; returns every process's result by side."""
    sides = {"reference": [], "current": []}
    for k, side in enumerate(ORDER):
        print(f"--- {arm} [{k + 1}/{len(ORDER)}] {side}", flush=True)
        if side == "reference":
            sides[side].append(run(arm, f"{tmp}/{arm}_{k}", [], tree=ref_tree))
        else:
            sides[side].append(run(arm, f"{tmp}/{arm}_{k}", extra))
    return sides


def judge(arm: str, sides: dict, recorded: dict | None = None) -> tuple[str, str]:
    """Verdict for one arm: ("pass" | "fail" | "inconclusive", line to print)."""
    ref_runs = [x for m in sides["reference"] for x in m["runs"]]
    cur_runs = [x for m in sides["current"] for x in m["runs"]]
    ref, cur = statistics.median(ref_runs), statistics.median(cur_runs)
    ratio = cur / ref
    first, last = (m["tok_s"] for m in sides["reference"])
    wander = abs(last - first) / ref
    delta = (f"{cur:.1f} vs {ref:.1f} tok/s ({ratio - 1:+.1%}; reference moved {wander:.1%} "
             f"within the session)")
    drift = ""
    if recorded is not None:
        drift = f"\n      reference now vs recorded {recorded['tok_s']}: {ref / recorded['tok_s'] - 1:+.1%}"
    if wander > UNSTEADY:
        return "inconclusive", (f"{arm}: INCONCLUSIVE  {delta}: the reference itself moved more than "
                                f"{UNSTEADY:.1%}; rerun when the machine is steady{drift}")
    if ratio < 1 - TOLERANCE:
        return "fail", f"{arm}: REGRESSED  {delta}{drift}"
    if ratio > 1 + TOLERANCE:
        return "fail", (f"{arm}: FASTER than the reference by more than {TOLERANCE:.0%}  {delta}\n"
                        f"      rerun with --update so the next change is measured against this.{drift}")
    return "pass", f"{arm}: ok  {delta}{drift}"


def uncommitted() -> bool:
    """Tracked files differ from HEAD. Checked before anything runs: the
    reference goes first, and a refusal after its first process would cost a
    minute to say what one command can."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def host_state() -> str:
    raw = os.environ.get("MINISERVE_HOST_POWER")
    return raw if raw else "(not recorded)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update", action="store_true",
                    help="measure the working commit and make it the reference")
    ap.add_argument("--repeat", type=int, default=1,
                    help="with --update: measure this many times and take the median of the medians")
    ap.add_argument("--baseline", default=None, help="default: the device profile's results directory")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="development only: measure uncommitted code")
    args = ap.parse_args()

    device, profile = sidecar.device_profile()
    path = args.baseline or f"{profile.results_dir}/{BASELINE_NAME}"
    extra = ["--allow-dirty"] if args.allow_dirty else []
    print(f"gate: host {host_state()}", flush=True)

    if args.update:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="miniserve-gate-") as tmp:
            takes = [{arm: measure(arm, f"{tmp}/{k}_{arm}", extra) for arm in ARMS}
                     for k in range(args.repeat)]
        measured = {}
        for arm in ARMS:
            per_take = sorted(t[arm]["tok_s"] for t in takes)
            pick = min(takes, key=lambda t: abs(t[arm]["tok_s"] - statistics.median(per_take)))
            measured[arm] = dict(pick[arm], takes=per_take)
        commits = {m["commit"] for m in measured.values()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"device": device, "tolerance": TOLERANCE, "reference_commit": commits.pop(),
                       "arms": measured}, f, indent=2)
            f.write("\n")
        for arm, m in measured.items():
            print(f"{arm}: {m['tok_s']} tok/s  spread {m['spread']:.2%}  takes {m['takes']}")
        print(f"gate: baseline written to {path} ({time.monotonic() - t0:.0f} s)")
        return 0

    if not os.path.exists(path):
        print(f"gate: no baseline at {path}; record one with --update", file=sys.stderr)
        return 2
    with open(path) as f:
        base = json.load(f)
    if base["device"] != device:
        print(f"gate: the baseline is for {base['device']}, this is {device}", file=sys.stderr)
        return 2
    for arm in ARMS:
        was = base["arms"].get(arm)
        if was is None or was["args"] != ARMS[arm]:
            print(f"gate: the baseline has no {arm} arm with the current arguments; rerun with --update",
                  file=sys.stderr)
            return 2
    ref_commit = base.get("reference_commit") or base["arms"]["unique"]["commit"]
    if not args.allow_dirty and uncommitted():
        print("gate: the working tree has uncommitted changes; commit them first", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    verdicts = {}
    with tempfile.TemporaryDirectory(prefix="miniserve-gate-") as tmp:
        with checkout(ref_commit, f"{tmp}/reference") as ref_tree:
            results = {arm: run_abba(arm, tmp, extra, ref_tree) for arm in ARMS}
    elapsed = time.monotonic() - t0
    for arm in ARMS:
        verdicts[arm], line = judge(arm, results[arm], base["arms"][arm])
        print(line)
    if "inconclusive" in verdicts.values():
        verdict, code = "INCONCLUSIVE", 3
    elif "fail" in verdicts.values():
        verdict, code = "FAIL", 1
    else:
        verdict, code = "pass", 0
    # Everything on one stream: a caller that redirects output wants the
    # verdict next to the lines it followed from, and the exit code is what
    # says pass, fail or inconclusive.
    print(f"gate: {verdict} against reference {ref_commit[:12]} ({elapsed:.0f} s)", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
