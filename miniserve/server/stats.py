"""What the engine did, counted per step, for a benchmark to read back.

A load generator sees requests and tokens; it cannot see the running batch. On a
card whose KV pool is smaller than the offered concurrency that difference is the
whole story: 256 requests in flight against a pool that holds a few dozen
sequences is a queue, not a batch, and reporting the concurrency without the
batch it actually produced overstates what the engine did. So the engine counts
its own steps and a benchmark reads the counters before and after each run.

The counters are written by the engine thread (through ``AsyncEngine.step_hook``,
which runs there) and read by the event loop. Python's integer and float
attribute assignments are atomic under the GIL, and a reader takes a snapshot of
all of them at once; a snapshot taken while a step is being counted can mix a
step's contributions with the one before it, which moves a mean by one step out
of thousands. That is worth far more than a lock on the engine's hot path.
"""

from __future__ import annotations

import dataclasses

from miniserve.engine.scheduler import Phase


@dataclasses.dataclass
class EngineStats:
    """Cumulative since the server started. Differences between two snapshots
    describe the window between them."""

    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    mixed_steps: int = 0
    # Decode rows summed over steps that had any: divided by those steps, the
    # running batch the engine sustained.
    decode_rows: int = 0
    steps_with_decode: int = 0
    prefill_tokens: int = 0  # tokens actually computed, cached prefixes excluded
    preemptions: int = 0
    # Sampled at the last step, not accumulated: the queue at that moment.
    running: int = 0
    waiting: int = 0

    def record(self, engine, batch) -> None:
        """Count one step. Called on the engine thread, once per ``step()``."""
        self.steps += 1
        if batch.phase is Phase.PREFILL:
            self.prefill_steps += 1
        elif batch.phase is Phase.DECODE:
            self.decode_steps += 1
        else:
            self.mixed_steps += 1
        if batch.num_decode:
            self.decode_rows += batch.num_decode
            self.steps_with_decode += 1
        self.prefill_tokens += sum(batch.extend_lens[batch.num_decode :])
        self.preemptions += len(batch.preempted)
        sched = engine.scheduler
        self.running = len(sched.running)
        self.waiting = len(sched.waiting)

    def snapshot(self) -> dict:
        d = dataclasses.asdict(self)
        d["running_batch_mean"] = round(self.decode_rows / self.steps_with_decode, 2) if self.steps_with_decode else 0.0
        return d
