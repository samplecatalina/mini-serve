"""The main-caliber harness: the rules that decide whether a run may write results.

These are the parts that have to hold before any number exists, so they are
tested without a GPU, a server or a load generator.
"""

from __future__ import annotations

import argparse
import json

import pytest

from bench import serving


def args(**over) -> argparse.Namespace:
    base = dict(
        url="http://127.0.0.1:8000", server_kind="miniserve", label="miniserve",
        model_name="Qwen/Qwen3-0.6B", tokenizer="/cache/qwen", concurrency=[1, 256],
        scenario=["D(1024,256)"], max_time_per_run=3, max_requests_per_run=2048,
        warmup_s=30.0, sampling={"temperature": 0, "ignore_eos": True},
        server_engine=None, server_version=None, image="sha256:abc",
    )
    return argparse.Namespace(**(base | over))


SERVER_INFO = dict(
    server="miniserve",
    model=dict(repo_id="Qwen/Qwen3-0.6B", revision="c1899de", path="/cache/qwen"),
    engine=dict(kv_pool_tokens=32768, cuda_graph_buckets=[1, 2, 4], dtype="bfloat16"),
    runtime=dict(torch="2.9.1+cu128", flashinfer="0.6.7.post3", git_dirty=False),
)


def record(**over):
    a = args(**over)
    engine, runtime, model = serving.normalize_server_info("miniserve", SERVER_INFO)
    return serving.fairness_record(a, SERVER_INFO, engine, runtime, model, ratio=0.1667)


@pytest.fixture(autouse=True)
def a_named_client(monkeypatch):
    """The load generator lives in its own image; these tests do not run inside it."""
    monkeypatch.setattr(serving, "client_version", lambda: "0.0.5")


class TestFairnessRecord:
    def test_every_required_field_is_filled(self):
        assert serving.missing_fairness(record()) == []

    @pytest.mark.parametrize("field", serving.FAIRNESS)
    def test_a_missing_field_is_reported(self, field):
        r = record()
        r[field] = None
        assert serving.missing_fairness(r) == [field]

    def test_it_carries_the_settings_the_server_reports_not_the_ones_asked_for(self):
        # The KV pool defaults to whatever memory allows, so the command line does
        # not know it; only the server does.
        assert record()["server_engine"]["kv_pool_tokens"] == 32768

    def test_an_unnamed_client_image_is_a_missing_field(self):
        assert serving.missing_fairness(record(image="")) == ["image"]

    def test_a_client_that_cannot_name_its_version_is_a_missing_field(self, monkeypatch):
        # "unknown" would pass a presence check and make the run unrepeatable.
        monkeypatch.setattr(serving, "client_version", lambda: None)
        assert serving.missing_fairness(record()) == ["client"]


class TestWarmupWindow:
    def test_seconds_become_the_fraction_of_the_run_they_are(self):
        assert serving.warmup_ratio(30, 3) == pytest.approx(30 / 180)

    def test_a_warmup_that_eats_half_the_run_is_refused(self):
        # Discarding half of a run leaves too little measured to be worth reporting,
        # and it means the run is too short for the clocks it is waiting on.
        with pytest.raises(SystemExit, match="raise --max-time-per-run"):
            serving.warmup_ratio(30, 1)

    def test_the_refusal_says_how_long_the_run_has_to_be(self):
        with pytest.raises(SystemExit, match="at least 2 minutes"):
            serving.warmup_ratio(35, 1)


class TestServerInfo:
    def test_this_engine_is_read_field_by_field(self):
        engine, runtime, model = serving.normalize_server_info("miniserve", SERVER_INFO)
        assert engine["dtype"] == "bfloat16"
        assert model["revision"] == "c1899de"
        assert runtime["flashinfer"] == "0.6.7.post3"

    def test_sglang_reports_one_flat_object_and_is_kept_whole(self):
        info = dict(model_path="Qwen/Qwen3-0.6B", max_total_num_tokens=335435,
                    attention_backend="flashinfer", version="0.5.10")
        engine, runtime, model = serving.normalize_server_info("sglang", info)
        assert engine["max_total_num_tokens"] == 335435  # nothing is dropped in translation
        assert runtime["attention_backend"] == "flashinfer"
        assert model["repo_id"] == "Qwen/Qwen3-0.6B"


class TestFlatten:
    def run_json(self, tmp_path, **over):
        stats = {
            name: dict(min=1.0, max=9.0, mean=5.0, stddev=1.0, sum=50.0,
                       p25=2.0, p50=5.0, p75=7.0, p90=8.0, p95=8.5, p99=9.0)
            for name in ("ttft", "tpot", "e2e_latency", "num_input_tokens", "num_output_tokens")
        }
        d = dict(
            _time_unit="ms",
            aggregated_metrics=dict(
                scenario="D(1024,256)", num_concurrency=256, run_duration=180.0,
                num_requests=2048, num_completed_requests=2048, num_error_requests=0,
                error_rate=0.0, requests_per_second=11.4,
                mean_output_throughput_tokens_per_s=2915.02,
                mean_total_tokens_throughput_tokens_per_s=14575.1,
                stats=stats | over.pop("stats", {}),
            ),
            individual_request_metrics=[],
        )
        p = tmp_path / "D1024_256_text-to-text_num_concurrency_256_time_180s.json"
        p.write_text(json.dumps(d))
        return str(p)

    def test_one_row_per_scenario_and_concurrency(self, tmp_path):
        rows = serving.flatten(self.run_json(tmp_path))
        assert len(rows) == 1
        assert (rows[0]["scenario"], rows[0]["concurrency"]) == ("D(1024,256)", 256)

    def test_it_carries_the_caliber_the_project_reports(self, tmp_path):
        row = serving.flatten(self.run_json(tmp_path))[0]
        for key in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p99_ms", "output_tok_s"):
            assert row[key] is not None, key
        assert row["output_tok_s"] == 2915.0
        assert row["time_unit"] == "ms"

    def test_errors_are_carried_not_dropped(self, tmp_path):
        row = serving.flatten(self.run_json(tmp_path))[0]
        assert row["num_errors"] == 0 and row["error_rate"] == 0.0
        # a run that lost requests must be visible in the row, not only in a log
        assert "num_completed" in row and "num_requests" in row

    def test_a_missing_percentile_becomes_empty_not_zero(self, tmp_path):
        path = self.run_json(tmp_path, stats={"tpot": dict(p50=5.0)})
        row = serving.flatten(path)[0]
        assert row["tpot_p50_ms"] == 5.0
        assert row["tpot_p99_ms"] is None  # zero would read as "no time between tokens"


class TestLoadedClocks:
    def sample(self, util, sm):
        return serving.sidecar.GpuSample(
            t=0.0, name="gpu", sm_mhz=sm, mem_mhz=9001, temp_c=60, power_w=100.0,
            power_limit_w=140.0, reasons="0x0", util=util, mem_used_mib=4096,
        )

    def test_the_idle_start_of_the_client_does_not_drag_the_clock_down(self):
        # genai-bench loads a tokenizer before it sends anything; those samples are
        # of an idle GPU and would understate the clock the run happened at.
        samples = [self.sample(0, 210)] * 10 + [self.sample(99, 2500)] * 10
        assert serving.loaded_gpu(samples)["sm_mhz"]["mean"] == 2500
        assert serving.loaded_gpu(samples)["num_samples"] == 10

    def test_no_loaded_samples_is_empty_rather_than_invented(self):
        assert serving.loaded_gpu([self.sample(0, 210)]) == {}


class TestEngineCounters:
    """The running batch is the one number a load generator cannot see: it reports
    the concurrency it offered, not the batch the engine managed to form from it."""

    def snapshot(self, **over):
        base = dict(steps=0, prefill_steps=0, decode_steps=0, mixed_steps=0, decode_rows=0,
                    steps_with_decode=0, prefill_tokens=0, preemptions=0, running=0, waiting=0)
        return base | over

    def test_the_running_batch_is_the_mean_over_steps_that_decoded(self):
        before = self.snapshot(steps=10, decode_rows=100, steps_with_decode=10)
        after = self.snapshot(steps=110, decode_rows=1700, steps_with_decode=110)
        assert serving.stats_delta(before, after)["running_batch_mean"] == 16.0

    def test_it_is_the_window_not_the_whole_server_lifetime(self):
        # Warmup and an earlier concurrency level ran on the same server.
        before = self.snapshot(steps=1000, decode_rows=1000, steps_with_decode=1000)  # batch 1 so far
        after = self.snapshot(steps=1100, decode_rows=4200, steps_with_decode=1100)
        assert serving.stats_delta(before, after)["running_batch_mean"] == 32.0

    def test_the_queue_at_the_end_is_a_reading_not_a_difference(self):
        d = serving.stats_delta(self.snapshot(waiting=5, running=8), self.snapshot(waiting=224, running=32))
        assert (d["waiting_at_end"], d["running_at_end"]) == (224, 32)

    def test_preemptions_accumulate(self):
        d = serving.stats_delta(self.snapshot(preemptions=3), self.snapshot(preemptions=47))
        assert d["preemptions"] == 44

    def test_a_server_without_counters_leaves_the_field_empty(self):
        # Another engine answers no such route; the row says nothing rather than guessing.
        assert serving.stats_delta(None, self.snapshot()) == {}
        assert serving.stats_delta(self.snapshot(), None) == {}

    def test_a_window_with_no_decode_step_does_not_divide_by_zero(self):
        assert serving.stats_delta(self.snapshot(), self.snapshot(steps=3, prefill_steps=3))["running_batch_mean"] == 0.0
