"""The sglang speculative-decoding harness, the parts that run without sglang: which engine
settings each arm asks for, and the workload file both engines replay."""

from __future__ import annotations

import argparse
import json

import pytest

from bench import sglang_spec


def args(**kw):
    base = dict(max_running=8, kv_pool_tokens=65536, chunked_prefill_size=2048, max_prefill_tokens=8192,
                gamma=4, eagle_head="tengyunw", tree=None)
    return argparse.Namespace(**{**base, **kw})


WL = {"model_path": "/m/8B", "draft_path": "/m/0.6B"}


def test_the_chain_is_the_default_shape(monkeypatch):
    monkeypatch.setattr(sglang_spec, "eagle3_path", lambda head="tengyunw": f"/heads/{head}")
    kw = sglang_spec.engine_args("eagle3", WL, args())
    assert (kw["speculative_num_steps"], kw["speculative_eagle_topk"], kw["speculative_num_draft_tokens"]) == (4, 1, 5)
    assert kw["speculative_draft_model_path"] == "/heads/tengyunw"


def test_a_tree_shape_and_another_head(monkeypatch):
    monkeypatch.setattr(sglang_spec, "eagle3_path", lambda head="tengyunw": f"/heads/{head}")
    kw = sglang_spec.engine_args("eagle3", WL, args(tree="3,4,8", eagle_head="angelslim"))
    assert (kw["speculative_num_steps"], kw["speculative_eagle_topk"], kw["speculative_num_draft_tokens"]) == (3, 4, 8)
    assert kw["speculative_draft_model_path"] == "/heads/angelslim"


def test_the_two_model_arm_keeps_the_chain_whatever_the_tree(monkeypatch):
    kw = sglang_spec.engine_args("standalone", WL, args(tree="3,4,8"))
    assert kw["speculative_algorithm"] == "STANDALONE" and kw["speculative_eagle_topk"] == 1


@pytest.mark.parametrize("bad", ["3,4,20", "0,1,1", "4,1,6", "3,0,4"])
def test_impossible_tree_shapes_are_refused(bad):
    with pytest.raises(SystemExit):
        sglang_spec.tree_shape(bad)


def test_both_heads_are_pinned():
    for repo, rev in sglang_spec.EAGLE3_HEADS.values():
        assert "/" in repo and len(rev) == 40


@pytest.mark.slow
def test_the_mtbench_workload_file_carries_its_stop_token(tmp_path):
    """Needs the pinned MT-Bench file and the Qwen3 tokenizer in the cache (`python -m bench.mtbench`)."""
    out = tmp_path / "w.json"
    assert sglang_spec.dump(["--out", str(out), "--model", "0.6B", "--max-running", "8", "--requests", "12",
                             "--workload", "mtbench", "--output-len", "64"]) == 0
    wl = json.loads(out.read_text())
    assert wl["caliber"]["workload"] == "mtbench" and len(wl["prompts"]) == 12
    assert wl["throughput_stop_ids"] == [151645]  # <|im_end|>
