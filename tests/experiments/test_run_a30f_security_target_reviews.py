from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/experiments/run_a30f_security_target_reviews.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("a30f_review_runner_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def test_estimate_counts_three_isolated_calls_per_case():
    summary = mod.estimate([{"prompt": "one"}, {"prompt": "two"}])
    assert summary["case_count"] == 2
    assert summary["planned_calls"] == 6
    assert summary["estimated_cost_usd"] > 0


def test_dry_run_is_resumable_and_does_not_write(tmp_path):
    manifest = [
        {"case_id": "a", "prompt": "prompt"},
        {"case_id": "b", "prompt": "prompt"},
    ]
    result = mod.run_pass(
        manifest,
        results_dir=tmp_path,
        pass_index=1,
        execute=False,
        model="gpt-5.4-mini",
        reasoning_effort="none",
        timeout=1,
        max_attempts=1,
    )
    assert result == {"pass_index": 1, "eligible": 2, "complete": 0, "pending": 2}
    assert not list(tmp_path.iterdir())


def test_tiebreak_selection_uses_only_complete_three_way_ties(tmp_path):
    for pass_index, label in enumerate((-1, 0, 1), start=1):
        path = tmp_path / f"pass_{pass_index}_results.jsonl"
        path.write_text(
            json.dumps(
                {"case_id": "tie", "status": "ok", "scores": {"security": label}}
            )
            + "\n"
        )
    assert mod.tiebreak_case_ids(tmp_path, expected_case_ids={"tie"}) == {"tie"}
