"""Tests for the complete-development Conflict review preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.reconcile_twinkl_qtwz_review import (  # noqa: E402
    _derive_outcomes,
    reconcile,
)
from scripts.experiments.review_twinkl_qtwz_remaining_development import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_live,
    prepare,
    validate,
)
from src.vif.drift_target import sha256_file  # noqa: E402


def test_live_complement_matches_the_registered_scope():
    config, _protocol, selected, reviewed, cases_by_id, _paths = _load_live(
        DEFAULT_CONFIG
    )

    assert len(reviewed) == 106
    assert selected.height == 186
    assert selected["trajectory_length"].sum() == 1483
    assert set(selected["canonical_case_id"]) == set(cases_by_id)
    assert config["review"]["direct_api_calls"] is False


def test_prepare_freezes_seven_complete_blind_shards(tmp_path: Path):
    output = tmp_path / "review"
    args = argparse.Namespace(config=str(DEFAULT_CONFIG), output=str(output))

    prepare(args)
    validate(args)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["remaining_case_count"] == 186
    assert manifest["remaining_entry_count"] == 1483
    assert len(manifest["shards"]) == 7
    assert sum(shard["case_count"] for shard in manifest["shards"]) == 186
    assert sum(shard["entry_count"] for shard in manifest["shards"]) == 1483


def test_reconcile_requires_and_covers_both_complete_lanes(tmp_path: Path):
    output = tmp_path / "review"
    args = argparse.Namespace(config=str(DEFAULT_CONFIG), output=str(output))
    prepare(args)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for lane in ("reviewer_a", "reviewer_b"):
        lane_dir = output / "reviews" / lane
        lane_dir.mkdir(parents=True)
        for shard in manifest["shards"]:
            packet = json.loads(Path(shard["packet_path"]).read_text(encoding="utf-8"))
            response = {
                "schema_version": "twinkl-qtwz-review-v1",
                "cohort_version": "twinkl-qtwz-complete-development-review-v1",
                "shard_id": shard["shard_id"],
                "packet_sha256": shard["packet_sha256"],
                "response_schema_sha256": manifest["response_schema_sha256"],
                "reviewer_prompt_version": "twinkl-qtwz-packet-only-v1",
                "reviewer_id": f"test-{lane}",
                "reviewer_runtime": "test",
                "reviewed_at": "2026-07-14T12:00:00+00:00",
                "cases": [
                    {
                        "review_case_id": case["review_case_id"],
                        "entry_assessments": [
                            {
                                "position": entry["position"],
                                "observable_conflict": "no",
                                "reason_code": "direct_aligned_or_neutral_behavior",
                                "confidence": "high",
                            }
                            for entry in case["entries"]
                        ],
                        "sustained_conflict": "no",
                        "delivery_state": "none",
                        "rationale": "No displayed Conflict.",
                    }
                    for case in packet["cases"]
                ],
            }
            (lane_dir / f"{shard['shard_id']}.json").write_text(
                json.dumps(response, indent=2) + "\n", encoding="utf-8"
            )

    reconcile(args)

    entry_path = output / "results" / "entry_target.parquet"
    assert sha256_file(entry_path)
    entries = pl.read_parquet(entry_path)
    assert entries.height == 1483
    assert entries["resolution_status"].unique().to_list() == ["resolved"]


def test_outcome_derivation_uses_one_maximal_drift():
    _config, protocol, _selected, _reviewed, _cases, _paths = _load_live(DEFAULT_CONFIG)
    selected = pl.DataFrame(
        {
            "canonical_case_id": ["p1:security"],
            "persona_id": ["p1"],
            "dimension": ["security"],
            "historical_split": ["training"],
            "analysis_role": ["development_only"],
            "cohort_role": ["previously_unreviewed"],
            "trajectory_length": [4],
            "case_content_sha256": ["content"],
        }
    )
    entries = pl.DataFrame(
        [
            {
                "canonical_case_id": "p1:security",
                "position": position,
                "t_index": position - 1,
                "date": f"2026-01-{position:02d}",
                "final_conflict": label,
                "resolution_method": "agreement",
            }
            for position, label in enumerate([False, True, True, True], start=1)
        ]
    )

    outcomes, episodes = _derive_outcomes(
        entries,
        selected,
        protocol=protocol,
        cohort_sha256="cohort",
    )

    assert outcomes["has_drift"][0] is True
    assert outcomes["episode_count"][0] == 1
    assert episodes["supporting_positions"][0].to_list() == [2, 3, 4]


def test_outcome_derivation_preserves_empty_drift_schema():
    _config, protocol, _selected, _reviewed, _cases, _paths = _load_live(DEFAULT_CONFIG)
    selected = pl.DataFrame(
        {
            "canonical_case_id": ["p1:security"],
            "persona_id": ["p1"],
            "dimension": ["security"],
            "historical_split": ["training"],
            "analysis_role": ["development_only"],
            "cohort_role": ["previously_unreviewed"],
            "trajectory_length": [2],
            "case_content_sha256": ["content"],
        }
    )
    entries = pl.DataFrame(
        [
            {
                "canonical_case_id": "p1:security",
                "position": position,
                "t_index": position - 1,
                "date": f"2026-01-{position:02d}",
                "final_conflict": False,
                "resolution_method": "agreement",
            }
            for position in (1, 2)
        ]
    )

    _outcomes, episodes = _derive_outcomes(
        entries,
        selected,
        protocol=protocol,
        cohort_sha256="cohort",
    )

    assert episodes.is_empty()
    assert episodes.schema["episode_id"] == pl.String
    assert episodes.schema["supporting_positions"] == pl.List(pl.Int64)
