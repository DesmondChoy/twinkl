"""Receipt checks for the blind Opus resolution of four ``twinkl-752.5`` cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    REPO_ROOT
    / "logs/experiments/artifacts/twinkl_752_5_opus_null_resolution_20260714"
)
RESULTS = ARTIFACT_ROOT / "results"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_blind_packet_hides_parent_context_and_prior_labels():
    packet_text = (ARTIFACT_ROOT / "reviewer_packet.json").read_text(
        encoding="utf-8"
    )
    packet = json.loads(packet_text)
    prompt = (ARTIFACT_ROOT / "review_prompt.md").read_text(encoding="utf-8")
    key = _read_json(ARTIFACT_ROOT / "parent_reconciliation_key.json")

    assert packet["rubric"]["rubric_id"] == "drift_v1_conflict_rubric_v1"
    assert all(case["core_value_context"] for case in packet["cases"])
    assert prompt.endswith(
        "REVIEW PACKET\n"
        + json.dumps(packet, indent=2, ensure_ascii=False)
        + "\n"
    )
    assert len(key["cases"]) == 4
    for case in key["cases"]:
        assert case["canonical_case_id"] not in packet_text
    for forbidden in (
        "persona_id",
        "canonical_case_id",
        "historical_split",
        "cohort_role",
        "analysis_role",
        "prior_label",
        "reviewer_a",
        "reviewer_b",
        "adjudicator",
        "vif",
    ):
        assert forbidden not in packet_text.lower()


def test_receipt_proves_opus_and_expected_four_labels():
    receipt = _read_json(ARTIFACT_ROOT / "opus_claude_output.json")
    assert any("opus" in model.lower() for model in receipt["modelUsage"])

    key = _read_json(ARTIFACT_ROOT / "parent_reconciliation_key.json")
    labels = _read_json(RESULTS / "opus_labels.json")
    canonical_by_review = {
        case["review_case_id"]: case["canonical_case_id"] for case in key["cases"]
    }
    actual = {
        (
            canonical_by_review[case["review_case_id"]],
            case["entry_adjudications"][0]["position"],
        ): (
            case["entry_adjudications"][0]["observable_conflict"],
            case["entry_adjudications"][0]["confidence"],
        )
        for case in labels["cases"]
    }

    assert actual == {
        ("799f3751:hedonism", 1): ("yes", "medium"),
        ("65ed1278:benevolence", 1): ("yes", "medium"),
        ("5943c186:hedonism", 4): ("yes", "low"),
        ("3cfa2ebf:universalism", 10): ("no", "low"),
    }


def test_resolution_completes_cohort_without_adding_drift():
    summary = _read_json(RESULTS / "summary.json")
    assert summary["case_count"] == 104
    assert summary["resolved_case_count"] == 104
    assert summary["unresolved_case_count"] == 0
    assert summary["reviewed_drift_count"] == 31
    assert summary["reviewed_drift_trajectory_count"] == 26
    assert summary["all_known_development_union"] == {
        "trajectory_count": 106,
        "resolved_trajectory_count": 106,
        "drift_count": 33,
        "drift_trajectory_count": 28,
    }

    outcomes = pl.read_parquet(RESULTS / "case_outcomes_opus_resolved.parquet")
    resolved_ids = {
        "799f3751:hedonism",
        "65ed1278:benevolence",
        "5943c186:hedonism",
        "3cfa2ebf:universalism",
    }
    resolved = outcomes.filter(pl.col("canonical_case_id").is_in(resolved_ids))
    assert resolved.height == 4
    assert resolved["case_resolution"].to_list() == ["resolved"] * 4
    assert not resolved["has_drift"].any()

    source = pl.read_parquet(
        REPO_ROOT
        / "logs/experiments/artifacts/"
        "twinkl_752_4_legacy_drift_review_20260713/results/entry_target_final.parquet"
    )
    revised = pl.read_parquet(RESULTS / "entry_target_opus_resolved.parquet")
    stable_source = source.filter(pl.col("resolution_status") != "unresolved").sort(
        "canonical_case_id", "position"
    )
    stable_revised = revised.filter(
        pl.col("resolution_method") != "opus_adjudication"
    ).select(source.columns)
    assert stable_source.equals(stable_revised)
    opus_rows = revised.filter(
        pl.col("resolution_method") == "opus_adjudication"
    ).select(
        "canonical_case_id",
        "position",
        "final_conflict",
        "opus_adjudicator_confidence",
    )
    assert {
        (row["canonical_case_id"], row["position"]): (
            row["final_conflict"],
            row["opus_adjudicator_confidence"],
        )
        for row in opus_rows.to_dicts()
    } == {
        ("799f3751:hedonism", 1): (True, "medium"),
        ("65ed1278:benevolence", 1): (True, "medium"),
        ("5943c186:hedonism", 4): (True, "low"),
        ("3cfa2ebf:universalism", 10): (False, "low"),
    }


def test_all_frozen_and_result_hashes_match_manifests():
    for manifest_path, sections in (
        (ARTIFACT_ROOT / "audit_manifest.json", ("source_files", "frozen_files")),
        (RESULTS / "audit_manifest.json", ("claude_output", "results")),
    ):
        manifest = _read_json(manifest_path)
        for section in sections:
            for relative_path, expected in manifest[section].items():
                assert _sha256(REPO_ROOT / relative_path) == expected
