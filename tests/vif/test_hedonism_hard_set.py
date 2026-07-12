"""Tests for the twinkl-748 Hedonism paired-review workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from src.vif.hedonism_hard_set import (
    REVIEW_PROMPT_VERSION,
    REVIEW_SCHEMA_VERSION,
    TARGET_VERSION,
    build_review_bundle,
    load_candidate_spec,
    materialize_reviewed_hard_set,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "config/evals/twinkl_748_hedonism_hard_set_v1.yaml"


def _response(bundle: Path, reviewer_id: str) -> dict:
    packet_path = bundle / "reviewer_packet/blind_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    key = json.loads(
        (bundle / "parent_control/reconciliation_key.json").read_text(
            encoding="utf-8"
        )
    )
    key_cases = {case["review_pair_id"]: case for case in key["cases"]}
    cases = []
    for packet_case in packet["cases"]:
        key_case = key_cases[packet_case["review_pair_id"]]
        key_entries = {
            entry["review_entry_id"]: entry for entry in key_case["entries"]
        }
        cases.append(
            {
                "review_pair_id": packet_case["review_pair_id"],
                "entry_reviews": [
                    {
                        "review_entry_id": entry["review_entry_id"],
                        "hedonism_label": key_entries[entry["review_entry_id"]][
                            "author_label"
                        ],
                        "confidence": "high",
                        "rationale": (
                            "The displayed choice clearly supports this label."
                        ),
                    }
                    for entry in packet_case["entries"]
                ],
                "comparable_except_choice": "yes",
                "realistic": "yes",
                "issue_codes": [],
                "accept_pair": "yes",
                "pair_rationale": "The pair changes one relevant choice.",
            }
        )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "packet_sha256": sha256_file(packet_path),
        "reviewer_id": reviewer_id,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "reviewer_runtime": "test-runtime",
        "requested_runtime": "5.6 Sol / Light",
        "pair_reviews": cases,
    }


def test_checked_in_candidate_spec_is_valid():
    spec = load_candidate_spec(SPEC)

    assert len(spec["pairs"]) == 24
    assert len({pair["family"] for pair in spec["pairs"]}) == 6


def test_candidate_spec_rejects_banned_term(tmp_path: Path):
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    payload["pairs"][0]["variants"][0]["text"] += " Hedonism."
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="leaks banned term"):
        load_candidate_spec(path)


def test_candidate_spec_rejects_boolean_author_label(tmp_path: Path):
    payload = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    payload["pairs"][0]["variants"][0]["author_label"] = True
    path = tmp_path / "bad-label.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="author labels must be"):
        load_candidate_spec(path)


def test_review_packet_hides_author_intent_and_randomizes_ids(tmp_path: Path):
    bundle = tmp_path / "bundle"
    build_review_bundle(spec_path=SPEC, output_dir=bundle, root=ROOT)

    packet_text = (bundle / "reviewer_packet/blind_packet.json").read_text(
        encoding="utf-8"
    )
    assert "author_label" not in packet_text
    assert "isolated_behavior" not in packet_text
    assert "family" not in packet_text
    assert "hp_001" not in packet_text
    assert "pair_001" in packet_text
    assert not (bundle / "reviewer_packet/reconciliation_key.json").exists()
    assert (bundle / "parent_control/reconciliation_key.json").is_file()


def test_materialization_freezes_only_paired_agreement(tmp_path: Path):
    bundle = tmp_path / "bundle"
    build_review_bundle(spec_path=SPEC, output_dir=bundle, root=ROOT)
    response_a = _response(bundle, "reviewer-a")
    response_b = _response(bundle, "reviewer-b")
    response_b["pair_reviews"][0]["accept_pair"] = "no"
    path_a = bundle / "reviewer_packet/reviewer_a.json"
    path_b = bundle / "reviewer_packet/reviewer_b.json"
    path_a.write_text(json.dumps(response_a, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps(response_b, indent=2), encoding="utf-8")

    summary = materialize_reviewed_hard_set(
        bundle_dir=bundle,
        reviewer_a_path=path_a,
        reviewer_b_path=path_b,
    )
    frozen = pl.read_parquet(
        bundle / "parent_control/frozen_hedonism_hard_set.parquet"
    )

    assert summary["accepted_pair_count"] == 23
    assert summary["excluded_pair_count"] == 1
    assert frozen.height == 46
    assert set(frozen["hedonism_target"].unique()) == {-1, 1}


def test_materialization_rejects_duplicate_reviewer_identity(tmp_path: Path):
    bundle = tmp_path / "bundle"
    build_review_bundle(spec_path=SPEC, output_dir=bundle, root=ROOT)
    response = _response(bundle, "same-reviewer")
    path_a = bundle / "reviewer_packet/reviewer_a.json"
    path_b = bundle / "reviewer_packet/reviewer_b.json"
    path_a.write_text(json.dumps(response, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps(deepcopy(response), indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct reviewer IDs"):
        materialize_reviewed_hard_set(
            bundle_dir=bundle,
            reviewer_a_path=path_a,
            reviewer_b_path=path_b,
        )


def test_materialization_rejects_boolean_reviewer_label(tmp_path: Path):
    bundle = tmp_path / "bundle"
    build_review_bundle(spec_path=SPEC, output_dir=bundle, root=ROOT)
    response_a = _response(bundle, "reviewer-a")
    response_b = _response(bundle, "reviewer-b")
    response_a["pair_reviews"][0]["entry_reviews"][0]["hedonism_label"] = True
    response_b["pair_reviews"][0]["entry_reviews"][0]["hedonism_label"] = True
    path_a = bundle / "reviewer_packet/reviewer_a.json"
    path_b = bundle / "reviewer_packet/reviewer_b.json"
    path_a.write_text(json.dumps(response_a, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps(response_b, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Hedonism label"):
        materialize_reviewed_hard_set(
            bundle_dir=bundle,
            reviewer_a_path=path_a,
            reviewer_b_path=path_b,
        )
