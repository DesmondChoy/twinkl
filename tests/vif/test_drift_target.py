"""Tests for the non-destructive student-visible drift target workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift_target import (
    REVIEW_PROMPT_VERSION,
    REVIEW_SCHEMA_VERSION,
    TARGET_VERSION,
    apply_student_visible_target,
    build_blind_packet,
    build_full_trajectory_cases,
    build_paired_target_delta,
    derive_target_split,
    record_promotion_threshold_receipt,
    sha256_file,
    validate_audit_manifest,
    validate_promotion_threshold_receipt,
    validate_review_bundle_cases,
    validate_target_manifest,
    write_review_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_labels() -> pl.DataFrame:
    rows = []
    for t_index, label in enumerate([-1, -1, 0]):
        row = {
            "persona_id": "p1",
            "t_index": t_index,
            "date": f"2026-01-0{t_index + 1}",
            "alignment_vector": [0] * len(SCHWARTZ_VALUE_ORDER),
            "rationales_json": "{}",
        }
        for dimension in SCHWARTZ_VALUE_ORDER:
            row[f"alignment_{dimension}"] = label if dimension == "security" else 0
            row[f"confidence_{dimension}"] = "unanimous"
            row[f"consensus_agreement_{dimension}"] = 5
            row[f"label_changed_{dimension}"] = False
        rows.append(row)
    return pl.DataFrame(rows)


def _registry() -> pl.DataFrame:
    return pl.DataFrame({"persona_id": ["p1"], "core_values": [["Security"]]})


def _cases() -> list[dict]:
    entries = [
        {
            "t_index": index,
            "date": f"2026-01-0{index + 1}",
            "initial_entry": text,
        }
        for index, text in enumerate(
            [
                "I skipped the safety inspection because I wanted to leave early.",
                "I ignored the same inspection again despite knowing the risk.",
                "I completed the inspection properly today.",
            ]
        )
    ]
    return [
        {
            "case_id": "development:p1:security",
            "split": "development",
            "persona_id": "p1",
            "dimension": "security",
            "entries": entries,
        }
    ]


def _response(
    *,
    labels: list[str],
    sustained_conflict: str = "yes",
    reviewer_id: str = "test-reviewer-a",
    split: str = "development",
    packet_sha256: str = "packet-hash",
) -> dict:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "split": split,
        "packet_sha256": packet_sha256,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "reviewer_id": reviewer_id,
        "reviewed_at": "2026-07-11T09:00:00+00:00",
        "cases": [
            {
                "review_case_id": "case_001",
                "entry_assessments": [
                    {
                        "position": position,
                        "observable_negative": label,
                        "reason_code": (
                            "direct_behavior_or_choice"
                            if label == "yes"
                            else "direct_aligned_or_neutral_behavior"
                        ),
                        "confidence": "high",
                    }
                    for position, label in enumerate(labels, start=1)
                ],
                "sustained_conflict": sustained_conflict,
                "delivery_state": "recovered",
                "rationale": "The entries state the choices directly.",
            }
        ],
    }


def test_blind_packet_uses_full_trajectory_without_source_metadata():
    packet, key = build_blind_packet(_cases(), split="development")

    case = packet["cases"][0]
    assert [entry["position"] for entry in case["entries"]] == [1, 2, 3]
    packet_text = str(packet)
    assert "p1" not in packet_text
    assert "2026-01" not in packet_text
    assert "t_index" not in packet_text
    assert "alignment" not in packet_text
    assert key["cases"][0]["persona_id"] == "p1"


def test_blind_packet_includes_all_runtime_entry_text_components():
    case = _cases()[0]
    case["entries"][0]["nudge_text"] = "What happened next?"
    case["entries"][0]["response_text"] = "I skipped it again."
    packet, _key = build_blind_packet([case], split="development")

    rendered = packet["cases"][0]["entries"][0]["journal_entry"]
    assert rendered == (
        "I skipped the safety inspection because I wanted to leave early.\n\n"
        'Nudge: "What happened next?"\n\nResponse: I skipped it again.'
    )


def test_review_bundle_separates_reviewer_packet_from_parent_control(
    tmp_path: Path,
):
    write_review_bundle(
        output_dir=tmp_path / "bundle",
        root=tmp_path,
        source_paths={},
        cases=_cases(),
        split="development",
    )

    reviewer_packet = tmp_path / "bundle/reviewer_packet"
    parent_control = tmp_path / "bundle/parent_control"
    assert (reviewer_packet / "blind_packet.json").is_file()
    assert (reviewer_packet / "response_schema.json").is_file()
    assert not (reviewer_packet / "reconciliation_key.json").exists()
    assert (parent_control / "reconciliation_key.json").is_file()
    assert (parent_control / "audit_manifest.json").is_file()
    assert not (tmp_path / "bundle/blind_packet.json").exists()
    assert not (tmp_path / "bundle/reconciliation_key.json").exists()


def test_paired_delta_overlays_a_copy_and_preserves_original_labels():
    base = _base_labels()
    original_rows = deepcopy(base.to_dicts())
    packet, key = build_blind_packet(_cases(), split="development")
    delta, summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=_response(labels=["yes", "yes", "no"]),
        reviewer_b=_response(
            labels=["yes", "yes", "no"], reviewer_id="test-reviewer-b"
        ),
        base_labels_df=base,
        registry_df=_registry(),
        split="development",
        packet_sha256="packet-hash",
    )

    variant = apply_student_visible_target(
        base, delta, _registry(), retired_persona_ids=[]
    )

    assert base.to_dicts() == original_rows
    assert delta["student_visible_label"].to_list() == [-1, -1, 0]
    assert summary["qualification_agreement_rate"] == 1.0
    assert variant["alignment_security"].to_list() == [-1, -1, 0]
    assert variant["confidence_security"].to_list() == [
        "student_visible",
        "student_visible",
        "student_visible",
    ]
    assert variant["alignment_vector"].to_list()[0][5] == -1


def test_qualification_disagreement_marks_a_promotion_case_unresolved():
    packet, key = build_blind_packet(
        [{**_cases()[0], "split": "promotion", "case_id": "promotion:p1:security"}],
        split="promotion",
    )
    delta, summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=_response(
            labels=["yes", "yes", "no"],
            sustained_conflict="yes",
            split="promotion",
        ),
        reviewer_b=_response(
            labels=["yes", "no", "no"],
            sustained_conflict="no",
            reviewer_id="test-reviewer-b",
            split="promotion",
        ),
        base_labels_df=_base_labels(),
        registry_df=_registry(),
        split="promotion",
        packet_sha256="packet-hash",
    )

    assert summary["promotable"] is False
    assert delta["reconciliation_status"].unique().to_list() == ["unresolved"]
    assert delta["student_visible_label"].null_count() == 3


def test_uncertain_promotion_decisions_are_not_promotable():
    packet, key = build_blind_packet(
        [{**_cases()[0], "split": "promotion", "case_id": "promotion:p1:security"}],
        split="promotion",
    )
    delta, summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=_response(
            labels=["uncertain", "no", "no"],
            sustained_conflict="uncertain",
            split="promotion",
        ),
        reviewer_b=_response(
            labels=["uncertain", "no", "no"],
            sustained_conflict="uncertain",
            reviewer_id="test-reviewer-b",
            split="promotion",
        ),
        base_labels_df=_base_labels(),
        registry_df=_registry(),
        split="promotion",
        packet_sha256="packet-hash",
    )

    assert summary["promotable"] is False
    assert delta["reconciliation_status"].unique().to_list() == ["unresolved"]


def test_uncertain_entry_labels_block_promotion_even_when_case_reviews_agree():
    packet, key = build_blind_packet(
        [{**_cases()[0], "split": "promotion", "case_id": "promotion:p1:security"}],
        split="promotion",
    )
    delta, summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=_response(
            labels=["uncertain", "no", "no"],
            sustained_conflict="no",
            split="promotion",
        ),
        reviewer_b=_response(
            labels=["uncertain", "no", "no"],
            sustained_conflict="no",
            reviewer_id="test-reviewer-b",
            split="promotion",
        ),
        base_labels_df=_base_labels(),
        registry_df=_registry(),
        split="promotion",
        packet_sha256="packet-hash",
    )

    assert summary["unresolved_case_ids"] == []
    assert summary["unresolved_entry_count"] == 1
    assert summary["promotable"] is False
    assert delta["student_visible_label"].null_count() == 1


def test_paired_reviews_require_distinct_ids_and_the_exact_packet():
    packet, key = build_blind_packet(_cases(), split="development")
    with pytest.raises(ValueError, match="distinct reviewer_id"):
        build_paired_target_delta(
            packet=packet,
            reconciliation_key=key,
            reviewer_a=_response(labels=["yes", "yes", "no"]),
            reviewer_b=_response(labels=["yes", "yes", "no"]),
            base_labels_df=_base_labels(),
            registry_df=_registry(),
            split="development",
            packet_sha256="packet-hash",
        )

    with pytest.raises(ValueError, match="reviewed packet"):
        build_paired_target_delta(
            packet=packet,
            reconciliation_key=key,
            reviewer_a=_response(labels=["yes", "yes", "no"]),
            reviewer_b=_response(
                labels=["yes", "yes", "no"],
                reviewer_id="test-reviewer-b",
                packet_sha256="different-packet",
            ),
            base_labels_df=_base_labels(),
            registry_df=_registry(),
            split="development",
            packet_sha256="packet-hash",
        )


def test_response_case_responses_alias_is_supported_for_v2_packet_reviews():
    packet, key = build_blind_packet(_cases(), split="development")
    reviewer_a = _response(labels=["yes", "yes", "no"])
    reviewer_a["case_responses"] = reviewer_a.pop("cases")
    delta, _summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=reviewer_a,
        reviewer_b=_response(
            labels=["yes", "yes", "no"], reviewer_id="test-reviewer-b"
        ),
        base_labels_df=_base_labels(),
        registry_df=_registry(),
        split="development",
        packet_sha256="packet-hash",
    )

    assert delta.height == 3


def test_review_bundle_must_cover_every_declared_core_value():
    packet, key = build_blind_packet(_cases(), split="development")
    expected_cases = [
        _cases()[0],
        {
            **_cases()[0],
            "case_id": "development:p1:power",
            "dimension": "power",
        },
    ]

    with pytest.raises(ValueError, match="every expected full trajectory"):
        validate_review_bundle_cases(
            packet=packet,
            reconciliation_key=key,
            expected_cases=expected_cases,
            split="development",
        )


def test_review_bundle_rejects_packet_text_that_differs_from_source_cases():
    packet, key = build_blind_packet(_cases(), split="development")
    tampered_packet = deepcopy(packet)
    tampered_packet["cases"][0]["entries"][0]["journal_entry"] = "Different text."

    with pytest.raises(ValueError, match="journal text mismatch"):
        validate_review_bundle_cases(
            packet=tampered_packet,
            reconciliation_key=key,
            expected_cases=_cases(),
            split="development",
        )


def test_review_bundle_manifest_rejects_tampered_packet(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("source", encoding="utf-8")
    write_review_bundle(
        output_dir=tmp_path / "bundle",
        root=tmp_path,
        source_paths={"source": source_path},
        cases=_cases(),
        split="development",
    )
    packet_path = tmp_path / "bundle/reviewer_packet/blind_packet.json"
    key_path = tmp_path / "bundle/parent_control/reconciliation_key.json"
    schema_path = tmp_path / "bundle/reviewer_packet/response_schema.json"
    manifest_path = tmp_path / "bundle/parent_control/audit_manifest.json"

    validate_audit_manifest(
        manifest_path,
        root=tmp_path,
        packet_path=packet_path,
        key_path=key_path,
        schema_path=schema_path,
        expected_cases=_cases(),
        split="development",
    )
    packet_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_audit_manifest(
            manifest_path,
            root=tmp_path,
            packet_path=packet_path,
            key_path=key_path,
            schema_path=schema_path,
            expected_cases=_cases(),
            split="development",
        )


def test_promotion_threshold_receipt_binds_the_frozen_file(tmp_path: Path):
    manifest_path = tmp_path / "audit_manifest.json"
    manifest_path.write_text('{"split": "promotion"}\n', encoding="utf-8")
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text('{"thresholds": {}}\n', encoding="utf-8")

    receipt = record_promotion_threshold_receipt(
        manifest_path,
        root=tmp_path,
        threshold_path=threshold_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert receipt["threshold_sha256"] == sha256_file(threshold_path)
    assert validate_promotion_threshold_receipt(manifest, root=tmp_path) == receipt

    threshold_path.write_text('{"thresholds": {"changed": true}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_promotion_threshold_receipt(manifest, root=tmp_path)


def test_target_overlay_rejects_a_retired_persona():
    packet, key = build_blind_packet(_cases(), split="development")
    delta, _summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=_response(labels=["yes", "yes", "no"]),
        reviewer_b=_response(
            labels=["yes", "yes", "no"], reviewer_id="test-reviewer-b"
        ),
        base_labels_df=_base_labels(),
        registry_df=_registry(),
        split="development",
        packet_sha256="packet-hash",
    )

    with pytest.raises(ValueError, match="retired frozen-test persona"):
        apply_student_visible_target(
            _base_labels(), delta, _registry(), retired_persona_ids=["p1"]
        )


def test_live_manifest_derives_the_locked_fresh_promotion_population():
    registry = pl.read_parquet(REPO_ROOT / "logs/registry/personas.parquet")
    entries = pl.read_parquet(
        REPO_ROOT / "logs/judge_labels/consensus_labels.parquet"
    ).select("persona_id", "t_index", "date")
    manifest_path = REPO_ROOT / "config/evals/drift_v1_student_visible_v1.yaml"
    manifest, split = validate_target_manifest(manifest_path, registry, entries)

    assert manifest["target_version"] == TARGET_VERSION
    assert len(split.development_persona_ids) == 28
    assert len(split.retired_persona_ids) == 27
    assert len(split.promotion_persona_ids) == 24
    assert (
        tuple(manifest["locked_promotion_persona_ids"]) == split.promotion_persona_ids
    )
    assert set(split.retired_persona_ids).isdisjoint(split.promotion_persona_ids)
    assert "abf1ce49" in split.promotion_persona_ids


def test_split_derivation_rejects_overlapping_original_groups(tmp_path: Path):
    manifest = tmp_path / "holdout.yaml"
    manifest.write_text(
        "source_persona_count: 2\n"
        "train_persona_ids: [p1]\n"
        "val_persona_ids: [p1]\n"
        "test_persona_ids: [p2]\n",
        encoding="utf-8",
    )
    registry = pl.DataFrame(
        {"persona_id": ["p1", "p2", "p3"], "core_values": [["Security"]] * 3}
    )

    with pytest.raises(ValueError, match="overlap"):
        derive_target_split(registry, manifest)


def test_full_trajectory_cases_fail_on_missing_entry_text():
    entries = pl.DataFrame(
        {
            "persona_id": ["p1"],
            "t_index": [0],
            "date": ["2026-01-01"],
            "initial_entry": [None],
        }
    )
    with pytest.raises(ValueError, match="missing journal text"):
        build_full_trajectory_cases(entries, _registry(), ["p1"], split="development")
