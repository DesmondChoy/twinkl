"""Mechanical case eligibility and response-inclusive retrieval regressions."""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime

import numpy as np
import pytest
from pydantic import ValidationError

from scripts.experiments.nsm_cases import (
    ROOT,
    build_cases,
    consistency_sample,
    document_text,
    rank_case,
    retrieval_config,
)
from src.demo.contracts import CORE_VALUE_ORDER
from src.north_star.review import SourceEntry

RECORD = ROOT / "logs/experiments/reports/north_star_20260906/nsm_experiment.json"


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


@pytest.fixture(scope="module")
def prepared(record: dict) -> dict:
    return build_cases(record)


def test_complete_frozen_case_inventory_and_failed_reviews(prepared: dict) -> None:
    cases = prepared["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == 501
    assert len({case["persona_id"] for case in cases}) == 105
    assert Counter(case["weekly_state"] for case in cases) == {
        "active_drift": 24,
        "no_active_drift": 443,
        "insufficient_evidence": 34,
    }
    assert Counter(case["split"] for case in cases) == {
        "development": 391,
        "final": 110,
    }
    assert prepared["validation"]["history_counts"]["responses"] == 400
    assert prepared["validation"]["sparse_history_final_weeks"] == 87
    assert len(prepared["provenance"]["unreviewed_empty_weeks"]) == 3
    failed = [case for case in cases if case["upstream"]["status"] != "ok"]
    assert {case["case_id"] for case in failed} == {
        "742c98d6:week:2025-10-06",
        "ed67c9cc:week:2025-02-10",
    }
    for case in failed:
        for decision in case["upstream"]["current_decisions"]:
            assert decision["review_status"] == "invalid"
            assert decision["verdict"] == "abstain"
            assert decision["evidence_quote"] == ""
    historical = next(
        case for case in cases if case["case_id"] == "621be543:week:2025-09-29"
    )
    assert historical["upstream"]["status"] == "ok"
    assert historical["upstream"]["historical_v2_status_retained"] is True


def test_eligibility_uses_both_onset_order_and_date(prepared: dict) -> None:
    same_day_earlier = 0
    for case in prepared["cases"]:
        assert datetime.fromisoformat(case["cutoff"]).weekday() == 0
        assert case["cutoff"][:10] > case["week_end"]
        for value in case["values"]:
            indices = []
            for source in value["sources"]:
                assert set(source) == {"entry_id", "journal_entry", "nudge_response"}
                SourceEntry.model_validate(source)
                metadata = value["source_metadata"][source["entry_id"]]
                assert metadata["owner_id"] == case["persona_id"]
                assert metadata["date"] <= case["week_end"]
                assert metadata["available_at"] < case["cutoff"]
                indices.append(metadata["t_index"])
                if case["weekly_state"] == "active_drift":
                    assert metadata["t_index"] < case["onset_t_index"]
                    assert metadata["date"] <= case["onset_date"]
                    same_day_earlier += metadata["date"] == case["onset_date"]
                if source["nudge_response"]:
                    assert metadata["response_available_at"] < case["cutoff"]
                    assert metadata["entry_sequence"] < metadata["nudge_sequence"]
                    assert metadata["nudge_sequence"] < metadata["response_sequence"]
                    if case["onset_available_at"]:
                        assert (
                            metadata["response_available_at"]
                            < case["onset_available_at"]
                        )
            assert indices == sorted(indices, reverse=True)
    assert same_day_earlier > 0


def test_profile_and_active_priority_are_application_order(prepared: dict) -> None:
    for case in prepared["cases"]:
        assert case["core_values"] == [
            value for value in CORE_VALUE_ORDER if value in case["core_values"]
        ]
        assert "questionnaire" not in case["profile"]
        if case["weekly_state"] == "active_drift":
            active = [
                value
                for value in case["core_values"]
                if case["drift_result"]["core_value_states"][value] == "active_drift"
            ]
            winner = max(
                active,
                key=lambda value: case["drift_result"]["core_value_details"][value][
                    "current_run_length"
                ],
            )
            assert [value["core_value"] for value in case["values"]] == [winner]
        elif case["weekly_state"] == "no_active_drift":
            assert [value["core_value"] for value in case["values"]] == case[
                "core_values"
            ]
        else:
            assert case["values"] == []
            assert case["source_availability"]["insufficient_evidence_control"] is True
    empty = [
        case
        for case in prepared["cases"]
        if case["context_reason"] == "no_eligible_writing"
    ]
    assert len(empty) == 1
    assert empty[0]["weekly_state"] == "active_drift"
    assert empty[0]["onset_t_index"] == 0


def test_consistency_sample_is_stable_and_disjoint_by_persona(prepared: dict) -> None:
    sample = prepared["consistency_case_ids"]
    assert sample == consistency_sample(list(reversed(prepared["cases"])))
    lookup = {case["case_id"]: case for case in prepared["cases"]}
    assert len(sample) == len({lookup[key]["persona_id"] for key in sample}) == 20
    assert all(lookup[key]["split"] == "development" for key in sample)
    assert all(
        any(value["sources"] for value in lookup[key]["values"]) for key in sample
    )


def test_modified_frozen_source_hash_fails_before_cases(record: dict) -> None:
    damaged = copy.deepcopy(record)
    damaged["audits"]["upstream_weekly_inputs"]["source_hashes"][
        "src/drift_detector.py"
    ] = "0" * 64
    with pytest.raises(ValueError, match="source hashes disagree|source hash mismatch"):
        build_cases(damaged)


def test_partition_overlap_is_rejected(record: dict) -> None:
    damaged = copy.deepcopy(record)
    damaged["partition"]["final_persona_ids"][0] = damaged["partition"][
        "development_persona_ids"
    ][0]
    with pytest.raises(ValueError, match="disjoint 81/24"):
        build_cases(damaged)


def test_retrieval_serialization_preserves_source_boundaries() -> None:
    source = {
        "entry_id": "p:entry:0",
        "journal_entry": "I planted two trees.",
        "nudge_response": "I also watered them.",
    }
    assert document_text(source) == (
        "search_document: Journal Entry:\nI planted two trees."
        "\n\nPersona nudge response:\nI also watered them."
    )
    assert document_text({**source, "nudge_response": None}) == (
        "search_document: Journal Entry:\nI planted two trees."
    )
    with pytest.raises(ValidationError):
        document_text({**source, "cohort": "known_drift"})
    with pytest.raises(ValidationError):
        document_text({**source, "nudge_text": "An AI-written suggestion"})
    config = retrieval_config()
    assert config["local_files_only"] is True
    assert config["model_max_seq_length"] == 8192
    assert config["truncation"] is False
    assert "config_sha256" in config


def test_retrieval_selects_distinct_candidates_then_restores_recency() -> None:
    sources = [
        {
            "entry_id": f"p:entry:{index}",
            "journal_entry": str(index),
            "nudge_response": None,
        }
        for index in reversed(range(5))
    ]
    case = {
        "case_id": "p:week:2026-01-05",
        "values": [
            {
                "core_value": "benevolence",
                "sources": sources,
                "source_metadata": {
                    f"p:entry:{index}": {"t_index": index} for index in range(5)
                },
            }
        ],
    }
    vectors = {
        f"p:entry:{index}": np.array([score])
        for index, score in enumerate([0.8, 0.9, 0.7, 0.8, 0.1])
    }
    result = rank_case(case, vectors, {"benevolence": np.array([1.0])})
    ranking = result["values"]["benevolence"]
    assert ranking["top_entry_ids"] == ["p:entry:1", "p:entry:3", "p:entry:0"]
    assert result["candidate_ids_by_value"]["benevolence"] == [
        "p:entry:3",
        "p:entry:1",
        "p:entry:0",
    ]
    assert [row["rank"] for row in ranking["ranking"]] == [1, 2, 3, 4, 5]
    case["values"][0]["sources"] = sources[:2]
    shorter = rank_case(case, vectors, {"benevolence": np.array([1.0])})
    assert len(shorter["values"]["benevolence"]["top_entry_ids"]) == 2
    case["values"][0]["sources"] = []
    assert rank_case(case, vectors, {"benevolence": np.array([1.0])})[
        "candidate_ids_by_value"
    ] == {"benevolence": []}


def test_retrieval_rejects_duplicate_or_nonfinite_candidates() -> None:
    source = {"entry_id": "p:entry:0", "journal_entry": "Words", "nudge_response": None}
    case = {
        "case_id": "p:week:2026-01-05",
        "values": [
            {
                "core_value": "benevolence",
                "sources": [source, source],
                "source_metadata": {source["entry_id"]: {"t_index": 0}},
            }
        ],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        rank_case(case, {}, {})
    case["values"][0]["sources"] = [source]
    with pytest.raises(ValueError, match="Nonfinite"):
        rank_case(
            case, {"p:entry:0": np.array([np.nan])}, {"benevolence": np.array([1.0])}
        )
