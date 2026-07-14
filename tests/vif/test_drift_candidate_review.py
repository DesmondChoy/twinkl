"""Tests for the full legacy-discoverable Drift review cohort."""

from __future__ import annotations

import polars as pl
import pytest

from src.vif.drift_candidate_review import (
    ADJUDICATION_PROMPT_VERSION,
    ADJUDICATION_SCHEMA_VERSION,
    COHORT_VERSION,
    REVIEW_PROMPT_VERSION,
    REVIEW_SCHEMA_VERSION,
    ReviewProtocol,
    apply_adjudication,
    build_adjudication_packet,
    build_adjudication_response_schema,
    build_blind_shard,
    build_legacy_trajectory_inventory,
    build_review_response_schema,
    derive_review_outcomes,
    match_legacy_negative_controls,
    reconcile_reviews,
    selected_case_metadata,
    sha256_json,
    shard_review_cases,
    validate_review_response,
    validate_shard_coverage,
)
from src.vif.drift_target import TargetSplit


def _case(persona_id: str, dimension: str, length: int = 3) -> dict:
    return {
        "case_id": f"development_only:{persona_id}:{dimension}",
        "split": "development_only",
        "persona_id": persona_id,
        "dimension": dimension,
        "entries": [
            {
                "t_index": index,
                "date": f"2026-01-{index + 1:02d}",
                "initial_entry": f"Entry {index} for {persona_id}.",
                "nudge_text": None,
                "response_text": None,
            }
            for index in range(length)
        ],
    }


def _labels(spec: dict[str, list[int]]) -> pl.DataFrame:
    rows = []
    for persona_id, labels in spec.items():
        for index, label in enumerate(labels):
            rows.append(
                {
                    "persona_id": persona_id,
                    "t_index": index,
                    "date": f"2026-01-{index + 1:02d}",
                    "alignment_security": label,
                }
            )
    return pl.DataFrame(rows)


def _split(*persona_ids: str) -> TargetSplit:
    return TargetSplit(
        training_persona_ids=tuple(persona_ids),
        development_persona_ids=(),
        retired_persona_ids=(),
        promotion_persona_ids=(),
    )


def test_candidate_requires_adjacent_negatives_from_one_source():
    cases = [_case("p1", "security", length=2)]
    consensus = _labels({"p1": [-1, 0]})
    persisted = _labels({"p1": [0, -1]})

    inventory = build_legacy_trajectory_inventory(
        cases, consensus, persisted, _split("p1")
    )

    assert not inventory["legacy_candidate"][0]


def test_control_matching_is_deterministic_and_never_reuses_a_persona():
    cases = [
        _case("candidate_a", "security", 3),
        _case("candidate_b", "security", 4),
        _case("control_a", "security", 3),
        _case("control_b", "security", 4),
        _case("control_c", "security", 5),
    ]
    consensus = _labels(
        {
            "candidate_a": [-1, -1, 0],
            "candidate_b": [0, -1, -1, 0],
            "control_a": [0, 0, 0],
            "control_b": [0, 0, 0, 0],
            "control_c": [0, 0, 0, 0, 0],
        }
    )
    persisted = consensus.clone()
    split = _split(*(case["persona_id"] for case in cases))
    inventory = build_legacy_trajectory_inventory(cases, consensus, persisted, split)

    first = match_legacy_negative_controls(inventory)
    second = match_legacy_negative_controls(inventory)

    assert first.to_dicts() == second.to_dicts()
    assert first["control_persona_id"].n_unique() == 2
    assert set(first["control_persona_id"].to_list()).isdisjoint(
        {"candidate_a", "candidate_b"}
    )
    assert first["length_delta"].sum() == 0


def test_control_matching_fails_when_unique_people_are_insufficient():
    cases = [
        _case("candidate_a", "security"),
        _case("candidate_b", "security"),
        _case("control_a", "security"),
    ]
    consensus = _labels(
        {
            "candidate_a": [-1, -1, 0],
            "candidate_b": [-1, -1, 0],
            "control_a": [0, 0, 0],
        }
    )
    inventory = build_legacy_trajectory_inventory(
        cases, consensus, consensus, _split(*(case["persona_id"] for case in cases))
    )

    with pytest.raises(ValueError, match="unique-person control"):
        match_legacy_negative_controls(inventory)


def test_shards_cover_every_selected_case_once():
    inventory = pl.DataFrame(
        {
            "canonical_case_id": ["a", "b", "c"],
            "trajectory_length": [5, 4, 3],
        }
    )
    shards = shard_review_cases(inventory, max_cases=2, max_entries=8)

    validate_shard_coverage(shards, ["a", "b", "c"])
    with pytest.raises(ValueError, match="overlap"):
        validate_shard_coverage({"one": ["a"], "two": ["a", "b", "c"]}, ["a", "b", "c"])


def test_custom_review_protocol_versions_packets_and_schemas():
    protocol = ReviewProtocol(
        cohort_version="custom-cohort-v1",
        review_schema_version="custom-review-v1",
        review_prompt_version="custom-prompt-v1",
        adjudication_schema_version="custom-adjudication-v1",
        adjudication_prompt_version="custom-adjudication-prompt-v1",
    )
    case = _case("p1", "security")

    packet, key = build_blind_shard(
        shard_id="shard_001",
        case_ids=["p1:security"],
        cases_by_id={"p1:security": case},
        protocol=protocol,
    )
    review_schema = build_review_response_schema(protocol)
    adjudication_schema = build_adjudication_response_schema(protocol)

    assert packet["schema_version"] == "custom-review-v1"
    assert packet["cohort_version"] == "custom-cohort-v1"
    assert key["cohort_version"] == "custom-cohort-v1"
    assert review_schema["reviewer_prompt_version"] == "custom-prompt-v1"
    assert adjudication_schema["schema_version"] == "custom-adjudication-v1"


def _response(
    packet: dict,
    *,
    labels: list[str],
    reviewer_id: str,
    packet_sha256: str,
    schema_sha256: str,
) -> dict:
    has_pair = any(
        first == second == "yes"
        for first, second in zip(labels, labels[1:], strict=False)
    )
    sustained = "yes" if has_pair else "no"
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "cohort_version": COHORT_VERSION,
        "shard_id": packet["shard_id"],
        "packet_sha256": packet_sha256,
        "response_schema_sha256": schema_sha256,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "reviewer_id": reviewer_id,
        "reviewer_runtime": "test",
        "reviewed_at": "2026-07-13T12:00:00+00:00",
        "cases": [
            {
                "review_case_id": packet["cases"][0]["review_case_id"],
                "entry_assessments": [
                    {
                        "position": index,
                        "observable_conflict": label,
                        "reason_code": (
                            "direct_behavior_or_choice"
                            if label == "yes"
                            else "direct_aligned_or_neutral_behavior"
                        ),
                        "confidence": "high",
                    }
                    for index, label in enumerate(labels, start=1)
                ],
                "sustained_conflict": sustained,
                "delivery_state": "active" if has_pair else "none",
                "rationale": "The displayed behavior supports these labels.",
            }
        ],
    }


def test_reconciliation_preserves_entry_disagreement():
    case = _case("p1", "security")
    packet, key = build_blind_shard(
        shard_id="shard_001",
        case_ids=["p1:security"],
        cases_by_id={"p1:security": case},
    )
    packet_sha256 = sha256_json(packet)
    schema_sha256 = "schema"
    first = _response(
        packet,
        labels=["yes", "yes", "no"],
        reviewer_id="reviewer-a",
        packet_sha256=packet_sha256,
        schema_sha256=schema_sha256,
    )
    second = _response(
        packet,
        labels=["yes", "no", "no"],
        reviewer_id="reviewer-b",
        packet_sha256=packet_sha256,
        schema_sha256=schema_sha256,
    )

    rows = reconcile_reviews(
        packet=packet,
        key=key,
        reviewer_a=first,
        reviewer_b=second,
        packet_sha256=packet_sha256,
        response_schema_sha256=schema_sha256,
        reviewer_a_sha256="a",
        reviewer_b_sha256="b",
    )

    assert [row["final_conflict"] for row in rows] == [True, None, False]


def test_response_rejects_inconsistent_sustained_conflict():
    case = _case("p1", "security")
    packet, _key = build_blind_shard(
        shard_id="shard_001",
        case_ids=["p1:security"],
        cases_by_id={"p1:security": case},
    )
    response = _response(
        packet,
        labels=["yes", "yes", "no"],
        reviewer_id="reviewer-a",
        packet_sha256="packet",
        schema_sha256="schema",
    )
    response["cases"][0]["sustained_conflict"] = "no"

    with pytest.raises(ValueError, match="Inconsistent sustained_conflict"):
        validate_review_response(
            response,
            packet=packet,
            packet_sha256="packet",
            response_schema_sha256="schema",
        )


def test_uncertain_entry_breaks_a_run_without_making_the_case_sustained():
    case = _case("p1", "security")
    packet, _key = build_blind_shard(
        shard_id="shard_001",
        case_ids=["p1:security"],
        cases_by_id={"p1:security": case},
    )
    response = _response(
        packet,
        labels=["yes", "uncertain", "yes"],
        reviewer_id="reviewer-a",
        packet_sha256="packet",
        schema_sha256="schema",
    )
    response["cases"][0]["sustained_conflict"] = "uncertain"

    validate_review_response(
        response,
        packet=packet,
        packet_sha256="packet",
        response_schema_sha256="schema",
    )


def _selected_case(case_id: str, length: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "canonical_case_id": [case_id],
            "persona_id": ["p1"],
            "dimension": ["security"],
            "cohort_role": ["candidate"],
            "analysis_role": ["development_reference"],
            "historical_split": ["training"],
            "pair_id": ["pair_001"],
            "matched_case_id": ["control:security"],
            "trajectory_length": [length],
            "consensus_candidate": [True],
            "persisted_candidate": [True],
            "consensus_window_count": [1],
            "persisted_window_count": [1],
            "case_content_sha256": ["content"],
        }
    )


def _entry_target(case_id: str, labels: list[bool | None]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "canonical_case_id": case_id,
                "position": index,
                "t_index": index - 1,
                "date": f"2026-01-{index:02d}",
                "final_conflict": label,
                "resolution_status": (
                    "resolved" if label is not None else "unresolved"
                ),
                "resolution_method": (
                    "agreement" if label is not None else "unresolved"
                ),
            }
            for index, label in enumerate(labels, start=1)
        ]
    )


@pytest.mark.parametrize(
    ("labels", "expected_episodes", "expected_drift"),
    [
        ([True, True, True], 1, True),
        ([True, True, False, True, True], 2, True),
        ([True, None, True], 0, None),
    ],
)
def test_episode_derivation_uses_maximal_runs_and_fail_closed_cases(
    labels: list[bool | None], expected_episodes: int, expected_drift: bool | None
):
    case_id = "p1:security"
    outcomes, episodes = derive_review_outcomes(
        _entry_target(case_id, labels),
        _selected_case(case_id, len(labels)),
        cohort_sha256="cohort",
    )

    assert episodes.height == expected_episodes
    assert outcomes["has_drift"][0] is expected_drift
    assert outcomes["entry_agreement_count"][0] == sum(
        label is not None for label in labels
    )


def test_outcomes_distinguish_initial_agreement_from_adjudicated_resolution():
    case_id = "p1:security"
    entries = _entry_target(case_id, [True, True])
    entries = entries.with_columns(
        pl.Series("resolution_method", ["agreement", "adjudication"])
    )

    outcomes, _episodes = derive_review_outcomes(
        entries,
        _selected_case(case_id, 2),
        cohort_sha256="cohort",
    )

    assert outcomes["entry_agreement_count"][0] == 1


def test_selected_metadata_keeps_candidate_and_control_roles():
    inventory = pl.DataFrame(
        {
            "canonical_case_id": ["candidate", "control"],
            "persona_id": ["p1", "p2"],
            "dimension": ["security", "security"],
            "historical_split": ["training", "training"],
            "trajectory_length": [3, 3],
            "displayed_character_count": [30, 30],
            "consensus_candidate": [True, False],
            "persisted_candidate": [True, False],
            "consensus_window_count": [1, 0],
            "persisted_window_count": [1, 0],
            "consensus_episode_count": [1, 0],
            "persisted_episode_count": [1, 0],
            "case_content_sha256": ["one", "two"],
            "legacy_candidate": [True, False],
        }
    )
    pairs = pl.DataFrame(
        {
            "candidate_case_id": ["candidate"],
            "candidate_persona_id": ["p1"],
            "candidate_historical_split": ["training"],
            "control_case_id": ["control"],
            "control_persona_id": ["p2"],
            "control_historical_split": ["training"],
            "dimension": ["security"],
            "candidate_length": [3],
            "control_length": [3],
            "split_match_tier": [0],
            "length_delta": [0],
            "stable_tie_sha256": ["tie"],
            "analysis_role": ["development_reference"],
        }
    )

    selected = selected_case_metadata(inventory, pairs)

    assert set(selected["cohort_role"].to_list()) == {"candidate", "control"}
    assert set(selected["matched_case_id"].to_list()) == {"candidate", "control"}


@pytest.mark.parametrize(
    ("decision", "expected_conflict", "expected_status"),
    [
        ("yes", True, "resolved"),
        ("no", False, "resolved"),
        ("uncertain", None, "unresolved"),
    ],
)
def test_adjudication_changes_only_disputed_entries(
    decision: str, expected_conflict: bool | None, expected_status: str
):
    cases = [
        {
            "canonical_case_id": "p1:security",
            "declared_core_value": "Security",
            "review_rationales": ["First rationale.", "Second rationale."],
            "entries": [
                {
                    "position": 1,
                    "journal_entry": "I completed the safety check.",
                    "resolution_status": "resolved",
                    "reviewer_a_disposition": "no",
                    "reviewer_a_reason": "direct_aligned_or_neutral_behavior",
                    "reviewer_a_confidence": "high",
                    "reviewer_b_disposition": "no",
                    "reviewer_b_reason": "direct_aligned_or_neutral_behavior",
                    "reviewer_b_confidence": "high",
                },
                {
                    "position": 2,
                    "journal_entry": "I skipped the safety check.",
                    "resolution_status": "unresolved",
                    "reviewer_a_disposition": "yes",
                    "reviewer_a_reason": "direct_behavior_or_choice",
                    "reviewer_a_confidence": "high",
                    "reviewer_b_disposition": "no",
                    "reviewer_b_reason": "ambiguous",
                    "reviewer_b_confidence": "low",
                },
            ],
        }
    ]
    packet, key = build_adjudication_packet(cases)
    case_id = packet["cases"][0]["adjudication_case_id"]
    response = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "cohort_version": COHORT_VERSION,
        "packet_sha256": "packet",
        "response_schema_sha256": "schema",
        "adjudicator_prompt_version": ADJUDICATION_PROMPT_VERSION,
        "adjudicator_id": "adjudicator",
        "adjudicator_runtime": "test",
        "adjudicated_at": "2026-07-13T12:00:00+00:00",
        "cases": [
            {
                "adjudication_case_id": case_id,
                "entry_adjudications": [
                    {
                        "position": 2,
                        "observable_conflict": decision,
                        "reason_code": (
                            "direct_behavior_or_choice"
                            if decision == "yes"
                            else "ambiguous"
                        ),
                        "confidence": "medium",
                    }
                ],
                "rationale": "The displayed choice decides the disagreement.",
            }
        ],
    }
    entry_target = pl.DataFrame(
        [
            {
                "canonical_case_id": "p1:security",
                "position": 1,
                "final_conflict": False,
                "resolution_method": "agreement",
                "resolution_status": "resolved",
            },
            {
                "canonical_case_id": "p1:security",
                "position": 2,
                "final_conflict": None,
                "resolution_method": "unresolved",
                "resolution_status": "unresolved",
            },
        ]
    )

    result = apply_adjudication(
        entry_target,
        packet=packet,
        key=key,
        response=response,
        packet_sha256="packet",
        response_schema_sha256="schema",
        response_sha256="response",
    ).sort("position")

    assert result["final_conflict"][0] is False
    assert result["resolution_method"][0] == "agreement"
    assert result["adjudicator_disposition"][0] is None
    assert result["final_conflict"][1] is expected_conflict
    assert result["resolution_status"][1] == expected_status
