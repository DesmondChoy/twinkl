"""Tests for the twinkl-j0ck hybrid soft-vote target builder."""

from __future__ import annotations

import json

import polars as pl
import pytest

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.soft_vote_target import (
    SOFT_TARGET_CLASS_ORDER,
    build_hybrid_soft_vote_target,
    resolve_activity_then_polarity,
)


def _scores(**overrides: int) -> dict[str, int]:
    scores = {value_name: 0 for value_name in SCHWARTZ_VALUE_ORDER}
    scores.update(overrides)
    return scores


def _result(entry_id: str, scores: dict[str, int]) -> dict:
    return {
        "entry_id": entry_id,
        "scores": scores,
        "rationales": {
            value_name: f"Evidence for {value_name}."
            for value_name, score in scores.items()
            if score != 0
        },
    }


def _label_row(
    *,
    persona_id: str,
    t_index: int,
    scores: dict[str, int],
    security_rationale: str,
) -> dict:
    return {
        "persona_id": persona_id,
        "t_index": t_index,
        "date": "2026-01-01",
        "alignment_vector": [scores[name] for name in SCHWARTZ_VALUE_ORDER],
        **{f"alignment_{name}": scores[name] for name in SCHWARTZ_VALUE_ORDER},
        "rationales_json": json.dumps({"security": security_rationale}),
    }


def _fixture() -> dict:
    persona_id = "example"
    t_index = 1
    entry_id = "example__1"
    manifest = pl.DataFrame(
        {
            "entry_id": [entry_id],
            "persona_id": [persona_id],
            "t_index": [t_index],
            "date": ["2026-01-01"],
        }
    )
    pass_scores = [
        _scores(achievement=1, benevolence=1),
        _scores(achievement=1, benevolence=1),
        _scores(achievement=0, benevolence=-1),
        _scores(achievement=0, benevolence=-1),
        _scores(achievement=-1, benevolence=0),
    ]
    passes = {
        index: [_result(entry_id, scores)]
        for index, scores in enumerate(pass_scores, start=1)
    }

    consensus_scores = _scores(achievement=1, benevolence=0, security=0)
    repaired_scores = _scores(achievement=0, benevolence=0, security=1)
    consensus = pl.DataFrame(
        [
            _label_row(
                persona_id=persona_id,
                t_index=t_index,
                scores=consensus_scores,
                security_rationale="Historical Security rationale.",
            )
        ]
    )
    repaired = pl.DataFrame(
        [
            _label_row(
                persona_id=persona_id,
                t_index=t_index,
                scores=repaired_scores,
                security_rationale="Active-state Security rationale.",
            )
        ]
    )
    security_target = pl.DataFrame(
        {
            "persona_id": [persona_id],
            "t_index": [t_index],
            "date": ["2026-01-01"],
            "new_label": [1],
            "vote_minus1": [1],
            "vote_neutral": [0],
            "vote_plus1": [3],
            "vote_count": [4],
            "agreement_count": [3],
            "decision_method": ["tie_break_review"],
            "target_policy": ["security_active_critic_state_v1"],
            "state_contract_version": ["active_critic_state_v1"],
        }
    )
    return {
        "twinkl_754_manifest": manifest,
        "twinkl_754_vote_passes": passes,
        "consensus_labels": consensus,
        "repaired_security_labels": repaired,
        "security_vote_target": security_target,
    }


def test_builds_value_major_soft_vector_and_matching_hard_labels():
    target = build_hybrid_soft_vote_target(**_fixture())

    assert target.height == 1
    row = target.to_dicts()[0]
    assert row["entry_id"] == "example__1"
    assert row["soft_target_class_order"] == list(SOFT_TARGET_CLASS_ORDER)
    assert row["soft_target_value_order"] == list(SCHWARTZ_VALUE_ORDER)
    assert len(row["soft_alignment_vector"]) == 30

    achievement_index = SCHWARTZ_VALUE_ORDER.index("achievement")
    achievement_slice = row["soft_alignment_vector"][
        achievement_index * 3 : achievement_index * 3 + 3
    ]
    assert achievement_slice == pytest.approx([0.2, 0.4, 0.4])
    assert row["alignment_achievement"] == 1
    assert row["vote_count_achievement"] == [1, 2, 2]
    assert row["vote_total_achievement"] == 5

    benevolence_index = SCHWARTZ_VALUE_ORDER.index("benevolence")
    assert row["alignment_benevolence"] == 0
    assert row["soft_alignment_vector"][
        benevolence_index * 3 : benevolence_index * 3 + 3
    ] == pytest.approx([0.4, 0.2, 0.4])

    security_index = SCHWARTZ_VALUE_ORDER.index("security")
    assert row["alignment_security"] == 1
    assert row["vote_count_security"] == [1, 0, 3]
    assert row["vote_total_security"] == 4
    assert row["soft_alignment_vector"][
        security_index * 3 : security_index * 3 + 3
    ] == pytest.approx([0.25, 0.0, 0.75])
    assert row["alignment_vector"] == [
        row[f"alignment_{name}"] for name in SCHWARTZ_VALUE_ORDER
    ]
    assert json.loads(row["rationales_json"])["security"] == (
        "Active-state Security rationale."
    )


def test_resolver_preserves_activity_then_polarity_behavior():
    assert resolve_activity_then_polarity([1, 1, 0, 0, -1]) == 1
    assert resolve_activity_then_polarity([1, 1, 0, -1, -1]) == 0
    assert resolve_activity_then_polarity([0, 0, 0, 1, -1]) == 0


def test_rejects_missing_five_pass_input():
    fixture = _fixture()
    fixture["twinkl_754_vote_passes"].pop(5)

    with pytest.raises(ValueError, match="exactly vote passes 1 through 5"):
        build_hybrid_soft_vote_target(**fixture)


def test_rejects_consensus_that_does_not_match_raw_votes():
    fixture = _fixture()
    fixture["consensus_labels"] = fixture["consensus_labels"].with_columns(
        pl.lit(0).alias("alignment_achievement")
    )

    with pytest.raises(ValueError, match="do not match the stored consensus"):
        build_hybrid_soft_vote_target(**fixture)


def test_rejects_unresolved_or_malformed_security_votes():
    fixture = _fixture()
    fixture["security_vote_target"] = fixture["security_vote_target"].with_columns(
        pl.lit(3).alias("vote_count")
    )

    with pytest.raises(ValueError, match="Invalid Security vote counts"):
        build_hybrid_soft_vote_target(**fixture)


def test_rejects_coordinate_date_drift():
    fixture = _fixture()
    fixture["repaired_security_labels"] = fixture[
        "repaired_security_labels"
    ].with_columns(pl.lit("2026-01-02").alias("date"))

    with pytest.raises(ValueError, match="dates do not match"):
        build_hybrid_soft_vote_target(**fixture)
