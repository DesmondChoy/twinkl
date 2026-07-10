"""Tests for the sustained-conflict decision benchmark."""

from __future__ import annotations

import polars as pl
import pytest

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift_benchmark import (
    build_detection_decisions,
    build_eligible_trajectories,
    build_reference_episodes,
    detect_sustained_conflict_episodes,
    episode_metrics,
    evidence_from_llm_records,
    evidence_from_ordinal_artifact,
    tune_detector_thresholds,
)


def _profiles(*rows: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"persona_id": persona_id, "core_values": values}
            for persona_id, values in rows
        ]
    )


def _label_rows(
    persona_id: str,
    dates: list[str],
    labels_by_dimension: dict[str, list[int | None]],
    *,
    confidence_by_dimension: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    rows = []
    for t_index, date in enumerate(dates):
        row = {"persona_id": persona_id, "t_index": t_index, "date": date}
        for dimension in SCHWARTZ_VALUE_ORDER:
            row[f"alignment_{dimension}"] = labels_by_dimension.get(
                dimension, [0] * len(dates)
            )[t_index]
            row[f"confidence_{dimension}"] = (confidence_by_dimension or {}).get(
                dimension, ["high"] * len(dates)
            )[t_index]
            row[f"consensus_agreement_{dimension}"] = 1.0
        rows.append(row)
    return pl.DataFrame(rows)


def _evidence_rows(
    persona_id: str,
    dimension: str,
    probabilities: list[float | None],
    uncertainties: list[float | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "source": "model",
                "persona_id": persona_id,
                "dimension": dimension,
                "t_index": index,
                "date": f"2026-01-{index + 1:02d}",
                "p_minus1": probability,
                "uncertainty": uncertainty,
                "predicted_class": -1
                if probability is not None and probability >= 0.5
                else 0,
                "evidence_kind": "soft_probability",
            }
            for index, (probability, uncertainty) in enumerate(
                zip(probabilities, uncertainties, strict=True)
            )
        ],
        schema_overrides={"p_minus1": pl.Float64, "uncertainty": pl.Float64},
    )


def test_reference_builder_tracks_extended_open_episode_and_confirmation():
    labels = _label_rows(
        "p1",
        ["2026-01-01", "2026-01-01", "2026-02-20"],
        {"security": [-1, -1, -1]},
    )

    episodes = build_reference_episodes(labels, _profiles(("p1", ["Security"])))

    assert episodes.height == 1
    episode = episodes.row(0, named=True)
    assert episode["onset_t_index"] == 0
    assert episode["confirmation_t_index"] == 1
    assert episode["end_t_index"] == 2
    assert episode["supporting_t_indices"] == [0, 1, 2]
    assert episode["length"] == 3
    assert episode["open_at_cutoff"] is True
    assert episode["delivery_state"] == "active"


def test_reference_builder_separates_runs_and_marks_recovery():
    labels = _label_rows(
        "p1",
        [f"2026-01-{day:02d}" for day in range(1, 6)],
        {"security": [-1, -1, 0, -1, -1]},
    )

    episodes = build_reference_episodes(labels, _profiles(("p1", ["Security"])))

    assert episodes["onset_t_index"].to_list() == [0, 3]
    assert episodes["delivery_state"].to_list() == ["recovered", "active"]


def test_reference_builder_keeps_values_independent_and_simultaneous():
    labels = _label_rows(
        "p1",
        ["2026-01-01", "2026-01-02"],
        {
            "security": [-1, -1],
            "benevolence": [-1, -1],
            "power": [1, 1],
        },
    )

    episodes = build_reference_episodes(
        labels,
        _profiles(("p1", ["Security", "Benevolence", "Power"])),
    )

    assert episodes.select("dimension").to_series().to_list() == [
        "benevolence",
        "security",
    ]


def test_reference_builder_treats_missing_or_no_majority_as_uncertain_break():
    labels = _label_rows(
        "p1",
        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        {"security": [-1, -1, 0, -1]},
        confidence_by_dimension={"security": ["high", "high", "no_majority", "high"]},
    )

    episodes = build_reference_episodes(labels, _profiles(("p1", ["Security"])))

    assert episodes.height == 1
    assert episodes["delivery_state"].item() == "uncertain"


def test_single_conflicts_do_not_form_reference_episode():
    labels = _label_rows(
        "p1",
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        {"security": [-1, 0, -1]},
    )

    episodes = build_reference_episodes(labels, _profiles(("p1", ["Security"])))

    assert episodes.is_empty()


def test_missing_t_index_breaks_reference_and_soft_evidence_runs():
    labels = _label_rows(
        "p1",
        ["2026-01-01", "2026-01-03"],
        {"security": [-1, -1]},
    ).with_columns(pl.Series("t_index", [0, 2]))
    evidence = _evidence_rows("p1", "security", [0.9, 0.9], [0.1, 0.1]).with_columns(
        pl.Series("t_index", [0, 2])
    )

    assert build_reference_episodes(labels, _profiles(("p1", ["Security"]))).is_empty()
    assert detect_sustained_conflict_episodes(
        evidence,
        probability_threshold=0.8,
        uncertainty_threshold=0.3,
    ).is_empty()


def test_invalid_declared_core_value_fails_clearly():
    labels = _label_rows("p1", ["2026-01-01", "2026-01-02"], {"security": [-1, -1]})

    with pytest.raises(ValueError, match="invalid core value"):
        build_reference_episodes(labels, _profiles(("p1", ["Not A Value"])))


def test_soft_detector_uses_probability_and_uncertainty_gates():
    evidence = _evidence_rows(
        "p1",
        "security",
        [0.7, 0.8, 0.9, 0.7],
        [0.1, 0.1, 0.7, 0.1],
    )

    episodes = detect_sustained_conflict_episodes(
        evidence,
        probability_threshold=0.6,
        uncertainty_threshold=0.3,
    )

    assert episodes.height == 1
    assert episodes["supporting_t_indices"].to_list()[0] == [0, 1]
    assert episodes["delivery_state"].item() == "uncertain"


def test_hard_label_detector_allows_missing_uncertainty_only_when_gate_disabled():
    evidence = _evidence_rows("p1", "security", [1.0, 1.0], [None, None])

    gated = detect_sustained_conflict_episodes(
        evidence,
        probability_threshold=1.0,
        uncertainty_threshold=0.3,
    )
    hard = detect_sustained_conflict_episodes(
        evidence,
        probability_threshold=1.0,
        uncertainty_threshold=None,
    )

    assert gated.is_empty()
    assert hard.height == 1


def test_pair_threshold_boundaries_are_inclusive():
    evidence = _evidence_rows("p1", "security", [0.5, 0.7], [0.3, 0.2])

    decisions = build_detection_decisions(
        evidence,
        probability_threshold=0.6,
        uncertainty_threshold=0.3,
    )

    assert decisions["pair_probability"].item() == pytest.approx(0.6)
    assert decisions["pair_uncertainty"].item() == pytest.approx(0.3)
    assert decisions["alert"].item() is True


def test_detector_keeps_sources_separate_and_rejects_duplicate_coordinates():
    first = _evidence_rows("p1", "security", [0.9, 0.9], [0.1, 0.1])
    second = first.with_columns(pl.lit("other_model").alias("source"))

    episodes = detect_sustained_conflict_episodes(
        pl.concat([first, second]),
        probability_threshold=0.8,
        uncertainty_threshold=0.3,
    )

    assert episodes.height == 2
    assert set(episodes["source"].to_list()) == {"model", "other_model"}
    with pytest.raises(ValueError, match="evidence contains duplicate"):
        build_detection_decisions(
            pl.concat([first, first]),
            probability_threshold=0.8,
            uncertainty_threshold=0.3,
        )


def test_episode_metrics_measure_hits_false_alarms_latency_and_recovery():
    profiles = _profiles(("positive", ["Security"]), ("negative", ["Security"]))
    positive_labels = _label_rows(
        "positive",
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        {"security": [-1, -1, 0]},
    )
    negative_labels = _label_rows(
        "negative",
        ["2026-01-01", "2026-01-02", "2026-01-03"],
        {"security": [0, 0, 0]},
    )
    labels = pl.concat([positive_labels, negative_labels])
    reference = build_reference_episodes(labels, profiles)
    eligible = build_eligible_trajectories(labels, profiles)
    predicted = pl.concat(
        [
            detect_sustained_conflict_episodes(
                _evidence_rows(
                    "positive", "security", [0.1, 0.8, 0.8], [0.1, 0.1, 0.1]
                ),
                probability_threshold=0.6,
                uncertainty_threshold=0.3,
            ),
            detect_sustained_conflict_episodes(
                _evidence_rows(
                    "negative", "security", [0.8, 0.8, 0.1], [0.1, 0.1, 0.1]
                ),
                probability_threshold=0.6,
                uncertainty_threshold=0.3,
            ),
        ]
    )

    decisions = pl.concat(
        [
            build_detection_decisions(
                _evidence_rows(
                    "positive", "security", [0.1, 0.8, 0.8], [0.1, 0.1, 0.1]
                ),
                probability_threshold=0.6,
                uncertainty_threshold=0.3,
            ),
            build_detection_decisions(
                _evidence_rows(
                    "negative", "security", [0.8, 0.8, 0.1], [0.1, 0.1, 0.1]
                ),
                probability_threshold=0.6,
                uncertainty_threshold=0.3,
            ),
        ]
    )
    metrics = episode_metrics(reference, predicted, eligible, decisions_df=decisions)

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["trajectory_false_alarm_rate"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(2 / 3)
    assert metrics["mean_latency_entries"] == pytest.approx(1.0)
    assert metrics["recovery_accuracy"] == pytest.approx(0.0)


def test_evidence_adapters_gate_to_declared_core_values():
    profiles = _profiles(("p1", ["Security"]))
    ordinal = pl.DataFrame(
        [
            {
                "persona_id": "p1",
                "t_index": 0,
                "date": "2026-01-01",
                "dimension": dimension,
                "predicted_class": -1,
                "uncertainty": 0.1,
                "class_probabilities": [0.8, 0.1, 0.1],
            }
            for dimension in ("security", "power")
        ]
    )
    llm_records = [
        {
            "status": "ok",
            "persona_id": "p1",
            "t_index": 0,
            "date": "2026-01-01",
            "core_values": ["Security"],
            "scores": {dimension: -1 for dimension in SCHWARTZ_VALUE_ORDER},
        }
    ]

    ordinal_evidence = evidence_from_ordinal_artifact(
        ordinal, profiles, source="ordinal"
    )
    llm_evidence = evidence_from_llm_records(llm_records, source="llm")

    assert ordinal_evidence["dimension"].to_list() == ["security"]
    assert ordinal_evidence["p_minus1"].item() == pytest.approx(0.8)
    assert llm_evidence["dimension"].to_list() == ["security"]
    assert llm_evidence["p_minus1"].item() == pytest.approx(1.0)
    assert llm_evidence["uncertainty"].item() is None


def test_threshold_tuning_prefers_guardrail_eligible_candidate():
    profiles = _profiles(("positive", ["Security"]), ("negative", ["Security"]))
    positive_labels = _label_rows(
        "positive",
        ["2026-01-01", "2026-01-02"],
        {"security": [-1, -1]},
    )
    negative_labels = _label_rows(
        "negative",
        ["2026-01-01", "2026-01-02"],
        {"security": [0, 0]},
    )
    labels = pl.concat([positive_labels, negative_labels])
    evidence = pl.concat(
        [
            _evidence_rows("positive", "security", [0.8, 0.8], [0.1, 0.1]),
            _evidence_rows("negative", "security", [0.55, 0.55], [0.1, 0.1]),
        ]
    )

    selected, grid = tune_detector_thresholds(
        evidence,
        build_reference_episodes(labels, profiles),
        build_eligible_trajectories(labels, profiles),
        probability_thresholds=[0.5, 0.7],
        uncertainty_thresholds=[0.3],
    )

    assert selected == {
        "probability_threshold": pytest.approx(0.7),
        "uncertainty_threshold": pytest.approx(0.3),
    }
    assert grid.filter(pl.col("meets_guardrails")).height == 1
