"""Protocol and receipt checks for the ``twinkl-752.5`` reassessment."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from scripts.experiments import reassess_twinkl_752_5 as study
from scripts.experiments import weekly_verifier_ablation as baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/evals/twinkl_752_5_reassessment_v1.yaml"


def _config() -> dict:
    return baseline._read_yaml(CONFIG_PATH)


def test_frozen_union_has_the_registered_counts_and_sources() -> None:
    config = _config()
    cases, summaries, targets, episodes, receipt = study._load_frozen_union(
        config, ROOT
    )

    assert receipt["counts"] == {
        "trajectories": 106,
        "personas": 105,
        "unique_entries": 882,
        "entry_value_cells": 894,
        "resolved_entry_value_cells": 892,
        "drifts": 33,
        "drift_trajectories": 28,
        "persona_weeks": 510,
    }
    assert len(cases) == summaries.height == 106
    assert targets.height == 894
    assert targets.filter(pl.col("final_conflict").is_null()).height == 2
    assert episodes.height == 33
    assert set(config["union"]["omitted_case_ids"]) <= {
        case["canonical_case_id"] for case in cases
    }
    assert receipt["opus_model_proof"]["raw_response_model_field"] is None
    assert receipt["opus_model_proof"]["recorded_runtime_models"] == ["claude-opus-4-8"]


def test_run_020_evidence_uses_full_profiles_and_matches_frozen_probabilities() -> None:
    config = _config()
    paths = study._artifact_paths(config, ROOT)
    full = pl.read_parquet(paths["run_020_full_evidence"])
    target = pl.read_parquet(paths["run_020_target_evidence"])
    provenance = json.loads(
        paths["run_020_full_evidence"]
        .with_suffix(".provenance.json")
        .read_text(encoding="utf-8")
    )
    manifest = baseline._read_json(paths["manifest"])

    assert full.height == 1302
    assert target.height == 894
    assert provenance["mc_seed"] == 7525001
    assert provenance["mc_samples"] == 50
    assert manifest["run_020_parity"]["overlap_cells"] == 85
    assert manifest["run_020_parity"]["max_p_minus1_absolute_delta"] <= 1e-6


def test_prompt_protocol_is_paired_and_trigger_schedule_is_frozen() -> None:
    config = _config()
    records, paths, manifest = study._load_prepared(config, ROOT)
    triggers = json.loads(paths["trigger_schedule"].read_text(encoding="utf-8"))
    opportunities = json.loads(
        paths["eligible_opportunities"].read_text(encoding="utf-8")
    )

    assert manifest["outcomes_inspected"] is False
    assert manifest["offline_diagnostic_scored"] is False
    assert manifest["prompt_counts"] == {
        study.WEEKLY_WITHOUT: 510,
        study.WEEKLY_WITH_RAW: 510,
        study.EARLY_WITHOUT: 19,
    }
    assert len(records) == 1039
    assert len(opportunities) == 671
    assert len(triggers) == 19
    assert len({(row["persona_id"], row["week_start"]) for row in triggers}) == 19
    assert all(
        date.fromisoformat(row["review_at_date"]).weekday() < 6 for row in triggers
    )

    by_key = {
        (row["persona_id"], row["week_start"], row["arm"]): row for row in records
    }
    for persona_id, week_start, arm in list(by_key):
        if arm != study.WEEKLY_WITHOUT:
            continue
        without = by_key[(persona_id, week_start, study.WEEKLY_WITHOUT)]
        with_raw = by_key[(persona_id, week_start, study.WEEKLY_WITH_RAW)]
        assert without["runtime_text_sha256"] == with_raw["runtime_text_sha256"]
        assert without["expected_coordinates"] == with_raw["expected_coordinates"]
        assert "CRITIC SIGNALS" not in without["prompt"]
        assert "CRITIC SIGNALS" in with_raw["prompt"]
    assert all(
        "CRITIC SIGNALS" not in row["prompt"]
        for row in records
        if row["arm"] == study.EARLY_WITHOUT
    )


def test_scheduler_recomputes_without_reference_labels() -> None:
    config = _config()
    paths = study._artifact_paths(config, ROOT)
    cases, _summaries, _targets, _episodes, _receipt = study._load_frozen_union(
        config, ROOT
    )
    evidence = pl.read_parquet(paths["run_020_target_evidence"])
    opportunities, triggers = study._scheduler_inputs(
        config, study._persona_records(cases), evidence
    )

    assert baseline._canonical_json(opportunities) == baseline._canonical_json(
        json.loads(paths["eligible_opportunities"].read_text(encoding="utf-8"))
    )
    assert baseline._canonical_json(triggers) == baseline._canonical_json(
        json.loads(paths["trigger_schedule"].read_text(encoding="utf-8"))
    )


def test_scoring_rejects_an_incomplete_response_set() -> None:
    config = _config()
    records, _paths, _manifest = study._load_prepared(config, ROOT)

    with pytest.raises(ValueError, match="Response set is incomplete"):
        study._assessment_map(
            [],
            records,
            repeats=int(config["study"]["repeats"]),
            requested_model=str(config["api"]["model"]),
        )


def test_completed_study_receipts_and_conclusions_are_frozen() -> None:
    config = _config()
    paths = study._artifact_paths(config, ROOT)
    responses = baseline._load_jsonl(paths["responses"])
    metrics = baseline._read_json(paths["metrics"])

    assert len(responses) == 3117
    assert sum(row["status"] == "ok" for row in responses) == 3077
    assert sum(row["status"] == "invalid" for row in responses) == 40
    assert baseline._sha256_file(paths["responses"]) == (
        "8c6a5127f14fb32a3ab1c3f465c44cce72cd6fc3051fb9e0778e0096d8111659"
    )
    assert baseline._sha256_file(paths["metrics"]) == (
        "b368c1ddd47e711700c6224f02bf1580c10e2d5a799f34e6e03d0bb990ed8f6b"
    )
    assert (
        metrics["architecture_questions"]["raw_score_value"][
            "old_conditional_rejection"
        ]
        == "inconclusive"
    )
    assert metrics["paired_trajectory_bootstrap"]["scheduling"]["deltas"]["recall"] == {
        "interval": [0.0, 0.0],
        "observed_median": 0.0,
    }
    assert metrics["offline_trigger_placement"]["observed_trigger_hits"] == 7
    assert metrics["actual_api_spend_usd"] == pytest.approx(4.887963)


def test_scheduled_review_adds_early_alerts_without_erasing_weekly_alerts() -> None:
    case = {
        "canonical_case_id": "p1:security",
        "persona_id": "p1",
        "dimension": "security",
        "entries": [
            {"t_index": 0, "date": "2026-07-13"},
            {"t_index": 1, "date": "2026-07-14"},
        ],
    }
    early_record = {
        "arm": study.EARLY_WITHOUT,
        "persona_id": "p1",
        "week_start": "2026-07-13",
        "review_event_id": "trigger:p1:2026-07-13",
        "review_at_date": "2026-07-14",
        "cutoff_t_index": 1,
    }

    def assessment(t_index: int, verdict: str) -> baseline.VerifierAssessment:
        return baseline.VerifierAssessment(
            t_index=t_index,
            dimension="security",
            verdict=verdict,
            confidence="high",
            reason_code="direct_behavior_or_choice",
            evidence_quote="evidence" if verdict == "conflict" else "",
        )

    weekly_predictions = {
        (study.WEEKLY_WITHOUT, 1, "p1", index, "security"): assessment(
            index, "conflict"
        )
        for index in (0, 1)
    }
    weekly_predictions.update(
        {
            (study.EARLY_WITHOUT, 1, "p1", index, "security"): assessment(
                index, "not_conflict"
            )
            for index in (0, 1)
        }
    )
    scheduled, _covered = study._setup_predictions(
        cases=[case],
        records=[early_record],
        predictions=weekly_predictions,
        setup=study.SCHEDULED,
        repeat=1,
    )
    assert len(scheduled) == 1
    assert scheduled[0]["alert_date"] == "2026-07-19"

    early_predictions = {
        (study.WEEKLY_WITHOUT, 1, "p1", index, "security"): assessment(
            index, "not_conflict"
        )
        for index in (0, 1)
    }
    early_predictions.update(
        {
            (study.EARLY_WITHOUT, 1, "p1", index, "security"): assessment(
                index, "conflict"
            )
            for index in (0, 1)
        }
    )
    scheduled, _covered = study._setup_predictions(
        cases=[case],
        records=[early_record],
        predictions=early_predictions,
        setup=study.SCHEDULED,
        repeat=1,
    )
    assert len(scheduled) == 1
    assert scheduled[0]["alert_date"] == "2026-07-14"
