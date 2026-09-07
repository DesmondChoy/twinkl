"""Behavioral checks for the definitions-only Weekly Drift comparison."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError

from scripts.experiments import weekly_drift_definitions as study
from src import weekly_drift_reviewer as reviewer
from src.drift_detector import detect_drift
from src.models.judge import SCHWARTZ_VALUE_ORDER


def _row(
    persona_id="persona-a",
    current=(0, 1),
    values=("benevolence",),
    origin=date(2025, 1, 6),
):
    start = origin + timedelta(days=min(current))
    start -= timedelta(days=start.weekday())
    request = reviewer.WeeklyDriftReviewerRequest(
        persona_id=persona_id,
        week_start=start.isoformat(),
        week_end=(start + timedelta(days=6)).isoformat(),
        core_values=list(values),
        history=[
            reviewer.WeeklyDriftReviewerEntry(
                t_index=index,
                date=(origin + timedelta(days=index)).isoformat(),
                text=f"I refused to help today. I left early on day {index}.",
            )
            for index in range(max(current) + 1)
        ],
        current_t_indices=list(current),
        prompt="Frozen request receipt",
        prompt_sha256="frozen-prompt",
        runtime_text_sha256="frozen-source",
    )
    variants = {}
    for variant in study.VARIANTS:
        payload = {
            "instructions": f"Frozen {variant} instructions",
            "input_data": "Frozen displayed Journal Entries",
            "schema": reviewer.WeeklyVerifierResponse.model_json_schema(),
            "prompt_version": "3.0" if variant == "baseline" else "4.0",
        }
        payload["request_sha256"] = study.digest(payload)
        variants[variant] = payload
    return {
        "case_id": f"{persona_id}:{start.isoformat()}",
        "request": request.model_dump(mode="json"),
        "variants": variants,
    }


def _parsed(row, verdicts=None):
    verdicts = verdicts or {}
    assessments = []
    for index in row["request"]["current_t_indices"]:
        for value in row["request"]["core_values"]:
            verdict = verdicts.get((index, value), "conflict")
            assessments.append(
                reviewer.VerifierAssessment(
                    t_index=index,
                    dimension=value,
                    verdict=verdict,
                    confidence="high",
                    reason_code=(
                        "direct_behavior_or_choice"
                        if verdict == "conflict"
                        else "direct_aligned_or_neutral_behavior"
                        if verdict == "not_conflict"
                        else "ambiguous"
                    ),
                    evidence_quote=(
                        "I refused to help today." if verdict == "conflict" else ""
                    ),
                )
            )
    return reviewer.WeeklyVerifierResponse(assessments=assessments)


def _result(row, variant, *, verdicts=None, status="ok", repeat=1):
    attempt = {
        "event": "finished",
        "status": status,
        "parsed": _parsed(row, verdicts).model_dump(mode="json"),
    }
    return study.terminal_result(row, variant, repeat, [attempt])


def _drifts(row, verdicts=None):
    result = _result(row, "baseline", verdicts=verdicts)
    decisions = [
        reviewer.WeeklyDriftReviewerDecision.model_validate(item)
        for item in result["decisions"]
    ]
    return [
        drift.model_dump(mode="json")
        for drift in detect_drift(
            decisions, persona_id=row["request"]["persona_id"]
        ).drifts
    ]


def test_score_aggregates_cross_week_history_without_counting_snapshots_twice():
    rows = [
        _row(current=(0,), origin=date(2025, 1, 12)),
        _row(current=(1, 2), origin=date(2025, 1, 12)),
    ]
    responses = [_result(row, variant) for row in rows for variant in study.VARIANTS]

    scores = study.score_rows(rows, responses, repeats=1)
    run = scores["runs"]["1"]

    assert run["before_total"] == run["after_total"] == run["unchanged"] == 1
    for variant in study.VARIANTS:
        assert run["drifts"][variant][0]["supporting_t_indices"] == [0, 1, 2]
    assert run["changed_entry_decisions"] == []
    assert set(run["per_core_value"]) == set(SCHWARTZ_VALUE_ORDER)
    assert scores["counts"]["power"]["baseline"]["runs"] == [0]
    assert sum(item["before"] for item in run["per_core_value"].values()) == 1


def test_equal_totals_still_report_core_value_redistribution():
    row = _row(values=("benevolence", "security"))
    responses = [
        _result(
            row,
            variant,
            verdicts={(index, excluded): "not_conflict" for index in (0, 1)},
        )
        for variant, excluded in (
            ("baseline", "security"),
            ("definitions", "benevolence"),
        )
    ]

    run = study.score_rows([row], responses, repeats=1)["runs"]["1"]

    assert run["before_total"] == run["after_total"] == 1
    assert run["delta"] == 0
    assert run["per_core_value"]["benevolence"] == {
        "before": 1,
        "after": 0,
        "delta": -1,
    }
    assert run["per_core_value"]["security"] == {
        "before": 0,
        "after": 1,
        "delta": 1,
    }
    assert len(run["changed_entry_decisions"]) == 4
    assert run["removed"][0]["core_value"] == "benevolence"
    assert run["added"][0]["core_value"] == "security"


def test_equal_per_value_counts_still_report_different_personas():
    before = _drifts(_row("persona-a"))
    after = _drifts(_row("persona-b"))

    comparison = study.compare_drifts(before, after)

    assert comparison["delta"] == 0
    assert comparison["per_core_value"]["benevolence"]["delta"] == 0
    assert comparison["unchanged"] == 0
    assert comparison["removed"][0]["persona_id"] == "persona-a"
    assert comparison["added"][0]["persona_id"] == "persona-b"


def test_same_drift_id_with_extended_end_is_a_boundary_change():
    before = _drifts(_row())
    after = _drifts(_row(current=(0, 1, 2)))
    assert before[0]["drift_id"] == after[0]["drift_id"]

    comparison = study.compare_drifts(before, after)

    assert comparison["unchanged"] == 0
    assert len(comparison["changed_boundaries"]) == 1
    assert comparison["changed_boundaries"][0]["before"]["end_t_index"] == 1
    assert comparison["changed_boundaries"][0]["after"]["end_t_index"] == 2


@pytest.mark.parametrize("reverse", [False, True], ids=["split", "merge"])
def test_overlapping_split_or_merge_is_not_an_arbitrary_one_to_one_match(reverse):
    row = _row(current=(0, 1, 2, 3, 4))
    continuous = _drifts(row)
    separated = _drifts(row, {(2, "benevolence"): "not_conflict"})
    before, after = (separated, continuous) if reverse else (continuous, separated)

    comparison = study.compare_drifts(before, after)

    assert comparison["delta"] == (-1 if reverse else 1)
    assert comparison["unchanged"] == 0
    assert len(comparison["overlap_changes"]) == 1
    assert comparison["added"] == comparison["removed"] == []


def test_changed_onset_with_overlap_is_reported_as_a_boundary_change():
    row = _row(current=(0, 1, 2))
    before = _drifts(row, {(2, "benevolence"): "not_conflict"})
    after = _drifts(row, {(0, "benevolence"): "not_conflict"})

    comparison = study.compare_drifts(before, after)

    assert comparison["unchanged"] == 0
    assert len(comparison["overlap_changes"]) == 1
    assert comparison["added"] == comparison["removed"] == []


def test_different_quote_alone_does_not_change_detected_drift_identity():
    before = _drifts(_row())
    after = deepcopy(before)
    after[0]["evidence_quotes"] = ["I left early on day 0.", "I left early on day 1."]

    comparison = study.compare_drifts(before, after)

    assert comparison["unchanged"] == 1
    assert comparison["changed_boundaries"] == []
    assert comparison["added"] == comparison["removed"] == []


def test_same_span_with_different_termination_is_reported_separately():
    row = _row(current=(0, 1, 2))
    before = _drifts(row, {(2, "benevolence"): "not_conflict"})
    after = _drifts(row, {(2, "benevolence"): "abstain"})

    comparison = study.compare_drifts(before, after)

    assert comparison["before_total"] == comparison["after_total"] == 1
    assert comparison["changed_boundaries"] == []
    assert len(comparison["changed_termination"]) == 1
    assert comparison["overlap_changes"][0]["kind"] == "termination_change"


def test_failed_review_keeps_abstain_coordinates_and_breaks_cross_week_drift():
    rows = [
        _row(current=(0,), origin=date(2025, 1, 12)),
        _row(current=(1,), origin=date(2025, 1, 12)),
    ]
    responses = [
        _result(
            row,
            variant,
            status="invalid" if variant == "definitions" and index == 1 else "ok",
        )
        for index, row in enumerate(rows)
        for variant in study.VARIANTS
    ]

    run = study.score_rows(rows, responses, repeats=1)["runs"]["1"]

    assert run["before_total"] == 1
    assert run["after_total"] == 0
    assert run["request_statuses"]["definitions"] == {"ok": 1, "invalid": 1}
    assert run["changed_entry_decisions"][0]["after"] == "abstain"


@pytest.mark.parametrize("corruption", ["missing_response", "duplicate_response"])
def test_scoring_rejects_missing_or_duplicate_terminal_requests(corruption):
    row = _row()
    responses = [_result(row, variant) for variant in study.VARIANTS]
    if corruption == "missing_response":
        responses.pop()
    else:
        responses[1] = deepcopy(responses[0])

    with pytest.raises(ValueError):
        study.score_rows([row], responses, repeats=1)


@pytest.mark.parametrize("corruption", ["missing_coordinate", "duplicate_coordinate"])
def test_scoring_rejects_missing_or_duplicate_effective_coordinates(corruption):
    row = _row()
    responses = [_result(row, variant) for variant in study.VARIANTS]
    if corruption == "missing_coordinate":
        responses[0]["decisions"].pop()
    else:
        responses[0]["decisions"][1] = deepcopy(responses[0]["decisions"][0])

    with pytest.raises(ValueError):
        study.score_rows([row], responses, repeats=1)


@pytest.mark.parametrize("field,value", [("variant", "baseline"), ("repeat", 2)])
def test_scoring_rejects_result_metadata_that_disagrees_with_request_key(field, value):
    row = _row()
    responses = [_result(row, variant) for variant in study.VARIANTS]
    responses[1][field] = value
    responses[1]["request_sha256"] = row["variants"][responses[1]["variant"]][
        "request_sha256"
    ]

    with pytest.raises(ValueError):
        study.score_rows([row], responses, repeats=1)


@pytest.mark.parametrize(
    "field,value",
    [("persona_id", "another-persona"), ("date", "2025-01-20")],
)
def test_scoring_rejects_effective_decisions_with_wrong_provenance(field, value):
    row = _row()
    responses = [_result(row, variant) for variant in study.VARIANTS]
    responses[1]["decisions"][0][field] = value

    with pytest.raises(ValueError):
        study.score_rows([row], responses, repeats=1)


def test_scoring_rejects_failed_terminal_status_with_successful_decisions():
    row = _row()
    responses = [_result(row, variant) for variant in study.VARIANTS]
    responses[1]["status"] = "error"

    with pytest.raises(ValueError):
        study.score_rows([row], responses, repeats=1)


def _client(*outcomes):
    return SimpleNamespace(
        responses=SimpleNamespace(parse=AsyncMock(side_effect=outcomes))
    )


def _provider_response(row):
    return SimpleNamespace(output_parsed=_parsed(row), model="gpt-5.6-luna")


def test_execute_request_sends_exact_frozen_messages_and_records_success():
    row = _row()
    client = _client(_provider_response(row))
    events = []

    result = asyncio.run(
        study.execute_request(
            client, row, "baseline", 1, study.SETTINGS, [], events.append
        )
    )

    call = client.responses.parse.call_args.kwargs
    assert call["instructions"] == row["variants"]["baseline"]["instructions"]
    assert call["input"] == row["variants"]["baseline"]["input_data"]
    assert call["model"] == "gpt-5.6-luna"
    assert call["reasoning"] == {"effort": "low"}
    assert call["text_format"] is reviewer.WeeklyVerifierResponse
    assert call["store"] is False
    assert result["status"] == "ok"
    assert [event["event"] for event in events] == ["started", "finished"]
    assert len(result["decisions"]) == 2


def test_invalid_output_is_terminal_and_fails_closed_without_retry():
    row = _row()
    invalid = SimpleNamespace(
        output_parsed=reviewer.WeeklyVerifierResponse(assessments=[])
    )
    client = _client(invalid, _provider_response(row))

    result = asyncio.run(
        study.execute_request(
            client, row, "definitions", 1, study.SETTINGS, [], lambda _: None
        )
    )

    assert client.responses.parse.call_count == 1
    assert result["status"] == "invalid"
    assert len(result["decisions"]) == 2
    assert all(item["verdict"] == "abstain" for item in result["decisions"])
    assert all(item["confidence"] is None for item in result["decisions"])


@pytest.mark.parametrize("recovers", [False, True])
def test_transient_provider_error_obeys_two_attempt_limit(monkeypatch, recovers):
    row = _row()
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    client = _client(timeout, _provider_response(row) if recovers else timeout)
    monkeypatch.setattr(study.asyncio, "sleep", AsyncMock())
    events = []

    result = asyncio.run(
        study.execute_request(
            client, row, "baseline", 1, study.SETTINGS, [], events.append
        )
    )

    assert client.responses.parse.call_count == 2
    assert result["attempts"] == 2
    assert result["status"] == ("ok" if recovers else "error")
    assert [event["attempt_number"] for event in events] == [1, 1, 2, 2]


def test_interrupted_reserved_attempt_is_not_sent_again():
    row = _row()
    client = _client(_provider_response(row))
    attempts = [{"event": "started", "attempt_number": 1}]
    events = []

    result = asyncio.run(
        study.execute_request(
            client, row, "baseline", 1, study.SETTINGS, attempts, events.append
        )
    )

    client.responses.parse.assert_not_called()
    assert events == []
    assert result["status"] == "error"
    assert result["unknown_interrupted_attempt"] is True
    assert all(item["verdict"] == "abstain" for item in result["decisions"])
