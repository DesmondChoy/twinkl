"""Tests for the maintained Weekly Drift Reviewer contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    VerifierAssessment,
    WeeklyVerifierResponse,
    build_weekly_drift_reviewer_request,
    persist_weekly_drift_reviewer_receipt,
    validate_weekly_drift_reviewer_response,
)


def _request():
    return build_weekly_drift_reviewer_request(
        persona_id="deadbeef",
        week_start="2025-01-06",
        week_end="2025-01-12",
        core_values=["benevolence"],
        history=[
            {
                "t_index": 0,
                "date": "2025-01-06",
                "text": "Cancelled dinner with my family to stay at work.",
            }
        ],
        current_t_indices=[0],
    )


def _assessment() -> VerifierAssessment:
    return VerifierAssessment(
        t_index=0,
        dimension="benevolence",
        verdict="conflict",
        confidence="high",
        reason_code="direct_behavior_or_choice",
        evidence_quote="Cancelled dinner with my family",
    )


def test_request_and_response_contract_excludes_vif_critic_input():
    request = _request()
    response = WeeklyVerifierResponse(assessments=[_assessment()])

    validate_weekly_drift_reviewer_response(response, request)

    assert "VIF Critic" not in request.prompt
    assert "Cancelled dinner with my family" in request.prompt
    assert request.expected_coordinates == {(0, "benevolence")}


def test_conflict_quote_must_be_an_exact_journal_entry_substring():
    request = _request()
    response = WeeklyVerifierResponse(
        assessments=[
            _assessment().model_copy(update={"evidence_quote": "Invented quote"})
        ]
    )

    with pytest.raises(ValueError, match="Evidence quote"):
        validate_weekly_drift_reviewer_response(response, request)


class _FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.kwargs = None

    async def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_parsed=self.parsed,
            model="gpt-5.6-luna",
            id="response-1",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            output=[],
        )


@pytest.mark.asyncio
async def test_openai_caller_persists_effective_decision_and_frozen_contract(
    tmp_path: Path,
):
    responses = _FakeResponses(WeeklyVerifierResponse(assessments=[_assessment()]))
    reviewer = OpenAIWeeklyDriftReviewer(client=SimpleNamespace(responses=responses))

    receipt = await reviewer(_request())
    path = persist_weekly_drift_reviewer_receipt(receipt, tmp_path / "review.json")
    payload = json.loads(path.read_text())

    assert receipt.status == "ok"
    assert receipt.decisions[0].verdict == "conflict"
    assert responses.kwargs["model"] == "gpt-5.6-luna"
    assert responses.kwargs["reasoning"] == {"effort": "low"}
    assert responses.kwargs["store"] is False
    assert payload["schema_version"] == "weekly-drift-reviewer-receipt-v1"


@pytest.mark.asyncio
async def test_invalid_response_fails_closed_to_abstain():
    responses = _FakeResponses(WeeklyVerifierResponse(assessments=[]))
    reviewer = OpenAIWeeklyDriftReviewer(client=SimpleNamespace(responses=responses))

    receipt = await reviewer(_request())

    assert receipt.status == "invalid"
    assert receipt.attempts == 1
    assert [decision.verdict for decision in receipt.decisions] == ["abstain"]
    assert receipt.decisions[0].review_status == "invalid"


@pytest.mark.asyncio
async def test_provider_error_fails_closed_to_abstain():
    class _FailingResponses:
        async def parse(self, **_kwargs):
            raise RuntimeError("provider unavailable")

    reviewer = OpenAIWeeklyDriftReviewer(
        client=SimpleNamespace(responses=_FailingResponses())
    )

    receipt = await reviewer(_request())

    assert receipt.status == "error"
    assert receipt.attempts == 1
    assert receipt.error_type == "RuntimeError"
    assert [decision.verdict for decision in receipt.decisions] == ["abstain"]
