"""Coach-only recovery preserves frozen weekly work and measured operation timing."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from starlette.testclient import TestClient

from src.demo.api import create_app
from src.demo.contracts import (
    AssessmentTimeAdvanceRequest,
    CoachRetryRequest,
    SessionResumeState,
    WeeklyCoachGeneratedEvent,
    WeeklyDigestBuiltEvent,
)
from tests.demo.test_experience_service import (
    _create_assessment,
    _receipt,
    _service,
    _submit_request,
    _valid_coach_response,
)


async def failed_week():
    service, _, reviewer = _service(
        [_receipt(decision="no_nudge", nudge_text=None)] * 2
    )
    create = await _create_assessment(service)
    await service.submit_journal_entry(
        _submit_request(create, index=0, expected_revision=0)
    )
    response = await service.advance_assessment_time(
        AssessmentTimeAdvanceRequest(
            operation="advance_assessment_time",
            request_id="close-first",
            idempotency_key="a" * 64,
            session_id=create.profile.session_id,
            expected_revision=1,
            action="close_week",
        )
    )
    assert response.operation == "advance_assessment_time"
    request = CoachRetryRequest(
        operation="retry_coach",
        request_id="retry",
        idempotency_key="b" * 64,
        session_id=create.profile.session_id,
        expected_revision=response.session.revision,
        week_start="2026-07-20",
    )
    return service, reviewer, create, response.session, request


def test_http_adapter_accepts_coach_retry_and_validates_its_week():
    service, _, _, _, request = asyncio.run(failed_week())

    async def recovered_coach(prompt, _format=None, _instructions=None):
        return _valid_coach_response()

    service._coach_llm_complete = recovered_coach
    with TestClient(create_app(service)) as client:
        payload = request.model_dump(mode="json")
        invalid = client.post(
            "/api/experience", json={**payload, "week_start": "2026-07-21"}
        )
        recovered = client.post("/api/experience", json=payload)

    assert invalid.status_code == 422
    assert invalid.json()["requested_operation"] == "retry_coach"
    assert recovered.status_code == 200
    assert recovered.json()["operation"] == "retry_coach"
    assert recovered.json()["session"]["weekly_digest"]["coach_narrative"] is not None


@pytest.mark.asyncio
async def test_retry_recovers_only_coach_and_reuses_completed_work(monkeypatch):
    service, reviewer, create, before, request = await failed_week()
    original_events = list(service._events[before.session_id])
    calls = []
    clock = [datetime(2026, 7, 25, 8, 30, tzinfo=UTC)]
    ticks = [100.0]
    service._now = lambda: clock[0]
    monkeypatch.setattr("src.demo.experience_service.perf_counter", lambda: ticks[0])

    async def recovered_coach(prompt, _format=None, _instructions=None):
        calls.append(prompt)
        clock[0] += timedelta(seconds=2.345)
        ticks[0] += 2.345
        return _valid_coach_response()

    service._coach_llm_complete = recovered_coach
    response = await service.handle(request)
    assert response.operation == "retry_coach"
    assert response.session.weekly_digest.coach_narrative is not None
    assert response.session.revision == before.revision + 1
    assert response.session.assessment_clock == before.assessment_clock
    assert response.session.journal_entries == before.journal_entries
    assert response.session.drift_result == before.drift_result
    assert (
        response.session.weekly_reviewer_decisions == before.weekly_reviewer_decisions
    )
    assert service._events[before.session_id][:-1] == original_events
    event = service._events[before.session_id][-1]
    assert isinstance(event, WeeklyCoachGeneratedEvent)
    digest_event = next(
        event for event in original_events if isinstance(event, WeeklyDigestBuiltEvent)
    )
    assert event.parent_event_id == digest_event.event_id
    assert event.input_hash == digest_event.input_hash
    assert event.duration_ms == 2345
    assert datetime.fromisoformat(event.completed_at) - datetime.fromisoformat(
        event.started_at
    ) == timedelta(seconds=2.345)

    duplicate = await service.retry_coach(
        request.model_copy(update={"request_id": "dup"})
    )
    assert duplicate.operation == "retry_coach"
    assert duplicate.request_id == "dup"
    assert duplicate.event_ids == response.event_ids
    repeated = await service.retry_coach(request.model_copy(update={
        "idempotency_key": "c" * 64,
        "expected_revision": response.session.revision,
    }))
    assert repeated.operation == "retry_coach"
    assert repeated.session.revision == response.session.revision
    assert repeated.event_ids == response.event_ids
    assert len(calls) == len(reviewer.requests) == 1

    fresh, _, _ = _service([])
    restored = await fresh.create_session(create.model_copy(update={
        "resume_state": SessionResumeState(
            session_id=before.session_id,
            revision=response.session.revision,
            journal_entries=response.session.journal_entries,
            nudges=response.session.nudges,
            assessment_clock=response.session.assessment_clock,
            trace_events=service._events[before.session_id],
        ),
    }))
    assert restored.operation == "create_session"
    assert restored.session.weekly_digest == response.session.weekly_digest


@pytest.mark.asyncio
async def test_historical_retry_does_not_replace_current_week():
    service, reviewer, create, first, request = await failed_week()
    entry = _submit_request(create, index=1, expected_revision=first.revision)
    entry = entry.model_copy(update={
        "journal_entry": entry.journal_entry.model_copy(update={"date": "2026-07-27"}),
    })
    saved = await service.submit_journal_entry(entry)
    assert saved.operation == "submit_journal_entry"
    second = await service.advance_assessment_time(AssessmentTimeAdvanceRequest(
        operation="advance_assessment_time", request_id="close-second",
        idempotency_key="d" * 64, session_id=first.session_id,
        expected_revision=saved.session.revision, action="close_week",
    ))
    assert second.operation == "advance_assessment_time"
    calls = []

    async def recovered_coach(prompt, _format=None, _instructions=None):
        calls.append(prompt)
        return _valid_coach_response()

    service._coach_llm_complete = recovered_coach
    recovered = await service.retry_coach(request.model_copy(update={
        "expected_revision": second.session.revision,
    }))
    assert recovered.operation == "retry_coach"
    assert recovered.session.weekly_digest == second.session.weekly_digest
    assert recovered.session.drift_result == second.session.drift_result
    assert recovered.session.assessment_clock == second.session.assessment_clock
    assert service._events[first.session_id][-1].status == "complete"
    assert service._events[first.session_id][-1].input_refs[0].id.endswith("2026-07-20")
    assert len(calls) == 1
    assert len(reviewer.requests) == 2


@pytest.mark.asyncio
async def test_failure_retry_is_idempotent_and_rejects_stale_or_missing_week():
    service, reviewer, _, before, request = await failed_week()
    calls = []

    async def failed_coach(prompt, _format=None, _instructions=None):
        calls.append(prompt)
        raise TimeoutError

    service._coach_llm_complete = failed_coach
    response = await service.retry_coach(request)
    assert response.operation == "retry_coach"
    repeated = await service.retry_coach(request)
    assert repeated == response
    assert service._events[before.session_id][-1].status == "failed"
    assert len(calls) == 1
    stale = await service.retry_coach(request.model_copy(update={
        "idempotency_key": "c" * 64,
    }))
    assert stale.operation == "error"
    assert stale.error.code == "session_conflict"
    missing = await service.retry_coach(request.model_copy(update={
        "idempotency_key": "d" * 64,
        "week_start": "2026-07-13",
        "expected_revision": response.session.revision,
    }))
    assert missing.operation == "error"
    assert missing.error.code == "coach_retry_unavailable"
    conflict = await service.retry_coach(request.model_copy(update={
        "week_start": "2026-07-13",
    }))
    assert conflict.operation == "error"
    assert conflict.error.code == "idempotency_conflict"
    assert len(calls) == len(reviewer.requests) == 1


@pytest.mark.asyncio
async def test_invalid_coach_can_retry_and_each_fresh_response_is_validated():
    service, reviewer, _, before, request = await failed_week()
    invalid = json.loads(_valid_coach_response())
    invalid["weekly_mirror"] = invalid["weekly_mirror"].replace('"', "")
    responses = [json.dumps(invalid), json.dumps(invalid), _valid_coach_response()]
    calls = []

    async def coach(prompt, _format=None, _instructions=None):
        calls.append(prompt)
        return responses.pop(0)

    service._coach_llm_complete = coach
    for index in range(3):
        response = await service.retry_coach(request.model_copy(update={
            "idempotency_key": str(index + 1) * 64,
            "expected_revision": before.revision + index,
        }))
        assert response.operation == "retry_coach"
        event = service._events[before.session_id][-1]
        assert isinstance(event, WeeklyCoachGeneratedEvent)
        if index < 2:
            assert event.status == "invalid"
            assert event.error is not None and event.error.retryable
            assert event.validation is not None and not event.validation.valid
            assert event.validation.errors == [
                "No quoted phrase from selected evidence was detected."
            ]
            assert response.session.weekly_digest.coach_narrative is None
        else:
            assert event.status == "complete"
            assert event.validation is not None and event.validation.valid
            assert response.session.weekly_digest.coach_narrative is not None
        assert response.session.assessment_clock == before.assessment_clock
        assert response.session.journal_entries == before.journal_entries
    assert len(calls) == 3
    assert len(reviewer.requests) == 1
