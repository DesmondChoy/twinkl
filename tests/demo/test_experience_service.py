"""Tests for the manual Journal Entry Experience boundary."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from src.demo.api import create_app, create_deployment_app
from src.demo.canonical_fixture import build_canonical_fixture
from src.demo.contracts import (
    JournalEntry,
    JournalEntrySubmitRequest,
    ScenarioLoadRequest,
    SessionCreateRequest,
    SessionResumeState,
    TraceReadRequest,
)
from src.demo.experience_service import InMemoryExperienceService
from src.nudge.runtime import NudgeRuntimeReceipt, NudgeRuntimeRequest
from src.weekly_drift_reviewer import (
    VerifierAssessment,
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerReceipt,
    WeeklyDriftReviewerRequest,
)

ROOT = Path(__file__).resolve().parents[2]


class QueueNudgeRuntime:
    def __init__(
        self,
        receipts: list[NudgeRuntimeReceipt | Exception],
    ) -> None:
        self.receipts = iter(receipts)
        self.requests: list[NudgeRuntimeRequest] = []

    async def __call__(self, request: NudgeRuntimeRequest) -> NudgeRuntimeReceipt:
        self.requests.append(request)
        result = next(self.receipts)
        if isinstance(result, Exception):
            raise result
        return result


class DeterministicWeeklyReviewer:
    def __init__(
        self,
        *,
        verdict: str = "not_conflict",
        statuses: list[str] | None = None,
    ) -> None:
        self.verdict = verdict
        self.statuses = list(statuses or [])
        self.requests: list[WeeklyDriftReviewerRequest] = []

    async def __call__(
        self,
        request: WeeklyDriftReviewerRequest,
    ) -> WeeklyDriftReviewerReceipt:
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else "ok"
        history = {entry.t_index: entry for entry in request.history}
        assessments = []
        decisions = []
        for t_index in request.current_t_indices:
            for core_value in request.core_values:
                verdict = self.verdict if status == "ok" else "abstain"
                evidence_quote = (
                    history[t_index].text.split("\n\n", 1)[0]
                    if verdict == "conflict"
                    else ""
                )
                if status == "ok":
                    assessments.append(
                        VerifierAssessment(
                            t_index=t_index,
                            dimension=core_value,
                            verdict=verdict,  # type: ignore[arg-type]
                            confidence="high",
                            reason_code=(
                                "direct_behavior_or_choice"
                                if verdict == "conflict"
                                else "direct_aligned_or_neutral_behavior"
                            ),
                            evidence_quote=evidence_quote,
                        )
                    )
                decisions.append(
                    WeeklyDriftReviewerDecision(
                        persona_id=request.persona_id,
                        week_start=request.week_start,
                        week_end=request.week_end,
                        t_index=t_index,
                        date=history[t_index].date,
                        core_value=core_value,
                        verdict=verdict,  # type: ignore[arg-type]
                        confidence="high" if status == "ok" else None,
                        reason_code=(
                            "direct_behavior_or_choice"
                            if status == "ok" and verdict == "conflict"
                            else "direct_aligned_or_neutral_behavior"
                            if status == "ok"
                            else None
                        ),
                        evidence_quote=evidence_quote,
                        review_status=status,  # type: ignore[arg-type]
                    )
                )
        return WeeklyDriftReviewerReceipt(
            created_at="2026-07-25T08:30:00+00:00",
            persona_id=request.persona_id,
            week_start=request.week_start,
            week_end=request.week_end,
            core_values=request.core_values,
            current_t_indices=request.current_t_indices,
            prompt_name="weekly_vif_verifier",
            prompt_version="2.0",
            prompt_sha256=request.prompt_sha256,
            runtime_text_sha256=request.runtime_text_sha256,
            requested_model="gpt-5.6-luna",
            reasoning_effort="low",
            status=status,  # type: ignore[arg-type]
            attempts=1,
            latency_seconds=0.25,
            resolved_model="gpt-5.6-luna",
            response_id="weekly-response-1",
            refusal="Unable to answer." if status == "refusal" else None,
            validation_error=(
                "Response coordinate mismatch." if status == "invalid" else None
            ),
            error_type="TimeoutError" if status == "error" else None,
            error="Provider timed out." if status == "error" else None,
            assessments=assessments,
            decisions=decisions,
        )


def _ids() -> Iterator[int]:
    yield from range(1, 100)


def _service(
    receipts: list[NudgeRuntimeReceipt | Exception],
    *,
    weekly_verdict: str = "not_conflict",
    weekly_statuses: list[str] | None = None,
) -> tuple[
    InMemoryExperienceService,
    QueueNudgeRuntime,
    DeterministicWeeklyReviewer,
]:
    sequence = _ids()
    runtime = QueueNudgeRuntime(receipts)
    weekly_reviewer = DeterministicWeeklyReviewer(
        verdict=weekly_verdict,
        statuses=weekly_statuses,
    )
    service = InMemoryExperienceService(
        nudge_runtime=runtime,
        weekly_reviewer=weekly_reviewer,
        now=lambda: datetime(2026, 7, 25, 8, 30, tzinfo=UTC),
        make_id=lambda prefix: f"{prefix}-{next(sequence)}",
    )
    return service, runtime, weekly_reviewer


def _receipt(
    status: str = "ok",
    *,
    decision: str | None = "elaboration",
    nudge_text: str | None = "What felt most true in that moment?",
) -> NudgeRuntimeReceipt:
    raw_response = None
    refusal = None
    validation_error = None
    error_type = None
    error = None
    reason_text = "The entry names a moment but leaves its meaning unexplored."
    reason: str | None = reason_text
    if status == "ok":
        raw_response = (
            '{"decision":"'
            + str(decision)
            + '","reason":"'
            + reason_text
            + '","nudge_text":'
            + (f'"{nudge_text}"' if nudge_text is not None else "null")
            + "}"
        )
    elif status == "refusal":
        refusal = "Unable to answer."
        decision = None
        reason = None
        nudge_text = None
    elif status == "invalid":
        raw_response = '{"decision":"elaboration","nudge_text":"Why?"}'
        validation_error = "Question must contain 2-12 words."
        decision = None
        reason = None
        nudge_text = None
    else:
        error_type = "TimeoutError"
        error = "Provider timed out."
        decision = None
        reason = None
        nudge_text = None
    return NudgeRuntimeReceipt(
        created_at="2026-07-25T08:30:00+00:00",
        prompt_name="nudge_decision_and_generation",
        prompt_version="1.0.0",
        prompt_sha256="a" * 64,
        requested_model="gpt-5.6-luna",
        reasoning_effort="none",
        status=status,  # type: ignore[arg-type]
        latency_seconds=0.125,
        resolved_model="gpt-5.6-luna",
        response_id="response-1",
        raw_response=raw_response,
        refusal=refusal,
        validation_error=validation_error,
        error_type=error_type,
        error=error,
        decision=decision,
        reason=reason,
        nudge_text=nudge_text,
    )


async def _create(
    service: InMemoryExperienceService,
) -> SessionCreateRequest:
    profile = build_canonical_fixture().session.profile
    request = SessionCreateRequest(
        operation="create_session",
        request_id="create-1",
        idempotency_key="1" * 64,
        profile=profile,
    )
    response = await service.create_session(request)
    assert response.operation == "create_session"
    return request


def _submit_request(
    create_request: SessionCreateRequest,
    *,
    index: int,
    expected_revision: int,
) -> JournalEntrySubmitRequest:
    return JournalEntrySubmitRequest(
        operation="submit_journal_entry",
        request_id=f"submit-{index}",
        idempotency_key=f"{index + 2:x}" * 64,
        session_id=create_request.profile.session_id,
        expected_revision=expected_revision,
        journal_entry=JournalEntry(
            journal_entry_id=f"manual-entry-{index}",
            t_index=index,
            date=f"2026-07-{25 + index:02d}",
            content=f"Journal Entry {index} says enough to invite reflection.",
        ),
    )


@pytest.mark.asyncio
async def test_submission_is_idempotent_and_emits_real_nudge_events() -> None:
    service, runtime, weekly_reviewer = _service([_receipt()])
    create_request = await _create(service)
    request = _submit_request(create_request, index=0, expected_revision=0)

    response = await service.submit_journal_entry(request)
    repeated = await service.submit_journal_entry(
        request.model_copy(update={"request_id": "submit-retry"})
    )
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-1",
            session_id=create_request.profile.session_id,
        )
    )

    assert response.operation == "submit_journal_entry"
    assert repeated.operation == "submit_journal_entry"
    assert response.event_ids == repeated.event_ids
    assert len(response.session.journal_entries) == 1
    assert response.session.revision == 1
    assert response.session.nudges[0].outcome == "displayed"
    assert len(runtime.requests) == 1
    assert len(weekly_reviewer.requests) == 1
    assert response.session.drift_result is not None
    assert response.session.weekly_digest is not None
    assert response.session.weekly_reviewer_decisions
    assert trace.operation == "read_trace"
    assert [event.event_type for event in trace.events] == [
        "profile_confirmed",
        "journal_entry_submitted",
        "nudge_suppression_checked",
        "nudge_decided",
        "nudge_generated",
        "weekly_review_requested",
        "weekly_review_completed",
        "drift_detected",
        "weekly_digest_built",
    ]
    decision = next(
        event for event in trace.events if event.event_type == "nudge_decided"
    )
    assert decision.model_contract is not None
    assert decision.model_contract.model == "gpt-5.6-luna"
    assert decision.model_contract.reasoning_effort == "none"
    weekly_request = weekly_reviewer.requests[0]
    assert weekly_request.core_values == ["benevolence"]
    assert "VIF Critic" not in weekly_request.prompt
    weekly_event = next(
        event for event in trace.events if event.event_type == "weekly_review_completed"
    )
    assert weekly_event.model_contract is not None
    assert weekly_event.model_contract.model == "gpt-5.6-luna"
    assert weekly_event.model_contract.reasoning_effort == "low"


@pytest.mark.asyncio
async def test_weekly_integration_forms_drift_and_cites_visible_entries() -> None:
    service, _, weekly_reviewer = _service(
        [
            _receipt(decision="no_nudge", nudge_text=None),
            _receipt(decision="no_nudge", nudge_text=None),
        ],
        weekly_verdict="conflict",
    )
    create_request = await _create(service)

    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    second = await service.submit_journal_entry(
        _submit_request(create_request, index=1, expected_revision=1)
    )
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-weekly-success",
            session_id=create_request.profile.session_id,
        )
    )

    assert first.operation == "submit_journal_entry"
    assert first.session.drift_result is not None
    assert first.session.drift_result.delivery_state == "stable"
    assert second.operation == "submit_journal_entry"
    assert second.session.drift_result is not None
    assert second.session.drift_result.delivery_state == "active"
    assert len(second.session.weekly_reviewer_decisions) == 2
    assert second.session.weekly_digest is not None
    assert [row.t_index for row in second.session.weekly_digest.evidence] == [0, 1]
    assert weekly_reviewer.requests[1].current_t_indices == [0, 1]
    assert all(
        "VIF Critic" not in request.prompt for request in weekly_reviewer.requests
    )

    assert trace.operation == "read_trace"
    latest_events = trace.events[-4:]
    assert [event.event_type for event in latest_events] == [
        "weekly_review_requested",
        "weekly_review_completed",
        "drift_detected",
        "weekly_digest_built",
    ]
    assert [event.parent_event_id for event in latest_events[1:]] == [
        event.event_id for event in latest_events[:-1]
    ]
    digest_event = latest_events[-1]
    assert digest_event.event_type == "weekly_digest_built"
    assert digest_event.details.cited_journal_entry_ids == [
        "manual-entry-0",
        "manual-entry-1",
    ]


@pytest.mark.parametrize(
    ("status", "event_status"),
    [
        ("refusal", "refused"),
        ("invalid", "invalid"),
        ("error", "failed"),
    ],
)
@pytest.mark.asyncio
async def test_weekly_integration_fails_closed_to_abstain(
    status: str,
    event_status: str,
) -> None:
    service, _, _ = _service(
        [_receipt(decision="no_nudge", nudge_text=None)],
        weekly_statuses=[status],
    )
    create_request = await _create(service)

    response = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id=f"trace-weekly-{status}",
            session_id=create_request.profile.session_id,
        )
    )

    assert response.operation == "submit_journal_entry"
    assert {
        decision.verdict for decision in response.session.weekly_reviewer_decisions
    } == {"abstain"}
    assert response.session.drift_result is not None
    assert response.session.drift_result.delivery_state == "stable"
    assert response.session.weekly_digest is not None
    assert response.session.weekly_digest.response_mode == "high_uncertainty"
    assert response.session.weekly_digest.mode_rationale == (
        "The Weekly Drift Reviewer could not return usable evidence for this week."
    )
    assert trace.operation == "read_trace"
    completed = next(
        event for event in trace.events if event.event_type == "weekly_review_completed"
    )
    assert completed.status == event_status
    assert completed.error is not None
    assert completed.details.receipt.error is None
    assert trace.events[-2].event_type == "drift_detected"
    assert trace.events[-1].event_type == "weekly_digest_built"


@pytest.mark.asyncio
async def test_third_entry_is_suppressed_after_two_displayed_nudges() -> None:
    service, runtime, _ = _service([_receipt(), _receipt()])
    create_request = await _create(service)

    for index in range(3):
        response = await service.submit_journal_entry(
            _submit_request(
                create_request,
                index=index,
                expected_revision=index,
            )
        )
        assert response.operation == "submit_journal_entry"

    assert len(runtime.requests) == 2
    assert response.session.revision == 3
    assert response.session.nudges[-1].outcome == "suppressed"
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-suppressed",
            session_id=create_request.profile.session_id,
        )
    )
    assert trace.operation == "read_trace"
    suppression = [
        event
        for event in trace.events
        if event.event_type == "nudge_suppression_checked"
    ][-1]
    assert suppression.details.suppressed is True


@pytest.mark.asyncio
async def test_rejects_conflicting_retries_and_invalid_journal_order() -> None:
    service, runtime, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
    create_request = await _create(service)
    first_request = _submit_request(
        create_request,
        index=0,
        expected_revision=0,
    )
    first = await service.submit_journal_entry(first_request)
    conflicting_retry = await service.submit_journal_entry(
        first_request.model_copy(
            update={
                "journal_entry": first_request.journal_entry.model_copy(
                    update={"content": "Different content under the same retry key."}
                )
            }
        )
    )
    stale = await service.submit_journal_entry(
        _submit_request(
            create_request,
            index=1,
            expected_revision=0,
        )
    )
    second_request = _submit_request(
        create_request,
        index=1,
        expected_revision=1,
    )
    earlier = await service.submit_journal_entry(
        second_request.model_copy(
            update={
                "idempotency_key": "4" * 64,
                "journal_entry": second_request.journal_entry.model_copy(
                    update={"date": "2026-07-24"}
                ),
            }
        )
    )

    assert first.operation == "submit_journal_entry"
    assert conflicting_retry.operation == "error"
    assert conflicting_retry.error.code == "idempotency_conflict"
    assert stale.operation == "error"
    assert stale.error.code == "journal_order_conflict"
    assert earlier.operation == "error"
    assert earlier.error.code == "journal_order_conflict"
    assert len(runtime.requests) == 1


@pytest.mark.asyncio
async def test_browser_state_restores_after_in_memory_service_restart() -> None:
    first_service, _, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
    create_request = await _create(first_service)
    first = await first_service.submit_journal_entry(
        _submit_request(
            create_request,
            index=0,
            expected_revision=0,
        )
    )
    first_trace = await first_service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-restart",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert first_trace.operation == "read_trace"

    restarted_service, restarted_runtime, _ = _service(
        [_receipt(decision="no_nudge", nudge_text=None)]
    )
    restored = await restarted_service.create_session(
        create_request.model_copy(
            update={
                "request_id": "create-after-restart",
                "idempotency_key": "9" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=first.session.revision,
                    journal_entries=first.session.journal_entries,
                    nudges=first.session.nudges,
                    trace_events=first_trace.events,
                ),
            }
        )
    )
    second = await restarted_service.submit_journal_entry(
        _submit_request(
            create_request,
            index=1,
            expected_revision=1,
        )
    )

    assert restored.operation == "create_session"
    assert restored.session.revision == 1
    assert (
        restored.session.weekly_reviewer_decisions
        == first.session.weekly_reviewer_decisions
    )
    assert restored.session.drift_result == first.session.drift_result
    assert restored.session.weekly_digest == first.session.weekly_digest
    assert second.operation == "submit_journal_entry"
    assert second.session.revision == 2
    assert [entry.t_index for entry in second.session.journal_entries] == [0, 1]
    assert len(restarted_runtime.requests) == 1


@pytest.mark.asyncio
async def test_browser_nudge_response_recomputes_affected_week() -> None:
    service, runtime, weekly_reviewer = _service([_receipt()])
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-response",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert before.operation == "read_trace"

    response_text = "I chose the challenge because it felt more honest."
    updated_entry = first.session.journal_entries[0].model_copy(
        update={"nudge_response": response_text}
    )
    updated_nudge = first.session.nudges[0].model_copy(
        update={"outcome": "answered", "response": response_text}
    )
    synchronized = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-response",
                "idempotency_key": "b" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=2,
                    journal_entries=[updated_entry],
                    nudges=[updated_nudge],
                    trace_events=before.events,
                ),
            }
        )
    )

    assert synchronized.operation == "create_session"
    assert synchronized.session.revision == 2
    assert synchronized.session.journal_entries[0].nudge_response == response_text
    assert synchronized.session.nudges[0].outcome == "answered"
    assert len(runtime.requests) == 1
    assert len(weekly_reviewer.requests) == 2
    assert response_text in weekly_reviewer.requests[-1].history[0].text


@pytest.mark.asyncio
async def test_browser_nudge_skip_recomputes_without_changing_entry() -> None:
    service, runtime, weekly_reviewer = _service([_receipt()])
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-skip",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert before.operation == "read_trace"

    synchronized = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-skip",
                "idempotency_key": "8" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=2,
                    journal_entries=first.session.journal_entries,
                    nudges=[
                        first.session.nudges[0].model_copy(
                            update={"outcome": "skipped"}
                        )
                    ],
                    trace_events=before.events,
                ),
            }
        )
    )

    assert synchronized.operation == "create_session"
    assert synchronized.session.revision == 2
    assert synchronized.session.journal_entries == first.session.journal_entries
    assert synchronized.session.nudges[0].outcome == "skipped"
    assert len(runtime.requests) == 1
    assert len(weekly_reviewer.requests) == 2


@pytest.mark.asyncio
async def test_browser_update_rejects_a_stale_trace() -> None:
    service, _, weekly_reviewer = _service([_receipt()])
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-stale-update",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert before.operation == "read_trace"

    response_text = "Keep this answer attached to the current trace."
    stale = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-stale-trace",
                "idempotency_key": "9" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=2,
                    journal_entries=[
                        first.session.journal_entries[0].model_copy(
                            update={"nudge_response": response_text}
                        )
                    ],
                    nudges=[
                        first.session.nudges[0].model_copy(
                            update={
                                "outcome": "answered",
                                "response": response_text,
                            }
                        )
                    ],
                    trace_events=before.events[:-1],
                ),
            }
        )
    )

    assert stale.operation == "error"
    assert stale.error.code == "session_conflict"
    assert len(weekly_reviewer.requests) == 1


@pytest.mark.asyncio
async def test_equal_revision_browser_mismatch_is_not_a_successful_no_op() -> None:
    service, _, _ = _service(
        [
            _receipt(decision="no_nudge", nudge_text=None),
            _receipt(decision="no_nudge", nudge_text=None),
            _receipt(decision="no_nudge", nudge_text=None),
        ]
    )
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    second = await service.submit_journal_entry(
        _submit_request(create_request, index=1, expected_revision=1)
    )
    browser_trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-divergence",
            session_id=create_request.profile.session_id,
        )
    )
    third = await service.submit_journal_entry(
        _submit_request(create_request, index=2, expected_revision=2)
    )
    assert first.operation == "submit_journal_entry"
    assert second.operation == "submit_journal_entry"
    assert browser_trace.operation == "read_trace"
    assert third.operation == "submit_journal_entry"

    removed_id = second.session.journal_entries[0].journal_entry_id
    mismatch = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "equal-revision-mismatch",
                "idempotency_key": "7" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=third.session.revision,
                    journal_entries=second.session.journal_entries[1:],
                    nudges=[
                        nudge
                        for nudge in second.session.nudges
                        if nudge.journal_entry_id != removed_id
                    ],
                    trace_events=browser_trace.events,
                ),
            }
        )
    )

    assert mismatch.operation == "error"
    assert mismatch.error.code == "session_conflict"


@pytest.mark.asyncio
async def test_browser_removal_recomputes_without_removed_journal_entry() -> None:
    service, runtime, weekly_reviewer = _service(
        [
            _receipt(decision="no_nudge", nudge_text=None),
            _receipt(decision="no_nudge", nudge_text=None),
        ]
    )
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    second = await service.submit_journal_entry(
        _submit_request(create_request, index=1, expected_revision=1)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-removal",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert second.operation == "submit_journal_entry"
    assert before.operation == "read_trace"

    removed_id = second.session.journal_entries[0].journal_entry_id
    synchronized = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-removal",
                "idempotency_key": "c" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=3,
                    journal_entries=second.session.journal_entries[1:],
                    nudges=[
                        nudge
                        for nudge in second.session.nudges
                        if nudge.journal_entry_id != removed_id
                    ],
                    trace_events=before.events,
                ),
            }
        )
    )

    assert synchronized.operation == "create_session"
    assert synchronized.session.revision == 3
    assert [
        entry.journal_entry_id for entry in synchronized.session.journal_entries
    ] == ["manual-entry-1"]
    assert {
        decision.t_index for decision in synchronized.session.weekly_reviewer_decisions
    } == {1}
    assert synchronized.session.weekly_digest is not None
    assert synchronized.session.weekly_digest.n_entries == 1
    assert len(runtime.requests) == 2
    assert len(weekly_reviewer.requests) == 3
    assert [entry.t_index for entry in weekly_reviewer.requests[-1].history] == [1]


@pytest.mark.asyncio
async def test_removed_journal_entry_index_cannot_be_reused() -> None:
    service, _, _ = _service(
        [
            _receipt(decision="no_nudge", nudge_text=None),
            _receipt(decision="no_nudge", nudge_text=None),
        ]
    )
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    second = await service.submit_journal_entry(
        _submit_request(create_request, index=1, expected_revision=1)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-index-removal",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert second.operation == "submit_journal_entry"
    assert before.operation == "read_trace"
    removed = second.session.journal_entries[-1]
    synchronized = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-index-removal",
                "idempotency_key": "6" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=3,
                    journal_entries=second.session.journal_entries[:-1],
                    nudges=[
                        nudge
                        for nudge in second.session.nudges
                        if nudge.journal_entry_id != removed.journal_entry_id
                    ],
                    trace_events=before.events,
                ),
            }
        )
    )
    assert synchronized.operation == "create_session"

    reused_request = _submit_request(
        create_request,
        index=1,
        expected_revision=3,
    )
    reused = await service.submit_journal_entry(
        reused_request.model_copy(
            update={
                "idempotency_key": "5" * 64,
                "journal_entry": reused_request.journal_entry.model_copy(
                    update={"journal_entry_id": "replacement-entry"}
                ),
            }
        )
    )

    assert reused.operation == "error"
    assert reused.error.code == "journal_order_conflict"
    assert reused.error.message == "This Journal Entry position was already used."


@pytest.mark.asyncio
async def test_removed_only_entry_stays_removed_after_service_restart() -> None:
    service, _, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
    create_request = await _create(service)
    first = await service.submit_journal_entry(
        _submit_request(create_request, index=0, expected_revision=0)
    )
    before = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-before-empty-removal",
            session_id=create_request.profile.session_id,
        )
    )
    assert first.operation == "submit_journal_entry"
    assert before.operation == "read_trace"

    synchronized = await service.create_session(
        create_request.model_copy(
            update={
                "request_id": "sync-empty-removal",
                "idempotency_key": "d" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=2,
                    journal_entries=[],
                    nudges=[],
                    trace_events=before.events,
                ),
            }
        )
    )
    after = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-after-empty-removal",
            session_id=create_request.profile.session_id,
        )
    )
    assert synchronized.operation == "create_session"
    assert after.operation == "read_trace"

    restarted, _, _ = _service([])
    restored = await restarted.create_session(
        create_request.model_copy(
            update={
                "request_id": "restore-empty-removal",
                "idempotency_key": "e" * 64,
                "resume_state": SessionResumeState(
                    session_id=create_request.profile.session_id,
                    revision=synchronized.session.revision,
                    journal_entries=[],
                    nudges=[],
                    trace_events=after.events,
                ),
            }
        )
    )

    assert restored.operation == "create_session"
    assert restored.session.journal_entries == []
    assert restored.session.weekly_reviewer_decisions == []
    assert restored.session.drift_result is None
    assert restored.session.weekly_digest is None


@pytest.mark.parametrize("status", ["refusal", "invalid"])
@pytest.mark.asyncio
async def test_non_retryable_nudge_failures_keep_the_journal_entry(
    status: str,
) -> None:
    service, runtime, _ = _service([_receipt(status)])
    create_request = await _create(service)
    request = _submit_request(create_request, index=0, expected_revision=0)

    response = await service.submit_journal_entry(request)
    repeated = await service.submit_journal_entry(request)

    assert response.operation == "submit_journal_entry"
    assert repeated.operation == "submit_journal_entry"
    assert response.session.journal_entries[0].content.startswith("Journal Entry 0")
    assert response.session.revision == 1
    assert len(runtime.requests) == 1
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-failure",
            session_id=create_request.profile.session_id,
        )
    )
    assert trace.operation == "read_trace"
    nudge_event = next(
        event for event in trace.events if event.event_type == "nudge_decided"
    )
    assert nudge_event.status == status.replace("refusal", "refused")


@pytest.mark.asyncio
async def test_explicit_retry_reruns_only_failed_nudge_step() -> None:
    service, runtime, _ = _service([_receipt("error"), _receipt()])
    create_request = await _create(service)
    request = _submit_request(create_request, index=0, expected_revision=0)

    failed = await service.submit_journal_entry(request)
    recovered = await service.submit_journal_entry(
        request.model_copy(update={"request_id": "submit-explicit-retry"})
    )

    assert failed.operation == "submit_journal_entry"
    assert recovered.operation == "submit_journal_entry"
    assert len(recovered.session.journal_entries) == 1
    assert recovered.session.revision == 1
    assert recovered.session.nudges[0].outcome == "displayed"
    assert len(runtime.requests) == 2
    assert (
        len(
            [
                event
                for event in (
                    await service.read_trace(
                        TraceReadRequest(
                            operation="read_trace",
                            request_id="trace-retry",
                            session_id=create_request.profile.session_id,
                        )
                    )
                ).events
                if event.event_type == "journal_entry_submitted"
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_unexpected_injected_nudge_failure_uses_retryable_receipt_path() -> None:
    service, runtime, _ = _service(
        [
            RuntimeError("Injected nudge runtime failed."),
            _receipt(decision="no_nudge", nudge_text=None),
        ]
    )
    create_request = await _create(service)
    request = _submit_request(create_request, index=0, expected_revision=0)

    failed = await service.submit_journal_entry(request)
    recovered = await service.submit_journal_entry(
        request.model_copy(update={"request_id": "retry-injected-failure"})
    )

    assert failed.operation == "submit_journal_entry"
    assert recovered.operation == "submit_journal_entry"
    assert len(runtime.requests) == 2
    assert len(recovered.session.journal_entries) == 1
    trace = await service.read_trace(
        TraceReadRequest(
            operation="read_trace",
            request_id="trace-injected-failure",
            session_id=create_request.profile.session_id,
        )
    )
    assert trace.operation == "read_trace"
    nudge_events = [
        event for event in trace.events if event.event_type == "nudge_decided"
    ]
    assert [event.status for event in nudge_events] == ["failed", "complete"]


def test_http_adapter_validates_and_serves_contract_responses() -> None:
    service, _, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
    profile = build_canonical_fixture().session.profile

    with TestClient(create_app(service)) as client:
        health = client.get("/api/health")
        created = client.post(
            "/api/experience",
            json=SessionCreateRequest(
                operation="create_session",
                request_id="create-http",
                idempotency_key="f" * 64,
                profile=profile,
            ).model_dump(mode="json"),
        )
        submitted = client.post(
            "/api/experience",
            json=JournalEntrySubmitRequest(
                operation="submit_journal_entry",
                request_id="submit-http",
                idempotency_key="e" * 64,
                session_id=profile.session_id,
                expected_revision=0,
                journal_entry=JournalEntry(
                    journal_entry_id="http-entry",
                    t_index=0,
                    date="2026-07-25",
                    content="A specific moment that needs no follow-up.",
                ),
            ).model_dump(mode="json"),
        )
        traced = client.post(
            "/api/experience",
            json=TraceReadRequest(
                operation="read_trace",
                request_id="trace-http",
                session_id=profile.session_id,
            ).model_dump(mode="json"),
        )

    assert health.json() == {"status": "ok"}
    assert created.status_code == 200
    assert created.json()["operation"] == "create_session"
    assert created.json()["session"]["profile"]["top_values"] == ["benevolence"]
    assert submitted.status_code == 200
    assert submitted.json()["session"]["journal_entries"][0]["content"].startswith(
        "A specific moment"
    )
    assert traced.status_code == 200
    assert [event["event_type"] for event in traced.json()["events"]] == [
        "profile_confirmed",
        "journal_entry_submitted",
        "nudge_suppression_checked",
        "nudge_decided",
        "weekly_review_requested",
        "weekly_review_completed",
        "drift_detected",
        "weekly_digest_built",
    ]


def test_http_adapter_serves_the_built_experience_and_public_health(
    tmp_path: Path,
) -> None:
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<h1>Twinkl Experience</h1>")
    (static_root / "scenario.json").write_text('{"source":"saved_replay"}')

    with TestClient(create_app(static_root=static_root)) as client:
        railway_health = client.get("/health")
        api_health = client.get("/api/health")
        index = client.get("/")
        client_route = client.get("/journal/history")
        scenario = client.get("/scenario.json")
        missing_asset = client.get("/missing.js")

    assert railway_health.status_code == 200
    assert railway_health.text == "ok"
    assert api_health.json() == {"status": "ok"}
    assert "Twinkl Experience" in index.text
    assert "Twinkl Experience" in client_route.text
    assert scenario.json() == {"source": "saved_replay"}
    assert missing_asset.status_code == 404


def test_demo_credentials_protect_static_files_and_api_but_not_health(
    tmp_path: Path,
) -> None:
    static_root = tmp_path / "dist"
    static_root.mkdir()
    (static_root / "index.html").write_text("<h1>Protected demo</h1>")

    with TestClient(
        create_app(
            static_root=static_root,
            demo_credentials=("professor", "bounded-demo"),
        )
    ) as client:
        health = client.get("/health")
        anonymous_index = client.get("/")
        anonymous_api = client.get("/api/health")
        authorized_index = client.get(
            "/",
            auth=("professor", "bounded-demo"),
        )
        authorized_api = client.get(
            "/api/health",
            auth=("professor", "bounded-demo"),
        )

    assert health.status_code == 200
    assert anonymous_index.status_code == 401
    assert anonymous_api.status_code == 401
    assert anonymous_api.headers["www-authenticate"].startswith("Basic ")
    assert "professor" not in anonymous_api.text
    assert "bounded-demo" not in anonymous_api.text
    assert "Protected demo" in authorized_index.text
    assert authorized_api.json() == {"status": "ok"}


def test_provider_enabled_public_demo_requires_access_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TWINKL_PUBLIC_DEMO", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-provider-key")
    monkeypatch.delenv("TWINKL_DEMO_USERNAME", raising=False)
    monkeypatch.delenv("TWINKL_DEMO_PASSWORD", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Provider-enabled public demos require",
    ):
        create_deployment_app()


def test_public_demo_credentials_must_be_configured_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("TWINKL_DEMO_USERNAME", "professor")
    monkeypatch.delenv("TWINKL_DEMO_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="must be set together"):
        create_deployment_app()


def test_http_adapter_loads_each_saved_persona_at_its_first_week() -> None:
    service, _, _ = _service([])
    scenario_ids = [
        "stable-meera",
        "active-wei-jun",
        "recovered-marc",
        "uncertain-noor",
        "two-values-lukas",
    ]

    with TestClient(create_app(service, scenario_root=ROOT)) as client:
        for scenario_id in scenario_ids:
            loaded = client.post(
                "/api/experience",
                json=ScenarioLoadRequest(
                    operation="load_scenario",
                    request_id=f"load-{scenario_id}",
                    scenario_id=scenario_id,
                ).model_dump(mode="json"),
            )

            assert loaded.status_code == 200
            payload = loaded.json()
            first_week = payload["scenario"]["weeks"][0]
            visible_ids = set(first_week["journal_entry_ids"])
            assert payload["operation"] == "load_scenario"
            assert payload["scenario"]["source"] == "saved_replay"
            assert (
                payload["session"]["selection"]["selected_week"]
                == first_week["week_id"]
            )
            assert {
                entry["journal_entry_id"]
                for entry in payload["session"]["journal_entries"]
            } == visible_ids
            assert set(payload["session"]["trace_event_ids"]) == set(
                first_week["event_ids"]
            )

        missing = client.post(
            "/api/experience",
            json=ScenarioLoadRequest(
                operation="load_scenario",
                request_id="load-missing",
                scenario_id="missing-persona",
            ).model_dump(mode="json"),
        )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "scenario_not_found"


def test_http_adapter_returns_a_safe_scenario_integrity_error(
    tmp_path: Path,
) -> None:
    scenario_directory = tmp_path / "frontend/onboarding/public/scenarios"
    shutil.copytree(
        ROOT / "frontend/onboarding/public/scenarios",
        scenario_directory,
    )
    changed = scenario_directory / "stable-meera.json"
    changed.write_bytes(changed.read_bytes() + b" ")

    service, _, _ = _service([])
    with TestClient(create_app(service, scenario_root=tmp_path)) as client:
        response = client.post(
            "/api/experience",
            json=ScenarioLoadRequest(
                operation="load_scenario",
                request_id="load-tampered",
                scenario_id="stable-meera",
            ).model_dump(mode="json"),
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "scenario_integrity_error",
        "message": "The saved persona catalog failed its integrity check.",
        "retryable": False,
    }
    assert str(tmp_path) not in response.text
