"""Tests for the manual Journal Entry Experience boundary."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from starlette.testclient import TestClient

from src.demo.api import create_app
from src.demo.canonical_fixture import build_canonical_fixture
from src.demo.contracts import (
    JournalEntry,
    JournalEntrySubmitRequest,
    SessionCreateRequest,
    SessionResumeState,
    TraceReadRequest,
)
from src.demo.experience_service import InMemoryExperienceService
from src.nudge.runtime import NudgeRuntimeReceipt, NudgeRuntimeRequest


class QueueNudgeRuntime:
    def __init__(self, receipts: list[NudgeRuntimeReceipt]) -> None:
        self.receipts = iter(receipts)
        self.requests: list[NudgeRuntimeRequest] = []

    async def __call__(self, request: NudgeRuntimeRequest) -> NudgeRuntimeReceipt:
        self.requests.append(request)
        return next(self.receipts)


def _ids() -> Iterator[int]:
    yield from range(1, 100)


def _service(
    receipts: list[NudgeRuntimeReceipt],
) -> tuple[InMemoryExperienceService, QueueNudgeRuntime]:
    sequence = _ids()
    runtime = QueueNudgeRuntime(receipts)
    service = InMemoryExperienceService(
        nudge_runtime=runtime,
        now=lambda: datetime(2026, 7, 25, 8, 30, tzinfo=UTC),
        make_id=lambda prefix: f"{prefix}-{next(sequence)}",
    )
    return service, runtime


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
    reason = "The entry names a moment but leaves its meaning unexplored."
    if status == "ok":
        raw_response = (
            '{"decision":"'
            + str(decision)
            + '","reason":"'
            + reason
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
    service, runtime = _service([_receipt()])
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
    assert trace.operation == "read_trace"
    assert [event.event_type for event in trace.events] == [
        "profile_confirmed",
        "journal_entry_submitted",
        "nudge_suppression_checked",
        "nudge_decided",
        "nudge_generated",
    ]
    decision = trace.events[-2]
    assert decision.model_contract is not None
    assert decision.model_contract.model == "gpt-5.6-luna"
    assert decision.model_contract.reasoning_effort == "none"


@pytest.mark.asyncio
async def test_third_entry_is_suppressed_after_two_displayed_nudges() -> None:
    service, runtime = _service([_receipt(), _receipt()])
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
    service, runtime = _service([_receipt(decision="no_nudge", nudge_text=None)])
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
    first_service, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
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

    restarted_service, restarted_runtime = _service(
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
    assert second.operation == "submit_journal_entry"
    assert second.session.revision == 2
    assert [entry.t_index for entry in second.session.journal_entries] == [0, 1]
    assert len(restarted_runtime.requests) == 1


@pytest.mark.parametrize("status", ["refusal", "invalid"])
@pytest.mark.asyncio
async def test_non_retryable_nudge_failures_keep_the_journal_entry(
    status: str,
) -> None:
    service, runtime = _service([_receipt(status)])
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
    assert trace.events[-1].status == status.replace("refusal", "refused")


@pytest.mark.asyncio
async def test_explicit_retry_reruns_only_failed_nudge_step() -> None:
    service, runtime = _service([_receipt("error"), _receipt()])
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


def test_http_adapter_validates_and_serves_contract_responses() -> None:
    service, _ = _service([_receipt(decision="no_nudge", nudge_text=None)])
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
    ]
