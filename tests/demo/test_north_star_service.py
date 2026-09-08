"""Independent NSM lifecycle checks at the live Experience boundary."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from src.demo.contracts import (
    AssessmentTimeAdvanceRequest,
    NorthStarReviewedEvent,
    NorthStarReviewRequest,
    SessionDeleteRequest,
    SessionResumeState,
    TraceReadRequest,
)
from src.north_star.runtime import (
    LIVE_POLICY_PATH,
    NorthStarRequest,
    OpenAINorthStarRuntime,
    _PrivateResponseLedger,
    pending_north_star_record,
)
from tests.demo.test_experience_service import (
    _create_assessment,
    _receipt,
    _service,
    _submit_request,
)
from tests.north_star.test_runtime import FakeProvider, count_requests


class ControlledRuntime:
    def __init__(self, *, blocked: bool = False, fail: bool = False):
        self.requests: list[NorthStarRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.fail = fail
        self.forgotten = []

    async def __call__(self, request: NorthStarRequest, *, retry: bool = False):
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        return pending_north_star_record(request).model_copy(
            update={
                "status": "failed" if self.fail else "not_eligible",
                "reason": "injected_failure" if self.fail else "injected_omission",
                "retryable": self.fail,
            }
        )

    def forget(self, record):
        self.forgotten.append(record.input_hash)


async def reviewed_session(runtime: ControlledRuntime):
    service, _, reviewer = _service([_receipt(decision=None, nudge_text=None)])
    service._north_star_runtime = runtime
    create = await _create_assessment(service)
    first = await service.submit_journal_entry(
        _submit_request(create, index=0, expected_revision=0)
    )
    assert first.operation == "submit_journal_entry"
    response = await service.advance_assessment_time(
        AssessmentTimeAdvanceRequest(
            operation="advance_assessment_time",
            request_id="close",
            idempotency_key="e" * 64,
            session_id=create.profile.session_id,
            expected_revision=1,
            action="close_week",
        )
    )
    assert response.operation == "advance_assessment_time"
    request = NorthStarReviewRequest(
        operation="review_north_star",
        request_id="moment",
        session_id=create.profile.session_id,
        expected_revision=response.session.revision,
        week_start="2026-07-20",
    )
    return service, create, reviewer, response.session, request


@pytest.mark.asyncio
async def test_fresh_live_policy_passes_service_publication_and_reuse_validation(
    tmp_path,
):
    service, _, reviewer, session, request = await reviewed_session(ControlledRuntime())
    snapshot = service._north_star_request(session, request.week_start)
    ledger = _PrivateResponseLedger(tmp_path / "live-budget.json", LIVE_POLICY_PATH)
    provider = FakeProvider(
        tmp_path,
        supportive={(snapshot.core_values[0], snapshot.writing[0].entry_id)},
        ledger=ledger,
    )
    service._north_star_runtime = OpenAINorthStarRuntime(
        provider=provider,
        count_requests=count_requests,
        policy_path=LIVE_POLICY_PATH,
    )
    weekly_events = list(service._events[session.session_id])
    first = await service.review_north_star(request)
    assert first.operation == "north_star_reviewed"
    event = service._events[session.session_id][-1]
    assert isinstance(event, NorthStarReviewedEvent)
    assert event.source == "live_run"
    assert event.details.record.status == "complete"
    assert event.details.record.selected.entry_id == snapshot.writing[0].entry_id
    assert event.model_contract.model == "gpt-5.6-luna"
    generated = provider.generated
    charged = ledger.path.read_bytes()

    reused = await service.review_north_star(request)
    assert reused.operation == "north_star_reviewed"
    assert reused.event_ids == first.event_ids
    assert provider.generated == generated
    assert ledger.path.read_bytes() == charged
    assert len(reviewer.requests) == 1
    assert [
        event for event in service._events[session.session_id]
        if not isinstance(event, NorthStarReviewedEvent)
    ] == weekly_events


@pytest.mark.asyncio
async def test_weekly_result_precedes_nsm_and_duplicate_requests_coalesce():
    runtime = ControlledRuntime(blocked=True)
    service, _, reviewer, session, request = await reviewed_session(runtime)
    assert session.weekly_digest is not None
    assert runtime.requests == []
    before = session.weekly_digest.model_dump()
    first = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    second = asyncio.create_task(service.review_north_star(request))
    trace = await asyncio.wait_for(
        service.read_trace(
            TraceReadRequest(
                operation="read_trace",
                request_id="pending",
                session_id=session.session_id,
            )
        ),
        1,
    )
    assert trace.operation == "read_trace"
    assert trace.events[-1].event_type == "north_star_reviewed"
    assert trace.events[-1].status == "running"
    runtime.release.set()
    one, two = await asyncio.gather(first, second)
    assert one.operation == two.operation == "north_star_reviewed"
    assert one.event_ids == two.event_ids
    assert len(runtime.requests) == 1
    assert len(reviewer.requests) == 1
    assert one.session.weekly_digest.model_dump() == before
    reused = await service.review_north_star(request)
    assert reused.operation == "north_star_reviewed"
    assert len(runtime.requests) == 1
    assert (
        len(
            [
                e
                for e in service._events[session.session_id]
                if isinstance(e, NorthStarReviewedEvent)
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_nsm_failure_and_retry_do_not_repeat_weekly_review():
    runtime = ControlledRuntime(fail=True)
    service, _, reviewer, session, request = await reviewed_session(runtime)
    result = await service.review_north_star(request)
    assert result.operation == "north_star_reviewed"
    assert result.session.weekly_digest == session.weekly_digest
    await service.review_north_star(request)
    assert len(runtime.requests) == 1
    runtime.fail = False
    retried = await service.review_north_star(
        request.model_copy(update={"retry": True})
    )
    assert retried.operation == "north_star_reviewed"
    assert len(runtime.requests) == 2
    assert len(reviewer.requests) == 1


@pytest.mark.asyncio
async def test_transient_count_failure_recovers_without_repeating_weekly_work(tmp_path):
    calls = 0

    async def measure(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary count service failure")
        return await count_requests(*args)

    service, _, reviewer, session, request = await reviewed_session(ControlledRuntime())
    provider = FakeProvider(tmp_path)
    service._north_star_runtime = OpenAINorthStarRuntime(
        provider=provider, count_requests=measure
    )
    weekly_events = list(service._events[session.session_id])
    first = await service.review_north_star(request)
    assert first.operation == "north_star_reviewed"
    failed = service._events[session.session_id][-1]
    assert isinstance(failed, NorthStarReviewedEvent)
    assert failed.details.record.retryable
    assert failed.error.retryable
    assert provider.generated == 0
    repeated = await service.review_north_star(request)
    assert repeated.event_ids == first.event_ids
    assert calls == 1
    recovered = await service.review_north_star(
        request.model_copy(update={"retry": True})
    )
    assert recovered.operation == "north_star_reviewed"
    assert service._events[session.session_id][-1].details.record.status == "complete"
    assert recovered.session.weekly_digest == session.weekly_digest
    assert [
        event
        for event in service._events[session.session_id]
        if not isinstance(event, NorthStarReviewedEvent)
    ] == weekly_events
    assert len(reviewer.requests) == 1


@pytest.mark.asyncio
async def test_not_ready_review_can_be_retried_without_a_provider_call():
    runtime = ControlledRuntime()
    service, _, _ = _service([])
    service._north_star_runtime = runtime
    created = await _create_assessment(service)
    result = await service.review_north_star(
        NorthStarReviewRequest(
            operation="review_north_star",
            request_id="before-weekly-review",
            session_id=created.profile.session_id,
            expected_revision=0,
            week_start="2026-07-20",
        )
    )
    assert result.operation == "error"
    assert result.error.code == "north_star_not_ready"
    assert result.error.retryable
    assert runtime.requests == []


@pytest.mark.asyncio
async def test_delete_session_during_nsm_does_not_resurrect_it():
    runtime = ControlledRuntime(blocked=True)
    service, _, _, session, request = await reviewed_session(runtime)
    pending = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    deleted = await asyncio.wait_for(
        service.delete_session(
            SessionDeleteRequest(
                operation="delete_session",
                request_id="delete",
                session_id=session.session_id,
            )
        ),
        1,
    )
    assert deleted.deleted
    runtime.release.set()
    result = await pending
    assert result.operation == "error"
    assert session.session_id not in service._sessions
    assert session.session_id not in service._events
    assert runtime.forgotten


@pytest.mark.asyncio
async def test_removal_invalidates_pending_nsm_and_preserves_trace_links():
    runtime = ControlledRuntime(blocked=True)
    service, create, _, session, request = await reviewed_session(runtime)
    pending = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    events = list(service._events[session.session_id])
    resumed = await service.create_session(
        create.model_copy(
            update={
                "request_id": "remove",
                "idempotency_key": "f" * 64,
                "resume_state": SessionResumeState(
                    session_id=session.session_id,
                    revision=session.revision + 1,
                    journal_entries=[],
                    nudges=[],
                    assessment_clock=session.assessment_clock,
                    trace_events=events,
                ),
            }
        )
    )
    assert resumed.operation == "create_session"
    runtime.release.set()
    result = await pending
    assert result.operation == "error"
    assert not any(
        isinstance(e, NorthStarReviewedEvent)
        for e in service._events[session.session_id]
    )
    known = set()
    for event in service._events[session.session_id]:
        assert event.parent_event_id is None or event.parent_event_id in known
        known.add(event.event_id)


@pytest.mark.asyncio
async def test_old_session_without_source_availability_never_borrows_writing():
    runtime = ControlledRuntime()
    service, _, _, session, request = await reviewed_session(runtime)
    service._events[session.session_id] = [
        event
        for event in service._events[session.session_id]
        if event.event_type != "journal_entry_submitted"
    ]
    built = service._north_star_request(session, request.week_start)
    assert built.writing == []


@pytest.mark.asyncio
async def test_response_has_independent_server_availability():
    service, _, _ = _service([_receipt()])
    create = await _create_assessment(service)
    first = await service.submit_journal_entry(
        _submit_request(create, index=0, expected_revision=0)
    )
    assert first.operation == "submit_journal_entry"
    response = "I actually chose to help."
    resumed = await service.create_session(
        create.model_copy(
            update={
                "request_id": "reply",
                "idempotency_key": "b" * 64,
                "resume_state": SessionResumeState(
                    session_id=first.session.session_id,
                    revision=2,
                    journal_entries=[
                        first.session.journal_entries[0].model_copy(
                            update={"nudge_response": response}
                        )
                    ],
                    nudges=[
                        first.session.nudges[0].model_copy(
                            update={"outcome": "answered", "response": response}
                        )
                    ],
                    assessment_clock=first.session.assessment_clock,
                    trace_events=service._events[first.session.session_id],
                ),
            }
        )
    )
    assert resumed.operation == "create_session"
    events = service._events[first.session.session_id]
    assert events[-1].event_type == "nudge_response_recorded"
    assert events[-1].details.response == response
    reviewed, _ = await service.run_due_weekly_reviews(
        session_id=first.session.session_id, as_of=date(2026, 7, 27)
    )
    built = service._north_star_request(reviewed, "2026-07-20")
    assert built.writing[0].response_available_at == events[-1].started_at
    assert built.writing[0].nudge_response == response


@pytest.mark.asyncio
async def test_disconnected_waiter_still_finishes_cleanup_after_deletion():
    runtime = ControlledRuntime(blocked=True)
    service, _, _, session, request = await reviewed_session(runtime)
    waiter = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    job = next(iter(service._north_star_inflight.values()))
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await service.delete_session(
        SessionDeleteRequest(
            operation="delete_session",
            request_id="delete",
            session_id=session.session_id,
        )
    )
    runtime.release.set()
    await asyncio.wait_for(job, 1)
    assert not service._north_star_inflight
    assert runtime.forgotten
    assert session.session_id not in service._sessions


@pytest.mark.asyncio
async def test_disconnected_waiter_still_publishes_result_for_resume():
    runtime = ControlledRuntime(blocked=True)
    service, _, _, session, request = await reviewed_session(runtime)
    waiter = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    job = next(iter(service._north_star_inflight.values()))
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    runtime.release.set()
    await asyncio.wait_for(job, 1)
    event = service._events[session.session_id][-1]
    assert isinstance(event, NorthStarReviewedEvent)
    assert event.details.record.status == "not_eligible"
    reused = await service.review_north_star(request)
    assert reused.operation == "north_star_reviewed"
    assert len(runtime.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"t_index": 1}, {"date": "2026-07-24"}])
async def test_restored_source_cannot_change_submitted_chronology(change):
    runtime = ControlledRuntime()
    service, _, _, session, request = await reviewed_session(runtime)
    restored = session.model_copy(
        update={
            "journal_entries": [session.journal_entries[0].model_copy(update=change)],
        }
    )
    built = service._north_star_request(restored, request.week_start)
    assert built.writing == []


@pytest.mark.asyncio
async def test_restored_pending_record_reuses_the_original_event():
    runtime = ControlledRuntime()
    service, _, _, session, request = await reviewed_session(runtime)
    snapshot = service._north_star_request(session, request.week_start)
    saved = service._north_star_event(
        pending_north_star_record(snapshot),
        event_id="restored-pending",
        parent_event_id=session.trace_event_ids[-1],
    )
    service._events[session.session_id].append(saved)
    service._append_session(session, event_ids=[saved.event_id])
    completed = await service.review_north_star(request)
    assert completed.operation == "north_star_reviewed"
    assert completed.event_ids == [saved.event_id]
    moments = [
        e
        for e in service._events[session.session_id]
        if isinstance(e, NorthStarReviewedEvent)
    ]
    assert len(moments) == 1
    assert moments[0].status == "complete"


@pytest.mark.asyncio
async def test_earlier_review_remains_addressable_after_later_week():
    runtime = ControlledRuntime()
    service, _, _, session, request = await reviewed_session(runtime)
    expected = service._north_star_request(session, request.week_start)
    # The latest session summary is only a projection, not the earlier week's authority.
    later = session.model_copy(
        update={
            "weekly_digest": session.weekly_digest.model_copy(
                update={
                    "week_start": "2026-07-27",
                    "week_end": "2026-08-02",
                }
            ),
            "drift_result": session.drift_result.model_copy(
                update={
                    "cutoff_date": "2026-08-01",
                    "cutoff_t_index": 8,
                }
            ),
        }
    )
    assert service._north_star_request(later, request.week_start) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_before_edit", [False, True])
async def test_browser_can_remove_before_receiving_independent_nsm_trace(
    complete_before_edit,
):
    runtime = ControlledRuntime(blocked=True)
    service, create, _, session, request = await reviewed_session(runtime)
    browser_events = list(service._events[session.session_id])
    pending = asyncio.create_task(service.review_north_star(request))
    await asyncio.wait_for(runtime.started.wait(), 1)
    if complete_before_edit:
        runtime.release.set()
        await pending
    result = await service.create_session(
        create.model_copy(
            update={
                "request_id": "remove-with-older-trace",
                "idempotency_key": "b" * 64,
                "resume_state": SessionResumeState(
                    session_id=session.session_id,
                    revision=session.revision + 1,
                    journal_entries=[],
                    nudges=[],
                    assessment_clock=session.assessment_clock,
                    trace_events=browser_events,
                ),
            }
        )
    )
    assert result.operation == "create_session"
    assert result.session.journal_entries == []
    runtime.release.set()
    await pending
    assert not any(
        isinstance(e, NorthStarReviewedEvent)
        for e in service._events[session.session_id]
    )


def test_nsm_trace_reconciliation_keeps_writing_and_record_identity_checks():
    from src.demo.experience_service import InMemoryExperienceService

    assert InMemoryExperienceService._same_writing_trace([], [])
    # Unknown NSM identities cannot be injected as server-generated history.
    from tests.north_star.test_runtime import request as build_request

    service, _, _ = _service([])
    event = service._north_star_event(
        pending_north_star_record(build_request()),
        event_id="not-server-owned",
        parent_event_id=None,
    )
    assert not service._same_writing_trace([event], [])
    assert not service._same_writing_trace(
        [event.model_copy(update={"input_hash": "forged"})], [event]
    )
