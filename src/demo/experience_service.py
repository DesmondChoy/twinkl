"""In-memory Experience session boundary for manual Journal Entries."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, cast
from uuid import uuid4

from src.demo.contracts import (
    ApiErrorResponse,
    ApiResponse,
    EventStatus,
    EventValidation,
    ExperienceSession,
    JournalEntrySubmitRequest,
    JournalEntrySubmittedDetails,
    JournalEntrySubmittedEvent,
    JournalEntrySubmittedResponse,
    ModelContract,
    NudgeDecidedEvent,
    NudgeDecisionDetails,
    NudgeGeneratedDetails,
    NudgeGeneratedEvent,
    NudgeInteraction,
    NudgeSuppressionCheckedEvent,
    NudgeSuppressionDetails,
    ProfileConfirmedDetails,
    ProfileConfirmedEvent,
    ResourceRef,
    SafeError,
    SessionCreatedResponse,
    SessionCreateRequest,
    SessionSelection,
    TraceEvent,
    TraceReadRequest,
    TraceReadResponse,
)
from src.models.nudge import NUDGE_CATEGORIES, NudgeCategory
from src.nudge.decision import should_suppress_nudge
from src.nudge.runtime import (
    NudgeRuntimeReceipt,
    NudgeRuntimeRequest,
    OpenAINudgeRuntime,
    build_nudge_runtime_request,
)

NudgeRuntime = Callable[[NudgeRuntimeRequest], Awaitable[NudgeRuntimeReceipt]]
Operation = Literal["create_session", "submit_journal_entry", "read_trace"]


def _hash_payload(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _safe_raw_response(value: str | None) -> Any | None:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@dataclass
class _IdempotentResult:
    fingerprint: str
    response: SessionCreatedResponse | JournalEntrySubmittedResponse
    retryable: bool = False
    retry_parent_event_id: str | None = None


class InMemoryExperienceService:
    """Small contract-driven session store for the capstone Experience."""

    def __init__(
        self,
        *,
        nudge_runtime: NudgeRuntime | None = None,
        now: Callable[[], datetime] | None = None,
        make_id: Callable[[str], str] | None = None,
    ) -> None:
        self._nudge_runtime = nudge_runtime or OpenAINudgeRuntime()
        self._now = now or (lambda: datetime.now(UTC))
        self._make_id = make_id or (lambda prefix: f"{prefix}-{uuid4()}")
        self._sessions: dict[str, ExperienceSession] = {}
        self._events: dict[str, list[TraceEvent]] = {}
        self._idempotency: dict[tuple[str, str], _IdempotentResult] = {}
        self._lock = asyncio.Lock()

    def _timestamp(self) -> str:
        return self._now().isoformat()

    @staticmethod
    def _fingerprint(
        request: SessionCreateRequest | JournalEntrySubmitRequest,
    ) -> str:
        return _hash_payload(
            request.model_dump(
                mode="json",
                exclude={"request_id", "idempotency_key"},
            )
        )

    @staticmethod
    def _error(
        *,
        requested_operation: Operation,
        request_id: str,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> ApiErrorResponse:
        return ApiErrorResponse(
            operation="error",
            requested_operation=requested_operation,
            request_id=request_id,
            status="error",
            error=SafeError(code=code, message=message, retryable=retryable),
        )

    @staticmethod
    def _with_request_id(
        response: SessionCreatedResponse | JournalEntrySubmittedResponse,
        request_id: str,
    ) -> SessionCreatedResponse | JournalEntrySubmittedResponse:
        return response.model_copy(update={"request_id": request_id})

    def _terminal_event_fields(self, *, duration_ms: int = 0) -> dict[str, Any]:
        timestamp = self._timestamp()
        return {
            "started_at": timestamp,
            "completed_at": timestamp,
            "duration_ms": duration_ms,
        }

    async def create_session(
        self,
        request: SessionCreateRequest,
    ) -> SessionCreatedResponse | ApiErrorResponse:
        async with self._lock:
            key = (request.operation, request.idempotency_key)
            fingerprint = self._fingerprint(request)
            cached = self._idempotency.get(key)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="idempotency_conflict",
                        message=(
                            "This retry key was already used for different "
                            "Profile data."
                        ),
                    )
                response = cast(SessionCreatedResponse, cached.response)
                return cast(
                    SessionCreatedResponse,
                    self._with_request_id(response, request.request_id),
                )

            existing = self._sessions.get(request.profile.session_id)
            if existing is not None:
                if existing.profile != request.profile:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="session_conflict",
                        message="This session already belongs to a different Profile.",
                    )
                response = SessionCreatedResponse(
                    operation="create_session",
                    request_id=request.request_id,
                    status="ok",
                    session=existing,
                )
                self._idempotency[key] = _IdempotentResult(
                    fingerprint=fingerprint,
                    response=response,
                )
                return response

            if request.resume_state is not None:
                resume_state = request.resume_state
                events = list(resume_state.trace_events)
                session = ExperienceSession(
                    session_id=request.profile.session_id,
                    revision=resume_state.revision,
                    profile=request.profile,
                    journal_entries=resume_state.journal_entries,
                    nudges=resume_state.nudges,
                    trace_event_ids=[event.event_id for event in events],
                    selection=SessionSelection(view="experience"),
                    updated_at=self._timestamp(),
                )
            else:
                event_id = self._make_id("profile")
                event = ProfileConfirmedEvent(
                    event_id=event_id,
                    session_id=request.profile.session_id,
                    parent_event_id=None,
                    event_type="profile_confirmed",
                    status="complete",
                    source="live_run",
                    input_hash=_hash_payload(request.profile.model_dump(mode="json")),
                    result_refs=[
                        ResourceRef(kind="profile", id=request.profile.session_id)
                    ],
                    details=ProfileConfirmedDetails(profile=request.profile),
                    **self._terminal_event_fields(),
                )
                events = [event]
                session = ExperienceSession(
                    session_id=request.profile.session_id,
                    revision=0,
                    profile=request.profile,
                    trace_event_ids=[event_id],
                    selection=SessionSelection(view="experience"),
                    updated_at=self._timestamp(),
                )
            self._sessions[session.session_id] = session
            self._events[session.session_id] = events
            response = SessionCreatedResponse(
                operation="create_session",
                request_id=request.request_id,
                status="ok",
                session=session,
            )
            self._idempotency[key] = _IdempotentResult(
                fingerprint=fingerprint,
                response=response,
            )
            return response

    @staticmethod
    def _ordering_error(
        session: ExperienceSession,
        request: JournalEntrySubmitRequest,
    ) -> str | None:
        entry = request.journal_entry
        if request.expected_revision != session.revision:
            return "The Journal Entry is based on an older session revision."
        if any(
            existing.journal_entry_id == entry.journal_entry_id
            for existing in session.journal_entries
        ):
            return "This Journal Entry has already been saved."
        if any(
            existing.t_index == entry.t_index for existing in session.journal_entries
        ):
            return "This Journal Entry position is already in use."
        if (
            session.journal_entries
            and entry.t_index <= session.journal_entries[-1].t_index
        ):
            return "Journal Entries must be saved in chronological order."
        try:
            entry_date = date.fromisoformat(entry.date)
            previous_date = (
                date.fromisoformat(session.journal_entries[-1].date)
                if session.journal_entries
                else None
            )
        except ValueError:
            return "Journal Entry dates must use YYYY-MM-DD."
        if previous_date is not None and entry_date < previous_date:
            return "Journal Entries must be saved in chronological order."
        return None

    @staticmethod
    def _displayed_history(session: ExperienceSession) -> list[bool]:
        displayed = {
            nudge.journal_entry_id
            for nudge in session.nudges
            if nudge.outcome in {"displayed", "skipped", "answered"}
        }
        return [
            entry.journal_entry_id in displayed for entry in session.journal_entries
        ]

    def _append_session(
        self,
        session: ExperienceSession,
        *,
        journal_entries: list[Any] | None = None,
        nudges: list[NudgeInteraction] | None = None,
        event_ids: list[str] | None = None,
        increment_revision: bool = False,
    ) -> ExperienceSession:
        payload = session.model_dump(mode="json")
        payload.update(
            {
                "revision": session.revision + int(increment_revision),
                "journal_entries": journal_entries
                if journal_entries is not None
                else session.journal_entries,
                "nudges": nudges if nudges is not None else session.nudges,
                "trace_event_ids": [
                    *session.trace_event_ids,
                    *(event_ids or []),
                ],
                "updated_at": self._timestamp(),
            }
        )
        updated = ExperienceSession.model_validate(payload)
        self._sessions[updated.session_id] = updated
        return updated

    async def _run_nudge(
        self,
        *,
        session: ExperienceSession,
        request: JournalEntrySubmitRequest,
        parent_event_id: str,
    ) -> tuple[ExperienceSession, list[TraceEvent], bool]:
        entry = request.journal_entry
        previous_entries = [
            existing.model_dump(mode="json")
            for existing in session.journal_entries
            if existing.journal_entry_id != entry.journal_entry_id
        ]
        runtime_request = build_nudge_runtime_request(
            entry_content=entry.content,
            entry_date=entry.date,
            previous_entries=previous_entries,
        )
        receipt = await self._nudge_runtime(runtime_request)
        event_status: EventStatus = {
            "ok": "complete",
            "refusal": "refused",
            "invalid": "invalid",
            "error": "failed",
        }[receipt.status]
        should_nudge = receipt.status == "ok" and receipt.decision != "no_nudge"
        category: NudgeCategory | None = None
        if should_nudge and receipt.decision in NUDGE_CATEGORIES:
            category = cast(NudgeCategory, receipt.decision)
        should_nudge = category is not None
        validation = EventValidation(
            valid=receipt.status == "ok",
            schema_name="NudgeDecisionAndGeneration",
            errors=[receipt.validation_error]
            if receipt.validation_error is not None
            else [],
        )
        safe_error = None
        retryable = receipt.status == "error"
        if receipt.status == "refusal":
            safe_error = SafeError(
                code="nudge_refused",
                message="The nudge check could not return a usable result.",
                retryable=False,
            )
        elif receipt.status == "invalid":
            safe_error = SafeError(
                code="nudge_invalid",
                message="The nudge check returned an invalid result.",
                retryable=False,
            )
        elif receipt.status == "error":
            safe_error = SafeError(
                code="nudge_failed",
                message="The nudge check could not finish. Try again.",
                retryable=True,
            )

        decision_event_id = self._make_id("nudge-decision")
        decision_event = NudgeDecidedEvent(
            event_id=decision_event_id,
            session_id=session.session_id,
            parent_event_id=parent_event_id,
            event_type="nudge_decided",
            status=event_status,
            source="live_run",
            input_refs=[ResourceRef(kind="journal_entry", id=entry.journal_entry_id)],
            model_contract=ModelContract(
                provider="openai",
                model=receipt.requested_model,
                reasoning_effort=receipt.reasoning_effort,
            ),
            prompt=runtime_request.prompt,
            raw_response=_safe_raw_response(receipt.raw_response),
            validation=validation,
            result_refs=(
                [ResourceRef(kind="event", id=decision_event_id)]
                if receipt.status == "ok"
                else []
            ),
            input_hash=runtime_request.prompt_sha256,
            error=safe_error,
            details=NudgeDecisionDetails(
                should_nudge=should_nudge,
                category=category,
                reason=receipt.reason if receipt.status == "ok" else None,
            ),
            **self._terminal_event_fields(
                duration_ms=max(0, round(receipt.latency_seconds * 1000))
            ),
        )
        events: list[TraceEvent] = [decision_event]
        nudges = list(session.nudges)

        if receipt.status == "ok" and not should_nudge:
            nudges.append(
                NudgeInteraction(
                    nudge_id=self._make_id("nudge"),
                    journal_entry_id=entry.journal_entry_id,
                    outcome="no_nudge",
                )
            )
        elif receipt.status == "ok" and receipt.nudge_text is not None:
            nudge = NudgeInteraction(
                nudge_id=self._make_id("nudge"),
                journal_entry_id=entry.journal_entry_id,
                outcome="displayed",
                category=category,
                reason=receipt.reason,
                text=receipt.nudge_text,
            )
            generated_event_id = self._make_id("nudge-generated")
            generated_event = NudgeGeneratedEvent(
                event_id=generated_event_id,
                session_id=session.session_id,
                parent_event_id=decision_event_id,
                event_type="nudge_generated",
                status="complete",
                source="live_run",
                input_refs=[
                    ResourceRef(kind="journal_entry", id=entry.journal_entry_id),
                    ResourceRef(kind="event", id=decision_event_id),
                ],
                validation=EventValidation(
                    valid=True,
                    schema_name="NudgeQuestionLength",
                ),
                result_refs=[ResourceRef(kind="nudge", id=nudge.nudge_id)],
                input_hash=runtime_request.prompt_sha256,
                details=NudgeGeneratedDetails(
                    nudge=nudge,
                    word_count=len(receipt.nudge_text.split()),
                    attempts=receipt.attempts,
                ),
                **self._terminal_event_fields(),
            )
            nudges.append(nudge)
            events.append(generated_event)

        updated = self._append_session(
            session,
            nudges=nudges,
            event_ids=[event.event_id for event in events],
        )
        self._events[session.session_id].extend(events)
        return updated, events, retryable

    async def submit_journal_entry(
        self,
        request: JournalEntrySubmitRequest,
    ) -> JournalEntrySubmittedResponse | ApiErrorResponse:
        async with self._lock:
            key = (request.operation, request.idempotency_key)
            fingerprint = self._fingerprint(request)
            cached = self._idempotency.get(key)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="idempotency_conflict",
                        message=(
                            "This retry key was already used for a different "
                            "Journal Entry."
                        ),
                    )
                if not cached.retryable:
                    response = cast(JournalEntrySubmittedResponse, cached.response)
                    return cast(
                        JournalEntrySubmittedResponse,
                        self._with_request_id(response, request.request_id),
                    )
                cached_session = self._sessions[request.session_id]
                updated, retry_events, retryable = await self._run_nudge(
                    session=cached_session,
                    request=request,
                    parent_event_id=cast(str, cached.retry_parent_event_id),
                )
                response = JournalEntrySubmittedResponse(
                    operation="submit_journal_entry",
                    request_id=request.request_id,
                    status="ok",
                    session=updated,
                    event_ids=[event.event_id for event in retry_events],
                )
                self._idempotency[key] = _IdempotentResult(
                    fingerprint=fingerprint,
                    response=response,
                    retryable=retryable,
                    retry_parent_event_id=(
                        retry_events[-1].event_id if retryable else None
                    ),
                )
                return response

            session = self._sessions.get(request.session_id)
            if session is None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_not_found",
                    message=(
                        "Start the Experience session before saving a Journal Entry."
                    ),
                    retryable=True,
                )
            ordering_error = self._ordering_error(session, request)
            if ordering_error is not None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="journal_order_conflict",
                    message=ordering_error,
                )

            entry_event_id = self._make_id("journal")
            entry_event = JournalEntrySubmittedEvent(
                event_id=entry_event_id,
                session_id=session.session_id,
                parent_event_id=session.trace_event_ids[-1],
                event_type="journal_entry_submitted",
                status="complete",
                source="live_run",
                input_refs=[ResourceRef(kind="profile", id=session.profile.session_id)],
                result_refs=[
                    ResourceRef(
                        kind="journal_entry",
                        id=request.journal_entry.journal_entry_id,
                    )
                ],
                input_hash=fingerprint,
                details=JournalEntrySubmittedDetails(
                    journal_entry=request.journal_entry,
                    ordering_valid=True,
                ),
                **self._terminal_event_fields(),
            )
            previous_ids = [
                entry.journal_entry_id for entry in session.journal_entries[-3:]
            ]
            suppressed = should_suppress_nudge(self._displayed_history(session))
            suppression_event_id = self._make_id("nudge-suppression")
            suppression_event = NudgeSuppressionCheckedEvent(
                event_id=suppression_event_id,
                session_id=session.session_id,
                parent_event_id=entry_event_id,
                event_type="nudge_suppression_checked",
                status="complete",
                source="live_run",
                input_refs=[
                    ResourceRef(
                        kind="journal_entry",
                        id=request.journal_entry.journal_entry_id,
                    )
                ],
                input_hash=_hash_payload(
                    {
                        "entry_id": request.journal_entry.journal_entry_id,
                        "previous_entry_ids": previous_ids,
                        "suppressed": suppressed,
                    }
                ),
                details=NudgeSuppressionDetails(
                    previous_entry_ids=previous_ids,
                    suppressed=suppressed,
                ),
                **self._terminal_event_fields(),
            )
            session = self._append_session(
                session,
                journal_entries=[
                    *session.journal_entries,
                    request.journal_entry,
                ],
                event_ids=[entry_event_id, suppression_event_id],
                increment_revision=True,
            )
            self._events[session.session_id].extend([entry_event, suppression_event])
            events: list[TraceEvent] = [entry_event, suppression_event]
            retryable = False

            if suppressed:
                suppressed_nudge = NudgeInteraction(
                    nudge_id=self._make_id("nudge"),
                    journal_entry_id=request.journal_entry.journal_entry_id,
                    outcome="suppressed",
                )
                session = self._append_session(
                    session,
                    nudges=[*session.nudges, suppressed_nudge],
                )
            else:
                session, nudge_events, retryable = await self._run_nudge(
                    session=session,
                    request=request,
                    parent_event_id=suppression_event_id,
                )
                events.extend(nudge_events)

            response = JournalEntrySubmittedResponse(
                operation="submit_journal_entry",
                request_id=request.request_id,
                status="ok",
                session=session,
                event_ids=[event.event_id for event in events],
            )
            self._idempotency[key] = _IdempotentResult(
                fingerprint=fingerprint,
                response=response,
                retryable=retryable,
                retry_parent_event_id=events[-1].event_id if retryable else None,
            )
            return response

    async def read_trace(
        self,
        request: TraceReadRequest,
    ) -> TraceReadResponse | ApiErrorResponse:
        async with self._lock:
            events = self._events.get(request.session_id)
            if events is None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_not_found",
                    message="No Experience trace exists for this session.",
                )
            if request.after_event_id is not None:
                event_ids = [event.event_id for event in events]
                if request.after_event_id not in event_ids:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="trace_cursor_not_found",
                        message="The requested Inspect event is not in this session.",
                    )
                events = events[event_ids.index(request.after_event_id) + 1 :]
            return TraceReadResponse(
                operation="read_trace",
                request_id=request.request_id,
                status="ok",
                session_id=request.session_id,
                events=events,
            )

    async def handle(
        self,
        request: SessionCreateRequest | JournalEntrySubmitRequest | TraceReadRequest,
    ) -> ApiResponse:
        if isinstance(request, SessionCreateRequest):
            return await self.create_session(request)
        if isinstance(request, JournalEntrySubmitRequest):
            return await self.submit_journal_entry(request)
        return await self.read_trace(request)
