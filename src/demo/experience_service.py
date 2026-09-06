"""In-memory Experience session boundary for manual Journal Entries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol, cast
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.coach.llm_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_llm_complete,
)
from src.coach.schemas import WeeklyDigest
from src.coach.weekly_digest import (
    LLMCompleteFn,
    attach_coach_artifacts,
    build_weekly_drift_reviewer_digest_from_entries,
    generate_weekly_digest_coach,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)
from src.demo.contracts import (
    ApiErrorResponse,
    ApiResponse,
    AssessmentClock,
    AssessmentTimeAdvancedDetails,
    AssessmentTimeAdvancedEvent,
    AssessmentTimeAdvancedResponse,
    AssessmentTimeAdvanceRequest,
    ContractFixtureSet,
    DriftDetectedDetails,
    DriftDetectedEvent,
    EventStatus,
    EventValidation,
    ExperienceSession,
    JournalEntrySubmitRequest,
    JournalEntrySubmittedDetails,
    JournalEntrySubmittedEvent,
    JournalEntrySubmittedResponse,
    ModelContract,
    NorthStarReviewedDetails,
    NorthStarReviewedEvent,
    NorthStarReviewedResponse,
    NorthStarReviewRequest,
    NudgeDecidedEvent,
    NudgeDecisionDetails,
    NudgeGeneratedDetails,
    NudgeGeneratedEvent,
    NudgeInteraction,
    NudgeResponseRecordedDetails,
    NudgeResponseRecordedEvent,
    NudgeSuppressionCheckedEvent,
    NudgeSuppressionDetails,
    ProfileConfirmedDetails,
    ProfileConfirmedEvent,
    ResourceRef,
    SafeError,
    ScenarioLoadedResponse,
    ScenarioLoadRequest,
    SessionCreatedResponse,
    SessionCreateRequest,
    SessionDeletedResponse,
    SessionDeleteRequest,
    SessionResumeState,
    SessionSelection,
    TraceEvent,
    TraceReadRequest,
    TraceReadResponse,
    WeeklyCoachGeneratedDetails,
    WeeklyCoachGeneratedEvent,
    WeeklyDigestBuiltDetails,
    WeeklyDigestBuiltEvent,
    WeeklyDriftReviewerDecisionContract,
    WeeklyDriftReviewerReceiptContract,
    WeeklyDriftReviewerRequestContract,
    WeeklyReviewCompletedDetails,
    WeeklyReviewCompletedEvent,
    WeeklyReviewRequestedDetails,
    WeeklyReviewRequestedEvent,
    build_drift_rule_steps,
)
from src.drift_detector import DriftDetectorResult, detect_drift
from src.models.nudge import NUDGE_CATEGORIES, NudgeCategory
from src.north_star.live_runtime import create_live_north_star_runtime
from src.north_star.runtime import (
    NorthStarRecord,
    NorthStarRequest,
    SourceWriting,
    build_north_star_request,
    pending_north_star_record,
    profile_reference,
    validate_north_star_record,
)
from src.nudge.decision import should_suppress_nudge
from src.nudge.runtime import (
    NudgeRuntimeReceipt,
    NudgeRuntimeRequest,
    OpenAINudgeRuntime,
    build_failed_nudge_runtime_receipt,
    build_nudge_runtime_request,
)
from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    WeeklyDriftReviewerEntry,
    WeeklyDriftReviewerFn,
    WeeklyDriftReviewerReceipt,
    build_weekly_drift_reviewer_request,
)

NudgeRuntime = Callable[[NudgeRuntimeRequest], Awaitable[NudgeRuntimeReceipt]]
Operation = Literal[
    "create_session",
    "submit_journal_entry",
    "advance_assessment_time",
    "delete_session",
    "load_scenario",
    "read_trace",
    "review_north_star",
]


class NorthStarRuntime(Protocol):
    async def __call__(
        self, request: NorthStarRequest, *, retry: bool = False
    ) -> NorthStarRecord: ...


_DEFAULT_COACH = object()


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
    response: (
        SessionCreatedResponse
        | JournalEntrySubmittedResponse
        | AssessmentTimeAdvancedResponse
    )
    retryable: bool = False
    retry_parent_event_id: str | None = None


class InMemoryExperienceService:
    """Small contract-driven session store for the capstone Experience."""

    def __init__(
        self,
        *,
        nudge_runtime: NudgeRuntime | None = None,
        weekly_reviewer: WeeklyDriftReviewerFn | None = None,
        coach_llm_complete: LLMCompleteFn | None | object = _DEFAULT_COACH,
        coach_model_contract: ModelContract | None = None,
        north_star_runtime: NorthStarRuntime | None = None,
        now: Callable[[], datetime] | None = None,
        make_id: Callable[[str], str] | None = None,
    ) -> None:
        self._nudge_runtime = nudge_runtime or OpenAINudgeRuntime()
        self._weekly_reviewer = weekly_reviewer or OpenAIWeeklyDriftReviewer()
        self._coach_llm_complete = (
            build_llm_complete()
            if coach_llm_complete is _DEFAULT_COACH
            else cast(LLMCompleteFn | None, coach_llm_complete)
        )
        provider = os.environ.get("TWINKL_COACH_PROVIDER", "openai").strip().lower()
        default_model = (
            DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_OPENAI_MODEL
        )
        self._coach_model_contract = coach_model_contract or ModelContract(
            provider=provider,
            model=os.environ.get("TWINKL_COACH_MODEL", default_model),
            reasoning_effort="none",
        )
        self._now = now or (lambda: datetime.now(UTC))
        self._make_id = make_id or (lambda prefix: f"{prefix}-{uuid4()}")
        self._sessions: dict[str, ExperienceSession] = {}
        self._events: dict[str, list[TraceEvent]] = {}
        self._idempotency: dict[tuple[str, str], _IdempotentResult] = {}
        self._lock = asyncio.Lock()
        self._north_star_runtime = (
            north_star_runtime or create_live_north_star_runtime()
        )
        self._north_star_inflight: dict[
            tuple[str, str], asyncio.Task[NorthStarRecord | None]
        ] = {}

    def _timestamp(self) -> str:
        return self._now().isoformat()

    @staticmethod
    def _fingerprint(
        request: (
            SessionCreateRequest
            | JournalEntrySubmitRequest
            | AssessmentTimeAdvanceRequest
        ),
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
        response: (
            SessionCreatedResponse
            | JournalEntrySubmittedResponse
            | AssessmentTimeAdvancedResponse
        ),
        request_id: str,
    ) -> (
        SessionCreatedResponse
        | JournalEntrySubmittedResponse
        | AssessmentTimeAdvancedResponse
    ):
        return cast(
            SessionCreatedResponse
            | JournalEntrySubmittedResponse
            | AssessmentTimeAdvancedResponse,
            response.model_copy(update={"request_id": request_id}),
        )

    def _terminal_event_fields(self, *, duration_ms: int = 0) -> dict[str, Any]:
        timestamp = self._timestamp()
        return {
            "started_at": timestamp,
            "completed_at": timestamp,
            "duration_ms": duration_ms,
        }

    @staticmethod
    def _week_bounds(raw: str) -> tuple[date, date]:
        entry_date = date.fromisoformat(raw)
        week_start = entry_date - timedelta(days=entry_date.weekday())
        return week_start, week_start + timedelta(days=6)

    def _initial_assessment_clock(
        self,
        *,
        timezone: str | None,
        journal_entries: list[Any],
    ) -> AssessmentClock | None:
        if timezone is None:
            return None
        try:
            zone = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("The assessment timezone is not supported.") from error
        current = self._now().astimezone(zone).date()
        if journal_entries:
            current = max(
                current,
                date.fromisoformat(journal_entries[-1].date),
            )
        return AssessmentClock(
            current_date=current.isoformat(),
            timezone=timezone,
        )

    @staticmethod
    def _displayed_entry_text(
        session: ExperienceSession,
        *,
        journal_entry_id: str,
    ) -> str:
        entry = next(
            row
            for row in session.journal_entries
            if row.journal_entry_id == journal_entry_id
        )
        nudge = next(
            (row for row in session.nudges if row.journal_entry_id == journal_entry_id),
            None,
        )
        parts = [entry.content]
        if nudge is not None and nudge.text:
            parts.append(f'Nudge: "{nudge.text}"')
        response = (
            nudge.response
            if nudge is not None and nudge.response
            else entry.nudge_response
        )
        if response:
            parts.append(f"Response: {response}")
        return "\n\n".join(parts)

    @staticmethod
    def _restored_weekly_state(
        events: list[TraceEvent],
    ) -> tuple[
        list[WeeklyDriftReviewerDecisionContract],
        DriftDetectorResult | None,
        WeeklyDigest | None,
    ]:
        drift_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event, DriftDetectedEvent)
            ),
            None,
        )
        digest_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event, WeeklyDigestBuiltEvent)
            ),
            None,
        )
        digest = digest_event.details.digest if digest_event is not None else None
        coach_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event, WeeklyCoachGeneratedEvent)
                and event.status == "complete"
                and event.details.narrative is not None
                and event.details.validation is not None
                and digest_event is not None
                and event.parent_event_id == digest_event.event_id
            ),
            None,
        )
        if digest is not None and coach_event is not None:
            digest = attach_coach_artifacts(
                digest,
                coach_event.details.narrative,
                coach_event.details.validation,
            )
        return (
            list(drift_event.details.decisions) if drift_event is not None else [],
            drift_event.details.result if drift_event is not None else None,
            digest,
        )

    @staticmethod
    def _reviewed_week_starts(events: list[TraceEvent]) -> set[date]:
        return {
            date.fromisoformat(event.details.receipt.week_start)
            for event in events
            if isinstance(event, WeeklyReviewCompletedEvent)
        }

    def _week_is_finalized(
        self,
        session: ExperienceSession,
        *,
        week_start: date,
    ) -> bool:
        week_end = week_start + timedelta(days=6)
        current_entry_ids = {
            entry.journal_entry_id
            for entry in session.journal_entries
            if week_start <= date.fromisoformat(entry.date) <= week_end
        }
        nudges = {
            nudge.journal_entry_id: nudge
            for nudge in session.nudges
            if nudge.journal_entry_id in current_entry_ids
        }
        if any(nudge.outcome == "displayed" for nudge in nudges.values()):
            return False

        latest_decision_status: dict[str, EventStatus] = {}
        for event in self._events[session.session_id]:
            if not isinstance(event, NudgeDecidedEvent):
                continue
            journal_entry_id = next(
                (
                    reference.id
                    for reference in event.input_refs
                    if reference.kind == "journal_entry"
                ),
                None,
            )
            if journal_entry_id in current_entry_ids:
                latest_decision_status[cast(str, journal_entry_id)] = event.status
        return all(
            entry_id in nudges
            or latest_decision_status.get(entry_id)
            in {"complete", "refused", "invalid"}
            for entry_id in current_entry_ids
        )

    @staticmethod
    def _same_writing_trace(
        browser_events: list[TraceEvent], server_events: list[TraceEvent]
    ) -> bool:
        """NSM may finish independently of the browser's writing revision."""
        server_moments = {
            event.event_id: event.input_hash
            for event in server_events
            if isinstance(event, NorthStarReviewedEvent)
        }
        return (
            [e for e in browser_events if not isinstance(e, NorthStarReviewedEvent)]
            == [e for e in server_events if not isinstance(e, NorthStarReviewedEvent)]
            and all(
                event.event_id in server_moments
                and event.input_hash == server_moments[event.event_id]
                for event in browser_events
                if isinstance(event, NorthStarReviewedEvent)
            )
        )

    @staticmethod
    def _resume_update_date(
        existing: ExperienceSession,
        resume_state: SessionResumeState,
    ) -> date | None:
        if resume_state.assessment_clock != existing.assessment_clock:
            return None
        if resume_state.revision != existing.revision + 1:
            return None

        previous_entries = existing.journal_entries
        next_entries = resume_state.journal_entries
        previous_ids = [entry.journal_entry_id for entry in previous_entries]
        next_ids = [entry.journal_entry_id for entry in next_entries]

        removed_ids = set(previous_ids) - set(next_ids)
        if len(removed_ids) == 1 and not (set(next_ids) - set(previous_ids)):
            removed_id = removed_ids.pop()
            if next_entries != [
                entry
                for entry in previous_entries
                if entry.journal_entry_id != removed_id
            ]:
                return None
            if resume_state.nudges != [
                nudge
                for nudge in existing.nudges
                if nudge.journal_entry_id != removed_id
            ]:
                return None
            removed_entry = next(
                entry
                for entry in previous_entries
                if entry.journal_entry_id == removed_id
            )
            return date.fromisoformat(removed_entry.date)

        if previous_ids != next_ids or len(existing.nudges) != len(resume_state.nudges):
            return None
        changed_entries = [
            (previous, updated)
            for previous, updated in zip(
                previous_entries,
                next_entries,
                strict=True,
            )
            if previous != updated
        ]
        changed_nudges = [
            (previous, updated)
            for previous, updated in zip(
                existing.nudges,
                resume_state.nudges,
                strict=True,
            )
            if previous != updated
        ]
        if len(changed_nudges) != 1:
            return None

        previous_nudge, updated_nudge = changed_nudges[0]
        previous_entry = next(
            (
                entry
                for entry in previous_entries
                if entry.journal_entry_id == previous_nudge.journal_entry_id
            ),
            None,
        )
        updated_entry = next(
            (
                entry
                for entry in next_entries
                if entry.journal_entry_id == updated_nudge.journal_entry_id
            ),
            None,
        )
        if (
            previous_entry is None
            or updated_entry is None
            or previous_entry.model_copy(update={"nudge_response": None})
            != updated_entry.model_copy(update={"nudge_response": None})
            or previous_nudge.nudge_id != updated_nudge.nudge_id
            or previous_nudge.outcome != "displayed"
            or updated_nudge.outcome not in {"answered", "skipped"}
            or previous_nudge.model_copy(
                update={
                    "outcome": updated_nudge.outcome,
                    "response": updated_nudge.response,
                }
            )
            != updated_nudge
        ):
            return None
        if updated_nudge.outcome == "answered":
            if (
                len(changed_entries) != 1
                or changed_entries[0] != (previous_entry, updated_entry)
                or not updated_nudge.response
                or updated_entry.nudge_response != updated_nudge.response
            ):
                return None
        elif (
            changed_entries
            or updated_nudge.response is not None
            or updated_entry.nudge_response is not None
        ):
            return None
        return date.fromisoformat(updated_entry.date)

    async def _apply_resume_update(
        self,
        *,
        existing: ExperienceSession,
        resume_state: SessionResumeState,
        affected_date: date,
    ) -> ExperienceSession:
        reviewed_weeks = self._reviewed_week_starts(self._events[existing.session_id])
        events = list(resume_state.trace_events)
        affected_week, _ = self._week_bounds(affected_date.isoformat())
        removed_parents = {
            event.event_id: event.parent_event_id
            for event in events
            if isinstance(event, NorthStarReviewedEvent)
            and date.fromisoformat(event.details.record.week_start) >= affected_week
        }
        forget = getattr(self._north_star_runtime, "forget", None)
        if forget is not None:
            for event in events:
                if (
                    isinstance(event, NorthStarReviewedEvent)
                    and event.event_id in removed_parents
                ):
                    forget(event.details.record)
        retained_events: list[TraceEvent] = []
        for event in events:
            if event.event_id in removed_parents:
                continue
            parent = event.parent_event_id
            while parent in removed_parents:
                parent = removed_parents[parent]
            retained_events.append(event.model_copy(update={"parent_event_id": parent}))
        events = retained_events
        old_nudges = {nudge.nudge_id: nudge for nudge in existing.nudges}
        for nudge in resume_state.nudges:
            previous = old_nudges.get(nudge.nudge_id)
            if (
                nudge.response
                and previous is not None
                and nudge.response != previous.response
            ):
                events.append(
                    NudgeResponseRecordedEvent(
                        event_id=self._make_id("nudge-response-recorded"),
                        session_id=existing.session_id,
                        parent_event_id=events[-1].event_id,
                        event_type="nudge_response_recorded",
                        status="complete",
                        source="live_run",
                        input_refs=[
                            ResourceRef(kind="journal_entry", id=nudge.journal_entry_id)
                        ],
                        input_hash=_hash_payload(nudge.model_dump(mode="json")),
                        details=NudgeResponseRecordedDetails(
                            journal_entry_id=nudge.journal_entry_id,
                            nudge_id=nudge.nudge_id,
                            response=nudge.response,
                        ),
                        **self._terminal_event_fields(),
                    )
                )
        decisions = [
            decision
            for decision in existing.weekly_reviewer_decisions
            if date.fromisoformat(decision.week_start) < affected_week
        ]
        prior_digest = next(
            (
                event.details.digest
                for event in reversed(events)
                if isinstance(event, WeeklyDigestBuiltEvent)
                and date.fromisoformat(event.details.digest.week_start) < affected_week
            ),
            None,
        )
        base_payload = existing.model_dump(mode="json")
        base_payload.update(
            {
                "revision": resume_state.revision,
                "journal_entries": resume_state.journal_entries,
                "nudges": resume_state.nudges,
                "weekly_reviewer_decisions": decisions,
                "drift_result": (
                    detect_drift(
                        decisions,
                        persona_id=existing.profile.user_id,
                    )
                    if decisions
                    else None
                ),
                "weekly_digest": prior_digest,
                "trace_event_ids": [event.event_id for event in events],
                "updated_at": self._timestamp(),
            }
        )
        updated = ExperienceSession.model_validate(base_payload)
        self._sessions[updated.session_id] = updated
        self._events[updated.session_id] = events

        weeks_to_recompute = sorted(
            {
                self._week_bounds(entry.date)[0]
                for entry in updated.journal_entries
                if self._week_bounds(entry.date)[0] >= affected_week
                and self._week_bounds(entry.date)[0] in reviewed_weeks
            }
        )
        for week_start in weeks_to_recompute:
            updated, _ = await self._run_weekly_review(
                session=updated,
                parent_event_id=updated.trace_event_ids[-1],
                week_start_override=week_start,
            )
        return updated

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

            if request.assessment_timezone is not None:
                try:
                    ZoneInfo(request.assessment_timezone)
                except ZoneInfoNotFoundError:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="invalid_assessment_timezone",
                        message="The assessment timezone is not supported.",
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
                if request.resume_state is not None:
                    resume_state = request.resume_state
                    server_events = self._events[existing.session_id]
                    if self._same_writing_trace(
                        resume_state.trace_events, server_events
                    ):
                        resume_state = resume_state.model_copy(
                            update={"trace_events": list(server_events)}
                        )
                    affected_date = self._resume_update_date(
                        existing,
                        resume_state,
                    )
                    if affected_date is not None:
                        if (
                            resume_state.trace_events
                            != self._events[existing.session_id]
                        ):
                            return self._error(
                                requested_operation=request.operation,
                                request_id=request.request_id,
                                code="session_conflict",
                                message=(
                                    "The browser-held Experience trace is not current."
                                ),
                            )
                        existing = await self._apply_resume_update(
                            existing=existing,
                            resume_state=resume_state,
                            affected_date=affected_date,
                        )
                    elif resume_state.revision != existing.revision:
                        return self._error(
                            requested_operation=request.operation,
                            request_id=request.request_id,
                            code="session_conflict",
                            message=(
                                "The browser-held Experience update is not based "
                                "on the current session."
                            ),
                        )
                    elif (
                        resume_state.journal_entries != existing.journal_entries
                        or resume_state.nudges != existing.nudges
                        or resume_state.assessment_clock
                        != existing.assessment_clock
                        or resume_state.trace_events
                        != self._events[existing.session_id]
                    ):
                        return self._error(
                            requested_operation=request.operation,
                            request_id=request.request_id,
                            code="session_conflict",
                            message=(
                                "The browser-held Experience state is not current."
                            ),
                        )
                if existing.assessment_clock is None and request.assessment_timezone:
                    try:
                        assessment_clock = self._initial_assessment_clock(
                            timezone=request.assessment_timezone,
                            journal_entries=existing.journal_entries,
                        )
                    except ValueError as error:
                        return self._error(
                            requested_operation=request.operation,
                            request_id=request.request_id,
                            code="invalid_assessment_timezone",
                            message=str(error),
                        )
                    existing = self._append_session(
                        existing,
                        assessment_clock=assessment_clock,
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
                decisions, _, restored_digest = self._restored_weekly_state(events)
                entry_indices = {
                    entry.t_index for entry in resume_state.journal_entries
                }
                decisions = [
                    decision
                    for decision in decisions
                    if decision.t_index in entry_indices
                ]
                drift_result = (
                    detect_drift(
                        decisions,
                        persona_id=request.profile.user_id,
                    )
                    if decisions
                    else None
                )
                populated_weeks = {
                    self._week_bounds(entry.date)[0].isoformat()
                    for entry in resume_state.journal_entries
                }
                weekly_digest = next(
                    (
                        restored_digest
                        for event in reversed(events)
                        if isinstance(event, WeeklyDigestBuiltEvent)
                        and event.details.digest.week_start in populated_weeks
                        and restored_digest is not None
                    ),
                    None,
                )
                session = ExperienceSession(
                    session_id=request.profile.session_id,
                    revision=resume_state.revision,
                    profile=request.profile,
                    journal_entries=resume_state.journal_entries,
                    nudges=resume_state.nudges,
                    weekly_reviewer_decisions=decisions,
                    drift_result=drift_result,
                    weekly_digest=weekly_digest,
                    assessment_clock=(
                        resume_state.assessment_clock
                        or self._initial_assessment_clock(
                            timezone=request.assessment_timezone,
                            journal_entries=resume_state.journal_entries,
                        )
                    ),
                    trace_event_ids=[event.event_id for event in events],
                    selection=SessionSelection(view="experience"),
                    updated_at=self._timestamp(),
                )
            else:
                try:
                    assessment_clock = self._initial_assessment_clock(
                        timezone=request.assessment_timezone,
                        journal_entries=[],
                    )
                except ValueError as error:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="invalid_assessment_timezone",
                        message=str(error),
                    )
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
                    assessment_clock=assessment_clock,
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

    async def load_saved_scenario(
        self,
        request: ScenarioLoadRequest,
        fixture: ContractFixtureSet,
    ) -> ScenarioLoadedResponse:
        """Load the first deterministic week without provider work."""
        from src.demo.scenarios import project_scenario_week

        first_week = fixture.scenario.weeks[0]
        session, events = project_scenario_week(fixture, first_week.week_id)
        async with self._lock:
            self._sessions[session.session_id] = session
            self._events[session.session_id] = list(events)
        return ScenarioLoadedResponse(
            operation="load_scenario",
            request_id=request.request_id,
            status="ok",
            session=session,
            scenario=fixture.scenario,
            event_ids=list(session.trace_event_ids),
        )

    def _ordering_error(
        self,
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
        if any(
            isinstance(event, JournalEntrySubmittedEvent)
            and event.details.journal_entry.t_index == entry.t_index
            for event in self._events[session.session_id]
        ):
            return "This Journal Entry position was already used."
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
        if (
            session.assessment_clock is not None
            and entry.date != session.assessment_clock.current_date
        ):
            return "Journal Entries must use the current simulated date."
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
        weekly_reviewer_decisions: (
            list[WeeklyDriftReviewerDecisionContract] | None
        ) = None,
        drift_result: DriftDetectorResult | None = None,
        weekly_digest: WeeklyDigest | None = None,
        assessment_clock: AssessmentClock | None = None,
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
                "weekly_reviewer_decisions": (
                    weekly_reviewer_decisions
                    if weekly_reviewer_decisions is not None
                    else session.weekly_reviewer_decisions
                ),
                "drift_result": (
                    drift_result if drift_result is not None else session.drift_result
                ),
                "weekly_digest": (
                    weekly_digest
                    if weekly_digest is not None
                    else session.weekly_digest
                ),
                "assessment_clock": (
                    assessment_clock
                    if assessment_clock is not None
                    else session.assessment_clock
                ),
                "trace_event_ids": [
                    *session.trace_event_ids,
                    *(event_ids or []),
                ],
                "updated_at": self._timestamp(),
            }
        )
        updated = ExperienceSession.model_validate(payload)
        self._sessions[updated.session_id] = updated
        return cast(ExperienceSession, updated)

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
        try:
            receipt = await self._nudge_runtime(runtime_request)
        except Exception as error:  # noqa: BLE001 - injected runtime boundary
            receipt = build_failed_nudge_runtime_receipt(runtime_request, error)
        event_status = cast(
            EventStatus,
            {
                "ok": "complete",
                "refusal": "refused",
                "invalid": "invalid",
                "error": "failed",
            }[receipt.status],
        )
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

    @staticmethod
    def _weekly_error(receipt: WeeklyDriftReviewerReceipt) -> SafeError | None:
        if receipt.status == "refusal":
            return SafeError(
                code="weekly_review_refused",
                message="The weekly review could not return a usable result.",
                retryable=False,
            )
        if receipt.status == "invalid":
            return SafeError(
                code="weekly_review_invalid",
                message="The weekly review returned an invalid result.",
                retryable=False,
            )
        if receipt.status == "error":
            return SafeError(
                code="weekly_review_failed",
                message="The weekly review could not finish.",
                retryable=False,
            )
        return None

    async def _run_coach_digest(
        self,
        *,
        digest: WeeklyDigest,
        session_id: str,
        parent_event_id: str,
        week_id: str,
    ) -> tuple[WeeklyDigest, WeeklyCoachGeneratedEvent]:
        prompt = render_digest_prompt(digest)
        narrative = None
        validation = None
        generation_failed = self._coach_llm_complete is None
        if self._coach_llm_complete is not None:
            try:
                narrative, prompt = await generate_weekly_digest_coach(
                    digest,
                    self._coach_llm_complete,
                )
            except Exception:
                generation_failed = True
            else:
                generation_failed = narrative is None

        if narrative is not None:
            validation = validate_weekly_digest_narrative(digest, narrative)
        validation_errors = (
            [check.details for check in validation.checks if not check.passed]
            if validation is not None
            else ["No valid Coach Digest response was available."]
        )
        valid = validation is not None and not validation_errors
        status: EventStatus = (
            "complete" if valid else "failed" if generation_failed else "invalid"
        )
        error = None
        if status == "failed":
            error = SafeError(
                code=(
                    "coach_provider_unavailable"
                    if self._coach_llm_complete is None
                    else "coach_response_unavailable"
                ),
                message="The Coach Digest could not return a valid response.",
                retryable=self._coach_llm_complete is not None,
            )
        elif status == "invalid":
            error = SafeError(
                code="coach_response_invalid",
                message="The Coach Digest response did not pass validation.",
                retryable=False,
            )

        event = WeeklyCoachGeneratedEvent(
            event_id=self._make_id("weekly-coach-generated"),
            session_id=session_id,
            parent_event_id=parent_event_id,
            event_type="weekly_coach_generated",
            status=status,
            source="live_run",
            input_refs=[ResourceRef(kind="weekly_digest", id=week_id)],
            model_contract=self._coach_model_contract,
            prompt=prompt,
            raw_response=(
                narrative.model_dump(mode="json") if narrative is not None else None
            ),
            validation=EventValidation(
                valid=valid,
                schema_name="WeeklyDigestCoachNarrative",
                errors=validation_errors,
            ),
            result_refs=(
                [ResourceRef(kind="weekly_coach", id=week_id)] if valid else []
            ),
            input_hash=_hash_payload(digest.model_dump(mode="json")),
            error=error,
            details=WeeklyCoachGeneratedDetails(
                narrative=narrative,
                validation=validation,
            ),
            **self._terminal_event_fields(),
        )
        return (
            attach_coach_artifacts(digest, narrative, validation) if valid else digest,
            event,
        )

    async def _run_weekly_review(
        self,
        *,
        session: ExperienceSession,
        parent_event_id: str,
        week_start_override: date | None = None,
    ) -> tuple[ExperienceSession, list[TraceEvent]]:
        if week_start_override is None:
            latest_entry = session.journal_entries[-1]
            week_start, week_end = self._week_bounds(latest_entry.date)
        else:
            week_start = week_start_override
            week_end = week_start + timedelta(days=6)
        current_entries = [
            entry
            for entry in session.journal_entries
            if week_start <= date.fromisoformat(entry.date) <= week_end
        ]
        history_entries = [
            entry
            for entry in session.journal_entries
            if date.fromisoformat(entry.date) <= week_end
        ]
        reviewer_history = [
            WeeklyDriftReviewerEntry(
                t_index=entry.t_index,
                date=entry.date,
                text=self._displayed_entry_text(
                    session,
                    journal_entry_id=entry.journal_entry_id,
                ),
            )
            for entry in history_entries
        ]
        reviewer_request = build_weekly_drift_reviewer_request(
            persona_id=session.profile.user_id,
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            core_values=session.profile.top_values,
            history=reviewer_history,
            current_t_indices=[entry.t_index for entry in current_entries],
        )
        week_id = f"{session.session_id}:{week_start.isoformat()}"
        model_contract = ModelContract(
            provider="openai",
            model="gpt-5.6-luna",
            reasoning_effort="low",
        )
        requested_event_id = self._make_id("weekly-review-requested")
        requested_event = WeeklyReviewRequestedEvent(
            event_id=requested_event_id,
            session_id=session.session_id,
            parent_event_id=parent_event_id,
            event_type="weekly_review_requested",
            status="complete",
            source="live_run",
            input_refs=[
                ResourceRef(kind="week", id=week_id),
                *[
                    ResourceRef(
                        kind="journal_entry",
                        id=entry.journal_entry_id,
                    )
                    for entry in history_entries
                ],
            ],
            model_contract=model_contract,
            prompt=reviewer_request.prompt,
            result_refs=[ResourceRef(kind="weekly_review", id=week_id)],
            input_hash=reviewer_request.runtime_text_sha256,
            details=WeeklyReviewRequestedDetails(
                request=WeeklyDriftReviewerRequestContract.model_validate(
                    reviewer_request.model_dump(mode="json")
                )
            ),
            **self._terminal_event_fields(),
        )

        receipt = await self._weekly_reviewer(reviewer_request)
        completed_status = cast(
            EventStatus,
            {
                "ok": "complete",
                "refusal": "refused",
                "invalid": "invalid",
                "error": "failed",
            }[receipt.status],
        )
        validation_errors = (
            [receipt.validation_error] if receipt.validation_error is not None else []
        )
        if receipt.assessments:
            raw_response: Any | None = {
                "assessments": [
                    assessment.model_dump(mode="json")
                    for assessment in receipt.assessments
                ]
            }
        elif receipt.refusal is not None:
            raw_response = {"refusal": receipt.refusal}
        else:
            raw_response = None
        receipt_payload = receipt.model_dump(mode="json")
        receipt_payload["error"] = None
        completed_event_id = self._make_id("weekly-review-completed")
        completed_event = WeeklyReviewCompletedEvent(
            event_id=completed_event_id,
            session_id=session.session_id,
            parent_event_id=requested_event_id,
            event_type="weekly_review_completed",
            status=completed_status,
            source="live_run",
            input_refs=[
                ResourceRef(kind="weekly_review", id=week_id),
                ResourceRef(kind="event", id=requested_event_id),
            ],
            model_contract=model_contract,
            prompt=reviewer_request.prompt,
            raw_response=raw_response,
            validation=EventValidation(
                valid=receipt.status == "ok",
                schema_name="WeeklyVerifierResponse",
                errors=validation_errors,
            ),
            result_refs=[ResourceRef(kind="weekly_review", id=week_id)],
            input_hash=reviewer_request.runtime_text_sha256,
            error=self._weekly_error(receipt),
            details=WeeklyReviewCompletedDetails(
                receipt=WeeklyDriftReviewerReceiptContract.model_validate(
                    receipt_payload
                )
            ),
            **self._terminal_event_fields(
                duration_ms=max(0, round(receipt.latency_seconds * 1000))
            ),
        )

        decisions = [
            decision
            for decision in session.weekly_reviewer_decisions
            if decision.week_start != reviewer_request.week_start
        ]
        decisions.extend(
            WeeklyDriftReviewerDecisionContract.model_validate(
                decision.model_dump(mode="json")
            )
            for decision in receipt.decisions
        )
        decisions.sort(key=lambda row: (row.t_index, row.core_value))
        drift_result = detect_drift(
            decisions,
            persona_id=session.profile.user_id,
        )
        drift_event_id = self._make_id("drift-detected")
        drift_event = DriftDetectedEvent(
            event_id=drift_event_id,
            session_id=session.session_id,
            parent_event_id=completed_event_id,
            event_type="drift_detected",
            status="complete",
            source="live_run",
            input_refs=[ResourceRef(kind="weekly_review", id=week_id)],
            result_refs=[ResourceRef(kind="drift", id=week_id)],
            input_hash=_hash_payload(
                [decision.model_dump(mode="json") for decision in decisions]
            ),
            details=DriftDetectedDetails(
                decisions=decisions,
                steps=build_drift_rule_steps(decisions),
                result=drift_result,
            ),
            **self._terminal_event_fields(),
        )

        weekly_digest = build_weekly_drift_reviewer_digest_from_entries(
            persona_id=session.profile.user_id,
            persona_name=None,
            week_start=reviewer_request.week_start,
            week_end=reviewer_request.week_end,
            core_values=list(session.profile.top_values),
            entries=reviewer_history,
            decisions=decisions,
            drift_result=drift_result,
        )
        journal_id_by_index = {
            entry.t_index: entry.journal_entry_id for entry in session.journal_entries
        }
        cited_journal_entry_ids = [
            journal_id_by_index[evidence.t_index] for evidence in weekly_digest.evidence
        ]
        digest_event_id = self._make_id("weekly-digest-built")
        digest_event = WeeklyDigestBuiltEvent(
            event_id=digest_event_id,
            session_id=session.session_id,
            parent_event_id=drift_event_id,
            event_type="weekly_digest_built",
            status="complete",
            source="live_run",
            input_refs=[
                ResourceRef(kind="week", id=week_id),
                ResourceRef(kind="drift", id=week_id),
            ],
            result_refs=[ResourceRef(kind="weekly_digest", id=week_id)],
            input_hash=_hash_payload(weekly_digest.model_dump(mode="json")),
            details=WeeklyDigestBuiltDetails(
                digest=weekly_digest,
                cited_journal_entry_ids=cited_journal_entry_ids,
            ),
            **self._terminal_event_fields(),
        )
        weekly_digest, coach_event = await self._run_coach_digest(
            digest=weekly_digest,
            session_id=session.session_id,
            parent_event_id=digest_event_id,
            week_id=week_id,
        )
        events: list[TraceEvent] = [
            requested_event,
            completed_event,
            drift_event,
            digest_event,
            coach_event,
        ]
        updated = self._append_session(
            session,
            weekly_reviewer_decisions=decisions,
            drift_result=drift_result,
            weekly_digest=weekly_digest,
            event_ids=[event.event_id for event in events],
        )
        self._events[session.session_id].extend(events)
        return updated, events

    async def _run_due_weekly_reviews(
        self,
        *,
        session: ExperienceSession,
        as_of: date,
        parent_event_id: str,
    ) -> tuple[ExperienceSession, list[TraceEvent]]:
        reviewed_weeks = self._reviewed_week_starts(self._events[session.session_id])
        due_weeks = sorted(
            {
                week_start
                for entry in session.journal_entries
                for week_start, week_end in [self._week_bounds(entry.date)]
                if week_end < as_of
                and week_start not in reviewed_weeks
                and self._week_is_finalized(session, week_start=week_start)
            }
        )
        events: list[TraceEvent] = []
        updated = session
        for week_start in due_weeks:
            updated, weekly_events = await self._run_weekly_review(
                session=updated,
                parent_event_id=parent_event_id,
                week_start_override=week_start,
            )
            events.extend(weekly_events)
            parent_event_id = weekly_events[-1].event_id
        return updated, events

    async def run_due_weekly_reviews(
        self,
        *,
        session_id: str,
        as_of: date,
    ) -> tuple[ExperienceSession, list[TraceEvent]]:
        """Review finalized Monday-Sunday weeks closed before local ``as_of``."""
        async with self._lock:
            session = self._sessions[session_id]
            return await self._run_due_weekly_reviews(
                session=session,
                as_of=as_of,
                parent_event_id=session.trace_event_ids[-1],
            )

    async def advance_assessment_time(
        self,
        request: AssessmentTimeAdvanceRequest,
    ) -> AssessmentTimeAdvancedResponse | ApiErrorResponse:
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
                            "This retry key was already used for another time change."
                        ),
                    )
                response = cast(AssessmentTimeAdvancedResponse, cached.response)
                return cast(
                    AssessmentTimeAdvancedResponse,
                    self._with_request_id(response, request.request_id),
                )

            session = self._sessions.get(request.session_id)
            if session is None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_not_found",
                    message="No Experience session exists for this time change.",
                )
            if session.assessment_clock is None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="assessment_time_unavailable",
                    message="Simulated time is not available for this session.",
                )
            if request.expected_revision != session.revision:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_conflict",
                    message="The simulated date is based on an older session revision.",
                )
            if any(nudge.outcome == "displayed" for nudge in session.nudges):
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="assessment_time_not_ready",
                    message="Answer or skip the current follow-up before time moves.",
                )

            previous = date.fromisoformat(session.assessment_clock.current_date)
            current_week, _ = self._week_bounds(previous.isoformat())
            if request.action == "close_week":
                current_entries = [
                    entry
                    for entry in session.journal_entries
                    if self._week_bounds(entry.date)[0] == current_week
                ]
                if not current_entries:
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="assessment_time_not_ready",
                        message="Write a Journal Entry before closing this week.",
                    )
                if not self._week_is_finalized(session, week_start=current_week):
                    return self._error(
                        requested_operation=request.operation,
                        request_id=request.request_id,
                        code="assessment_time_not_ready",
                        message=(
                            "Finish the current Journal Entry before closing this week."
                        ),
                    )
                current = previous + timedelta(days=7 - previous.weekday())
            else:
                current = previous + timedelta(days=1)

            clock = session.assessment_clock.model_copy(
                update={"current_date": current.isoformat()}
            )
            event = AssessmentTimeAdvancedEvent(
                event_id=self._make_id("assessment-time-advanced"),
                session_id=session.session_id,
                parent_event_id=session.trace_event_ids[-1],
                event_type="assessment_time_advanced",
                status="complete",
                source="live_run",
                input_refs=[ResourceRef(kind="assessment_time", id=session.session_id)],
                result_refs=[
                    ResourceRef(kind="assessment_time", id=session.session_id)
                ],
                input_hash=_hash_payload(
                    {
                        "action": request.action,
                        "current_date": previous.isoformat(),
                        "revision": session.revision,
                    }
                ),
                details=AssessmentTimeAdvancedDetails(
                    action=request.action,
                    previous_date=previous.isoformat(),
                    current_date=current.isoformat(),
                ),
                **self._terminal_event_fields(),
            )
            session = self._append_session(
                session,
                assessment_clock=clock,
                event_ids=[event.event_id],
                increment_revision=True,
            )
            self._events[session.session_id].append(event)
            events: list[TraceEvent] = [event]
            if request.action == "close_week":
                session, weekly_events = await self._run_due_weekly_reviews(
                    session=session,
                    as_of=current,
                    parent_event_id=event.event_id,
                )
                events.extend(weekly_events)

            response = AssessmentTimeAdvancedResponse(
                operation="advance_assessment_time",
                request_id=request.request_id,
                status="ok",
                session=session,
                event_ids=[event.event_id for event in events],
            )
            self._idempotency[key] = _IdempotentResult(
                fingerprint=fingerprint,
                response=response,
            )
            return response

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
                updated, nudge_retry_events, retryable = await self._run_nudge(
                    session=cached_session,
                    request=request,
                    parent_event_id=cast(str, cached.retry_parent_event_id),
                )
                cached_retry_parent_event_id = (
                    nudge_retry_events[-1].event_id if retryable else None
                )
                updated, weekly_events = await self._run_due_weekly_reviews(
                    session=updated,
                    as_of=date.fromisoformat(request.journal_entry.date),
                    parent_event_id=nudge_retry_events[-1].event_id,
                )
                retry_events = [*nudge_retry_events, *weekly_events]
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
                    retry_parent_event_id=cached_retry_parent_event_id,
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
            retry_parent_event_id: str | None = None

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
                retry_parent_event_id = nudge_events[-1].event_id if retryable else None

            session, weekly_events = await self._run_due_weekly_reviews(
                session=session,
                as_of=date.fromisoformat(request.journal_entry.date),
                parent_event_id=events[-1].event_id,
            )
            events.extend(weekly_events)

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
                retry_parent_event_id=retry_parent_event_id,
            )
            return response

    def _north_star_request(
        self, session: ExperienceSession, week_start: str
    ) -> NorthStarRequest:
        events = self._events[session.session_id]
        review_event = next(
            (
                event
                for event in reversed(events)
                if isinstance(event, WeeklyReviewRequestedEvent)
                and event.details.request.week_start == week_start
            ),
            None,
        )
        if review_event is None:
            raise ValueError("A live closed-week review is required")
        completed = next(
            (
                event
                for event in events
                if isinstance(event, WeeklyReviewCompletedEvent)
                and event.parent_event_id == review_event.event_id
            ),
            None,
        )
        drift = next(
            (
                event
                for event in events
                if isinstance(event, DriftDetectedEvent)
                and completed is not None
                and event.parent_event_id == completed.event_id
            ),
            None,
        )
        digest_event = next(
            (
                event
                for event in events
                if isinstance(event, WeeklyDigestBuiltEvent)
                and drift is not None
                and event.parent_event_id == drift.event_id
            ),
            None,
        )
        if drift is None or digest_event is None:
            raise ValueError("A completed weekly review is required")
        digest = digest_event.details.digest
        submissions = {
            event.details.journal_entry.journal_entry_id: event
            for event in events
            if isinstance(event, JournalEntrySubmittedEvent)
        }
        responses = {
            event.details.journal_entry_id: event
            for event in events
            if isinstance(event, NudgeResponseRecordedEvent)
        }
        writing = []
        for entry in session.journal_entries:
            submitted = submissions.get(entry.journal_entry_id)
            # Old browser state without source availability cannot supply evidence.
            if (
                submitted is None
                or submitted.details.journal_entry.content != entry.content
                or submitted.details.journal_entry.t_index != entry.t_index
                or submitted.details.journal_entry.date != entry.date
            ):
                continue
            response = responses.get(entry.journal_entry_id)
            response_at = (
                response.started_at
                if response is not None
                and response.details.response == entry.nudge_response
                else None
            )
            writing.append(
                SourceWriting(
                    owner_id=session.profile.user_id,
                    entry_id=entry.journal_entry_id,
                    t_index=entry.t_index,
                    date=entry.date,
                    journal_entry=entry.content,
                    nudge_response=entry.nudge_response,
                    available_at=submitted.started_at,
                    response_available_at=response_at,
                )
            )
        return build_north_star_request(
            session_id=session.session_id,
            owner_id=session.profile.user_id,
            profile_ref=profile_reference(session.profile.model_dump(mode="json")),
            core_values=list(session.profile.top_values),
            week_start=digest.week_start,
            week_end=digest.week_end,
            cutoff_at=review_event.started_at,
            drift_result=drift.details.result,
            writing=writing,
        )

    def _north_star_event(
        self, record: NorthStarRecord, *, event_id: str, parent_event_id: str | None
    ) -> NorthStarReviewedEvent:
        pending = record.status == "pending"
        failed = record.status == "failed"
        return NorthStarReviewedEvent(
            event_id=event_id,
            session_id=record.session_id,
            parent_event_id=parent_event_id,
            event_type="north_star_reviewed",
            status="running" if pending else "failed" if failed else "complete",
            source="live_run",
            input_refs=[
                ResourceRef(kind="week", id=f"{record.session_id}:{record.week_start}")
            ],
            model_contract=ModelContract(
                provider="openai", model="gpt-5.6-luna", reasoning_effort="low"
            ),
            input_hash=record.input_hash,
            validation=EventValidation(
                valid=record.status == "complete",
                schema_name="NorthStarRecord",
                errors=[record.reason] if failed else [],
            ),
            error=SafeError(
                code="north_star_unavailable",
                message="This moment could not be reviewed.",
                retryable=record.retryable,
            )
            if failed
            else None,
            details=NorthStarReviewedDetails(record=record),
            **(
                {"started_at": self._timestamp()}
                if pending
                else self._terminal_event_fields()
            ),
        )

    async def _execute_north_star(
        self, snapshot: NorthStarRequest, event_id: str, *, retry: bool
    ) -> NorthStarRecord | None:
        """Finish publication and cleanup even if the HTTP waiter disconnects."""
        key = (snapshot.session_id, snapshot.input_hash)
        try:
            try:
                record = validate_north_star_record(
                    await self._north_star_runtime(snapshot, retry=retry), snapshot
                )
            except Exception:
                record = pending_north_star_record(snapshot).model_copy(
                    update={
                        "status": "failed",
                        "reason": "runtime_error",
                        "retryable": False,
                    }
                )
            async with self._lock:
                current = self._sessions.get(snapshot.session_id)
                try:
                    still_current = (
                        current is not None
                        and self._north_star_request(
                            current, snapshot.week_start
                        ).input_hash
                        == snapshot.input_hash
                    )
                except ValueError:
                    still_current = False
                events = self._events.get(snapshot.session_id, [])
                index = next(
                    (i for i, event in enumerate(events) if event.event_id == event_id),
                    None,
                )
                if not still_current or index is None:
                    forget = getattr(self._north_star_runtime, "forget", None)
                    if forget is not None:
                        forget(record)
                    if index is not None:
                        omitted = pending_north_star_record(snapshot).model_copy(
                            update={
                                "status": "not_eligible",
                                "reason": "inputs_changed",
                            }
                        )
                        events[index] = self._north_star_event(
                            omitted,
                            event_id=event_id,
                            parent_event_id=events[index].parent_event_id,
                        )
                    return None
                events[index] = self._north_star_event(
                    record,
                    event_id=event_id,
                    parent_event_id=events[index].parent_event_id,
                )
                self._north_star_inflight.pop(key, None)
                return record
        finally:
            self._north_star_inflight.pop(key, None)

    async def review_north_star(
        self, request: NorthStarReviewRequest
    ) -> NorthStarReviewedResponse | ApiErrorResponse:
        """Review outside the session lock and verify the snapshot before publishing."""
        async with self._lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_not_found",
                    message="This session is no longer available.",
                )
            if session.revision != request.expected_revision:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_conflict",
                    message="The weekly review has changed.",
                )
            try:
                snapshot = self._north_star_request(session, request.week_start)
            except ValueError:
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="north_star_not_ready",
                    message="Complete this week's review first.",
                    retryable=True,
                )
            matching = next(
                (
                    event
                    for event in reversed(self._events[session.session_id])
                    if isinstance(event, NorthStarReviewedEvent)
                    and event.input_hash == snapshot.input_hash
                ),
                None,
            )
            if matching is not None and matching.details.record.status != "pending":
                try:
                    validate_north_star_record(matching.details.record, snapshot)
                except ValueError:
                    matching = None
                else:
                    if not (request.retry and matching.details.record.retryable):
                        return NorthStarReviewedResponse(
                            operation="north_star_reviewed",
                            request_id=request.request_id,
                            status="ok",
                            session=session,
                            event_ids=[matching.event_id],
                        )
            key = (session.session_id, snapshot.input_hash)
            task = self._north_star_inflight.get(key)
            if task is None:
                if matching is not None and matching.details.record.status == "pending":
                    pending_event = matching
                else:
                    pending_event = self._north_star_event(
                        pending_north_star_record(snapshot),
                        event_id=self._make_id("north-star-reviewed"),
                        parent_event_id=session.trace_event_ids[-1],
                    )
                    self._events[session.session_id].append(pending_event)
                    self._append_session(session, event_ids=[pending_event.event_id])
                task = asyncio.create_task(
                    self._execute_north_star(
                        snapshot, pending_event.event_id, retry=request.retry
                    )
                )
                self._north_star_inflight[key] = task
            event_id = next(
                event.event_id
                for event in reversed(self._events[session.session_id])
                if isinstance(event, NorthStarReviewedEvent)
                and event.input_hash == snapshot.input_hash
            )

        record = await asyncio.shield(task)
        async with self._lock:
            current = self._sessions.get(request.session_id)
            if (
                record is None
                or current is None
                or event_id not in current.trace_event_ids
            ):
                return self._error(
                    requested_operation=request.operation,
                    request_id=request.request_id,
                    code="session_conflict",
                    message="The writing changed while this moment was reviewed.",
                )
            return NorthStarReviewedResponse(
                operation="north_star_reviewed",
                request_id=request.request_id,
                status="ok",
                session=current,
                event_ids=[event_id],
            )

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

    async def delete_session(
        self,
        request: SessionDeleteRequest,
    ) -> SessionDeletedResponse:
        """Delete one in-memory session and its request receipts."""
        async with self._lock:
            forget = getattr(self._north_star_runtime, "forget", None)
            if forget is not None:
                for event in self._events.get(request.session_id, []):
                    if isinstance(event, NorthStarReviewedEvent):
                        forget(event.details.record)
            session_removed = self._sessions.pop(request.session_id, None) is not None
            events_removed = self._events.pop(request.session_id, None) is not None
            receipt_keys = [
                key
                for key, result in self._idempotency.items()
                if result.response.session.session_id == request.session_id
            ]
            for key in receipt_keys:
                del self._idempotency[key]
            return SessionDeletedResponse(
                operation="delete_session",
                request_id=request.request_id,
                status="ok",
                session_id=request.session_id,
                deleted=session_removed or events_removed or bool(receipt_keys),
            )

    async def handle(
        self,
        request: (
            SessionCreateRequest
            | JournalEntrySubmitRequest
            | AssessmentTimeAdvanceRequest
            | SessionDeleteRequest
            | TraceReadRequest
            | NorthStarReviewRequest
        ),
    ) -> ApiResponse:
        if isinstance(request, SessionCreateRequest):
            return await self.create_session(request)
        if isinstance(request, JournalEntrySubmitRequest):
            return await self.submit_journal_entry(request)
        if isinstance(request, AssessmentTimeAdvanceRequest):
            return await self.advance_assessment_time(request)
        if isinstance(request, SessionDeleteRequest):
            return await self.delete_session(request)
        if isinstance(request, NorthStarReviewRequest):
            return await self.review_north_star(request)
        return await self.read_trace(request)
