"""Shared, source-bound North Star Moment runtime for both Experience paths.

Semantic support comes only from the factual AI assessment. This module owns
source availability, chronology, exact quotation checks, and one-card selection.
The default ledger keeps spend receipts on disk and response text in memory;
session deletion calls ``forget`` to release that memory too.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.drift_detector import DriftDetectorResult, DriftRecord
from src.north_star import assessment, input_budget
from src.north_star.provider import (
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    ProviderAttempt,
    ProviderRole,
    stable_hash,
)
from src.north_star.review import (
    ReviewBatch,
    ReviewResult,
    ReviewValidationError,
    SourceEntry,
)

ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_POLICY_PATH = ROOT / "config/evals/north_star_integration_v1.json"
DEFAULT_DIRECTORY = ROOT / "logs/experiments/reports/north_star_integration_20260906"
PROTOCOL_VERSION = "north-star-integration-v1"
RecordStatus = Literal["pending", "complete", "failed", "not_eligible"]
MomentMode = Literal["reflection", "encouragement", "reminder"]


def profile_reference(profile: dict[str, Any]) -> str:
    """Hash Profile JSON with the same integral-number representation as browsers."""

    def normalize(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    payload = json.dumps(
        normalize(profile), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("North Star Moment timestamps require a timezone")
    return result.astimezone(UTC)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceWriting(_Model):
    owner_id: str
    entry_id: str
    t_index: int = Field(ge=0)
    date: str
    journal_entry: str
    nudge_response: str | None = None
    available_at: str
    response_available_at: str | None = None

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        date.fromisoformat(self.date)
        _timestamp(self.available_at)
        if self.response_available_at is not None:
            if _timestamp(self.response_available_at) < _timestamp(self.available_at):
                raise ValueError("Response cannot precede its Journal Entry")
        if not self.owner_id.strip() or not self.entry_id.strip():
            raise ValueError("Source identity must be nonempty")
        return self


class NorthStarValue(_Model):
    core_value: str
    user_phrase: str
    approved_definition: str


class NorthStarRequest(_Model):
    schema_version: Literal["north-star-request-v1"] = "north-star-request-v1"
    protocol_version: str = PROTOCOL_VERSION
    prompt_version: str = assessment.SOURCE_PROMPT_VERSION
    assessment_schema_version: str = assessment.SOURCE_SCHEMA_VERSION
    session_id: str
    owner_id: str
    profile_ref: str
    core_values: list[str]
    value_definitions: dict[str, NorthStarValue]
    week_start: str
    week_end: str
    cutoff_at: str
    drift_result: DriftDetectorResult
    writing: list[SourceWriting]
    input_hash: str


class NorthStarSelection(_Model):
    entry_id: str
    t_index: int
    date: str
    quote_source: Literal["journal_entry", "nudge_response"]
    evidence_quote: str


class NorthStarValueReview(_Model):
    core_value: str
    value_phrase: str
    approved_definition: str
    prompt_version: str = assessment.SOURCE_PROMPT_VERSION
    schema_version: str = assessment.SOURCE_SCHEMA_VERSION
    sources: list[SourceWriting]
    provider_request: dict[str, Any]
    input_receipt: dict[str, Any] | None = None
    provider_attempts: list[ProviderAttempt] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class NorthStarRecord(_Model):
    schema_version: Literal["north-star-record-v1"] = "north-star-record-v1"
    session_id: str
    owner_id: str
    profile_ref: str
    week_start: str
    week_end: str
    cutoff_at: str
    input_hash: str
    status: RecordStatus
    mode: MomentMode | None = None
    reason: str
    core_value: str | None = None
    value_phrase: str | None = None
    selected: NorthStarSelection | None = None
    source_ids: list[str] = Field(default_factory=list)
    sources: list[SourceWriting] = Field(default_factory=list)
    onset_t_index: int | None = None
    onset_date: str | None = None
    onset_available_at: str | None = None
    reviews: list[NorthStarValueReview] = Field(default_factory=list)
    validation_evidence: list[str] = Field(default_factory=list)
    attempts: int = 0
    retryable: bool = False
    created_at: str


def build_north_star_request(
    *,
    session_id: str,
    owner_id: str,
    profile_ref: str,
    core_values: list[str],
    week_start: str,
    week_end: str,
    cutoff_at: str,
    drift_result: DriftDetectorResult,
    writing: list[SourceWriting],
) -> NorthStarRequest:
    """Bind the confirmed Profile, review context, and complete source state."""
    if not session_id or not owner_id or not profile_ref:
        raise ValueError("North Star Moment requires session and Profile identity")
    if date.fromisoformat(week_start) > date.fromisoformat(week_end):
        raise ValueError("Invalid reviewed week")
    _timestamp(cutoff_at)
    if drift_result.persona_id != owner_id:
        raise ValueError("Weekly Drift Detection belongs to another owner")
    if date.fromisoformat(drift_result.cutoff_date) > date.fromisoformat(week_end):
        raise ValueError("Weekly Drift Detection extends beyond the reviewed week")
    if not core_values or len(set(core_values)) != len(core_values):
        raise ValueError("Core Values must be nonempty and distinct")
    configured = yaml.safe_load((ROOT / "config/schwartz_values.yaml").read_text())
    values = {
        name.lower().replace("-", "_"): NorthStarValue(
            core_value=name.lower().replace("-", "_"),
            user_phrase=details["user_phrase"].strip(),
            approved_definition=details["definition"].strip(),
        )
        for name, details in configured["values"].items()
    }
    if any(value not in values for value in core_values):
        raise ValueError("Unknown Core Value")
    if set(drift_result.core_value_states) != set(core_values):
        raise ValueError("Weekly Drift Detection uses different Core Values")
    source_ids = [source.entry_id for source in writing]
    coordinates = [(source.owner_id, source.t_index) for source in writing]
    if len(set(source_ids)) != len(source_ids) or len(set(coordinates)) != len(writing):
        raise ValueError("Duplicate source identity or order")
    cutoff = _timestamp(cutoff_at)
    available = []
    for source in writing:
        if (
            source.owner_id != owner_id
            or source.t_index > drift_result.cutoff_t_index
            or source.date > week_end
            or _timestamp(source.available_at) > cutoff
        ):
            continue
        response_eligible = (
            source.response_available_at is not None
            and _timestamp(source.response_available_at) <= cutoff
        )
        available.append(
            source.model_copy(
                update={
                    "nudge_response": source.nudge_response
                    if response_eligible
                    else None,
                    "response_available_at": source.response_available_at
                    if response_eligible
                    else None,
                }
            )
        )
    available.sort(key=lambda source: source.t_index)
    if any(
        left.date > right.date
        for left, right in zip(available, available[1:], strict=False)
    ):
        raise ValueError("Journal Entry dates disagree with their stored order")
    payload = {
        "schema_version": "north-star-request-v1",
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": assessment.SOURCE_PROMPT_VERSION,
        "assessment_schema_version": assessment.SOURCE_SCHEMA_VERSION,
        "session_id": session_id,
        "owner_id": owner_id,
        "profile_ref": profile_ref,
        "core_values": core_values,
        "value_definitions": {
            value: values[value].model_dump() for value in core_values
        },
        "week_start": week_start,
        "week_end": week_end,
        "cutoff_at": cutoff_at,
        "drift_result": drift_result.model_dump(mode="json"),
        "writing": [source.model_dump() for source in available],
    }
    return NorthStarRequest.model_validate(
        {**payload, "input_hash": stable_hash(payload)}
    )


def _validate_request(request: NorthStarRequest) -> None:
    expected = build_north_star_request(
        **request.model_dump(
            include={
                "session_id",
                "owner_id",
                "profile_ref",
                "core_values",
                "week_start",
                "week_end",
                "cutoff_at",
            }
        ),
        drift_result=request.drift_result,
        writing=request.writing,
    )
    if expected != request:
        raise ValueError("North Star Moment request is stale or changed")


def _context(
    request: NorthStarRequest,
) -> tuple[list[str], list[SourceWriting], DriftRecord | None, str | None, str]:
    result = request.drift_result
    if result.delivery_state == "insufficient_evidence":
        return [], [], None, None, "insufficient_evidence"
    values = list(request.core_values)
    onset = None
    onset_at = None
    if result.delivery_state == "active_drift":
        active = [v for v in values if result.core_value_states[v] == "active_drift"]
        if not active or any(v not in result.core_value_details for v in active):
            return [], [], None, None, "missing_active_drift_context"
        value = max(
            active, key=lambda v: result.core_value_details[v].current_run_length
        )
        matches = [
            row
            for row in result.drifts
            if row.core_value == value
            and row.termination_reason is None
            and row.end_t_index == result.core_value_details[value].last_t_index
            and row.persona_id == request.owner_id
        ]
        if len(matches) != 1:
            return [], [], None, None, "missing_active_drift_onset"
        onset = matches[0]
        onset_sources = [
            source
            for source in request.writing
            if source.owner_id == request.owner_id
            and source.t_index == onset.onset_t_index
            and source.date == onset.onset_date
        ]
        if len(onset_sources) != 1:
            return [], [], onset, None, "missing_onset_availability"
        onset_at = onset_sources[0].available_at
        values = [value]
    cutoff = _timestamp(request.cutoff_at)
    if onset_at is not None and _timestamp(onset_at) > cutoff:
        return [], [], onset, onset_at, "onset_unavailable_at_cutoff"
    sources = []
    for source in request.writing:
        if (
            source.owner_id != request.owner_id
            or source.t_index > result.cutoff_t_index
            or source.date > request.week_end
            or _timestamp(source.available_at) > cutoff
        ):
            continue
        if onset is not None and (
            source.t_index >= onset.onset_t_index
            or source.date > onset.onset_date
            or (
                onset_at is not None
                and _timestamp(source.available_at) >= _timestamp(onset_at)
            )
        ):
            continue
        response_eligible = (
            source.response_available_at is not None
            and _timestamp(source.response_available_at) <= cutoff
            and (
                onset_at is None
                or _timestamp(source.response_available_at) < _timestamp(onset_at)
            )
        )
        eligible = source.model_copy(
            update={
                "nudge_response": source.nudge_response if response_eligible else None,
                "response_available_at": source.response_available_at
                if response_eligible
                else None,
            }
        )
        if eligible.journal_entry.strip() or (
            eligible.nudge_response and eligible.nudge_response.strip()
        ):
            sources.append(eligible)
    sources.sort(key=lambda source: source.t_index, reverse=True)
    return (
        values,
        sources,
        onset,
        onset_at,
        "eligible" if sources else "no_eligible_writing",
    )


def _review_sources(sources: Sequence[SourceWriting]) -> list[SourceEntry]:
    return [
        SourceEntry(
            entry_id=source.entry_id,
            journal_entry=source.journal_entry,
            nudge_response=source.nudge_response,
        )
        for source in sources
    ]


def source_review_requests(request: NorthStarRequest) -> list[dict[str, Any]]:
    """Return exact complete runtime inputs, without provider calls or truncation."""
    _validate_request(request)
    values, sources, _, _, reason = _context(request)
    if reason != "eligible":
        return []
    requests = []
    for value in values:
        system, prompt = assessment.build_source_prompt(
            **request.value_definitions[value].model_dump(),
            sources=_review_sources(sources),
        )
        payload = json.loads(prompt)
        # Bind cache receipts to session and source state without exposing Drift
        # judgments as semantic evidence to the source assessor.
        payload["context_hash"] = request.input_hash
        requests.append(
            {
                "system": system,
                "prompt": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                "schema": assessment.source_json_schema(),
                "provider": "openai",
                "role": "runtime",
                "purpose": f"nsm-integration:{request.session_id}",
            }
        )
    return requests


def _record(request: NorthStarRequest, **updates: Any) -> NorthStarRecord:
    _, _, onset, onset_at, _ = _context(request)
    fields = request.model_dump(
        include={
            "session_id",
            "owner_id",
            "profile_ref",
            "week_start",
            "week_end",
            "cutoff_at",
            "input_hash",
        }
    )
    return NorthStarRecord(
        **fields,
        onset_t_index=onset.onset_t_index if onset else None,
        onset_date=onset.onset_date if onset else None,
        onset_available_at=onset_at,
        created_at=datetime.now(UTC).isoformat(),
        **updates,
    )


def pending_north_star_record(request: NorthStarRequest) -> NorthStarRecord:
    _validate_request(request)
    return _record(request, status="pending", reason="awaiting_ai_review")


def _pick(
    request: NorthStarRequest,
    reviewed: Sequence[tuple[str, ReviewBatch]],
    sources: Sequence[SourceWriting],
) -> tuple[str, ReviewResult, SourceWriting] | None:
    candidates = []
    for value, batch in reviewed:
        results = {result.entry_id: result for result in batch.results}
        for source in sources:
            result = results[source.entry_id]
            if result.decision == "supportive":
                candidates.append((value, result, source))
    if request.drift_result.delivery_state != "active_drift":
        current = [row for row in candidates if request.week_start <= row[2].date]
        if current:
            return current[0]
    return candidates[0] if candidates else None


class RuntimeProvider(Protocol):
    ledger: BudgetLedger

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict,
        provider: Literal["openai", "gemini"],
        purpose: str,
        retry: bool = False,
        role: ProviderRole | None = None,
    ) -> ProviderAttempt: ...

    def invalidate(self, attempt: ProviderAttempt, reason: str) -> ProviderAttempt: ...


class _PrivateResponseLedger(BudgetLedger):
    """Persist budget accounting while session-owned receipts retain source text."""

    def __init__(self, path: Path, policy_path: Path):
        super().__init__(path, policy_path)
        self.raw_responses: dict[tuple[str, int], str] = {}

    def reserve(self, request: dict, *, retry: bool) -> ProviderAttempt:
        attempt = super().reserve(request, retry=retry)
        key = attempt.request_hash, attempt.attempt_number
        if key in self.raw_responses:
            attempt.raw_text = self.raw_responses[key]
        return attempt

    def finish(self, attempt: ProviderAttempt) -> ProviderAttempt:
        key = attempt.request_hash, attempt.attempt_number
        if attempt.raw_text is not None:
            self.raw_responses[key] = attempt.raw_text
        super().finish(attempt.model_copy(update={"raw_text": None}))
        return attempt


CountRequests = Callable[[list[dict], dict, Path], Awaitable[dict]]


class OpenAINorthStarRuntime:
    def __init__(
        self,
        *,
        provider: RuntimeProvider | None = None,
        count_requests: CountRequests | None = None,
        ledger_path: Path | None = None,
        counts_path: Path | None = None,
        policy_path: Path = INTEGRATION_POLICY_PATH,
    ):
        self.provider = provider
        self.count_requests = count_requests or input_budget.measure_requests
        self.ledger_path = ledger_path or DEFAULT_DIRECTORY / "budget.json"
        self.counts_path = counts_path or DEFAULT_DIRECTORY / "input-counts.json"
        self.policy_path = policy_path
        self.policy = json.loads(policy_path.read_text())
        self._injected_provider = provider is not None
        settings = self.policy["runtime"]
        if (
            settings["provider"] != "openai"
            or settings["model"] != "gpt-5.6-luna"
            or settings["reasoning_effort"] != "low"
            or self.policy["max_attempts"] != 2
            or self.policy["input_token_limit"] != input_budget.INPUT_TOKEN_LIMIT
            or self.policy["per_attempt_usd"] > 0.25
            or self.policy["budget_usd"] > 20
        ):
            raise ValueError("Runtime must use the approved Luna integration policy")
        if provider is not None and provider.ledger.policy != self.policy:
            raise ValueError("Provider and runtime budget policies differ")

    def forget(self, record: NorthStarRecord) -> None:
        """Release transient provider text when its owning session is deleted."""
        if self.provider and isinstance(self.provider.ledger, _PrivateResponseLedger):
            for review in record.reviews:
                for attempt in review.provider_attempts:
                    self.provider.ledger.raw_responses.pop(
                        (attempt.request_hash, attempt.attempt_number), None
                    )

    async def __call__(
        self, request: NorthStarRequest, *, retry: bool = False
    ) -> NorthStarRecord:
        _validate_request(request)
        values, sources, _, _, reason = _context(request)
        if reason != "eligible":
            return _record(request, status="not_eligible", reason=reason)
        if not self._injected_provider and not os.environ.get("OPENAI_API_KEY"):
            return _record(
                request, status="failed", reason="provider_unavailable", retryable=True
            )
        if self.provider is None:
            self.provider = BudgetedProvider(
                _PrivateResponseLedger(self.ledger_path, self.policy_path)
            )
        provider = self.provider
        assert provider is not None
        requests = source_review_requests(request)
        reviews: list[NorthStarValueReview] = []
        reviewed: list[tuple[str, ReviewBatch]] = []
        attempts = 0
        for value, provider_request in zip(values, requests, strict=True):
            metadata = request.value_definitions[value]
            record_review = NorthStarValueReview(
                core_value=value,
                value_phrase=metadata.user_phrase,
                approved_definition=metadata.approved_definition,
                sources=list(sources),
                provider_request=provider_request,
            )
            reviews.append(record_review)
            try:
                counts = await self.count_requests(
                    [provider_request], self.policy, self.counts_path
                )
                receipt = counts["counts"].get(stable_hash(provider_request))
                count = input_budget.validate_receipt(
                    provider_request, self.policy, receipt
                )
                record_review = record_review.model_copy(
                    update={"input_receipt": receipt}
                )
                reviews[-1] = record_review
                if count > input_budget.INPUT_TOKEN_LIMIT:
                    raise input_budget.InputBudgetError("complete_input_exceeds_16000")
            except Exception as exc:
                code = getattr(exc, "status_code", getattr(exc, "code", None))
                retryable = code in (408, 429, 500, 502, 503, 504) or any(
                    word in type(exc).__name__.lower()
                    for word in ("timeout", "connection")
                )
                return _record(
                    request,
                    status="failed",
                    reason=f"input_budget:{type(exc).__name__}",
                    reviews=reviews,
                    attempts=attempts,
                    retryable=retryable,
                    validation_evidence=[str(exc)]
                    if isinstance(exc, input_budget.InputBudgetError)
                    else [],
                )
            call_retry = retry
            for _ in range(self.policy["max_attempts"]):
                try:
                    attempt = await provider.complete(
                        **provider_request, retry=call_retry
                    )
                except BudgetError as exc:
                    return _record(
                        request,
                        status="failed",
                        reason="budget_unavailable",
                        reviews=reviews,
                        attempts=attempts,
                        validation_evidence=[str(exc)],
                    )
                attempts += int(not attempt.reused)
                record_review.provider_attempts.append(attempt)
                if attempt.status == "completed":
                    try:
                        batch = assessment.validate_source_review(
                            attempt.raw_text or "",
                            core_value=value,
                            sources=_review_sources(sources),
                        )
                    except ReviewValidationError as exc:
                        # Reused completed receipts must take this same path.
                        # A crash between provider completion and app validation
                        # must never turn malformed output into a terminal cache.
                        attempt = provider.invalidate(attempt, str(exc))
                        record_review.provider_attempts[-1] = attempt
                        record_review.validation_errors.extend(exc.errors)
                    else:
                        reviewed.append((value, batch))
                        break
                if (
                    attempt.retryable
                    and attempt.attempt_number < self.policy["max_attempts"]
                    and (not attempt.reused or retry or attempt.status == "invalid")
                ):
                    call_retry = True
                    continue
                return _record(
                    request,
                    status="failed",
                    reason=f"provider_{attempt.status}",
                    reviews=reviews,
                    attempts=attempts,
                    retryable=attempt.retryable
                    and attempt.attempt_number < self.policy["max_attempts"],
                )
            else:
                return _record(
                    request,
                    status="failed",
                    reason="retry_limit_reached",
                    reviews=reviews,
                    attempts=attempts,
                )
        selected = _pick(request, reviewed, sources)
        common = {
            "status": "complete",
            "reviews": reviews,
            "attempts": attempts,
            "sources": sources,
            "source_ids": [source.entry_id for source in sources],
            "validation_evidence": [
                "same_owner",
                "source_available_at_cutoff",
                "complete_source_review",
                "exact_continuous_quote",
                "no_internal_value_labels",
                "ai_assessment_not_human_validation",
            ],
        }
        if selected is None:
            return _record(request, reason="no_supportive_source", **common)
        value, result, source = selected
        assert result.quote_source is not None
        mode: MomentMode = (
            "reflection"
            if request.drift_result.delivery_state == "active_drift"
            else "encouragement"
            if source.date >= request.week_start
            else "reminder"
        )
        record = _record(
            request,
            reason="supportive_action_selected",
            mode=mode,
            core_value=value,
            value_phrase=request.value_definitions[value].user_phrase,
            selected=NorthStarSelection(
                entry_id=source.entry_id,
                t_index=source.t_index,
                date=source.date,
                quote_source=result.quote_source,
                evidence_quote=result.evidence_quote,
            ),
            **common,
        )
        return validate_north_star_record(record, request)


def validate_north_star_record(
    record: NorthStarRecord, request: NorthStarRequest
) -> NorthStarRecord:
    """Revalidate saved results against current identity, context, and raw reviews."""
    _validate_request(request)
    for field in (
        "session_id",
        "owner_id",
        "profile_ref",
        "week_start",
        "week_end",
        "cutoff_at",
        "input_hash",
    ):
        if getattr(record, field) != getattr(request, field):
            raise ValueError(f"North Star Moment record mismatch: {field}")
    values, sources, onset, onset_at, reason = _context(request)
    if (record.onset_t_index, record.onset_date, record.onset_available_at) != (
        onset.onset_t_index if onset else None,
        onset.onset_date if onset else None,
        onset_at,
    ):
        raise ValueError("North Star Moment onset changed")
    if record.status != "complete":
        if record.selected is not None or record.mode is not None:
            raise ValueError("An incomplete North Star Moment cannot show a card")
        return record
    if reason != "eligible" or record.sources != sources:
        raise ValueError("North Star Moment source eligibility changed")
    if record.source_ids != [source.entry_id for source in sources]:
        raise ValueError("North Star Moment source identifiers changed")
    if len(record.reviews) != len(values):
        raise ValueError("North Star Moment review coverage is incomplete")
    reviewed = []
    policy = json.loads(INTEGRATION_POLICY_PATH.read_text())
    for value, saved, provider_request in zip(
        values, record.reviews, source_review_requests(request), strict=True
    ):
        metadata = request.value_definitions[value]
        if (
            saved.core_value != value
            or saved.sources != sources
            or saved.value_phrase != metadata.user_phrase
            or saved.approved_definition != metadata.approved_definition
            or saved.provider_request != provider_request
            or saved.prompt_version != assessment.SOURCE_PROMPT_VERSION
            or saved.schema_version != assessment.SOURCE_SCHEMA_VERSION
        ):
            raise ValueError("North Star Moment review inputs changed")
        if (
            not saved.provider_attempts
            or saved.provider_attempts[-1].status != "completed"
        ):
            raise ValueError("North Star Moment lacks a completed provider receipt")
        input_tokens = input_budget.validate_receipt(
            provider_request, policy, saved.input_receipt
        )
        if input_tokens > input_budget.INPUT_TOKEN_LIMIT:
            raise ValueError("North Star Moment complete input exceeds token limit")
        attempt = saved.provider_attempts[-1]
        expected_hash = stable_hash(
            {**provider_request, "policy_hash": stable_hash(policy)}
        )
        if (
            attempt.request_hash != expected_hash
            or attempt.requested_model != policy["runtime"]["model"]
            or attempt.reasoning_effort != "low"
            or attempt.provider != "openai"
            or attempt.role != "runtime"
            or not 1 <= attempt.attempt_number <= policy["max_attempts"]
        ):
            raise ValueError("North Star Moment provider receipt changed")
        reviewed.append(
            (
                value,
                assessment.validate_source_review(
                    saved.provider_attempts[-1].raw_text or "",
                    core_value=value,
                    sources=_review_sources(sources),
                ),
            )
        )
    selected = _pick(request, reviewed, sources)
    if selected is None:
        if record.selected is not None or record.mode is not None:
            raise ValueError("North Star Moment omission changed")
    else:
        value, result, source = selected
        assert result.quote_source is not None
        mode = (
            "reflection"
            if onset
            else "encouragement"
            if source.date >= request.week_start
            else "reminder"
        )
        expected = NorthStarSelection(
            entry_id=source.entry_id,
            t_index=source.t_index,
            date=source.date,
            quote_source=result.quote_source,
            evidence_quote=result.evidence_quote,
        )
        if (
            record.selected != expected
            or record.mode != mode
            or record.core_value != value
            or record.value_phrase != request.value_definitions[value].user_phrase
        ):
            raise ValueError("North Star Moment selection changed")
    return record
