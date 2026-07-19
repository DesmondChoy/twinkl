"""Weekly Drift Reviewer request, response, and persistence contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from prompts import get_prompt_metadata, load_prompt
from src.models.judge import SCHWARTZ_VALUE_ORDER

WEEKLY_DRIFT_REVIEWER_MODEL = "gpt-5.6-luna"
WEEKLY_DRIFT_REVIEWER_REASONING_EFFORT = "low"
WEEKLY_DRIFT_REVIEWER_PROMPT = "weekly_vif_verifier"
WEEKLY_DRIFT_REVIEWER_MAX_ATTEMPTS = 2

Verdict = Literal["conflict", "not_conflict", "abstain"]
Confidence = Literal["low", "medium", "high"]
ReasonCode = Literal[
    "ambiguous",
    "direct_aligned_or_neutral_behavior",
    "direct_behavior_or_choice",
    "external_constraint",
    "feeling_or_intent_only",
    "missing_text",
    "needs_hidden_context",
]
ReviewStatus = Literal["ok", "refusal", "invalid", "error"]


class VerifierAssessment(BaseModel):
    """One current-week Journal Entry and Core Value decision."""

    model_config = ConfigDict(extra="forbid")

    t_index: int = Field(ge=0)
    dimension: str
    verdict: Verdict
    confidence: Confidence
    reason_code: ReasonCode
    evidence_quote: str = ""


class WeeklyVerifierResponse(BaseModel):
    """Frozen structured response for one persona-week."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[VerifierAssessment]


class WeeklyDriftReviewerEntry(BaseModel):
    """Displayed Journal Entry text supplied to the Weekly Drift Reviewer."""

    t_index: int = Field(ge=0)
    date: str
    text: str


class WeeklyDriftReviewerRequest(BaseModel):
    """One weekly request with cumulative student-visible history."""

    persona_id: str
    week_start: str
    week_end: str
    core_values: list[str]
    history: list[WeeklyDriftReviewerEntry]
    current_t_indices: list[int]
    prompt: str
    prompt_sha256: str
    runtime_text_sha256: str

    @property
    def expected_coordinates(self) -> set[tuple[int, str]]:
        return {
            (t_index, core_value)
            for t_index in self.current_t_indices
            for core_value in self.core_values
        }


class WeeklyDriftReviewerDecision(BaseModel):
    """Persisted effective decision used by the Drift Detector."""

    persona_id: str
    week_start: str
    week_end: str
    t_index: int = Field(ge=0)
    date: str
    core_value: str
    verdict: Verdict
    confidence: Confidence | None = None
    reason_code: ReasonCode | None = None
    evidence_quote: str = ""
    review_status: ReviewStatus


class WeeklyDriftReviewerReceipt(BaseModel):
    """Versioned request receipt and effective Weekly Drift Reviewer Decisions."""

    schema_version: Literal["weekly-drift-reviewer-receipt-v1"] = (
        "weekly-drift-reviewer-receipt-v1"
    )
    created_at: str
    persona_id: str
    week_start: str
    week_end: str
    core_values: list[str]
    current_t_indices: list[int]
    prompt_name: str
    prompt_version: str
    prompt_sha256: str
    runtime_text_sha256: str
    requested_model: str
    reasoning_effort: str
    status: ReviewStatus
    attempts: int = Field(ge=1)
    latency_seconds: float = Field(ge=0.0)
    resolved_model: str | None = None
    response_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    refusal: str | None = None
    validation_error: str | None = None
    error_type: str | None = None
    error: str | None = None
    assessments: list[VerifierAssessment] = Field(default_factory=list)
    decisions: list[WeeklyDriftReviewerDecision]


WeeklyDriftReviewerFn = Callable[
    [WeeklyDriftReviewerRequest], Awaitable[WeeklyDriftReviewerReceipt]
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_weekly_drift_reviewer_request(
    *,
    persona_id: str,
    week_start: str,
    week_end: str,
    core_values: Sequence[str],
    history: Sequence[WeeklyDriftReviewerEntry | dict[str, Any]],
    current_t_indices: Sequence[int],
) -> WeeklyDriftReviewerRequest:
    """Render the frozen prompt without VIF Critic input."""
    normalized_history = [
        WeeklyDriftReviewerEntry.model_validate(row) for row in history
    ]
    normalized_history.sort(key=lambda row: row.t_index)
    normalized_values = list(dict.fromkeys(str(value) for value in core_values))
    current_indices = list(dict.fromkeys(int(index) for index in current_t_indices))

    if not normalized_values:
        raise ValueError("At least one Core Value is required")
    invalid_values = sorted(set(normalized_values) - set(SCHWARTZ_VALUE_ORDER))
    if invalid_values:
        raise ValueError("Invalid Core Values: " + ", ".join(invalid_values))
    if not normalized_history:
        raise ValueError("At least one Journal Entry is required")
    if not current_indices:
        raise ValueError("At least one current-week Journal Entry is required")

    history_by_index = {entry.t_index: entry for entry in normalized_history}
    if len(history_by_index) != len(normalized_history):
        raise ValueError("Journal Entry t_index values must be unique")
    missing_current = sorted(set(current_indices) - set(history_by_index))
    if missing_current:
        raise ValueError(
            "Current-week Journal Entries are missing from history: "
            + ", ".join(str(index) for index in missing_current)
        )

    cumulative_history = "\n\n".join(
        f"[ENTRY t_index={entry.t_index}]\n{entry.text}" for entry in normalized_history
    )
    current_week_entries = "\n".join(
        f"- t_index={t_index}" for t_index in current_indices
    )
    prompt = (
        load_prompt(WEEKLY_DRIFT_REVIEWER_PROMPT)
        .render(
            declared_values="\n".join(f"- {value}" for value in normalized_values),
            cumulative_history=cumulative_history,
            current_week_entries=current_week_entries,
        )
        .strip()
        + "\n"
    )
    runtime_payload = [entry.model_dump(mode="json") for entry in normalized_history]
    return WeeklyDriftReviewerRequest(
        persona_id=persona_id,
        week_start=week_start,
        week_end=week_end,
        core_values=normalized_values,
        history=normalized_history,
        current_t_indices=current_indices,
        prompt=prompt,
        prompt_sha256=_sha256_text(prompt),
        runtime_text_sha256=_sha256_text(_canonical_json(runtime_payload)),
    )


def validate_weekly_drift_reviewer_response(
    response: WeeklyVerifierResponse,
    request: WeeklyDriftReviewerRequest,
) -> None:
    """Validate complete coordinates and exact Conflict evidence quotes."""
    validate_weekly_drift_reviewer_assessments(
        response,
        expected_coordinates=request.expected_coordinates,
        entry_text_by_t_index={entry.t_index: entry.text for entry in request.history},
    )


def validate_weekly_drift_reviewer_assessments(
    response: WeeklyVerifierResponse,
    *,
    expected_coordinates: set[tuple[int, str]],
    entry_text_by_t_index: dict[int, str],
) -> None:
    """Validate the frozen response independently of its persistence envelope."""
    observed = {
        (assessment.t_index, assessment.dimension)
        for assessment in response.assessments
    }
    if len(observed) != len(response.assessments) or observed != expected_coordinates:
        raise ValueError(
            f"Response coordinate mismatch: expected={sorted(expected_coordinates)}, "
            f"observed={sorted(observed)}"
        )

    for assessment in response.assessments:
        quote = assessment.evidence_quote.strip()
        if assessment.verdict == "conflict" and not quote:
            raise ValueError("Conflict assessment is missing an evidence quote")
        if assessment.verdict == "conflict" and quote not in entry_text_by_t_index.get(
            assessment.t_index, ""
        ):
            raise ValueError(
                f"Evidence quote is not present in t_index={assessment.t_index}"
            )


def _usage_tokens(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _response_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def _is_transient_error(error: Exception) -> bool:
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:
        return False
    return isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    )


def _effective_decisions(
    request: WeeklyDriftReviewerRequest,
    *,
    status: ReviewStatus,
    response: WeeklyVerifierResponse | None,
) -> list[WeeklyDriftReviewerDecision]:
    assessments = (
        {
            (assessment.t_index, assessment.dimension): assessment
            for assessment in response.assessments
        }
        if response is not None and status == "ok"
        else {}
    )
    entry_by_index = {entry.t_index: entry for entry in request.history}
    decisions: list[WeeklyDriftReviewerDecision] = []
    for t_index in request.current_t_indices:
        for core_value in request.core_values:
            assessment = assessments.get((t_index, core_value))
            decisions.append(
                WeeklyDriftReviewerDecision(
                    persona_id=request.persona_id,
                    week_start=request.week_start,
                    week_end=request.week_end,
                    t_index=t_index,
                    date=entry_by_index[t_index].date,
                    core_value=core_value,
                    verdict=assessment.verdict if assessment else "abstain",
                    confidence=assessment.confidence if assessment else None,
                    reason_code=assessment.reason_code if assessment else None,
                    evidence_quote=assessment.evidence_quote if assessment else "",
                    review_status=status,
                )
            )
    return decisions


def _receipt(
    request: WeeklyDriftReviewerRequest,
    *,
    status: ReviewStatus,
    attempts: int,
    latency_seconds: float,
    response: WeeklyVerifierResponse | None = None,
    resolved_model: str | None = None,
    response_id: str | None = None,
    usage: dict[str, int] | None = None,
    refusal: str | None = None,
    validation_error: str | None = None,
    error: Exception | None = None,
) -> WeeklyDriftReviewerReceipt:
    prompt_metadata = get_prompt_metadata(WEEKLY_DRIFT_REVIEWER_PROMPT)
    return WeeklyDriftReviewerReceipt(
        created_at=datetime.now(UTC).isoformat(),
        persona_id=request.persona_id,
        week_start=request.week_start,
        week_end=request.week_end,
        core_values=request.core_values,
        current_t_indices=request.current_t_indices,
        prompt_name=WEEKLY_DRIFT_REVIEWER_PROMPT,
        prompt_version=str(prompt_metadata["version"]),
        prompt_sha256=request.prompt_sha256,
        runtime_text_sha256=request.runtime_text_sha256,
        requested_model=WEEKLY_DRIFT_REVIEWER_MODEL,
        reasoning_effort=WEEKLY_DRIFT_REVIEWER_REASONING_EFFORT,
        status=status,
        attempts=attempts,
        latency_seconds=latency_seconds,
        resolved_model=resolved_model,
        response_id=response_id,
        usage=usage or {},
        refusal=refusal,
        validation_error=validation_error,
        error_type=type(error).__name__ if error else None,
        error=str(error) if error else None,
        assessments=response.assessments if response is not None else [],
        decisions=_effective_decisions(request, status=status, response=response),
    )


class OpenAIWeeklyDriftReviewer:
    """Frozen Luna-low caller with transient-only retry and fail-closed decisions."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 2000,
        service_tier: str = "default",
    ) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._service_tier = service_tier

    async def __call__(
        self, request: WeeklyDriftReviewerRequest
    ) -> WeeklyDriftReviewerReceipt:
        started = time.monotonic()
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except Exception as error:  # noqa: BLE001 - provider boundary
                return _receipt(
                    request,
                    status="error",
                    attempts=1,
                    latency_seconds=time.monotonic() - started,
                    error=error,
                )

        last_error: Exception | None = None
        for attempt in range(1, WEEKLY_DRIFT_REVIEWER_MAX_ATTEMPTS + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": WEEKLY_DRIFT_REVIEWER_MODEL,
                    "input": request.prompt,
                    "text_format": WeeklyVerifierResponse,
                    "reasoning": {"effort": WEEKLY_DRIFT_REVIEWER_REASONING_EFFORT},
                    "max_output_tokens": self._max_output_tokens,
                    "store": False,
                    "service_tier": self._service_tier,
                    "timeout": self._timeout_seconds,
                }
                api_response = await self._client.responses.parse(**request_kwargs)
                parsed = getattr(api_response, "output_parsed", None)
                resolved_model = getattr(api_response, "model", None)
                response_id = getattr(api_response, "id", None)
                usage = _usage_tokens(api_response)
                if not isinstance(parsed, WeeklyVerifierResponse):
                    return _receipt(
                        request,
                        status="refusal",
                        attempts=attempt,
                        latency_seconds=time.monotonic() - started,
                        resolved_model=resolved_model,
                        response_id=response_id,
                        usage=usage,
                        refusal=_response_refusal(api_response),
                    )
                try:
                    validate_weekly_drift_reviewer_response(parsed, request)
                except ValueError as error:
                    return _receipt(
                        request,
                        status="invalid",
                        attempts=attempt,
                        latency_seconds=time.monotonic() - started,
                        response=parsed,
                        resolved_model=resolved_model,
                        response_id=response_id,
                        usage=usage,
                        validation_error=str(error),
                    )
                return _receipt(
                    request,
                    status="ok",
                    attempts=attempt,
                    latency_seconds=time.monotonic() - started,
                    response=parsed,
                    resolved_model=resolved_model,
                    response_id=response_id,
                    usage=usage,
                )
            except Exception as error:  # noqa: BLE001 - provider boundary
                last_error = error
                if (
                    attempt >= WEEKLY_DRIFT_REVIEWER_MAX_ATTEMPTS
                    or not _is_transient_error(error)
                ):
                    break
                await asyncio.sleep((2 ** (attempt - 1)) + random.random())

        return _receipt(
            request,
            status="error",
            attempts=attempt,
            latency_seconds=time.monotonic() - started,
            error=last_error,
        )


def persist_weekly_drift_reviewer_receipt(
    receipt: WeeklyDriftReviewerReceipt,
    path: Path,
) -> Path:
    """Persist one versioned Weekly Drift Reviewer receipt as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2) + "\n")
    return path
