"""Production-representative nudge runtime for Experience."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prompts import get_prompt_metadata, load_prompt
from src.nudge.decision import format_previous_entries
from src.nudge.schemas import (
    NUDGE_DECISION_AND_GENERATION_RESPONSE_FORMAT,
    NudgeDecisionAndGenerationResponse,
)

NUDGE_RUNTIME_MODEL = "gpt-5.6-luna"
NUDGE_RUNTIME_REASONING_EFFORT = "none"
NUDGE_RUNTIME_PROMPT = "nudge_decision_and_generation"
NUDGE_MIN_WORDS = 2
NUDGE_MAX_WORDS = 12

NudgeRuntimeStatus = Literal["ok", "refusal", "invalid", "error"]


class NudgeContextEntry(BaseModel):
    """One sanitized prior Journal Entry supplied as context."""

    model_config = ConfigDict(extra="forbid")

    date: str
    content: str


class NudgeRuntimeRequest(BaseModel):
    """Rendered request for one merged decision-and-generation call."""

    model_config = ConfigDict(extra="forbid")

    entry_date: str
    entry_content: str = Field(min_length=1)
    previous_entries: list[NudgeContextEntry] = Field(default_factory=list)
    prompt: str
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NudgeRuntimeReceipt(BaseModel):
    """Inspect-ready record of one GPT-5.6 Luna call without reasoning."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["nudge-runtime-receipt-v1"] = "nudge-runtime-receipt-v1"
    created_at: str
    prompt_name: str
    prompt_version: str
    prompt_sha256: str
    requested_model: str
    reasoning_effort: str
    status: NudgeRuntimeStatus
    attempts: Literal[1] = 1
    latency_seconds: float = Field(ge=0.0)
    resolved_model: str | None = None
    response_id: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: str | None = None
    refusal: str | None = None
    validation_error: str | None = None
    error_type: str | None = None
    error: str | None = None
    decision: str | None = None
    reason: str | None = None
    nudge_text: str | None = None


def build_nudge_runtime_request(
    *,
    entry_content: str,
    entry_date: str,
    previous_entries: list[dict[str, Any]] | None = None,
) -> NudgeRuntimeRequest:
    """Sanitize context and render the approved merged prompt."""
    content = entry_content.strip()
    if not content:
        raise ValueError("Journal Entry content cannot be blank")

    sanitized = format_previous_entries(previous_entries) or []
    context = [NudgeContextEntry.model_validate(entry) for entry in sanitized]
    prompt = (
        load_prompt(NUDGE_RUNTIME_PROMPT)
        .render(
            entry_content=content,
            entry_date=entry_date,
            previous_entries=[entry.model_dump() for entry in context],
            min_words=NUDGE_MIN_WORDS,
            max_words=NUDGE_MAX_WORDS,
        )
        .strip()
        + "\n"
    )
    return NudgeRuntimeRequest(
        entry_date=entry_date,
        entry_content=content,
        previous_entries=context,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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


def _receipt(
    request: NudgeRuntimeRequest,
    *,
    status: NudgeRuntimeStatus,
    latency_seconds: float,
    response: NudgeDecisionAndGenerationResponse | None = None,
    resolved_model: str | None = None,
    response_id: str | None = None,
    usage: dict[str, int] | None = None,
    raw_response: str | None = None,
    refusal: str | None = None,
    validation_error: str | None = None,
    error: Exception | None = None,
) -> NudgeRuntimeReceipt:
    metadata = get_prompt_metadata(NUDGE_RUNTIME_PROMPT)
    return NudgeRuntimeReceipt(
        created_at=datetime.now(UTC).isoformat(),
        prompt_name=NUDGE_RUNTIME_PROMPT,
        prompt_version=str(metadata["version"]),
        prompt_sha256=request.prompt_sha256,
        requested_model=NUDGE_RUNTIME_MODEL,
        reasoning_effort=NUDGE_RUNTIME_REASONING_EFFORT,
        status=status,
        latency_seconds=latency_seconds,
        resolved_model=resolved_model,
        response_id=response_id,
        usage=usage or {},
        raw_response=raw_response,
        refusal=refusal,
        validation_error=validation_error,
        error_type=type(error).__name__ if error else None,
        error=str(error) if error else None,
        decision=response.decision if response else None,
        reason=response.reason if response else None,
        nudge_text=response.nudge_text if response else None,
    )


class OpenAINudgeRuntime:
    """One-attempt GPT-5.6 Luna caller with reasoning disabled."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 300,
        service_tier: str = "default",
    ) -> None:
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._service_tier = service_tier

    async def __call__(self, request: NudgeRuntimeRequest) -> NudgeRuntimeReceipt:
        started = time.monotonic()
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            except Exception as error:  # noqa: BLE001 - provider boundary
                return _receipt(
                    request,
                    status="error",
                    latency_seconds=time.monotonic() - started,
                    error=error,
                )

        try:
            request_kwargs: dict[str, Any] = {
                "model": NUDGE_RUNTIME_MODEL,
                "input": request.prompt,
                "text": {"format": NUDGE_DECISION_AND_GENERATION_RESPONSE_FORMAT},
                "reasoning": {"effort": NUDGE_RUNTIME_REASONING_EFFORT},
                "max_output_tokens": self._max_output_tokens,
                "store": False,
                "service_tier": self._service_tier,
                "timeout": self._timeout_seconds,
            }
            api_response = await self._client.responses.create(**request_kwargs)
        except Exception as error:  # noqa: BLE001 - provider boundary
            return _receipt(
                request,
                status="error",
                latency_seconds=time.monotonic() - started,
                error=error,
            )

        raw_response = getattr(api_response, "output_text", None) or None
        resolved_model = getattr(api_response, "model", None)
        response_id = getattr(api_response, "id", None)
        usage = _usage_tokens(api_response)
        latency_seconds = time.monotonic() - started
        if raw_response is None:
            return _receipt(
                request,
                status="refusal",
                latency_seconds=latency_seconds,
                resolved_model=resolved_model,
                response_id=response_id,
                usage=usage,
                raw_response=raw_response,
                refusal=_response_refusal(api_response),
            )

        try:
            parsed = NudgeDecisionAndGenerationResponse.model_validate_json(
                raw_response
            )
        except (ValidationError, json.JSONDecodeError) as error:
            return _receipt(
                request,
                status="invalid",
                latency_seconds=latency_seconds,
                resolved_model=resolved_model,
                response_id=response_id,
                usage=usage,
                raw_response=raw_response,
                validation_error=str(error),
            )

        return _receipt(
            request,
            status="ok",
            latency_seconds=latency_seconds,
            response=parsed,
            resolved_model=resolved_model,
            response_id=response_id,
            usage=usage,
            raw_response=raw_response,
        )
