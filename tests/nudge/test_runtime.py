"""Tests for the production-representative Experience nudge runtime."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.nudge.runtime import (
    NUDGE_RUNTIME_MODEL,
    OpenAINudgeRuntime,
    build_nudge_runtime_request,
)
from src.nudge.schemas import NudgeDecisionAndGenerationResponse


def _api_response(
    payload: dict | None,
    *,
    refusal: str | None = None,
) -> SimpleNamespace:
    output = (
        [SimpleNamespace(content=[SimpleNamespace(refusal=refusal)])] if refusal else []
    )
    return SimpleNamespace(
        output_text=json.dumps(payload) if payload is not None else None,
        model=NUDGE_RUNTIME_MODEL,
        id="resp_nudge_123",
        usage=SimpleNamespace(input_tokens=120, output_tokens=24),
        output=output,
    )


def test_build_request_uses_approved_prompt_and_strips_metadata() -> None:
    request = build_nudge_runtime_request(
        entry_content="  Feeling fine, I guess.  ",
        entry_date="2026-07-25",
        previous_entries=[
            {
                "date": "2026-07-24",
                "content": "Yesterday was long.",
                "tone": "exhausted",
                "reflection_mode": "unsettled",
                "verbosity": "short",
            }
        ],
    )

    assert request.entry_content == "Feeling fine, I guess."
    assert request.previous_entries[0].model_dump() == {
        "date": "2026-07-24",
        "content": "Yesterday was long.",
    }
    assert "Nothing meaningful left unexplored" in request.prompt
    assert "2-12 words" in request.prompt
    assert "Yesterday was long." in request.prompt
    for forbidden in ("tone", "reflection_mode", "verbosity"):
        assert forbidden not in request.prompt.lower()
    assert len(request.prompt_sha256) == 64


def test_build_request_rejects_only_blank_content() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        build_nudge_runtime_request(entry_content=" \n ", entry_date="2026-07-25")

    short = build_nudge_runtime_request(
        entry_content="Bad day.",
        entry_date="2026-07-25",
    )
    assert "Bad day." in short.prompt


def test_response_requires_null_nudge_for_no_nudge() -> None:
    parsed = NudgeDecisionAndGenerationResponse(
        decision="no_nudge",
        reason="The entry is specific and leaves no meaningful thread unexplored.",
        nudge_text=None,
    )
    assert parsed.category is None

    with pytest.raises(ValidationError, match="requires null"):
        NudgeDecisionAndGenerationResponse(
            decision="no_nudge",
            reason="The entry is complete.",
            nudge_text="What happened next?",
        )


def test_response_requires_valid_question_for_nudge_category() -> None:
    parsed = NudgeDecisionAndGenerationResponse(
        decision="elaboration",
        reason="The desire for connection is present but remains unexplored.",
        nudge_text="What did you want Karen to ask?",
    )
    assert parsed.category == "elaboration"
    assert parsed.nudge_text == "What did you want Karen to ask?"

    with pytest.raises(ValidationError, match="requires nudge_text"):
        NudgeDecisionAndGenerationResponse(
            decision="clarification",
            reason="The entry is too vague to understand.",
            nudge_text=None,
        )

    with pytest.raises(ValidationError, match="2-12 words"):
        NudgeDecisionAndGenerationResponse(
            decision="tension_surfacing",
            reason="The writer dismisses an emotion immediately after naming it.",
            nudge_text="Why?",
        )


@pytest.mark.asyncio
async def test_runtime_uses_fixed_luna_none_contract_once() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=_api_response(
                    {
                        "decision": "elaboration",
                        "reason": (
                            "The writer wants deeper connection but leaves that "
                            "desire unexplored."
                        ),
                        "nudge_text": "What did you want Karen to ask?",
                    }
                )
            )
        )
    )
    request = build_nudge_runtime_request(
        entry_content=(
            "Karen asked if I was okay and I said I was fine. "
            "Part of me wanted her to push it."
        ),
        entry_date="2026-07-25",
    )

    receipt = await OpenAINudgeRuntime(client=client)(request)

    assert receipt.status == "ok"
    assert receipt.attempts == 1
    assert receipt.decision == "elaboration"
    assert receipt.nudge_text == "What did you want Karen to ask?"
    assert receipt.prompt_name == "nudge_decision_and_generation"
    assert receipt.prompt_version == "1.0.0"
    assert receipt.response_id == "resp_nudge_123"
    assert receipt.usage == {
        "input_tokens": 120,
        "output_tokens": 24,
        "total_tokens": 144,
    }
    client.responses.create.assert_awaited_once()
    kwargs = client.responses.create.await_args.kwargs
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["reasoning"] == {"effort": "none"}
    assert kwargs["store"] is False
    assert kwargs["input"] == request.prompt
    assert kwargs["text"]["format"]["name"] == "NudgeDecisionAndGeneration"


@pytest.mark.asyncio
async def test_runtime_preserves_no_nudge_reason() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=_api_response(
                    {
                        "decision": "no_nudge",
                        "reason": (
                            "The entry is specific, emotionally clear, and "
                            "sufficiently explored."
                        ),
                        "nudge_text": None,
                    }
                )
            )
        )
    )
    request = build_nudge_runtime_request(
        entry_content="I called Maya and apologized. We agreed to talk tomorrow.",
        entry_date="2026-07-25",
    )

    receipt = await OpenAINudgeRuntime(client=client)(request)

    assert receipt.status == "ok"
    assert receipt.decision == "no_nudge"
    assert receipt.reason is not None
    assert receipt.nudge_text is None


@pytest.mark.asyncio
async def test_runtime_marks_invalid_output_without_retry() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=_api_response(
                    {
                        "decision": "clarification",
                        "reason": "The entry is vague.",
                        "nudge_text": "Why?",
                    }
                )
            )
        )
    )
    request = build_nudge_runtime_request(
        entry_content="Bad day.",
        entry_date="2026-07-25",
    )

    receipt = await OpenAINudgeRuntime(client=client)(request)

    assert receipt.status == "invalid"
    assert receipt.validation_error is not None
    assert receipt.raw_response is not None
    client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_records_refusal_without_retry() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(return_value=_api_response(None, refusal="Cannot comply"))
        )
    )
    request = build_nudge_runtime_request(
        entry_content="A valid Journal Entry.",
        entry_date="2026-07-25",
    )

    receipt = await OpenAINudgeRuntime(client=client)(request)

    assert receipt.status == "refusal"
    assert receipt.refusal == "Cannot comply"
    client.responses.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_records_provider_error_without_retry() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(side_effect=TimeoutError("provider timed out"))
        )
    )
    request = build_nudge_runtime_request(
        entry_content="A valid Journal Entry.",
        entry_date="2026-07-25",
    )

    receipt = await OpenAINudgeRuntime(client=client)(request)

    assert receipt.status == "error"
    assert receipt.error_type == "TimeoutError"
    assert receipt.error == "provider timed out"
    client.responses.create.assert_awaited_once()
