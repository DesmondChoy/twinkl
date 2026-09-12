"""Exercise SDK retries through a local HTTP transport without model calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from src.coach.llm_client import build_llm_complete
from src.nudge.runtime import OpenAINudgeRuntime, build_nudge_runtime_request
from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    build_weekly_drift_reviewer_request,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["nudge", "weekly", "coach"])
@pytest.mark.parametrize("status", [400, 429])
async def test_http_attempt_count_matches_runtime_retry_policy(
    monkeypatch, role, status
):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "Local test response",
                    "type": "test_error",
                    "code": "test",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        clients = []

        def client_factory(**kwargs):
            assert kwargs["max_retries"] == 0
            client = AsyncOpenAI(
                api_key="offline-test", http_client=http_client, **kwargs
            )
            clients.append(client)
            return client

        monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
        monkeypatch.setattr("openai.AsyncOpenAI", client_factory)
        monkeypatch.setattr("src.weekly_drift_reviewer.asyncio.sleep", AsyncMock())
        if role == "nudge":
            receipt = await OpenAINudgeRuntime()(
                build_nudge_runtime_request(
                    entry_content="I called home.", entry_date="2026-09-12"
                )
            )
            assert receipt.status == "error"
            assert receipt.attempts == 1
        elif role == "weekly":
            receipt = await OpenAIWeeklyDriftReviewer()(
                build_weekly_drift_reviewer_request(
                    persona_id="test",
                    week_start="2026-09-07",
                    week_end="2026-09-13",
                    core_values=["benevolence"],
                    current_t_indices=[0],
                    history=[
                        {"t_index": 0, "date": "2026-09-12", "text": "I called home."}
                    ],
                )
            )
            assert receipt.status == "error"
            assert receipt.attempts == (2 if status == 429 else 1)
            assert all(decision.verdict == "abstain" for decision in receipt.decisions)
        else:
            complete = build_llm_complete(provider="openai")
            assert complete is not None
            assert await complete("test input", None) is None
        assert len(clients) == 1
        assert len(requests) == (2 if role == "weekly" and status == 429 else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,refusal",
    [
        ("incomplete", None),
        ("completed", "Cannot comply"),
        ("incomplete", "Cannot comply"),
        (None, None),
    ],
)
async def test_coach_rejects_text_from_failed_openai_envelope(
    monkeypatch, status, refusal
):
    response = SimpleNamespace(
        status=status,
        output_text='{"ok": true}',
        output=[SimpleNamespace(content=[SimpleNamespace(refusal=refusal)])],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **_kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=AsyncMock(return_value=response))
        ),
    )
    complete = build_llm_complete(provider="openai")
    assert complete is not None
    assert await complete("test input", None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,blocked,accepted",
    [
        ("MAX_TOKENS", None, False),
        ("SAFETY", None, False),
        ("STOP", "SAFETY", False),
        (None, None, False),
        ("STOP", "BLOCKED_REASON_UNSPECIFIED", True),
        ("STOP", None, True),
    ],
)
async def test_coach_rejects_text_from_failed_gemini_envelope(
    monkeypatch, reason, blocked, accepted
):
    from google.genai.types import FinishReason

    response = SimpleNamespace(
        text='{"ok": true}',
        prompt_feedback=SimpleNamespace(block_reason=blocked),
        candidates=[
            SimpleNamespace(finish_reason=FinishReason(reason) if reason else None)
        ],
    )
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(
        "google.genai.Client",
        lambda **_kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **_kwargs: response)
        ),
    )
    complete = build_llm_complete(provider="gemini")
    assert complete is not None
    assert await complete("test input", None) == ('{"ok": true}' if accepted else None)
