"""Tests for Coach Digest provider usage and latency capture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai.types.responses.response_usage import ResponseUsage

from src.coach.llm_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    build_llm_complete,
    calculate_openai_cost_usd,
    resolve_coach_model,
    summarize_llm_call_metrics,
)
from src.coach.schemas import LLMCallMetrics


def test_calculate_openai_cost_includes_cache_reads_and_writes():
    cost = calculate_openai_cost_usd(
        model="gpt-5.6-luna",
        service_tier="default",
        input_tokens=1_000,
        cached_input_tokens=100,
        cache_write_input_tokens=200,
        output_tokens=200,
    )

    assert cost == pytest.approx(0.000432)


def test_explicit_provider_uses_provider_default_model(monkeypatch):
    monkeypatch.setenv("TWINKL_COACH_PROVIDER", "gemini")
    monkeypatch.setenv("TWINKL_COACH_MODEL", "gemini-custom")

    assert resolve_coach_model() == ("gemini", "gemini-custom")
    assert resolve_coach_model(provider="openai") == (
        "openai",
        DEFAULT_OPENAI_MODEL,
    )
    assert resolve_coach_model(provider="gemini") == (
        "gemini",
        "gemini-custom",
    )
    assert resolve_coach_model(provider="openai", model="openai-custom") == (
        "openai",
        "openai-custom",
    )

    monkeypatch.delenv("TWINKL_COACH_MODEL")
    assert resolve_coach_model(provider="gemini") == (
        "gemini",
        DEFAULT_GEMINI_MODEL,
    )


@pytest.mark.asyncio
async def test_openai_call_records_usage_cost_and_latency(monkeypatch):
    captured: dict[str, object] = {}
    usage = ResponseUsage(
        input_tokens=1_000,
        input_tokens_details={"cached_tokens": 100},
        output_tokens=200,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=1_200,
    )

    class Responses:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_test",
                model="gpt-5.6-luna",
                service_tier="default",
                status="completed",
                usage=usage,
                output_text='{"ok":true}',
            )

    class Client:
        def __init__(self):
            self.responses = Responses()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    call_metrics: list[LLMCallMetrics] = []
    llm_complete = build_llm_complete(
        provider="openai",
        call_metrics=call_metrics,
    )
    assert llm_complete is not None

    raw = await llm_complete("input", None, "instructions")

    assert raw == '{"ok":true}'
    assert captured["service_tier"] == DEFAULT_OPENAI_SERVICE_TIER
    assert captured["reasoning"] == {
        "effort": DEFAULT_OPENAI_REASONING_EFFORT
    }
    assert len(call_metrics) == 1
    metric = call_metrics[0]
    assert metric.input_tokens == 1_000
    assert metric.cached_input_tokens == 100
    assert metric.output_tokens == 200
    assert metric.calculated_cost_usd == pytest.approx(0.000422)
    assert metric.latency_seconds >= 0

    summary = summarize_llm_call_metrics(call_metrics)
    assert summary["n_calls"] == 1
    assert summary["calculated_cost_usd"] == pytest.approx(0.000422)
