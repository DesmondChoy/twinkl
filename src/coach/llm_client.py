"""Provider-backed ``llm_complete`` adapters for Coach Digest responses.

The Coach Digest accepts an injected ``LLMCompleteFn`` so it stays testable and
provider-agnostic. This module provides OpenAI and Gemini
implementations for the demo path, selected via ``TWINKL_COACH_PROVIDER``
(defaults to ``openai``).

All adapters degrade gracefully: when the provider's API key is absent the
builder returns ``None`` and callers keep Weekly Drift Detection output without
a Coach Digest response.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from decimal import Decimal
from typing import Any

from src.coach.schemas import LLMCallMetrics
from src.coach.weekly_digest import LLMCompleteFn

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_REASONING_EFFORT = "none"
DEFAULT_OPENAI_SERVICE_TIER = "default"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048
OPENAI_LUNA_PRICING_SOURCE = (
    "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
)
OPENAI_LUNA_INPUT_USD_PER_MILLION = Decimal("0.20")
OPENAI_LUNA_CACHED_INPUT_USD_PER_MILLION = Decimal("0.02")
OPENAI_LUNA_OUTPUT_USD_PER_MILLION = Decimal("1.20")
OPENAI_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER = Decimal("2")
OPENAI_LONG_CONTEXT_OUTPUT_MULTIPLIER = Decimal("1.5")
OPENAI_LONG_CONTEXT_THRESHOLD = 272_000


def calculate_openai_cost_usd(
    *,
    model: str,
    service_tier: str | None,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Calculate published-rate cost for one supported OpenAI response."""
    if model != DEFAULT_OPENAI_MODEL or service_tier != DEFAULT_OPENAI_SERVICE_TIER:
        return None

    input_rate = OPENAI_LUNA_INPUT_USD_PER_MILLION
    cached_rate = OPENAI_LUNA_CACHED_INPUT_USD_PER_MILLION
    output_rate = OPENAI_LUNA_OUTPUT_USD_PER_MILLION
    if input_tokens > OPENAI_LONG_CONTEXT_THRESHOLD:
        input_rate *= OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER
        cached_rate *= OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER
        output_rate *= OPENAI_LONG_CONTEXT_OUTPUT_MULTIPLIER

    uncached_tokens = max(
        input_tokens - cached_input_tokens - cache_write_input_tokens,
        0,
    )
    cache_write_rate = input_rate * OPENAI_CACHE_WRITE_MULTIPLIER
    cost = (
        Decimal(uncached_tokens) * input_rate
        + Decimal(cached_input_tokens) * cached_rate
        + Decimal(cache_write_input_tokens) * cache_write_rate
        + Decimal(output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return float(cost)


def summarize_llm_call_metrics(
    call_metrics: list[LLMCallMetrics],
) -> dict[str, int | float | None]:
    """Aggregate token, cost, and latency metrics."""
    latencies = [metric.latency_seconds for metric in call_metrics]
    costs = [
        metric.calculated_cost_usd
        for metric in call_metrics
        if metric.calculated_cost_usd is not None
    ]
    return {
        "n_calls": len(call_metrics),
        "n_calls_with_usage": sum(
            metric.input_tokens is not None for metric in call_metrics
        ),
        "input_tokens": sum(metric.input_tokens or 0 for metric in call_metrics),
        "cached_input_tokens": sum(
            metric.cached_input_tokens or 0 for metric in call_metrics
        ),
        "cache_write_input_tokens": sum(
            metric.cache_write_input_tokens or 0 for metric in call_metrics
        ),
        "output_tokens": sum(metric.output_tokens or 0 for metric in call_metrics),
        "total_tokens": sum(metric.total_tokens or 0 for metric in call_metrics),
        "calculated_cost_usd": sum(costs) if len(costs) == len(call_metrics) else None,
        "total_latency_seconds": sum(latencies),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "max_latency_seconds": max(latencies) if latencies else None,
    }


def _openai_call_metrics(response: object, latency_seconds: float) -> LLMCallMetrics:
    usage = getattr(response, "usage", None)
    model = str(getattr(response, "model", DEFAULT_OPENAI_MODEL))
    service_tier = getattr(response, "service_tier", None)
    if usage is None:
        return LLMCallMetrics(
            provider="openai",
            model=model,
            reasoning_effort=DEFAULT_OPENAI_REASONING_EFFORT,
            service_tier=service_tier,
            response_id=getattr(response, "id", None),
            status=getattr(response, "status", None),
            latency_seconds=latency_seconds,
        )

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    input_tokens = int(getattr(usage, "input_tokens", 0))
    cached_input_tokens = int(getattr(input_details, "cached_tokens", 0))
    cache_write_input_tokens = int(
        getattr(input_details, "cache_write_tokens", 0)
    )
    output_tokens = int(getattr(usage, "output_tokens", 0))
    total_tokens = int(getattr(usage, "total_tokens", 0))
    cost = calculate_openai_cost_usd(
        model=model,
        service_tier=service_tier,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        output_tokens=output_tokens,
    )
    return LLMCallMetrics(
        provider="openai",
        model=model,
        reasoning_effort=DEFAULT_OPENAI_REASONING_EFFORT,
        service_tier=service_tier,
        response_id=getattr(response, "id", None),
        status=getattr(response, "status", None),
        latency_seconds=latency_seconds,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=int(
            getattr(output_details, "reasoning_tokens", 0)
        ),
        total_tokens=total_tokens,
        calculated_cost_usd=cost,
    )


def _unwrap_json_schema(response_format: dict | None) -> dict | None:
    """Extract the bare JSON Schema from an OpenAI-style response_format wrapper.

    The coach layer supplies ``{"type": "json_schema", "name": ..., "schema": {...}}``.
    Providers that take a raw schema (Gemini) need the inner ``schema`` object.
    """
    if not response_format:
        return None
    schema = response_format.get("schema")
    return schema if isinstance(schema, dict) else None


_DEFAULT_MODELS = {
    "openai": DEFAULT_OPENAI_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
}


def _resolve(provider: str | None, model: str | None) -> tuple[str, str]:
    """Resolve the provider and model ids that a builder would use."""
    resolved_provider = (
        (provider or os.environ.get("TWINKL_COACH_PROVIDER", "openai")).strip().lower()
    )
    default_model = _DEFAULT_MODELS.get(resolved_provider, DEFAULT_OPENAI_MODEL)
    resolved_model = model or os.environ.get("TWINKL_COACH_MODEL", default_model)
    return resolved_provider, resolved_model


def resolve_coach_model(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> str:
    """Return the ``provider:model`` id that ``build_llm_complete`` would use.

    The bare model id is ambiguous across providers, so evaluation reports
    record both parts.
    """
    resolved_provider, resolved_model = _resolve(provider, model)
    return f"{resolved_provider}:{resolved_model}"


def _build_openai_llm_complete(
    *,
    model: str | None,
    timeout: float,
    max_output_tokens: int,
    call_metrics: list[LLMCallMetrics] | None,
) -> LLMCompleteFn | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    _, resolved_model = _resolve("openai", model)

    async def llm_complete(
        prompt: str,
        response_format: dict | None,
        instructions: str | None = None,
    ) -> str | None:
        started = time.perf_counter()
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
                "reasoning": {"effort": DEFAULT_OPENAI_REASONING_EFFORT},
                "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
                "store": False,
                "timeout": timeout,
            }
            if instructions is not None:
                kwargs["instructions"] = instructions
            if response_format is not None:
                kwargs["text"] = {"format": response_format}

            response = await client.responses.create(**kwargs)
            if call_metrics is not None:
                call_metrics.append(
                    _openai_call_metrics(response, time.perf_counter() - started)
                )
            return getattr(response, "output_text", None) or None
        except Exception as exc:
            if call_metrics is not None:
                call_metrics.append(
                    LLMCallMetrics(
                        provider="openai",
                        model=resolved_model,
                        reasoning_effort=DEFAULT_OPENAI_REASONING_EFFORT,
                        service_tier=DEFAULT_OPENAI_SERVICE_TIER,
                        status="error",
                        latency_seconds=time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                )
            logger.warning(
                "Coach Digest OpenAI request failed for model %s; "
                "returning Weekly Drift Detection output without a "
                "Coach Digest response",
                resolved_model,
                exc_info=True,
            )
            return None

    return llm_complete


def _build_gemini_llm_complete(
    *,
    model: str | None,
    timeout: float,
    max_output_tokens: int,
    call_metrics: list[LLMCallMetrics] | None,
) -> LLMCompleteFn | None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    _, resolved_model = _resolve("gemini", model)

    def _generate(
        prompt: str,
        response_format: dict | None,
        instructions: str | None,
    ) -> str | None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            # google-genai expects milliseconds for the request timeout.
            "http_options": types.HttpOptions(timeout=int(timeout * 1000)),
            # Gemini flash models "think" by default, consuming the output budget
            # before emitting JSON and truncating it. Disable for this short task.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if instructions is not None:
            config_kwargs["system_instruction"] = instructions
        schema = _unwrap_json_schema(response_format)
        if schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = schema

        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return getattr(response, "text", None) or None

    async def llm_complete(
        prompt: str,
        response_format: dict | None,
        instructions: str | None = None,
    ) -> str | None:
        started = time.perf_counter()
        try:
            # Use the sync client off-thread: the genai async transport 404s in
            # this environment, and the coach cycle already runs off the UI loop.
            import asyncio

            raw = await asyncio.to_thread(
                _generate,
                prompt,
                response_format,
                instructions,
            )
            if call_metrics is not None:
                call_metrics.append(
                    LLMCallMetrics(
                        provider="gemini",
                        model=resolved_model,
                        reasoning_effort="none",
                        status="completed" if raw else "no_response",
                        latency_seconds=time.perf_counter() - started,
                    )
                )
            return raw
        except Exception as exc:
            if call_metrics is not None:
                call_metrics.append(
                    LLMCallMetrics(
                        provider="gemini",
                        model=resolved_model,
                        reasoning_effort="none",
                        status="error",
                        latency_seconds=time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                )
            logger.warning(
                "Coach Digest Gemini request failed for model %s; "
                "returning Weekly Drift Detection output without a "
                "Coach Digest response",
                resolved_model,
                exc_info=True,
            )
            return None

    return llm_complete


def build_llm_complete(
    *,
    provider: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    max_output_tokens: int | None = None,
    call_metrics: list[LLMCallMetrics] | None = None,
) -> LLMCompleteFn | None:
    """Build an ``llm_complete`` callable for the configured provider.

    Provider is chosen by ``TWINKL_COACH_PROVIDER`` (``openai`` or ``gemini``),
    defaulting to ``openai``. Returns ``None`` when the provider's API key is
    missing or the provider is unrecognised, so the demo stays runnable offline.
    """
    resolved_provider, _ = _resolve(provider, model)
    resolved_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
    resolved_max_tokens = (
        max_output_tokens
        if max_output_tokens is not None
        else DEFAULT_MAX_OUTPUT_TOKENS
    )

    builders = {
        "gemini": _build_gemini_llm_complete,
        "openai": _build_openai_llm_complete,
    }
    builder = builders.get(resolved_provider)
    if builder is None:
        return None

    return builder(
        model=model,
        timeout=resolved_timeout,
        max_output_tokens=resolved_max_tokens,
        call_metrics=call_metrics,
    )
