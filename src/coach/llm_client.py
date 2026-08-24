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

from src.coach.weekly_digest import LLMCompleteFn

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048


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
) -> LLMCompleteFn | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    _, resolved_model = _resolve("openai", model)

    async def llm_complete(
        prompt: str,
        response_format: dict | None,
        instructions: str | None = None,
    ) -> str | None:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            kwargs: dict[str, object] = {
                "model": resolved_model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
                "store": False,
                "timeout": timeout,
            }
            if instructions is not None:
                kwargs["instructions"] = instructions
            if response_format is not None:
                kwargs["text"] = {"format": response_format}

            response = await client.responses.create(**kwargs)
            return getattr(response, "output_text", None) or None
        except Exception:
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
        config_kwargs: dict[str, object] = {
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
        try:
            # Use the sync client off-thread: the genai async transport 404s in
            # this environment, and the coach cycle already runs off the UI loop.
            import asyncio

            return await asyncio.to_thread(
                _generate,
                prompt,
                response_format,
                instructions,
            )
        except Exception:
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
    )
