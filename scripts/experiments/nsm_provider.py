"""Experiment-only Luna calls with exact input counts and consolidated receipts.

One coordinator owns the mutable record and an atomic synchronous ``persist``
callback. Live application policies and their monetary guards are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from src.north_star.input_budget import validate_receipt
from src.north_star.provider import openai_input_payload, stable_hash

Validator = Callable[[dict, dict], dict]

DEFAULT_POLICY: dict = {
    "schema_version": "north-star-experiment-provider-v1",
    "budget_usd": None,
    "per_attempt_usd": None,
    "input_token_limit": 16_000,
    "max_output_tokens": 32_768,
    "max_attempts": 2,
    "sdk_retries": 0,
    "truncation": "disabled",
    "pricing_checked_at": "2026-09-06",
    "pricing_sources": ["https://developers.openai.com/api/docs/pricing"],
    "runtime": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "low",
        "timeout_seconds": 180,
        "service_tier": "default",
        "input_usd_per_million": 0.2,
        "cached_input_usd_per_million": 0.02,
        "cache_write_input_usd_per_million": 0.25,
        "output_usd_per_million": 1.2,
    },
    "reference": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "timeout_seconds": 300,
        "service_tier": "default",
        "input_usd_per_million": 0.2,
        "cached_input_usd_per_million": 0.02,
        "cache_write_input_usd_per_million": 0.25,
        "output_usd_per_million": 1.2,
    },
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {key: _jsonable(item) for key, item in vars(value).items()}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key:" + key)
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError("nonfinite_json_constant:" + value)


def _parse(raw: str) -> dict:
    parsed = json.loads(
        raw, object_pairs_hook=_strict_object, parse_constant=_invalid_constant
    )
    if not isinstance(parsed, dict):
        raise ValueError("response_must_be_json_object")
    return parsed


def _exception(exc: Exception) -> dict:
    status = getattr(exc, "status_code", None)
    transient = status in (408, 409, 429, 500, 502, 503, 504) or any(
        word in type(exc).__name__.lower() for word in ("timeout", "connection")
    )
    return {
        "status": "failed",
        "error_type": type(exc).__name__,
        "http_status": status,
        "provider_request_id": getattr(exc, "request_id", None),
        "retryable": transient,
        "failure_class": (
            "transient_provider_failure" if transient else "terminal_provider_failure"
        ),
    }


def _usage(attempt: dict, usage: Any, settings: dict) -> None:
    """Retain unknown costs when usage is missing or internally inconsistent."""
    attempt["usage"] = _jsonable(usage)
    if usage is None:
        return
    inputs: Any = getattr(usage, "input_tokens", None)
    outputs: Any = getattr(usage, "output_tokens", None)
    details = getattr(usage, "input_tokens_details", None)
    cached: Any = getattr(details, "cached_tokens", None)
    writes: Any = getattr(details, "cache_write_tokens", 0)
    attempt.update(
        input_tokens=inputs,
        output_tokens=outputs,
        cached_input_tokens=cached,
        cache_write_input_tokens=writes,
    )
    if not all(
        type(value) is int and value >= 0 for value in (inputs, outputs, cached, writes)
    ):
        return
    if cached + writes > inputs or getattr(usage, "total_tokens", inputs + outputs) != (
        inputs + outputs
    ):
        return
    attempt["calculated_cost_usd"] = (
        (inputs - cached - writes) * settings["input_usd_per_million"]
        + cached * settings["cached_input_usd_per_million"]
        + writes * settings["cache_write_input_usd_per_million"]
        + outputs * settings["output_usd_per_million"]
    ) / 1_000_000


class ExperimentProvider:
    """Persist every count and generation attempt; never replay a pending call.

    ``validator(parsed_output, frozen_request)`` must check the complete output
    contract and return a JSON-compatible normalized result, raising ValueError
    for invalid structured output. Original JSON and text remain in the attempt.
    Counts have their own bounded attempts; ``max_attempts=1`` also prevents count
    retries for a contradictory-judgment recheck. No SDK retries occur.
    """

    def __init__(
        self,
        record: dict,
        persist: Callable[[], None],
        *,
        validator: Validator | None = None,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.record = record
        self.persist = persist
        self.validator = validator
        self.client_factory = client_factory
        self._inflight: dict[str, asyncio.Task[dict]] = {}
        execution = record.setdefault("execution", {})
        self.policy = deepcopy(
            execution.setdefault("provider_policy", deepcopy(DEFAULT_POLICY))
        )
        if self.policy != DEFAULT_POLICY:
            raise ValueError("Experiment provider policy differs from frozen policy")
        self.policy_hash = stable_hash(self.policy)
        saved_hash = execution.setdefault("provider_policy_hash", self.policy_hash)
        if saved_hash != self.policy_hash:
            raise ValueError("Experiment provider policy hash changed")
        self.requests: dict = execution.setdefault("requests", {})
        self.persist()

    async def assess(
        self,
        request: dict,
        *,
        max_attempts: int = 2,
        validator: Validator | None = None,
    ) -> dict:
        validate = validator or self.validator
        if validate is None:
            raise ValueError("An exact structured-output validator is required")
        return await self._request(request, max_attempts, validate, count_only=False)

    async def measure(self, request: dict, *, max_attempts: int = 2) -> dict:
        """Count without generation; later assess reuses the frozen receipt."""
        return await self._request(request, max_attempts, None, count_only=True)

    async def _request(
        self,
        request: dict,
        max_attempts: int,
        validate: Validator | None,
        *,
        count_only: bool,
    ) -> dict:
        if type(max_attempts) is not int or max_attempts not in (1, 2):
            raise ValueError("At most two total attempts, or one recheck, are allowed")
        frozen = deepcopy(request)
        if (
            frozen.get("provider") != "openai"
            or frozen.get("role") not in ("runtime", "reference")
            or any(
                not isinstance(frozen.get(field), str) or not frozen[field]
                for field in ("purpose", "system", "prompt")
            )
            or not isinstance(frozen.get("schema"), dict)
        ):
            raise ValueError("Malformed experiment request")
        key = stable_hash(frozen)
        if key in self._inflight:
            receipt = await asyncio.shield(self._inflight[key])
            if receipt["max_attempts"] != max_attempts:
                raise ValueError("Saved request receipt differs from frozen request")
            if not count_only and receipt["status"] == "counted":
                return await self._request(
                    frozen, max_attempts, validate, count_only=False
                )
            return receipt

        async def run() -> dict:
            try:
                return await self._assess(
                    frozen, key, max_attempts, validate, count_only=count_only
                )
            finally:
                self._inflight.pop(key, None)

        task = asyncio.create_task(run())
        self._inflight[key] = task
        return await asyncio.shield(task)

    async def _assess(
        self,
        request: dict,
        key: str,
        max_attempts: int,
        validate: Validator | None,
        *,
        count_only: bool,
    ) -> dict:
        payload = openai_input_payload(request, self.policy)
        identity = {
            "request_hash": key,
            "policy_hash": self.policy_hash,
            "payload_hash": stable_hash(payload),
            "request": request,
            "max_attempts": max_attempts,
        }
        if key not in self.requests:
            self.requests[key] = {
                **identity,
                "created_at": _now(),
                "status": "prepared",
                "count_attempts": [],
                "count_receipt": None,
                "attempts": [],
                "result": None,
            }
            self.persist()
        receipt: dict = self.requests[key]
        if any(receipt.get(field) != value for field, value in identity.items()):
            raise ValueError("Saved request receipt differs from frozen request")
        for name in ("count_attempts", "attempts"):
            attempts = receipt[name]
            if len(attempts) > max_attempts:
                raise ValueError("Saved request exceeds its frozen attempt limit")
            if any(attempt["status"] == "pending" for attempt in attempts):
                receipt.update(
                    status="interrupted",
                    error_type="unresolved_pending_attempt",
                    failure_class="interrupted_attempt_not_replayed",
                )
                self.persist()
                return receipt
        if receipt["count_receipt"] is not None:
            validate_receipt(request, self.policy, receipt["count_receipt"])
        if receipt["attempts"]:
            last = receipt["attempts"][-1]
            if (
                last["status"] == "completed"
                or not last["retryable"]
                or (len(receipt["attempts"]) == max_attempts)
            ):
                receipt["status"] = last["status"]
                for field in ("error_type", "failure_class"):
                    if field in last:
                        receipt[field] = last[field]
                self.persist()
        if receipt["status"] == "completed":
            if count_only:
                return receipt
            assert validate is not None
            normalized = validate(
                deepcopy(receipt["attempts"][-1]["parsed_output"]), request
            )
            if normalized != receipt["result"]:
                raise ValueError(
                    "Saved normalized result differs from current validation"
                )
            return receipt
        if receipt["status"] not in ("prepared", "counting", "counted", "generating"):
            return receipt
        settings = self.policy[request["role"]]
        if receipt["count_receipt"] is None:
            if receipt["count_attempts"] and (
                not receipt["count_attempts"][-1]["retryable"]
                or len(receipt["count_attempts"]) == max_attempts
            ):
                receipt.update(
                    status="count_failed",
                    error_type=receipt["count_attempts"][-1]["error_type"],
                    failure_class=receipt["count_attempts"][-1]["failure_class"],
                )
                self.persist()
                return receipt
            await self._count(receipt, payload, settings)
            if receipt["count_receipt"] is None:
                return receipt
        count = validate_receipt(request, self.policy, receipt["count_receipt"])
        if count > self.policy["input_token_limit"]:
            receipt.update(
                status="input_limit_rejected",
                error_type="input_token_limit_exceeded",
                failure_class="input_limit_rejection",
            )
            self.persist()
            return receipt
        if count_only:
            receipt["status"] = "counted"
            self.persist()
            return receipt
        assert validate is not None
        await self._generate(receipt, payload, settings, validate)
        return receipt

    def _client(self, settings: dict) -> Any:
        factory = self.client_factory
        if factory is None:
            from openai import AsyncOpenAI

            factory = AsyncOpenAI
        return factory(max_retries=0, timeout=settings["timeout_seconds"])

    def _begin(self, receipt: dict, key: str, settings: dict) -> dict:
        attempt = {
            "attempt_number": len(receipt[key]) + 1,
            "created_at": _now(),
            "status": "pending",
            "retryable": False,
            "requested_model": settings["model"],
            "reasoning_effort": settings["reasoning_effort"],
            "timeout_seconds": settings["timeout_seconds"],
            "sdk_retries": 0,
            "calculated_cost_usd": None,
        }
        receipt[key].append(attempt)
        receipt["status"] = "counting" if key == "count_attempts" else "generating"
        self.persist()
        return attempt

    def _finish(self, attempt: dict, start: float) -> None:
        attempt.update(completed_at=_now(), latency_seconds=time.perf_counter() - start)
        self.persist()

    async def _count(self, receipt: dict, payload: dict, settings: dict) -> None:
        while len(receipt["count_attempts"]) < receipt["max_attempts"]:
            attempt = self._begin(receipt, "count_attempts", settings)
            start = time.perf_counter()
            validating = False
            try:
                async with asyncio.timeout(settings["timeout_seconds"]):
                    async with self._client(settings) as client:
                        response = await client.responses.input_tokens.count(**payload)
                counted = {
                    "request_hash": receipt["request_hash"],
                    "payload_hash": receipt["payload_hash"],
                    "model": settings["model"],
                    "input_tokens": getattr(response, "input_tokens", None),
                    "counted_at": _now(),
                    "provider_request_id": getattr(response, "_request_id", None),
                }
                attempt["raw_response"] = _jsonable(response)
                validating = True
                validate_receipt(receipt["request"], self.policy, counted)
                attempt.update(status="completed", **counted)
                receipt["count_receipt"] = counted
            except ValueError as exc:
                attempt.update(
                    {
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "failure_class": "invalid_input_token_receipt",
                    }
                    if validating
                    else _exception(exc)
                )
            except Exception as exc:
                attempt.update(_exception(exc))
            self._finish(attempt, start)
            if attempt["status"] == "completed":
                return
            if not attempt["retryable"]:
                break
        receipt.update(
            status="count_failed",
            error_type=receipt["count_attempts"][-1]["error_type"],
            failure_class=receipt["count_attempts"][-1]["failure_class"],
        )
        self.persist()

    async def _generate(
        self, receipt: dict, payload: dict, settings: dict, validate: Validator
    ) -> None:
        while len(receipt["attempts"]) < receipt["max_attempts"]:
            attempt = self._begin(receipt, "attempts", settings)
            start = time.perf_counter()
            validating = False
            try:
                async with asyncio.timeout(settings["timeout_seconds"]):
                    async with self._client(settings) as client:
                        response = await client.responses.create(
                            **payload,
                            max_output_tokens=self.policy["max_output_tokens"],
                            service_tier=settings["service_tier"],
                            store=False,
                        )
                raw = getattr(response, "output_text", None)
                attempt.update(
                    raw_text=raw,
                    actual_model=getattr(response, "model", None),
                    provider_response_id=getattr(response, "id", None),
                    provider_request_id=getattr(response, "_request_id", None),
                    provider_status=getattr(response, "status", None),
                    incomplete_details=_jsonable(
                        getattr(response, "incomplete_details", None)
                    ),
                    provider_error=_jsonable(getattr(response, "error", None)),
                )
                _usage(attempt, getattr(response, "usage", None), settings)
                refusals = [
                    _jsonable(content)
                    for item in getattr(response, "output", [])
                    if getattr(item, "type", None) == "message"
                    for content in getattr(item, "content", [])
                    if getattr(content, "type", None) == "refusal"
                ]
                if refusals:
                    attempt.update(
                        status="refused",
                        error_type="provider_refusal",
                        failure_class="provider_refusal",
                        refusals=refusals,
                    )
                elif getattr(response, "status", None) in (
                    "incomplete",
                    "completed",
                ) and (getattr(response, "status", None) == "incomplete" or not raw):
                    attempt.update(
                        status="incomplete",
                        error_type="incomplete_response",
                        failure_class="incomplete_response",
                        retryable=True,
                    )
                elif getattr(response, "status", None) == "completed":
                    validating = True
                    if not isinstance(raw, str):
                        raise ValueError("response_text_must_be_string")
                    parsed = _parse(raw)
                    attempt["parsed_output"] = deepcopy(parsed)
                    normalized = validate(parsed, receipt["request"])
                    json.dumps(normalized, allow_nan=False)
                    if not isinstance(normalized, dict):
                        raise TypeError(
                            "Validator must return a normalized JSON object"
                        )
                    attempt.update(status="completed", result=normalized)
                    receipt["result"] = normalized
                else:
                    code = getattr(getattr(response, "error", None), "code", None)
                    transient = code in ("server_error", "rate_limit_exceeded")
                    attempt.update(
                        status="failed",
                        error_type="provider_response_failed",
                        failure_class=(
                            "transient_provider_failure"
                            if transient
                            else "terminal_provider_failure"
                        ),
                        retryable=transient,
                    )
            except ValueError as exc:
                attempt.update(
                    {
                        "status": "invalid",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "failure_class": "invalid_structured_output",
                        "retryable": True,
                    }
                    if validating
                    else _exception(exc)
                )
            except Exception as exc:
                attempt.update(_exception(exc))
            self._finish(attempt, start)
            if attempt["status"] == "completed" or not attempt["retryable"]:
                break
        final = receipt["attempts"][-1]
        receipt["status"] = final["status"]
        for field in ("error_type", "failure_class"):
            if field in final:
                receipt[field] = final[field]
        self.persist()
