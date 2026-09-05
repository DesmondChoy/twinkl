"""Budgeted NSM provider attempts with explicit retries and reusable receipts.

The ledger spans offline preparation and local live QC processes. A reservation
is retained for interrupted or unmetered calls, so a lost response cannot silently
release budget. Application receipts decide whether a completed batch is valid.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "config/evals/north_star_moment_v1.json"
DEFAULT_LEDGER = (
    ROOT / "logs/experiments/reports/north_star_phase0b_20260905/budget.json"
)
T = TypeVar("T")


class BudgetError(RuntimeError):
    """No authorized attempt can be reserved."""


class ProviderAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_hash: str
    attempt_number: int
    purpose: str
    provider: Literal["openai", "gemini"]
    requested_model: str
    actual_model: str | None = None
    reasoning_effort: str
    created_at: str
    status: str
    retryable: bool = False
    raw_text: str | None = None
    error_type: str | None = None
    provider_response_id: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int | None = None
    latency_seconds: float = 0
    reserved_cost_usd: float
    calculated_cost_usd: float | None = None
    reused: bool = False


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _record_openai_usage(attempt: ProviderAttempt, usage: Any, settings: dict) -> None:
    """Only release a reservation when required usage is complete and consistent."""
    if usage is None:
        return
    inputs = _token_count(getattr(usage, "input_tokens", None))
    outputs = _token_count(getattr(usage, "output_tokens", None))
    attempt.input_tokens, attempt.output_tokens = inputs, outputs
    details = getattr(usage, "input_tokens_details", None)
    cached = _token_count(getattr(details, "cached_tokens", None))
    # Cache writes are an optional extended field, absent on standard responses.
    writes = _token_count(getattr(details, "cache_write_tokens", 0))
    if cached is not None:
        attempt.cached_input_tokens = cached
    if writes is not None:
        attempt.cache_write_input_tokens = writes
    if inputs is None or outputs is None or cached is None or writes is None:
        return
    total_raw = getattr(usage, "total_tokens", None)
    if total_raw is not None and _token_count(total_raw) != inputs + outputs:
        return
    if cached + writes > inputs:
        return
    attempt.calculated_cost_usd = (
        (inputs - cached - writes) * settings["input_usd_per_million"]
        + cached * settings["cached_input_usd_per_million"]
        + writes * settings["input_usd_per_million"] * 1.25
        + outputs * settings["output_usd_per_million"]
    ) / 1_000_000


def _record_gemini_usage(attempt: ProviderAttempt, usage: Any, settings: dict) -> None:
    """Count thinking in output; missing usage retains the authorized reservation."""
    if usage is None:
        return
    inputs = _token_count(getattr(usage, "prompt_token_count", None))
    attempt.input_tokens = inputs
    if inputs is None:
        return
    total_raw = getattr(usage, "total_token_count", None)
    candidate_raw = getattr(usage, "candidates_token_count", None)
    thought_raw = getattr(usage, "thoughts_token_count", None)
    candidates = _token_count(candidate_raw)
    thoughts = _token_count(thought_raw)
    if (candidate_raw is not None and candidates is None) or (
        thought_raw is not None and thoughts is None
    ):
        return
    # No tool calls are requested. Unexpected tool usage makes pricing uncertain.
    tool_raw = getattr(usage, "tool_use_prompt_token_count", None)
    if tool_raw is not None and _token_count(tool_raw) != 0:
        return
    if total_raw is not None:
        total = _token_count(total_raw)
        if total is None or total < inputs:
            return
        outputs = total - inputs
        known_output = (candidates or 0) + (thoughts or 0)
        if known_output > outputs or (
            candidates is not None and thoughts is not None and known_output != outputs
        ):
            return
    elif candidates is not None and thoughts is not None:
        outputs = candidates + thoughts
    else:
        return
    attempt.output_tokens = outputs
    cached_raw = getattr(usage, "cached_content_token_count", None)
    # Missing cache details get the full input rate, never an assumed discount.
    cached = 0 if cached_raw is None else _token_count(cached_raw)
    if cached is None or cached > inputs:
        return
    attempt.cached_input_tokens = cached
    attempt.calculated_cost_usd = (
        (inputs - cached) * settings["input_usd_per_million"]
        + cached
        * settings.get(
            "cached_input_usd_per_million", settings["input_usd_per_million"]
        )
        + outputs * settings["output_usd_per_million"]
    ) / 1_000_000


class BudgetLedger:
    def __init__(self, path: Path = DEFAULT_LEDGER, policy_path: Path = POLICY_PATH):
        self.path = path.resolve()
        self.policy = json.loads(policy_path.read_text())
        lock_key = hashlib.sha256(str(self.path).encode()).hexdigest()
        self.lock_path = Path(tempfile.gettempdir()) / f"twinkl-nsm-{lock_key}.lock"

    def transact(self, operation: Callable[[dict], T]) -> T:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lock a stable inode: replacing the data file must not let another
        # process acquire a different lock while this transaction is active.
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                try:
                    state = json.loads(self.path.read_text(encoding="utf-8"))
                    if (
                        not isinstance(state, dict)
                        or state.get("schema_version") != "north-star-budget-v1"
                        or not isinstance(state.get("attempts"), list)
                        or not isinstance(state.get("policy_hash"), str)
                    ):
                        raise ValueError("Malformed ledger")
                    for saved_attempt in state["attempts"]:
                        ProviderAttempt.model_validate(saved_attempt)
                except ValueError as exc:
                    raise BudgetError(
                        "Existing budget ledger is empty or corrupt"
                    ) from exc
            else:
                state = {
                    "schema_version": "north-star-budget-v1",
                    "policy_hash": stable_hash(self.policy),
                    "attempts": [],
                }
            if state["policy_hash"] != stable_hash(self.policy):
                raise BudgetError("The ledger's frozen budget policy has changed")
            result = operation(state)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as file:
                    temporary = Path(file.name)
                    json.dump(state, file, indent=2, sort_keys=True)
                    file.write("\n")
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return result

    def reserve(self, request: dict, *, retry: bool) -> ProviderAttempt:
        key = stable_hash(request)
        provider = request["provider"]
        settings = self.policy["runtime" if provider == "openai" else "reference"]
        # UTF-8 byte count plus protocol/schema margin conservatively bounds input
        # tokens for these text-only requests. Use a cache-write uplift for OpenAI.
        input_bound = len(json.dumps(request, ensure_ascii=False).encode()) + 2048
        if input_bound > 64_000:
            raise BudgetError("Request exceeds the frozen input envelope")
        reserved = (
            input_bound
            * settings["input_usd_per_million"]
            * (1.25 if provider == "openai" else 1)
            + self.policy["max_output_tokens"] * settings["output_usd_per_million"]
        ) / 1_000_000
        if reserved > self.policy["per_attempt_usd"]:
            raise BudgetError("Per-attempt budget exceeded")

        def commit(state: dict) -> ProviderAttempt:
            attempts = [a for a in state["attempts"] if a["request_hash"] == key]
            if attempts and attempts[-1]["status"] == "completed":
                return ProviderAttempt(**{**attempts[-1], "reused": True})
            if attempts:
                if attempts[-1]["status"] == "pending":
                    raise BudgetError(
                        "An identical provider attempt is already pending"
                    )
                if not retry or not attempts[-1]["retryable"]:
                    return ProviderAttempt(**{**attempts[-1], "reused": True})
                if len(attempts) >= self.policy["max_attempts"]:
                    raise BudgetError("Retry limit reached")
            spent = sum(
                a["calculated_cost_usd"]
                if a["calculated_cost_usd"] is not None
                else a["reserved_cost_usd"]
                for a in state["attempts"]
            )
            if spent + reserved > self.policy["budget_usd"]:
                raise BudgetError("Total authorized budget exhausted")
            attempt = ProviderAttempt(
                request_hash=key,
                attempt_number=len(attempts) + 1,
                purpose=request["purpose"],
                provider=provider,
                requested_model=settings["model"],
                reasoning_effort=settings.get(
                    "reasoning_effort", settings.get("thinking_level", "none")
                ),
                created_at=datetime.now(UTC).isoformat(),
                status="pending",
                reserved_cost_usd=reserved,
            )
            state["attempts"].append(attempt.model_dump())
            return attempt

        return self.transact(commit)

    def finish(self, attempt: ProviderAttempt) -> ProviderAttempt:
        def commit(state: dict) -> ProviderAttempt:
            for i, previous in enumerate(state["attempts"]):
                if (previous["request_hash"], previous["attempt_number"]) == (
                    attempt.request_hash,
                    attempt.attempt_number,
                ):
                    state["attempts"][i] = attempt.model_dump()
                    return attempt
            raise BudgetError("Attempt was not reserved")

        return self.transact(commit)

    def snapshot(self) -> dict:
        return self.transact(lambda state: state)


class BudgetedProvider:
    def __init__(self, ledger: BudgetLedger | None = None):
        self.ledger = ledger or BudgetLedger()
        self._inflight: dict[str, asyncio.Task[ProviderAttempt]] = {}

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict,
        provider: Literal["openai", "gemini"],
        purpose: str,
        retry: bool = False,
    ) -> ProviderAttempt:
        policy = self.ledger.policy
        request = {
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "provider": provider,
            "purpose": purpose,
            "policy_hash": stable_hash(policy),
        }
        key = stable_hash(request)
        pending = self._inflight.get(key)
        if pending is not None:
            reused: ProviderAttempt = (await asyncio.shield(pending)).model_copy(
                update={"reused": True}
            )
            return reused

        async def run() -> ProviderAttempt:
            try:
                return await self._complete(request, retry=retry)
            finally:
                self._inflight.pop(key, None)

        task = asyncio.create_task(run())
        self._inflight[key] = task
        return await asyncio.shield(task)

    async def _complete(self, request: dict, *, retry: bool) -> ProviderAttempt:
        policy = self.ledger.policy
        system, prompt, schema = request["system"], request["prompt"], request["schema"]
        provider = request["provider"]
        attempt = self.ledger.reserve(request, retry=retry)
        if attempt.reused:
            return attempt
        start = time.perf_counter()
        try:
            if provider == "openai":
                from openai import AsyncOpenAI

                async with AsyncOpenAI(
                    max_retries=0, timeout=policy["timeout_seconds"]
                ) as client:
                    response = await client.responses.create(
                        model=attempt.requested_model,
                        instructions=system,
                        input=prompt,
                        reasoning={"effort": "none"},
                        max_output_tokens=policy["max_output_tokens"],
                        service_tier="default",
                        store=False,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "nsm_review",
                                "strict": True,
                                "schema": schema,
                            }
                        },
                    )
                attempt.raw_text = response.output_text or None
                attempt.actual_model = response.model
                attempt.provider_response_id = response.id
                refused = any(
                    getattr(content, "type", None) == "refusal"
                    for item in response.output
                    if getattr(item, "type", None) == "message"
                    for content in getattr(item, "content", [])
                )
                if refused:
                    attempt.status = "refused"
                    attempt.error_type = "provider_refusal"
                else:
                    attempt.status = (
                        "completed"
                        if response.status == "completed" and attempt.raw_text
                        else "incomplete"
                    )
                _record_openai_usage(attempt, response.usage, policy["runtime"])
            else:
                result = await asyncio.to_thread(self._gemini, system, prompt, schema)
                attempt.raw_text = result.text or None
                attempt.actual_model = result.model_version
                attempt.provider_response_id = result.response_id
                reasons = [
                    str(candidate.finish_reason).rsplit(".", 1)[-1]
                    for candidate in result.candidates or []
                ]
                block_reason = getattr(
                    getattr(result, "prompt_feedback", None), "block_reason", None
                )
                blocked = (
                    block_reason is not None
                    and str(block_reason).rsplit(".", 1)[-1]
                    != "BLOCKED_REASON_UNSPECIFIED"
                )
                refusal_reasons = {
                    "SAFETY",
                    "RECITATION",
                    "BLOCKLIST",
                    "PROHIBITED_CONTENT",
                    "SPII",
                    "IMAGE_SAFETY",
                    "IMAGE_PROHIBITED_CONTENT",
                }
                if blocked or any(reason in refusal_reasons for reason in reasons):
                    attempt.status = "refused"
                    attempt.error_type = (
                        f"provider_blocked:{block_reason}"
                        if blocked
                        else "provider_refusal:" + ",".join(reasons)
                    )
                else:
                    attempt.status = (
                        "completed"
                        if reasons
                        and all(reason == "STOP" for reason in reasons)
                        and attempt.raw_text
                        else "incomplete"
                    )
                _record_gemini_usage(
                    attempt, result.usage_metadata, policy["reference"]
                )
            attempt.retryable = attempt.status == "incomplete"
        except Exception as exc:
            attempt.status = "failed"
            attempt.error_type = type(exc).__name__
            code = getattr(exc, "status_code", getattr(exc, "code", None))
            attempt.retryable = code in (408, 429, 500, 502, 503, 504) or any(
                word in type(exc).__name__.lower() for word in ("timeout", "connection")
            )
        attempt.latency_seconds = time.perf_counter() - start
        return self.ledger.finish(attempt)

    def _gemini(self, system: str, prompt: str, schema: dict) -> Any:
        from google import genai
        from google.genai import types

        policy = self.ledger.policy
        with genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(
                timeout=policy["timeout_seconds"] * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            return client.models.generate_content(
                model=policy["reference"]["model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    max_output_tokens=policy["max_output_tokens"],
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW, include_thoughts=False
                    ),
                ),
            )

    def invalidate(self, attempt: ProviderAttempt, reason: str) -> ProviderAttempt:
        attempt = attempt.model_copy(
            update={
                "status": "invalid",
                "error_type": reason,
                "retryable": True,
                "reused": False,
            }
        )
        return self.ledger.finish(attempt)
