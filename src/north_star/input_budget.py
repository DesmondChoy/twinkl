"""Count complete NSM OpenAI inputs without generating or truncating responses."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.north_star.provider import (
    BudgetError,
    openai_input_payload,
    provider_settings,
    stable_hash,
)

INPUT_TOKEN_LIMIT = 16_000
SCHEMA_VERSION = "north-star-input-counts-v1"


class InputBudgetError(ValueError):
    """A complete, matching input-token receipt is unavailable."""


def count_payload(request: dict, policy: dict) -> dict:
    """Mirror all input-bearing fields of the frozen OpenAI generation request."""
    if request.get("provider") != "openai":
        raise InputBudgetError("Input counting supports only OpenAI requests")
    try:
        return openai_input_payload(request, policy)
    except BudgetError as exc:
        raise InputBudgetError(str(exc)) from exc


def validate_receipt(request: dict, policy: dict, receipt: Any) -> int:
    """Return the measured count only for this exact request, model, and payload."""
    if not isinstance(receipt, dict):
        raise InputBudgetError("Missing or malformed input-token receipt")
    payload = count_payload(request, policy)
    if (
        receipt.get("request_hash") != stable_hash(request)
        or receipt.get("payload_hash") != stable_hash(payload)
        or receipt.get("model") != payload["model"]
    ):
        raise InputBudgetError(
            "Input-token receipt does not match request payload/model"
        )
    count = receipt.get("input_tokens")
    if type(count) is not int or count < 0:
        raise InputBudgetError("Input-token count must be a nonnegative integer")
    return count


def _read_counts(output: Path) -> dict:
    if not output.exists():
        return {"schema_version": SCHEMA_VERSION, "counts": {}}
    try:
        state = json.loads(output.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise InputBudgetError("Existing input-token counts are corrupt") from exc
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or not isinstance(state.get("counts"), dict)
    ):
        raise InputBudgetError("Existing input-token counts are malformed")
    return state


def _write_counts(output: Path, state: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(state, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def measure_requests(requests: list[dict], policy: dict, output: Path) -> dict:
    """Persist sequential count-only receipts; callers enforce the 16,000-token cap.

    Existing matching receipts avoid another API request. Invalid saved receipts
    fail closed instead of being replaced. A failed call retains earlier counts.
    """
    from openai import AsyncOpenAI

    state = _read_counts(output)
    pending = []
    for request in requests:
        key = stable_hash(request)
        if key in state["counts"]:
            validate_receipt(request, policy, state["counts"][key])
        else:
            # Validate payload construction before opening a provider connection.
            count_payload(request, policy)
            pending.append(request)
    if not pending:
        return state

    async with AsyncExitStack() as stack:
        clients = {}
        for request in pending:
            key = stable_hash(request)
            if key in state["counts"]:
                continue
            payload = count_payload(request, policy)
            settings = provider_settings(request, policy)
            timeout = settings.get("timeout_seconds", policy["timeout_seconds"])
            if timeout not in clients:
                clients[timeout] = await stack.enter_async_context(
                    AsyncOpenAI(max_retries=0, timeout=timeout)
                )
            client = clients[timeout]
            response = await client.responses.input_tokens.count(**payload)
            provider_request_id = getattr(response, "_request_id", None)
            receipt = {
                "request_hash": key,
                "payload_hash": stable_hash(payload),
                "model": payload["model"],
                "input_tokens": getattr(response, "input_tokens", None),
                "counted_at": datetime.now(UTC).isoformat(),
                "provider_request_id": (
                    provider_request_id
                    if isinstance(provider_request_id, str)
                    else None
                ),
            }
            validate_receipt(request, policy, receipt)
            state["counts"][key] = receipt
            _write_counts(output, state)
    return state
