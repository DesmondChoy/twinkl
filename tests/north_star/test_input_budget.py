"""Mocked count-only API checks for the full-history input ceiling."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.north_star.input_budget import (
    INPUT_TOKEN_LIMIT,
    SCHEMA_VERSION,
    InputBudgetError,
    count_payload,
    measure_requests,
    validate_receipt,
)
from src.north_star.provider import LUNA_POLICY_PATH, POLICY_PATH, stable_hash


@pytest.fixture
def policy():
    return json.loads(POLICY_PATH.read_text())


@pytest.fixture
def runtime_request(policy):
    return {
        "system": "Review all eligible Journal Entries for Security.",
        "prompt": "Approved Core Value definition. Entry 1. Entry 2. Entry 30.",
        "schema": {
            "type": "object",
            "properties": {"quote": {"type": "string"}},
            "required": ["quote"],
            "additionalProperties": False,
        },
        "provider": "openai",
        "purpose": "full-history-review",
        "policy_hash": stable_hash(policy),
    }


def mock_counter(monkeypatch, responses):
    count = AsyncMock(side_effect=responses)
    constructors = []

    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            # Deliberately expose no generation endpoint.
            self.responses = SimpleNamespace(input_tokens=SimpleNamespace(count=count))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    return count, constructors


def receipt(runtime_request, policy, count=123):
    return {
        "request_hash": stable_hash(runtime_request),
        "payload_hash": stable_hash(count_payload(runtime_request, policy)),
        "model": policy["runtime"]["model"],
        "input_tokens": count,
    }


def test_count_includes_instructions_all_context_and_schema(runtime_request, policy):
    payload = count_payload(runtime_request, policy)
    assert payload == {
        "model": policy["runtime"]["model"],
        "instructions": runtime_request["system"],
        "input": runtime_request["prompt"],
        "reasoning": {"effort": "none"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "nsm_review",
                "strict": True,
                "schema": runtime_request["schema"],
            }
        },
        "truncation": "disabled",
    }


def test_measure_saves_and_reuses_exact_receipt(
    tmp_path, monkeypatch, runtime_request, policy
):
    count, constructors = mock_counter(
        monkeypatch, [SimpleNamespace(input_tokens=1250, _request_id="req-count")]
    )
    output = tmp_path / "nested" / "counts.json"
    state = asyncio.run(
        measure_requests([runtime_request, runtime_request], policy, output)
    )
    saved = state["counts"][stable_hash(runtime_request)]
    assert validate_receipt(runtime_request, policy, saved) == 1250
    assert saved["provider_request_id"] == "req-count"
    assert saved["counted_at"]
    assert state["schema_version"] == SCHEMA_VERSION
    assert json.loads(output.read_text()) == state
    assert asyncio.run(measure_requests([runtime_request], policy, output)) == state
    count.assert_awaited_once_with(**count_payload(runtime_request, policy))
    assert constructors == [{"max_retries": 0, "timeout": policy["timeout_seconds"]}]


def test_over_budget_history_is_counted_whole_without_truncation(
    tmp_path, monkeypatch, runtime_request, policy
):
    runtime_request["prompt"] = "Journal Entry. " * 20_000
    original = deepcopy(runtime_request)
    count, _ = mock_counter(
        monkeypatch, [SimpleNamespace(input_tokens=INPUT_TOKEN_LIMIT + 1)]
    )
    state = asyncio.run(
        measure_requests([runtime_request], policy, tmp_path / "counts.json")
    )
    assert validate_receipt(
        runtime_request, policy, state["counts"][stable_hash(runtime_request)]
    ) == (INPUT_TOKEN_LIMIT + 1)
    assert runtime_request == original
    count.assert_awaited_once_with(**count_payload(original, policy))


@pytest.mark.parametrize("invalid", [None, -1, True, "1250", 1250.0])
def test_invalid_api_count_is_not_saved(
    tmp_path, monkeypatch, runtime_request, policy, invalid
):
    mock_counter(monkeypatch, [SimpleNamespace(input_tokens=invalid)])
    output = tmp_path / "counts.json"
    with pytest.raises(InputBudgetError, match="nonnegative integer"):
        asyncio.run(measure_requests([runtime_request], policy, output))
    assert not output.exists()


def test_missing_api_count_is_not_saved(tmp_path, monkeypatch, runtime_request, policy):
    mock_counter(monkeypatch, [SimpleNamespace()])
    output = tmp_path / "counts.json"
    with pytest.raises(InputBudgetError, match="nonnegative integer"):
        asyncio.run(measure_requests([runtime_request], policy, output))
    assert not output.exists()


@pytest.mark.parametrize("missing", [None, [], "", {}])
def test_missing_receipt_fails_closed(runtime_request, policy, missing):
    with pytest.raises(InputBudgetError):
        validate_receipt(runtime_request, policy, missing)


@pytest.mark.parametrize("field", ["system", "prompt", "schema", "purpose"])
def test_changed_request_rejects_saved_receipt(runtime_request, policy, field):
    saved = receipt(runtime_request, policy)
    runtime_request[field] = "changed"
    with pytest.raises(InputBudgetError, match="does not match"):
        validate_receipt(runtime_request, policy, saved)


@pytest.mark.parametrize("field", ["request_hash", "payload_hash", "model"])
def test_corrupt_receipt_binding_rejects_saved_count(runtime_request, policy, field):
    saved = receipt(runtime_request, policy)
    saved[field] = "changed"
    with pytest.raises(InputBudgetError, match="does not match"):
        validate_receipt(runtime_request, policy, saved)


def test_changed_runtime_model_rejects_saved_count(runtime_request, policy):
    saved = receipt(runtime_request, policy)
    policy["runtime"]["model"] = "another-model"
    with pytest.raises(InputBudgetError, match="does not match"):
        validate_receipt(runtime_request, policy, saved)


def test_invalid_saved_receipt_is_not_overwritten(
    tmp_path, monkeypatch, runtime_request, policy
):
    output = tmp_path / "counts.json"
    saved = receipt(runtime_request, policy, count=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "counts": {stable_hash(runtime_request): saved},
            }
        )
    )
    original = output.read_bytes()
    count, constructors = mock_counter(monkeypatch, [])
    with pytest.raises(InputBudgetError, match="nonnegative integer"):
        asyncio.run(measure_requests([runtime_request], policy, output))
    assert output.read_bytes() == original
    count.assert_not_awaited()
    assert not constructors


@pytest.mark.parametrize("contents", ["", "{", "null", "[]", '{"counts": {}}'])
def test_corrupt_saved_file_fails_closed(
    tmp_path, monkeypatch, runtime_request, policy, contents
):
    output = tmp_path / "counts.json"
    output.write_text(contents)
    count, constructors = mock_counter(monkeypatch, [])
    with pytest.raises(InputBudgetError, match="Existing input-token counts"):
        asyncio.run(measure_requests([runtime_request], policy, output))
    assert output.read_text() == contents
    count.assert_not_awaited()
    assert not constructors


def test_later_api_failure_preserves_completed_count(
    tmp_path, monkeypatch, runtime_request, policy
):
    other = {**runtime_request, "prompt": "Second history"}
    count, _ = mock_counter(
        monkeypatch, [SimpleNamespace(input_tokens=1250), RuntimeError("disconnected")]
    )
    output = tmp_path / "counts.json"
    with pytest.raises(RuntimeError, match="disconnected"):
        asyncio.run(measure_requests([runtime_request, other], policy, output))
    state = json.loads(output.read_text())
    assert list(state["counts"]) == [stable_hash(runtime_request)]
    assert count.await_count == 2


def test_interrupted_save_preserves_previous_count(
    tmp_path, monkeypatch, runtime_request, policy
):
    mock_counter(monkeypatch, [SimpleNamespace(input_tokens=1250)] * 2)
    output = tmp_path / "counts.json"
    asyncio.run(measure_requests([runtime_request], policy, output))
    original = output.read_bytes()

    def fail_replace(*args):
        raise OSError("Interrupted receipt save")

    monkeypatch.setattr("src.north_star.input_budget.os.replace", fail_replace)
    with pytest.raises(OSError, match="Interrupted receipt save"):
        asyncio.run(
            measure_requests(
                [{**runtime_request, "prompt": "Second history"}], policy, output
            )
        )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".counts.json.*.tmp"))


def test_reference_provider_is_rejected_before_api_connection(
    tmp_path, monkeypatch, runtime_request, policy
):
    runtime_request["provider"] = "gemini"
    count, constructors = mock_counter(monkeypatch, [])
    with pytest.raises(InputBudgetError, match="only OpenAI"):
        asyncio.run(
            measure_requests([runtime_request], policy, tmp_path / "counts.json")
        )
    count.assert_not_awaited()
    assert not constructors


def test_both_openai_roles_count_and_reuse_their_own_receipts(
    tmp_path, monkeypatch, runtime_request
):
    policy = json.loads(LUNA_POLICY_PATH.read_text())
    requests = [
        {**runtime_request, "role": role, "policy_hash": stable_hash(policy)}
        for role in ("runtime", "reference")
    ]
    count, constructors = mock_counter(
        monkeypatch,
        [SimpleNamespace(input_tokens=1250), SimpleNamespace(input_tokens=1260)],
    )
    output = tmp_path / "counts.json"
    state = asyncio.run(measure_requests(requests, policy, output))
    assert len(state["counts"]) == 2
    assert constructors == [
        {"max_retries": 0, "timeout": 180},
        {"max_retries": 0, "timeout": 300},
    ]
    assert [call.kwargs["reasoning"] for call in count.call_args_list] == [
        {"effort": "low"},
        {"effort": "xhigh"},
    ]
    for request, measured in zip(requests, (1250, 1260), strict=True):
        saved = state["counts"][stable_hash(request)]
        assert validate_receipt(request, policy, saved) == measured
    assert asyncio.run(measure_requests(requests, policy, output)) == state
    assert count.await_count == 2
    with pytest.raises(InputBudgetError, match="does not match"):
        validate_receipt(
            requests[1], policy, state["counts"][stable_hash(requests[0])]
        )


def test_changed_reasoning_invalidates_count_receipt(runtime_request, policy):
    saved = receipt(runtime_request, policy)
    policy["runtime"]["reasoning_effort"] = "low"
    with pytest.raises(InputBudgetError, match="does not match"):
        validate_receipt(runtime_request, policy, saved)
