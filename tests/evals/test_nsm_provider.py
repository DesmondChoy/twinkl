"""Exercise the real OpenAI SDK against local fake HTTP responses only."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from openai import AsyncOpenAI

from scripts.experiments import nsm_provider
from scripts.experiments.nsm_provider import DEFAULT_POLICY, ExperimentProvider
from src.north_star.input_budget import validate_receipt
from src.north_star.provider import openai_input_payload, stable_hash


def request(purpose="toy-runtime", role="runtime"):
    return {
        "provider": "openai",
        "role": role,
        "purpose": purpose,
        "system": "Return the requested JSON object.",
        "prompt": "Synthetic provider fixture; no actual journal content.",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }


def validate(parsed, frozen):
    assert frozen["provider"] == "openai"
    if parsed != {"answer": "accepted"}:
        raise ValueError("invalid_toy_contract")
    return {"normalized": parsed["answer"]}


def counted(tokens=125):
    return {"object": "response.input_tokens", "input_tokens": tokens}


def generated(raw='{"answer":"accepted"}', status="completed", refusal=False):
    return {
        "id": "resp_toy",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "gpt-5.6-luna",
        "output": [
            {
                "type": "message",
                "id": "msg_toy",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "refusal", "refusal": "Toy refusal"}
                    if refusal
                    else {"type": "output_text", "text": raw, "annotations": []}
                ],
            }
        ],
        "usage": {
            "input_tokens": 125,
            "output_tokens": 32,
            "total_tokens": 157,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    }


class Harness:
    def __init__(self, replies):
        self.replies = list(replies)
        self.record = {}
        self.snapshots = []
        self.calls = []
        self.client_options = []

    def persist(self):
        self.snapshots.append(deepcopy(self.record))

    async def handle(self, http_request):
        kind = (
            "count_attempts"
            if http_request.url.path.endswith("/input_tokens")
            else "attempts"
        )
        assert any(
            row[kind] and row[kind][-1]["status"] == "pending"
            for row in self.snapshots[-1]["execution"]["requests"].values()
        ), "The attempt must be durable before network execution"
        self.calls.append(
            (str(http_request.url.path), json.loads(http_request.content))
        )
        await asyncio.sleep(0)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        return httpx.Response(status, json=body, headers={"x-request-id": "req_toy"})

    def client(self, **kwargs):
        self.client_options.append(kwargs)
        return AsyncOpenAI(
            api_key="local-fake-key",
            base_url="https://provider.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
            **kwargs,
        )

    def provider(self):
        return ExperimentProvider(
            self.record, self.persist, validator=validate, client_factory=self.client
        )


@pytest.mark.asyncio
async def test_exact_input_payload_counted_and_all_attempts_persisted():
    harness = Harness([counted(), generated()])
    frozen = request()
    provider = harness.provider()
    receipt = await provider.assess(frozen)
    expected = openai_input_payload(frozen, DEFAULT_POLICY)
    assert harness.calls[0] == ("/v1/responses/input_tokens", expected)
    assert harness.calls[1] == (
        "/v1/responses",
        {
            **expected,
            "max_output_tokens": 32768,
            "service_tier": "default",
            "store": False,
        },
    )
    assert validate_receipt(frozen, DEFAULT_POLICY, receipt["count_receipt"]) == 125
    assert receipt["request_hash"] == stable_hash(frozen)
    assert receipt["payload_hash"] == stable_hash(expected)
    assert receipt["status"] == "completed"
    assert receipt["result"] == {"normalized": "accepted"}
    attempt = receipt["attempts"][0]
    assert attempt["parsed_output"] == {"answer": "accepted"}
    assert json.loads(attempt["raw_text"]) == {"answer": "accepted"}
    assert attempt["calculated_cost_usd"] == pytest.approx(0.0000598)
    assert attempt["usage"]["output_tokens_details"]["reasoning_tokens"] == 12
    assert attempt["latency_seconds"] >= 0
    assert attempt["provider_request_id"] == "req_toy"
    assert harness.snapshots[-1] == harness.record
    assert harness.client_options == [{"max_retries": 0, "timeout": 180}] * 2


@pytest.mark.asyncio
async def test_count_only_preflight_reuses_count_for_later_generation():
    harness = Harness([counted(), generated()])
    provider = harness.provider()
    provider.validator = None
    receipt = await provider.measure(request())
    assert receipt["status"] == "counted"
    assert len(harness.calls) == 1
    assert not receipt["attempts"]
    resumed = await harness.provider().assess(request())
    assert resumed["status"] == "completed"
    assert len(harness.calls) == 2
    assert len(resumed["count_attempts"]) == 1


@pytest.mark.asyncio
async def test_absolute_attempt_timeout_even_if_transport_does_not_time_out(
    monkeypatch,
):
    harness = Harness([counted()])
    original_timeout = asyncio.timeout
    durations = []

    def short_timeout(seconds):
        durations.append(seconds)
        return original_timeout(0.001)

    async def blocked_handler(http_request):
        await asyncio.sleep(0.1)
        raise AssertionError("The absolute attempt timeout must interrupt this wait")

    monkeypatch.setattr(nsm_provider.asyncio, "timeout", short_timeout)
    monkeypatch.setattr(harness, "handle", blocked_handler)
    receipt = await harness.provider().measure(request(), max_attempts=1)
    assert receipt["status"] == "count_failed"
    assert receipt["count_attempts"][0]["error_type"] == "TimeoutError"
    assert receipt["count_attempts"][0]["retryable"]
    assert durations == [180]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    ["not JSON", '{"answer":"wrong"}', '{"answer":"accepted","answer":"accepted"}'],
)
async def test_invalid_json_contract_and_duplicate_keys_retry_only_twice(raw):
    harness = Harness([counted(), generated(raw), generated(raw)])
    provider = harness.provider()
    receipt = await provider.assess(request())
    assert receipt["status"] == "invalid"
    assert len(receipt["attempts"]) == 2
    assert all(row["retryable"] for row in receipt["attempts"])
    assert receipt["result"] is None
    await provider.assess(request())
    assert len(harness.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first",
    [
        generated(status="incomplete"),
        generated("bad JSON"),
        (429, {"error": {"message": "Busy", "type": "rate_limit_error"}}),
        httpx.ReadTimeout("Toy timeout"),
    ],
)
async def test_only_explicit_provider_attempt_retries_and_keeps_failed_usage(first):
    harness = Harness([counted(), first, generated()])
    receipt = await harness.provider().assess(request())
    assert receipt["status"] == "completed"
    assert len(receipt["attempts"]) == 2
    assert len(harness.calls) == 3
    assert receipt["attempts"][0]["retryable"]
    assert receipt["attempts"][1]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,status",
    [
        (generated(refusal=True), "refused"),
        (generated(status="failed"), "failed"),
        ((401, {"error": {"message": "Unauthorized"}}), "failed"),
    ],
)
async def test_refusal_and_permanent_provider_errors_are_terminal(reply, status):
    harness = Harness([counted(), reply])
    provider = harness.provider()
    receipt = await provider.assess(request())
    assert receipt["status"] == status
    assert not receipt["attempts"][0]["retryable"]
    await provider.assess(request())
    assert len(harness.calls) == 2


@pytest.mark.asyncio
async def test_transient_count_failure_retries_and_preserves_exact_receipt():
    harness = Harness([(503, {"error": {"message": "Busy"}}), counted(), generated()])
    receipt = await harness.provider().assess(request())
    assert receipt["status"] == "completed"
    assert len(receipt["count_attempts"]) == 2
    assert receipt["count_attempts"][0]["failure_class"] == "transient_provider_failure"
    assert len(receipt["attempts"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replies",
    [
        [counted(-1)],
        [counted(True)],
        [(401, {"error": {"message": "Unauthorized"}})],
        [httpx.ReadTimeout("Timeout"), httpx.ReadTimeout("Timeout")],
    ],
)
async def test_count_failure_never_generates_or_retries_after_exhaustion(replies):
    harness = Harness(replies)
    provider = harness.provider()
    receipt = await provider.assess(request())
    assert receipt["status"] == "count_failed"
    assert not receipt["attempts"]
    assert len(harness.calls) == len(replies)
    await provider.assess(request())
    assert len(harness.calls) == len(replies)


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [16000, 16001])
async def test_measured_input_boundary_and_no_legacy_utf8_or_monetary_envelope(tokens):
    harness = Harness([counted(tokens)] + ([generated()] if tokens == 16000 else []))
    frozen = request()
    frozen["prompt"] = "Synthetic fixture. " * 5000
    assert len(json.dumps(frozen).encode()) > 64000
    receipt = await harness.provider().assess(frozen)
    assert receipt["status"] == (
        "completed" if tokens == 16000 else "input_limit_rejected"
    )
    assert harness.record["execution"]["provider_policy"]["budget_usd"] is None
    assert harness.record["execution"]["provider_policy"]["per_attempt_usd"] is None


@pytest.mark.asyncio
async def test_recheck_gets_one_generation_attempt_and_reference_timeout():
    harness = Harness([counted(), generated(status="incomplete")])
    frozen = request("fresh-recheck", role="reference")
    provider = harness.provider()
    receipt = await provider.assess(frozen, max_attempts=1)
    assert receipt["status"] == "incomplete"
    assert len(receipt["attempts"]) == 1
    assert harness.client_options == [{"max_retries": 0, "timeout": 300}] * 2
    with pytest.raises(ValueError, match="differs from frozen request"):
        await provider.assess(frozen, max_attempts=2)


@pytest.mark.asyncio
async def test_resume_and_concurrent_duplicates_reuse_completed_requests_only():
    harness = Harness([counted(), generated(), counted(), generated()])
    provider = harness.provider()
    first, second = await asyncio.gather(
        provider.assess(request()), provider.assess(request())
    )
    assert first == second
    assert len(harness.calls) == 2
    await harness.provider().assess(request())
    assert len(harness.calls) == 2
    repeat = await provider.assess(request("repeat-2"))
    assert first["request_hash"] != repeat["request_hash"]
    assert len(harness.calls) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_key", ["count_attempts", "attempts"])
async def test_saved_pending_attempts_are_never_replayed(pending_key):
    harness = Harness([counted(), generated()])
    receipt = await harness.provider().assess(request())
    receipt[pending_key][-1]["status"] = "pending"
    resumed = await harness.provider().assess(request())
    assert resumed["status"] == "interrupted"
    assert resumed["error_type"] == "unresolved_pending_attempt"
    assert len(harness.calls) == 2


@pytest.mark.asyncio
async def test_resume_between_completed_attempt_and_final_receipt_write_no_duplicate():
    harness = Harness([counted(), generated()])
    provider = harness.provider()
    normal_persist = harness.persist

    def interrupted_persist():
        normal_persist()
        rows = harness.record["execution"]["requests"].values()
        if any(
            row["status"] == "generating"
            and row["attempts"]
            and row["attempts"][-1]["status"] == "completed"
            for row in rows
        ):
            raise RuntimeError("Simulated process interruption after durable response")

    provider.persist = interrupted_persist
    with pytest.raises(RuntimeError, match="Simulated process"):
        await provider.assess(request())
    receipt = await harness.provider().assess(request())
    assert receipt["status"] == "completed"
    assert len(harness.calls) == 2


@pytest.mark.asyncio
async def test_tampered_count_receipts_cannot_authorize_generation_or_reuse():
    harness = Harness([counted(), generated()])
    receipt = await harness.provider().assess(request())
    receipt["count_receipt"]["payload_hash"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        await harness.provider().assess(request())
    assert len(harness.calls) == 2


@pytest.mark.asyncio
async def test_unmetered_usage_is_unknown_and_valid_cache_write_usage_is_priced():
    first, second = generated(), generated()
    first["usage"] = None
    second["usage"]["input_tokens_details"]["cache_write_tokens"] = 10
    harness = Harness([counted(), first, counted(), second])
    provider = harness.provider()
    unknown = await provider.assess(request())
    assert unknown["attempts"][0]["calculated_cost_usd"] is None
    priced = await provider.assess(request("separate"))
    assert priced["attempts"][0]["calculated_cost_usd"] == pytest.approx(0.0000603)
