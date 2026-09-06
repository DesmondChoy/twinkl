"""No paid calls: budget reservation, interrupted attempts, and explicit retries."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.north_star.provider import (
    LUNA_POLICY_PATH,
    POLICY_PATH,
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    stable_hash,
)


def ledger(tmp_path, **overrides):
    policy = json.loads(POLICY_PATH.read_text())
    policy.update(overrides)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return BudgetLedger(tmp_path / "ledger.json", path)


def request(prompt="Writing", provider="openai"):
    return {"provider": provider, "purpose": "injected-test", "prompt": prompt}


def test_pending_reservation_blocks_duplicate_and_survives_reopen(tmp_path):
    budget = ledger(tmp_path)
    attempt = budget.reserve(request(), retry=False)
    assert attempt.status == "pending"
    reopened = BudgetLedger(budget.path, tmp_path / "policy.json")
    with pytest.raises(BudgetError, match="pending"):
        reopened.reserve(request(), retry=False)
    assert reopened.snapshot()["attempts"][0]["reserved_cost_usd"] > 0


def test_completed_record_is_reused_even_if_retry_requested(tmp_path):
    budget = ledger(tmp_path)
    attempt = budget.reserve(request(), retry=False)
    attempt.status = "completed"
    attempt.raw_text = "saved response"
    attempt.calculated_cost_usd = 0.001
    budget.finish(attempt)
    reused = budget.reserve(request(), retry=True)
    assert reused.reused and reused.raw_text == "saved response"
    assert len(budget.snapshot()["attempts"]) == 1


def test_retry_requires_retryable_failure_and_is_bounded(tmp_path):
    budget = ledger(tmp_path)
    attempt = budget.reserve(request(), retry=False)
    attempt.status = "failed"
    attempt.retryable = True
    budget.finish(attempt)
    assert budget.reserve(request(), retry=False).reused
    retry = budget.reserve(request(), retry=True)
    assert retry.attempt_number == 2
    retry.status = "failed"
    retry.retryable = True
    budget.finish(retry)
    with pytest.raises(BudgetError, match="Retry limit"):
        budget.reserve(request(), retry=True)


def test_nonretryable_failure_does_not_reserve_another_call(tmp_path):
    budget = ledger(tmp_path)
    attempt = budget.reserve(request(), retry=False)
    attempt.status = "failed"
    budget.finish(attempt)
    assert budget.reserve(request(), retry=True).reused


def test_total_ceiling_counts_unmetered_attempts_conservatively(tmp_path):
    budget = ledger(tmp_path, budget_usd=0.02)
    attempt = budget.reserve(request(), retry=False)
    attempt.status = "failed"
    budget.finish(attempt)
    with pytest.raises(BudgetError, match="Total authorized"):
        budget.reserve(request("Another request"), retry=False)


def test_per_attempt_ceiling_and_oversized_input_prevent_calls(tmp_path):
    budget = ledger(tmp_path, per_attempt_usd=0.001)
    with pytest.raises(BudgetError, match="Per-attempt"):
        budget.reserve(request(), retry=False)
    with pytest.raises(BudgetError, match="input envelope"):
        budget.reserve(request("x" * 65_000), retry=False)


def test_concurrent_process_style_reservations_cannot_overspend(tmp_path):
    budget = ledger(tmp_path, budget_usd=0.02)

    def reserve(index):
        try:
            budget.reserve(request(str(index)), retry=False)
            return True
        except BudgetError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reserve, range(4)))
    assert sum(results) == 1
    assert len(budget.snapshot()["attempts"]) == 1


@pytest.mark.parametrize("contents", ["", "{", "null", "[]", '{"attempts": []}'])
def test_existing_empty_or_corrupt_ledger_never_resets_budget(tmp_path, contents):
    budget = ledger(tmp_path)
    budget.path.write_text(contents)
    with pytest.raises(BudgetError, match="empty or corrupt"):
        budget.reserve(request(), retry=False)
    assert budget.path.read_text() == contents


@pytest.mark.parametrize("failure_point", ["serialize", "replace", "fsync"])
def test_interrupted_write_preserves_previous_reservation(
    tmp_path, monkeypatch, failure_point
):
    budget = ledger(tmp_path)
    budget.reserve(request(), retry=False)
    original = budget.path.read_bytes()

    def fail(*args, **kwargs):
        if failure_point == "serialize":
            args[1].write('{"partial":')
        raise OSError("Injected interrupted write")

    targets = {
        "serialize": "src.north_star.provider.json.dump",
        "replace": "src.north_star.provider.os.replace",
        "fsync": "src.north_star.provider.os.fsync",
    }
    with monkeypatch.context() as patch:
        patch.setattr(targets[failure_point], fail)
        with pytest.raises(OSError, match="interrupted write"):
            budget.reserve(request("next"), retry=False)
    assert budget.path.read_bytes() == original
    assert len(budget.snapshot()["attempts"]) == 1
    assert not list(tmp_path.glob(".ledger.json.*.tmp"))


def mock_openai(monkeypatch, response):
    calls = []
    create = AsyncMock(return_value=response)

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.responses = SimpleNamespace(create=create)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    return create, calls


def openai_response(usage=None, refusal=False):
    content = [SimpleNamespace(type="output_text", text="{}")]
    if refusal:
        content.append(SimpleNamespace(type="refusal", refusal="Declined"))
    return SimpleNamespace(
        output_text="{}",
        output=[SimpleNamespace(type="message", content=content)],
        model="gpt-5.6-luna",
        id="injected-response",
        status="completed",
        usage=usage,
    )


def gemini_response(usage=None, reason="STOP", block_reason=None):
    return SimpleNamespace(
        text="{}",
        model_version="gemini-3.5-flash",
        response_id="injected-response",
        candidates=[SimpleNamespace(finish_reason=reason)],
        usage_metadata=usage,
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
    )


async def complete(provider, kind="openai", retry=False, role=None):
    return await provider.complete(
        system="Review instructions",
        prompt="Injected writing",
        schema={"type": "object"},
        provider=kind,
        purpose="injected-test",
        retry=retry,
        role=role,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(prompt_token_count=100),
        SimpleNamespace(prompt_token_count=100, candidates_token_count=20),
        SimpleNamespace(prompt_token_count=100, total_token_count=50),
        SimpleNamespace(prompt_token_count=-1, total_token_count=50),
        SimpleNamespace(prompt_token_count=100, total_token_count="120"),
        SimpleNamespace(
            prompt_token_count=100, total_token_count=120, candidates_token_count=30
        ),
        SimpleNamespace(
            prompt_token_count=100,
            total_token_count=120,
            cached_content_token_count=101,
        ),
    ],
)
async def test_unknown_or_inconsistent_gemini_usage_keeps_reservation(
    tmp_path, monkeypatch, usage
):
    provider = BudgetedProvider(ledger(tmp_path))
    monkeypatch.setattr(provider, "_gemini", lambda *args: gemini_response(usage))
    attempt = await complete(provider, "gemini")
    assert attempt.status == "completed"
    assert attempt.calculated_cost_usd is None
    saved = provider.ledger.snapshot()["attempts"][0]
    assert saved["calculated_cost_usd"] is None
    assert saved["reserved_cost_usd"] > 0


@pytest.mark.asyncio
async def test_gemini_total_usage_includes_thinking_and_cached_input(
    tmp_path, monkeypatch
):
    budget = ledger(tmp_path)
    provider = BudgetedProvider(budget)
    usage = SimpleNamespace(
        prompt_token_count=1000,
        total_token_count=1150,
        candidates_token_count=50,
        thoughts_token_count=100,
        cached_content_token_count=200,
    )
    monkeypatch.setattr(provider, "_gemini", lambda *args: gemini_response(usage))
    attempt = await complete(provider, "gemini")
    assert attempt.output_tokens == 150
    assert attempt.cached_input_tokens == 200
    assert attempt.calculated_cost_usd == pytest.approx(
        (800 * 1.5 + 200 * 0.15 + 150 * 9) / 1_000_000
    )


@pytest.mark.asyncio
async def test_gemini_complete_total_can_price_missing_breakdown(tmp_path, monkeypatch):
    provider = BudgetedProvider(ledger(tmp_path))
    usage = SimpleNamespace(prompt_token_count=1000, total_token_count=1150)
    monkeypatch.setattr(provider, "_gemini", lambda *args: gemini_response(usage))
    attempt = await complete(provider, "gemini")
    assert attempt.output_tokens == 150
    assert attempt.calculated_cost_usd == pytest.approx(
        (1000 * 1.5 + 150 * 9) / 1_000_000
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(input_tokens=100),
        SimpleNamespace(input_tokens=100, output_tokens=20),
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=101),
        ),
        SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            total_tokens=90,
        ),
    ],
)
async def test_unknown_or_inconsistent_openai_usage_keeps_reservation(
    tmp_path, monkeypatch, usage
):
    provider = BudgetedProvider(ledger(tmp_path))
    mock_openai(monkeypatch, openai_response(usage))
    attempt = await complete(provider)
    assert attempt.status == "completed"
    assert attempt.calculated_cost_usd is None


@pytest.mark.asyncio
async def test_openai_cache_writes_have_documented_uplift(tmp_path, monkeypatch):
    provider = BudgetedProvider(ledger(tmp_path))
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=50,
        total_tokens=1050,
        input_tokens_details=SimpleNamespace(cached_tokens=200, cache_write_tokens=300),
    )
    create, calls = mock_openai(monkeypatch, openai_response(usage))
    attempt = await complete(provider)
    assert attempt.cache_write_input_tokens == 300
    assert attempt.calculated_cost_usd == pytest.approx(
        (500 * 0.2 + 200 * 0.02 + 300 * 0.2 * 1.25 + 50 * 1.2) / 1_000_000
    )
    assert calls == [{"max_retries": 0, "timeout": 60}]
    assert create.call_count == 1


@pytest.mark.asyncio
async def test_openai_mixed_refusal_and_json_is_refused_without_retry(
    tmp_path, monkeypatch
):
    provider = BudgetedProvider(ledger(tmp_path))
    create, _ = mock_openai(monkeypatch, openai_response(refusal=True))
    attempt = await complete(provider)
    assert attempt.status == "refused"
    assert attempt.error_type == "provider_refusal"
    assert not attempt.retryable
    reused = await complete(provider, retry=True)
    assert reused.reused and reused.status == "refused"
    assert create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "block_reason"),
    [("SAFETY", None), ("PROHIBITED_CONTENT", None), ("STOP", "SAFETY")],
)
async def test_gemini_refusal_is_nonretryable_even_with_json(
    tmp_path, monkeypatch, reason, block_reason
):
    provider = BudgetedProvider(ledger(tmp_path))
    calls = []

    def fake(*args):
        calls.append(args)
        return gemini_response(reason=reason, block_reason=block_reason)

    monkeypatch.setattr(provider, "_gemini", fake)
    attempt = await complete(provider, "gemini")
    assert attempt.status == "refused" and not attempt.retryable
    reused = await complete(provider, "gemini", retry=True)
    assert reused.reused and reused.status == "refused"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_identical_inflight_calls_coalesce_including_explicit_retry(
    tmp_path, monkeypatch
):
    provider = BudgetedProvider(ledger(tmp_path))
    create, _ = mock_openai(monkeypatch, openai_response())
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(**kwargs):
        started.set()
        await release.wait()
        return openai_response()

    create.side_effect = delayed
    first = asyncio.create_task(complete(provider))
    await started.wait()
    followers = [
        asyncio.create_task(complete(provider, retry=retry)) for retry in (False, True)
    ]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, *followers)
    assert [result.reused for result in results] == [False, True, True]
    assert create.call_count == 1
    assert len(provider.ledger.snapshot()["attempts"]) == 1
    assert not provider._inflight


def luna_ledger(tmp_path, **overrides):
    policy = json.loads(LUNA_POLICY_PATH.read_text())
    policy.update(overrides)
    path = tmp_path / "luna-policy.json"
    path.write_text(json.dumps(policy))
    return BudgetLedger(tmp_path / "luna-ledger.json", path)


@pytest.mark.asyncio
async def test_openai_roles_have_distinct_calls_and_matching_count_payloads(
    tmp_path, monkeypatch
):
    from src.north_star.input_budget import count_payload

    budget = luna_ledger(tmp_path)
    provider = BudgetedProvider(budget)
    create, constructors = mock_openai(monkeypatch, openai_response())
    attempts = await asyncio.gather(
        complete(provider, role="runtime"), complete(provider, role="reference")
    )
    assert create.call_count == 2
    assert {attempt.role for attempt in attempts} == {"runtime", "reference"}
    assert {attempt.reasoning_effort for attempt in attempts} == {"low", "xhigh"}
    assert len({attempt.request_hash for attempt in attempts}) == 2
    assert constructors == [
        {"max_retries": 0, "timeout": 180},
        {"max_retries": 0, "timeout": 300},
    ]
    for attempt, call in zip(attempts, create.call_args_list, strict=True):
        counted_request = {
            "system": "Review instructions",
            "prompt": "Injected writing",
            "schema": {"type": "object"},
            "provider": "openai",
            "purpose": "injected-test",
            "policy_hash": stable_hash(budget.policy),
            "role": attempt.role,
        }
        assert attempt.request_hash == stable_hash(counted_request)
        payload = count_payload(counted_request, budget.policy)
        assert call.kwargs == {
            **payload,
            "max_output_tokens": 32768,
            "service_tier": "default",
            "store": False,
        }
        reused = await complete(provider, role=attempt.role)
        assert reused.reused and reused.request_hash == attempt.request_hash
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_reference_openai_usage_uses_reference_prices(tmp_path, monkeypatch):
    reference = json.loads(LUNA_POLICY_PATH.read_text())["reference"]
    reference.update(
        input_usd_per_million=1.1,
        cached_input_usd_per_million=0.1,
        output_usd_per_million=2.2,
    )
    provider = BudgetedProvider(luna_ledger(tmp_path, reference=reference))
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=50,
        total_tokens=1050,
        input_tokens_details=SimpleNamespace(cached_tokens=200),
    )
    mock_openai(monkeypatch, openai_response(usage))
    attempt = await complete(provider, role="reference")
    assert attempt.calculated_cost_usd == pytest.approx(
        (800 * 1.1 + 200 * 0.1 + 50 * 2.2) / 1_000_000
    )
    assert attempt.reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_omitted_role_preserves_legacy_request_identity(tmp_path, monkeypatch):
    budget = ledger(tmp_path)
    provider = BudgetedProvider(budget)
    create, _ = mock_openai(monkeypatch, openai_response())
    attempt = await complete(provider)
    assert attempt.request_hash == stable_hash(
        {
            "system": "Review instructions",
            "prompt": "Injected writing",
            "schema": {"type": "object"},
            "provider": "openai",
            "purpose": "injected-test",
            "policy_hash": stable_hash(budget.policy),
        }
    )
    assert attempt.role is None
    assert attempt.reasoning_effort == "none"
    assert create.call_args.kwargs["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["reference", "unknown"])
async def test_mismatched_role_fails_before_reservation_or_connection(
    tmp_path, monkeypatch, role
):
    provider = BudgetedProvider(ledger(tmp_path))
    create, constructors = mock_openai(monkeypatch, openai_response())
    with pytest.raises(BudgetError, match="role"):
        await complete(provider, role=role)
    assert not provider.ledger.path.exists()
    assert not constructors
    create.assert_not_awaited()


def test_fresh_ledger_still_counts_spend_from_prior_experiments(tmp_path):
    budget = luna_ledger(tmp_path, budget_usd=0.35)
    assert not budget.path.exists()
    with pytest.raises(BudgetError, match="Total authorized"):
        budget.reserve({**request(), "role": "runtime"}, retry=False)
    assert not budget.path.exists()


def test_prior_spend_combines_with_metered_and_unmetered_new_attempts(tmp_path):
    prior = 0.34732602
    budget = luna_ledger(tmp_path, budget_usd=prior + 0.06)
    attempt = budget.reserve(request(), retry=False)
    attempt.status = "completed"
    attempt.calculated_cost_usd = 0.001
    budget.finish(attempt)
    budget.reserve(request("unmetered"), retry=False)
    with pytest.raises(BudgetError, match="Total authorized"):
        budget.reserve(request("third"), retry=False)
    assert len(budget.snapshot()["attempts"]) == 2


@pytest.mark.parametrize("prior", [-1, float("nan"), float("inf"), True, "0.1"])
def test_invalid_prior_spend_cannot_bypass_budget(tmp_path, prior):
    budget = ledger(tmp_path, prior_spend_usd=prior)
    with pytest.raises(BudgetError, match="Prior spend"):
        budget.reserve(request(), retry=False)


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_paid_work(tmp_path, monkeypatch):
    provider = BudgetedProvider(ledger(tmp_path))
    create, _ = mock_openai(monkeypatch, openai_response())
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed(**kwargs):
        started.set()
        await release.wait()
        return openai_response()

    create.side_effect = delayed
    first = asyncio.create_task(complete(provider))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    follower = asyncio.create_task(complete(provider))
    await asyncio.sleep(0)
    release.set()
    result = await follower
    assert result.status == "completed" and result.reused
    assert create.call_count == 1
    assert provider.ledger.snapshot()["attempts"][0]["status"] == "completed"
    assert not provider._inflight
