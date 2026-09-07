"""Offline development-revision tests with real ledgers and synthetic transport."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from scripts.experiments import north_star_phase0b_v3 as runner
from src.north_star import review, review_v2
from src.north_star.provider import POLICY_PATH, BudgetedProvider, BudgetLedger
from tests.evals.test_north_star_phase0b_v2 import (
    ACTION,
    OTHER_ACTION,
    ScriptedProvider,
    completed,
    forbid_complete,
    response,
    seed_role,
    synthetic_case,
)


@pytest.fixture
def budget(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_bytes(POLICY_PATH.read_bytes())
    return BudgetLedger(tmp_path / "ledger.json", policy)


def revised_response(case, *, decision="supportive", quote=ACTION, reason=None):
    payload = response(case, decision=decision, quote=quote, reason=reason)
    payload["schema_version"] = review_v2.REVIEW_SCHEMA_VERSION
    for result in payload["results"]:
        del result["decision"]
        result.update(
            action_assessment="The writer reports helping neighbors file claims.",
            value_assessment="The reported action protected others' welfare.",
            conflict_assessment="The supplied writing contains no contrary action.",
        )
    return payload


def runtime(provider, case, *, allow_paid=False):
    return asyncio.run(
        runner.review_runtime(
            provider,
            case=case,
            sources=runner.sources_for(case)[:3],
            allow_paid=allow_paid,
        )
    )


def run_case(case, provider, directory, seed, *, unresolved=None, allow_paid=False):
    directory.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        runner.run_case(
            case,
            provider,
            asyncio.Semaphore(1),
            seed=seed,
            unresolved_keys=unresolved or set(),
            directory=directory,
            allow_paid=allow_paid,
        )
    )


@pytest.mark.parametrize("allow_paid", [False, True])
@pytest.mark.parametrize("decision", ["supportive", "not_supportive"])
def test_completed_runtime_replays_without_transport_or_budget_changes(
    budget, monkeypatch, allow_paid, decision
):
    case = synthetic_case()
    provider = ScriptedProvider(
        budget, [completed(revised_response(case, decision=decision))]
    )
    batch, receipts = runtime(provider, case, allow_paid=True)
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    replayed, replay_receipts = runtime(provider, case, allow_paid=allow_paid)

    assert batch is not None
    assert batch.results[0].decision == decision
    assert replayed == batch
    assert replay_receipts == receipts
    assert provider.transport_calls == 1
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_missing_runtime_replays_as_failure_without_transport(budget, monkeypatch):
    provider = ScriptedProvider(budget, [])
    complete = forbid_complete(monkeypatch, provider)
    batch, receipts = runtime(provider, synthetic_case())
    assert batch is None
    assert receipts == []
    assert budget.snapshot()["attempts"] == []
    complete.assert_not_awaited()


@pytest.mark.parametrize("allow_paid", [False, True])
def test_two_invalid_runtime_attempts_exhaust_then_replay_without_transport(
    budget, monkeypatch, allow_paid
):
    case = synthetic_case()
    # Legacy output includes the now-forbidden duplicate decision judgment.
    provider = ScriptedProvider(budget, [completed(response(case))] * 2)
    batch, receipts = runtime(provider, case, allow_paid=True)
    assert batch is None
    assert len(receipts) == provider.transport_calls == 2
    assert [r["attempt"]["status"] for r in receipts] == ["invalid", "invalid"]
    assert [r["attempt"]["attempt_number"] for r in receipts] == [1, 2]
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)

    assert runtime(provider, case, allow_paid=allow_paid) == (None, receipts)
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


@pytest.mark.parametrize("allow_paid", [False, True])
def test_orphan_pending_runtime_keeps_reservation_without_transport(
    budget, monkeypatch, allow_paid
):
    case = synthetic_case()
    request = runner.request_for(
        case, runner.sources_for(case), budget.policy, runtime=True
    )
    pending = budget.reserve(request, retry=False)
    provider = ScriptedProvider(budget, [])
    complete = forbid_complete(monkeypatch, provider)
    saved = budget.snapshot()

    batch, receipts = runtime(provider, case, allow_paid=allow_paid)

    assert batch is None
    assert receipts[0]["attempt"]["status"] == "pending"
    assert pending.reserved_cost_usd > 0
    assert receipts[0]["attempt"]["calculated_cost_usd"] is None
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


def test_persisted_invalid_completion_is_revalidated_before_remaining_retry(
    budget, monkeypatch
):
    case = synthetic_case()
    request = runner.request_for(
        case, runner.sources_for(case), budget.policy, runtime=True
    )
    attempt = budget.reserve(request, retry=False)
    budget.finish(attempt.model_copy(update=completed(response(case))))
    provider = ScriptedProvider(budget, [completed(revised_response(case))])

    with monkeypatch.context() as patch:
        complete = forbid_complete(patch, provider)
        batch, receipts = runtime(provider, case)
        assert batch is None
        assert len(receipts) == 1
        assert receipts[0]["attempt"]["status"] == "invalid"
        assert receipts[0]["attempt"]["calculated_cost_usd"] == 0.001
        complete.assert_not_awaited()

    batch, receipts = runtime(provider, case, allow_paid=True)
    assert batch is not None
    assert [r["attempt"]["status"] for r in receipts] == ["invalid", "completed"]
    assert provider.transport_calls == 1
    assert len(budget.snapshot()["attempts"]) == 2


def test_concurrent_runtime_reviews_share_one_pending_attempt(budget):
    case = synthetic_case()

    async def exercise():
        started, release = asyncio.Event(), asyncio.Event()

        class DelayedProvider(BudgetedProvider):
            transport_calls = 0

            async def _complete(self, request, *, retry):
                attempt = self.ledger.reserve(request, retry=retry)
                self.transport_calls += 1
                started.set()
                await release.wait()
                return self.ledger.finish(
                    attempt.model_copy(update=completed(revised_response(case)))
                )

        provider = DelayedProvider(budget)

        async def one_review():
            return await runner.review_runtime(
                provider, case=case, sources=runner.sources_for(case), allow_paid=True
            )

        first = asyncio.create_task(one_review())
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(one_review())
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert provider.transport_calls == 1
        return results

    first, second = asyncio.run(exercise())
    assert first == second
    assert first[0] is not None
    assert len(first[1]) == len(budget.snapshot()["attempts"]) == 1


@pytest.mark.parametrize("mutation", ["missing", "source", "definition", "unfinished"])
def test_frozen_reference_requires_completed_exact_original_source_request(
    budget, mutation
):
    case = synthetic_case()
    provider = ScriptedProvider(budget, [completed(response(case))])
    seed_role(provider, case, "reference")
    seed = budget.snapshot()
    if mutation == "missing":
        seed["attempts"] = []
    elif mutation == "source":
        case["all_eligible_sources_in_retrieval_order"][0]["journal_entry"] += (
            " Changed."
        )
    elif mutation == "definition":
        case["value"]["definition"] = "A changed approved definition."
    elif mutation == "unfinished":
        seed["attempts"][-1]["status"] = "pending"
    with pytest.raises(ValueError, match="Missing frozen exhaustive reference"):
        runner.frozen_reference(case, seed, budget.policy)


def test_both_paths_grade_their_exact_quote_and_replay_every_receipt(
    tmp_path, budget, monkeypatch
):
    case = synthetic_case()
    full_entry = case["all_eligible_sources_in_retrieval_order"][0]["journal_entry"]
    provider = ScriptedProvider(
        budget,
        [
            completed(response(case, quote=OTHER_ACTION)),
            completed(revised_response(case)),
            completed(response(case)),
            completed(response(case, quote=full_entry)),
        ],
    )
    seed_role(provider, case, "reference")
    seed = budget.snapshot()
    result = run_case(case, provider, tmp_path / "results", seed, allow_paid=True)

    assert result["status"] == "completed"
    assert result["selected"]["evidence_quote"] == ACTION
    assert result["retrieval_only_selected"]["evidence_quote"] == full_entry
    assert result["grade"]["accepted"]
    assert result["retrieval_only_grade"]["accepted"]
    quote_receipts = [
        r
        for r in result["attempts"]
        if r["attempt"]["purpose"] == "development-quote_reference"
    ]
    assert len(quote_receipts) == 2
    assert [json.loads(r["prompt"])["candidate_quote"] for r in quote_receipts] == [
        ACTION,
        full_entry,
    ]
    assert all(
        json.loads(r["attempt"]["raw_text"])["schema_version"]
        == review.REVIEW_SCHEMA_VERSION
        for r in quote_receipts
    )
    assert provider.transport_calls == 4
    saved = budget.snapshot()
    complete = forbid_complete(monkeypatch, provider)
    replayed = run_case(case, provider, tmp_path / "results", seed, allow_paid=True)
    assert replayed == result
    assert budget.snapshot() == saved
    complete.assert_not_awaited()


@pytest.mark.parametrize("primary", ["not_supportive", "abstain", "contradictory"])
def test_primary_rejection_abstention_or_contradiction_cannot_approve_either_path(
    tmp_path, budget, monkeypatch, primary
):
    case = synthetic_case()
    primary_payload = response(
        case,
        decision="supportive" if primary == "contradictory" else primary,
        reason="ambiguous" if primary == "abstain" else None,
    )
    provider = ScriptedProvider(
        budget, [completed(primary_payload), completed(revised_response(case))]
    )
    seed_role(provider, case, "reference")
    seed = budget.snapshot()
    runtime(provider, case, allow_paid=True)
    unresolved = (
        {runner.source_key(case, runner.sources_for(case)[0])}
        if primary == "contradictory"
        else set()
    )
    complete = forbid_complete(monkeypatch, provider)

    result = run_case(case, provider, tmp_path / "results", seed, unresolved=unresolved)

    assert result["status"] == "completed"
    for name in ("grade", "retrieval_only_grade"):
        assert not result[name]["accepted"]
        assert result[name]["status"] == (
            "contradictory_primary_reference"
            if primary == "contradictory"
            else "primary_" + primary
        )
    complete.assert_not_awaited()


@pytest.mark.parametrize("missing", ["runtime", "baseline_quote", "both_quote_checks"])
def test_missing_execution_evidence_remains_failed_in_case_and_summary(
    tmp_path, budget, monkeypatch, missing
):
    case = synthetic_case()
    primary = response(
        case, quote=OTHER_ACTION if missing == "both_quote_checks" else ACTION
    )
    provider = ScriptedProvider(
        budget, [completed(primary), completed(revised_response(case))]
    )
    seed_role(provider, case, "reference")
    seed = budget.snapshot()
    if missing != "runtime":
        runtime(provider, case, allow_paid=True)
    complete = forbid_complete(monkeypatch, provider)

    result = run_case(case, provider, tmp_path / "results", seed)
    summary = runner.summarize([result], budget.snapshot()["attempts"], [])

    assert result["status"] == "failed"
    assert result["retrieval_only_grade"]["status"] == "failed"
    assert summary["failed_cases"] == 1
    assert not summary["gate_passed"]
    complete.assert_not_awaited()


@pytest.mark.parametrize("replacement", [OTHER_ACTION, "I invented a new action."])
def test_candidate_check_cannot_replace_or_invent_the_requested_quote(
    tmp_path, budget, replacement
):
    case = synthetic_case()
    provider = ScriptedProvider(
        budget,
        [
            completed(response(case, quote=OTHER_ACTION)),
            completed(revised_response(case)),
            completed(response(case, quote=replacement)),
            completed(response(case, quote=replacement)),
            completed(response(case, quote=replacement)),
            completed(response(case, quote=replacement)),
        ],
    )
    seed_role(provider, case, "reference")
    result = run_case(
        case, provider, tmp_path / "results", budget.snapshot(), allow_paid=True
    )
    assert result["status"] == "failed"
    assert not result["grade"]["accepted"]
    assert not result["retrieval_only_grade"]["accepted"]
    checks = [
        a
        for a in budget.snapshot()["attempts"]
        if a["purpose"] == "development-quote_reference"
    ]
    assert len(checks) == 4
    assert all(a["status"] == "invalid" for a in checks)


def test_empty_history_has_no_reference_or_runtime_calls(tmp_path, budget, monkeypatch):
    provider = ScriptedProvider(budget, [])
    complete = forbid_complete(monkeypatch, provider)
    result = run_case(
        synthetic_case(empty=True), provider, tmp_path / "results", budget.snapshot()
    )
    assert result["status"] == "no_earlier_writing"
    assert result["selected"] is None
    assert result["retrieval_only_selected"] is None
    assert result["attempts"] == []
    complete.assert_not_awaited()


@pytest.mark.parametrize("difference", [None, "source", "definition"])
def test_contradictions_group_only_identical_source_and_requested_definition(
    budget, difference
):
    first, second = synthetic_case("first"), synthetic_case("second")
    shared = first["all_eligible_sources_in_retrieval_order"][0]
    second["all_eligible_sources_in_retrieval_order"].insert(0, deepcopy(shared))
    if difference == "source":
        second["all_eligible_sources_in_retrieval_order"][0]["journal_entry"] += (
            " More."
        )
    elif difference == "definition":
        second["value"]["definition"] += " Especially the natural environment."
    provider = ScriptedProvider(
        budget,
        [
            completed(response(first)),
            completed(response(second, decision="not_supportive")),
        ],
    )
    seed_role(provider, first, "reference")
    seed_role(provider, second, "reference")

    contradictions = runner.contradictory_references(
        [first, second], budget.snapshot(), budget.policy
    )

    if difference is None:
        assert len(contradictions) == 1
        assert contradictions[0]["entry_id"] == shared["entry_id"]
        assert {d["decision"] for d in contradictions[0]["decisions"]} == {
            "supportive",
            "not_supportive",
        }
    else:
        assert contradictions == []


def metric_case(index, *, abstention=False, all_abstain=False, selected=False):
    return {
        "case_id": f"sample-{index}:universalism:episode_01",
        "eligible_sources": 1,
        "status": "completed",
        "selected": {"entry_id": "example"} if selected else None,
        "retrieval_only_selected": {"entry_id": "example"},
        "grade": {"accepted": False, "status": "primary_abstain"},
        "retrieval_only_grade": {"accepted": False},
        "reference_no_example": True,
        "reference_has_abstention": abstention,
        "reference_all_abstain": all_abstain,
    }


def test_no_accepted_denominator_preserves_all_nine_histories_including_abstentions():
    results = [
        metric_case(i, abstention=i < 3, all_abstain=i == 0, selected=i < 4)
        for i in range(9)
    ]
    summary = runner.summarize(results, [], [])
    assert summary["correct_no_card"] == {
        "numerator": 5,
        "denominator": 9,
        "rate": 5 / 9,
    }
    assert summary["no_example_reference_strata"] == {
        "with_abstentions": 3,
        "all_abstain": 1,
        "resolved_rejections_only": 6,
    }
    assert summary["incorrect_displayed"] == 4
    assert not summary["gate_passed"]


def test_new_failure_rates_count_failed_retries_without_inherited_reference_dilution():
    inherited = [
        {"provider": "gemini", "status": "completed", "calculated_cost_usd": 0.001}
        for _ in range(100)
    ]
    new = [
        {"provider": provider, "status": status, "calculated_cost_usd": 0.001}
        for provider in ("gemini", "openai")
        for status in ("invalid", "completed")
    ]
    summary = runner.summarize([], inherited + new, new)
    assert summary["unexpected_provider_failures"] == {
        provider: {"numerator": 1, "denominator": 2, "rate": 0.5}
        for provider in ("gemini", "openai")
    }
    assert summary["new_actual_attempts"] == 4
    assert summary["evaluated_unique_attempts"] == 104
    assert summary["new_calculated_cost_usd"] == pytest.approx(0.004)
    assert not summary["gate_passed"]


def test_zero_shown_precision_is_undefined_and_cannot_pass_gate():
    result = metric_case(0)
    result["retrieval_only_selected"] = None
    summary = runner.summarize([result], [], [])
    for name in ("precision", "retrieval_only_quotation_precision"):
        assert summary[name] == {"numerator": 0, "denominator": 0, "rate": None}
    assert summary["quotation_precision_difference"] is None
    assert not summary["gate_passed"]


def test_preflight_bounds_long_nudge_candidate_and_retries(budget):
    case = synthetic_case()
    case["all_eligible_sources_in_retrieval_order"][0]["nudge_response"] = (
        "I brought clean water to the travelers. " * 150
    )
    provider = ScriptedProvider(budget, [completed(response(case))])
    seed_role(provider, case, "reference")
    source = runner.sources_for(case)[0]
    nudge_cost = runner.attempt_bound(
        runner.quote_request(
            case,
            source,
            {"evidence_quote": source.nudge_response, "quote_source": "nudge_response"},
            budget.policy,
        ),
        budget.policy,
    )

    preflight = runner.budget_preflight([case], budget.snapshot(), budget.policy)

    bounds = preflight["case_bounds"][0]
    assert bounds["candidate_attempt_usd"] >= nudge_cost
    assert preflight["maximum_new_spent_or_reserved_usd"] == pytest.approx(
        sum(
            bounds[key]
            for key in (
                "runtime_attempt_usd",
                "baseline_attempt_usd",
                "candidate_attempt_usd",
            )
        )
        * budget.policy["max_attempts"]
    )


def test_preflight_keeps_unmetered_reservation_and_stops_exhausted_envelope(budget):
    case = synthetic_case()
    provider = ScriptedProvider(budget, [completed(response(case))])
    seed_role(provider, case, "reference")
    seed = budget.snapshot()
    seed["attempts"][0]["calculated_cost_usd"] = None
    preflight = runner.budget_preflight([case], seed, budget.policy)
    assert (
        preflight["inherited_spent_or_reserved_usd"]
        == (seed["attempts"][0]["reserved_cost_usd"])
    )
    seed["attempts"][0]["calculated_cost_usd"] = budget.policy["budget_usd"]
    with pytest.raises(ValueError, match="exceeds remaining paid envelope"):
        runner.budget_preflight([case], seed, budget.policy)


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    names = (
        "src/north_star/review_v2.py",
        "docs/north_star/protocol.md",
        "data/development_sources.json",
        "config/policy.json",
        "retrieval.json",
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Original frozen source\n")
    policy_path = root / names[3]
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    seed_ledger = BudgetLedger(root / "seed.json", policy_path)
    seed_ledger.snapshot()
    monkeypatch.setattr(runner, "EXECUTION_SOURCES", names[:2])

    def synthetic_manifest(**kwargs):
        return {
            "schema_version": "north-star-development-v3",
            "cases": [],
            "source_hashes": {
                name: runner.recovery.file_hash(root / name) for name in names[2:]
            },
            "budget_preflight": {},
            "unresolved_reference_sources": [],
        }

    monkeypatch.setattr(runner, "revised_manifest", synthetic_manifest)
    directory = tmp_path / "run"
    runner.prepare(
        directory=directory,
        prior_ledger=seed_ledger.path,
        root=root,
        retrieval_path=root / names[4],
        policy_path=policy_path,
    )
    return root, directory, names


def test_prepare_freezes_inputs_and_offline_empty_run_never_calls_provider(
    frozen_run, monkeypatch
):
    root, directory, _ = frozen_run
    complete = AsyncMock(side_effect=AssertionError("Unexpected provider invocation"))
    monkeypatch.setattr(BudgetedProvider, "complete", complete)
    manifest, policy_path = runner.verify_run(directory, root=root)
    assert manifest["cases"] == []
    assert policy_path == root / "config/policy.json"
    assert not asyncio.run(runner.run(directory=directory, root=root))
    assert (directory / "report.json").is_file()
    complete.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation", ["prompt", "protocol", "source", "seed", "manifest"]
)
def test_freeze_mutation_fails_before_any_provider_call(
    frozen_run, monkeypatch, mutation
):
    root, directory, names = frozen_run
    target = {
        "prompt": root / names[0],
        "protocol": root / names[1],
        "source": root / names[2],
        "seed": directory / "budget_seed.json",
        "manifest": directory / "manifest.json",
    }[mutation]
    target.write_text(target.read_text() + "\nChanged after freeze\n")
    complete = AsyncMock(side_effect=AssertionError("Unexpected provider invocation"))
    monkeypatch.setattr(BudgetedProvider, "complete", complete)
    with pytest.raises(ValueError, match="freeze mismatch|source changed"):
        asyncio.run(runner.run(directory=directory, root=root, allow_paid=True))
    complete.assert_not_awaited()


def test_prepare_cannot_overwrite_existing_frozen_run(frozen_run):
    root, directory, names = frozen_run
    original = (directory / "execution_freeze.json").read_bytes()
    with pytest.raises(ValueError, match="new empty run directory"):
        runner.prepare(
            directory=directory,
            prior_ledger=root / "seed.json",
            root=root,
            retrieval_path=root / names[4],
            policy_path=root / names[3],
        )
    assert (directory / "execution_freeze.json").read_bytes() == original


@pytest.fixture
def historical_inputs(tmp_path, monkeypatch):
    """Synthetic historical artifacts exercise the real v3 provenance checks."""
    root = tmp_path / "historical_repository"
    historical = root / runner.HISTORICAL_DIRECTORY.relative_to(runner.ROOT)
    historical.mkdir(parents=True)
    policy_path = root / "config/policy.json"
    policy_path.parent.mkdir()
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    data_path, code_path = root / "development.json", root / "legacy.py"
    data_path.write_text("Original synthetic development writing\n")
    code_path.write_text("Original synthetic execution source\n")
    case = synthetic_case()
    manifest = {
        "cases": [case],
        "source_hashes": {"development.json": runner.recovery.file_hash(data_path)},
    }
    (historical / "manifest.json").write_text(json.dumps(manifest))
    (historical / "execution_freeze.json").write_text(
        json.dumps(
            {"source_hashes": {"legacy.py": runner.recovery.file_hash(code_path)}}
        )
    )
    report_path = historical / "report.json"
    report_path.write_text('{"gate_passed": false}\n')
    (historical / "validation.json").write_text(
        json.dumps(
            {
                "hashes": {
                    str(report_path.relative_to(root)): runner.recovery.file_hash(
                        report_path
                    )
                }
            }
        )
    )
    ledger = BudgetLedger(historical / "budget.json", policy_path)
    provider = ScriptedProvider(ledger, [completed(response(case))])
    seed_role(provider, case, "reference")
    hardening_path = root / runner.HARDENING_VALIDATION
    hardening_path.parent.mkdir(parents=True)
    hardening_path.write_text(
        json.dumps(
            {
                "historical_replay": {
                    "input_ledger_sha256": runner.recovery.file_hash(ledger.path)
                }
            }
        )
    )
    monkeypatch.setattr(runner, "build_manifest", lambda **kwargs: deepcopy(manifest))
    monkeypatch.setattr(runner, "EXECUTION_SOURCES", ("legacy.py",))
    return {
        "root": root,
        "historical": historical,
        "directory": tmp_path / "new_freeze",
        "policy_path": policy_path,
        "prior_ledger": ledger.path,
        "retrieval_path": root / "retrieval.json",
    }


def test_preparation_binds_verified_original_report_receipts_and_sources(
    historical_inputs,
):
    args = {k: v for k, v in historical_inputs.items() if k != "historical"}
    manifest = runner.prepare(**args)
    historical = historical_inputs["historical"]
    root = historical_inputs["root"]
    for name in ("manifest.json", "report.json", "budget.json", "validation.json"):
        path = historical / name
        assert manifest["source_hashes"][str(path.relative_to(root))] == (
            runner.recovery.file_hash(path)
        )
    assert str(runner.HARDENING_VALIDATION) in manifest["source_hashes"]
    assert (historical_inputs["directory"] / "execution_freeze.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("report", "Historical report differs"),
        ("budget", "Historical budget differs"),
        ("source", "Historical execution source changed"),
        ("execution", "Historical execution source changed"),
    ],
)
def test_changed_historical_evidence_is_rejected_before_new_freeze(
    historical_inputs, monkeypatch, mutation, error
):
    historical = historical_inputs["historical"]
    root = historical_inputs["root"]
    target = {
        "report": historical / "report.json",
        "budget": historical / "budget.json",
        "source": root / "development.json",
        "execution": root / "legacy.py",
    }[mutation]
    target.write_text(target.read_text() + "\n")
    complete = AsyncMock(side_effect=AssertionError("Unexpected provider invocation"))
    monkeypatch.setattr(BudgetedProvider, "complete", complete)
    args = {k: v for k, v in historical_inputs.items() if k != "historical"}
    with pytest.raises(ValueError, match=error):
        runner.prepare(**args)
    assert not historical_inputs["directory"].exists()
    complete.assert_not_awaited()
