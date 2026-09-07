"""Independent run directories retain one authoritative development allowance."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock, Mock

import pytest

from scripts.experiments import north_star_phase0b_v2 as runner
from src.north_star.provider import (
    POLICY_PATH,
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    stable_hash,
)
from src.north_star.review import SourceEntry, build_review_prompt, review_json_schema


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    for name in runner.EXECUTION_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic executable identity\n")
    policy_path = root / "policy.json"
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    retrieval_path = root / "retrieval.json"
    retrieval_path.write_text('{"frozen": true}\n')
    source = {
        "entry_id": "aaaa:entry:0",
        "journal_entry": "I helped neighbors file their claims.",
        "nudge_response": None,
    }
    case = {
        "case_id": "aaaa:universalism:episode_01",
        "core_value": "universalism",
        "value": {
            "user_phrase": "Care for people",
            "definition": "Protect the welfare of all people and nature.",
        },
        "all_eligible_sources_in_retrieval_order": [source],
        "runtime_entry_ids": [source["entry_id"]],
    }
    manifest = {
        "schema_version": "north-star-development-v2",
        "frozen_at": "2026-09-05T00:00:00+00:00",
        "cases": [case],
        "case_count": 1,
        "source_hashes": {
            name: runner.file_hash(root / name)
            for name in ("policy.json", "retrieval.json")
        },
    }
    monkeypatch.setattr(
        runner, "build_manifest", lambda **kwargs: copy.deepcopy(manifest)
    )
    prior_path = root / "prior-budget.json"
    prior = BudgetLedger(prior_path, policy_path)
    attempt = prior.reserve(request("inherited"), retry=False)
    attempt.status = "completed"
    attempt.calculated_cost_usd = 0.01
    prior.finish(attempt)
    return root, policy_path, retrieval_path, prior_path, case, manifest


def request(name):
    return {"provider": "openai", "purpose": "development-runtime", "prompt": name}


def prepare(project, name):
    root, policy_path, retrieval_path, prior_path, _, _ = project
    directory = root / name
    runner.prepare(
        directory=directory,
        prior_ledger=prior_path,
        root=root,
        retrieval_path=retrieval_path,
        policy_path=policy_path,
    )
    seed = json.loads((directory / "budget_seed.json").read_text())
    return directory, seed


def test_two_prepared_successors_cannot_fork_remaining_allowance(project):
    root, policy_path, _, prior_path, _, manifest = project
    policy = json.loads(policy_path.read_text())
    policy["budget_usd"] = 0.025
    policy_path.write_text(json.dumps(policy))
    manifest["source_hashes"]["policy.json"] = runner.file_hash(policy_path)
    previous = json.loads(prior_path.read_text())
    previous["policy_hash"] = stable_hash(policy)
    prior_path.write_text(json.dumps(previous))
    original_prior = prior_path.read_bytes()
    first_directory, first_seed = prepare(project, "first")
    second_directory, second_seed = prepare(project, "second")

    first = runner._paid_ledger(root, policy_path, first_seed)
    first_attempt = first.reserve(request("first"), retry=False)
    second = runner._paid_ledger(root, policy_path, second_seed)
    assert first.path == second.path
    with pytest.raises(BudgetError, match="Total authorized budget exhausted"):
        second.reserve(request("second"), retry=False)

    attempts = second.snapshot()["attempts"]
    assert len(attempts) == 2
    assert attempts[-1] == first_attempt.model_dump()
    assert first_attempt.calculated_cost_usd is None
    assert 0.01 + first_attempt.reserved_cost_usd <= policy["budget_usd"]
    assert prior_path.read_bytes() == original_prior
    for directory in (first_directory, second_directory):
        runner.verify_run(directory, root=root)


@pytest.mark.parametrize("final_status", ["completed", "invalid"])
def test_retry_and_receipt_reuse_span_prepared_successors(project, final_status):
    root, policy_path, _, _, _, _ = project
    _, first_seed = prepare(project, "first")
    _, second_seed = prepare(project, "second")
    first = runner._paid_ledger(root, policy_path, first_seed)
    payload = request("shared")
    attempt = first.reserve(payload, retry=False)
    attempt.status = "invalid"
    attempt.retryable = True
    attempt.calculated_cost_usd = 0.001
    first.finish(attempt)

    second = runner._paid_ledger(root, policy_path, second_seed)
    retry = second.reserve(payload, retry=True)
    assert retry.attempt_number == 2
    with pytest.raises(BudgetError, match="already pending"):
        first.reserve(payload, retry=True)
    retry.status = final_status
    retry.retryable = final_status == "invalid"
    retry.calculated_cost_usd = 0.001
    second.finish(retry)

    if final_status == "completed":
        reused = first.reserve(payload, retry=True)
        assert reused.reused
        assert reused.attempt_number == 2
    else:
        with pytest.raises(BudgetError, match="Retry limit reached"):
            first.reserve(payload, retry=True)
    assert len(first.snapshot()["attempts"]) == 3


@pytest.mark.parametrize("change", ["cost", "reservation", "request"])
def test_incompatible_inherited_receipts_cannot_replace_authority(project, change):
    root, policy_path, _, _, _, _ = project
    _, seed = prepare(project, "first")
    ledger = runner._paid_ledger(root, policy_path, seed)
    saved = ledger.snapshot()
    changed = copy.deepcopy(seed)
    attempt = changed["attempts"][0]
    field, value = {
        "cost": ("calculated_cost_usd", 0),
        "reservation": ("reserved_cost_usd", 0),
        "request": ("request_hash", "a-different-request"),
    }[change]
    attempt[field] = value

    with pytest.raises(ValueError, match="Inherited"):
        runner._paid_ledger(root, policy_path, changed)
    assert ledger.snapshot() == saved


def test_deleted_authoritative_ledger_cannot_reinitialize_allowance(project):
    root, policy_path, _, _, _, _ = project
    _, seed = prepare(project, "first")
    ledger = runner._paid_ledger(root, policy_path, seed)
    ledger.reserve(request("new-reservation"), retry=False)
    marker = ledger.path.with_suffix(".seed.json")
    original_marker = marker.read_bytes()
    ledger.path.unlink()

    with pytest.raises(ValueError, match="never reset"):
        runner._paid_ledger(root, policy_path, seed)
    assert not ledger.path.exists()
    assert marker.read_bytes() == original_marker


def test_offline_replay_recovers_authoritative_receipts_before_report_write(
    project, monkeypatch
):
    root, policy_path, _, prior_path, case, _ = project
    directory, seed = prepare(project, "interrupted")
    ledger = runner._paid_ledger(root, policy_path, seed)
    sources = [
        SourceEntry(**row) for row in case["all_eligible_sources_in_retrieval_order"]
    ]
    system, prompt = build_review_prompt(
        core_value=case["core_value"],
        user_phrase=case["value"]["user_phrase"],
        approved_definition=case["value"]["definition"],
        sources=sources,
    )
    response = {
        "schema_version": "north-star-moment-review-v1",
        "core_value": case["core_value"],
        "results": [
            {
                "entry_id": sources[0].entry_id,
                "decision": "supportive",
                "quote_source": "journal_entry",
                "evidence_quote": sources[0].journal_entry,
                "reason_code": "observable_choice",
            }
        ],
    }
    for role, provider in (("runtime", "openai"), ("reference", "gemini")):
        payload = {
            "system": system,
            "prompt": prompt,
            "schema": review_json_schema(),
            "provider": provider,
            "purpose": f"development-{role}",
            "policy_hash": stable_hash(ledger.policy),
        }
        attempt = ledger.reserve(payload, retry=False)
        attempt.status = "completed"
        attempt.raw_text = json.dumps(response)
        attempt.calculated_cost_usd = 0.001
        ledger.finish(attempt)
    assert len(json.loads((directory / "budget.json").read_text())["attempts"]) == 1
    original_authority = ledger.path.read_bytes()
    original_prior = prior_path.read_bytes()
    complete = AsyncMock(side_effect=AssertionError("Replay must not call providers"))
    load_env = Mock(side_effect=AssertionError("Replay must not load provider secrets"))
    monkeypatch.setattr(BudgetedProvider, "complete", complete)
    monkeypatch.setattr(runner, "load_dotenv", load_env)

    assert not asyncio.run(runner.run(directory=directory, root=root))

    report = json.loads((directory / "report.json").read_text())
    assert report["cases"][0]["status"] == "completed"
    assert report["cases"][0]["selected"]["evidence_quote"] == sources[0].journal_entry
    assert report["summary"]["attempts"] == 2
    assert report["budget_accounting"]["total_ledger_attempts"] == 3
    assert json.loads((directory / "budget.json").read_text()) == ledger.snapshot()
    assert ledger.path.read_bytes() == original_authority
    assert prior_path.read_bytes() == original_prior
    complete.assert_not_awaited()
    load_env.assert_not_called()
