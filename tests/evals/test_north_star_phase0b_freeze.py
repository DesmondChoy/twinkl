"""Run preparation and recovery enforce provenance before any transport."""

import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.experiments import north_star_phase0b_v2 as runner
from src.north_star.provider import POLICY_PATH, BudgetLedger


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    for name in runner.EXECUTION_SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen executable\n")
    policy_path = root / "policy.json"
    policy_path.write_bytes(POLICY_PATH.read_bytes())
    retrieval_path = root / "retrieval.json"
    retrieval_path.write_text('{"frozen": true}')
    manifest = {
        "schema_version": "north-star-development-v2",
        "frozen_at": "2026-09-05T00:00:00+00:00",
        "cases": [],
        "case_count": 0,
        "source_hashes": {
            name: runner.file_hash(root / name)
            for name in ("policy.json", "retrieval.json")
        },
    }
    monkeypatch.setattr(
        runner, "build_manifest", lambda **kwargs: copy.deepcopy(manifest)
    )
    seed_path = tmp_path / "previous-budget.json"
    ledger = BudgetLedger(seed_path, policy_path)
    previous = ledger.reserve(
        {"provider": "openai", "purpose": "prior-development", "prompt": "old"},
        retry=False,
    )
    previous.status = "failed"
    previous.retryable = True
    previous.calculated_cost_usd = 0.01
    ledger.finish(previous)
    directory = tmp_path / "future-run"
    original_seed = seed_path.read_bytes()
    runner.prepare(
        directory=directory,
        prior_ledger=seed_path,
        root=root,
        retrieval_path=retrieval_path,
        policy_path=policy_path,
    )
    assert seed_path.read_bytes() == original_seed
    return root, directory, seed_path, manifest


def test_prepared_run_is_replayable_and_carries_prior_spending(prepared):
    root, directory, seed_path, _ = prepared
    manifest, policy_path = runner.verify_run(directory, root=root)
    assert manifest["case_count"] == 0
    assert policy_path == root / "policy.json"
    assert not asyncio.run(runner.run(directory=directory, root=root))
    report = json.loads((directory / "report.json").read_text())
    assert report["budget_accounting"] == {
        "inherited_attempts": 1,
        "total_ledger_attempts": 1,
        "spent_or_reserved_usd": 0.01,
    }
    assert (
        json.loads(seed_path.read_text())["attempts"]
        == json.loads((directory / "budget.json").read_text())["attempts"]
    )
    assert "verification_lift" not in report["summary"]


@pytest.mark.parametrize(
    "change",
    [
        "code",
        "input",
        "manifest",
        "freeze_hash",
        "freeze_coverage",
        "missing_freeze",
        "seed",
        "missing_ledger",
        "lost_spending",
        "paid_amount",
    ],
)
def test_changed_freeze_or_accounting_blocks_before_provider(
    prepared, monkeypatch, change
):
    root, directory, _, _ = prepared
    if change in {"code", "input", "manifest", "seed"}:
        path = {
            "code": root / runner.EXECUTION_SOURCES[0],
            "input": root / "retrieval.json",
            "manifest": directory / "manifest.json",
            "seed": directory / "budget_seed.json",
        }[change]
        path.write_text(path.read_text() + " ")
    elif change.startswith("freeze_"):
        path = directory / "execution_freeze.json"
        freeze = json.loads(path.read_text())
        if change == "freeze_hash":
            freeze["source_hashes"][runner.EXECUTION_SOURCES[0]] = "stale"
        else:
            del freeze["source_hashes"][runner.EXECUTION_SOURCES[0]]
        path.write_text(json.dumps(freeze))
    elif change == "missing_freeze":
        (directory / "execution_freeze.json").unlink()
    elif change == "missing_ledger":
        (directory / "budget.json").unlink()
    else:
        path = directory / "budget.json"
        state = json.loads(path.read_text())
        if change == "lost_spending":
            state["attempts"] = []
        else:
            state["attempts"][0]["calculated_cost_usd"] = 0
        path.write_text(json.dumps(state))
    provider = Mock(side_effect=AssertionError("Provider must not be constructed"))
    load_env = Mock(side_effect=AssertionError("No environment access before checks"))
    monkeypatch.setattr(runner, "BudgetedProvider", provider)
    monkeypatch.setattr(runner, "load_dotenv", load_env)
    with pytest.raises((ValueError, FileNotFoundError)):
        asyncio.run(runner.run(directory=directory, root=root, allow_paid=True))
    provider.assert_not_called()
    load_env.assert_not_called()


def test_rebuilt_source_contract_must_match_manifest(prepared, monkeypatch):
    root, directory, _, manifest = prepared
    changed = copy.deepcopy(manifest)
    changed["case_count"] = 99
    monkeypatch.setattr(runner, "build_manifest", lambda **kwargs: changed)
    with pytest.raises(ValueError, match="verified development inputs"):
        runner.verify_run(directory, root=root)


def test_historical_and_existing_directories_are_never_overwritten(prepared):
    root, directory, seed_path, _ = prepared
    for target in (directory, runner.HISTORICAL_DIRECTORY / "new-child"):
        with pytest.raises(ValueError, match="overwrite|immutable"):
            runner.prepare(
                directory=target,
                prior_ledger=seed_path,
                root=root,
                retrieval_path=root / "retrieval.json",
                policy_path=root / "policy.json",
            )


def test_interrupted_preparation_cannot_run_without_final_freeze(prepared, monkeypatch):
    root, _, seed_path, _ = prepared
    directory = root.parent / "interrupted"
    original_open = Path.open

    def interrupted_open(path, *args, **kwargs):
        if path == directory / "execution_freeze.json":
            raise OSError("Injected interruption before execution freeze")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupted_open)
        with pytest.raises(OSError, match="Injected interruption"):
            runner.prepare(
                directory=directory,
                prior_ledger=seed_path,
                root=root,
                retrieval_path=root / "retrieval.json",
                policy_path=root / "policy.json",
            )
    assert (directory / "budget.json").read_bytes() == (
        directory / "budget_seed.json"
    ).read_bytes()
    with pytest.raises(FileNotFoundError):
        runner.verify_run(directory, root=root)


def test_preparation_cannot_advertise_unsupported_provider_settings(prepared):
    root, _, seed_path, _ = prepared
    policy_path = root / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy["runtime"]["reasoning_effort"] = "low"
    policy_path.write_text(json.dumps(policy))
    target = root.parent / "unsupported"
    with pytest.raises(ValueError, match="unsupported"):
        runner.prepare(
            directory=target,
            prior_ledger=seed_path,
            root=root,
            retrieval_path=root / "retrieval.json",
            policy_path=policy_path,
        )
    assert not target.exists()


def test_frozen_33_case_run_reconstructs_exactly_without_transport(
    tmp_path, monkeypatch
):
    historical = runner.HISTORICAL_DIRECTORY
    before = {
        path: runner.file_hash(path) for path in historical.rglob("*") if path.is_file()
    }
    freeze = json.loads((historical / "execution_freeze.json").read_text())
    for name, digest in freeze["source_hashes"].items():
        assert runner.file_hash(runner.ROOT / name) == digest
    expected = json.loads((historical / "report.json").read_text())
    directory = tmp_path / "replay"
    monkeypatch.setattr(runner, "PAID_LEDGER", tmp_path / "unused-global.json")

    async def forbidden(*args, **kwargs):
        raise AssertionError("Frozen replay must never request provider transport")

    monkeypatch.setattr(runner.BudgetedProvider, "complete", forbidden)
    runner.prepare(directory=directory, prior_ledger=historical / "budget.json")
    assert not asyncio.run(runner.run(directory=directory))
    actual = json.loads((directory / "report.json").read_text())
    for name in (
        "cases",
        "precision",
        "correct_no_card",
        "coverage",
        "failed_cases",
        "unexpected_provider_failures",
        "calculated_cost_usd",
        "attempts",
        "gate_passed",
    ):
        assert actual["summary"][name] == expected["summary"][name]
    for old, new in zip(expected["cases"], actual["cases"], strict=True):
        for field in ("case_id", "status", "selected"):
            assert new[field] == old[field]
    assert len(actual["cases"]) == 33
    assert len(json.loads((directory / "budget.json").read_text())["attempts"]) == 61
    assert before == {
        path: runner.file_hash(path) for path in historical.rglob("*") if path.is_file()
    }
    assert not runner.PAID_LEDGER.exists()
