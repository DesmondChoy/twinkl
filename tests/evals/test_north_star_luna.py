"""Fresh-run contracts with toy sources and mocked transport, never stored results."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import Mock

import polars as pl
import pytest

from scripts.experiments import north_star_luna as runner
from src.north_star import assessment, input_budget
from src.north_star.provider import BudgetedProvider, BudgetLedger, stable_hash
from src.north_star.review import SourceEntry


def episode(persona, number, onset, dimension="benevolence"):
    return {
        "episode_id": f"{persona}:{dimension}:{number}",
        "persona_id": persona,
        "dimension": dimension,
        **{
            f"{point}_{field}": value
            for point, index in (
                ("onset", onset),
                ("confirmation", onset),
                ("end", onset),
            )
            for field, value in (
                ("t_index", index),
                ("position", index + 1),
                ("date", f"2026-01-{index + 1:02d}"),
            )
        },
    }


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    rows = [
        {
            "t_index": i,
            "date": f"2026-01-{i + 1:02d}",
            "initial_entry": f"I cooked dinner for my friend on day {i}.",
            "nudge_response": "NUDGE_WITH_UNKNOWN_AVAILABILITY",
            "label": "GENERATION_LABEL_MUST_NOT_ENTER_PROMPTS",
        }
        for i in range(4)
    ]
    episodes = [
        episode("aaaa", 1, 1),
        episode("aaaa", 2, 2),
        episode("aaaa", 3, 1, "security"),
        episode("aaaa", 4, 0),
        episode("bbbb", 1, 1),
    ]
    path = root / runner.EPISODES
    path.parent.mkdir(parents=True)
    pl.DataFrame(episodes).write_parquet(path)
    runner.write_json(
        root / runner.COHORT,
        {
            "source_episode_sha256": runner.sha256(path),
            "development_persona_ids": ["aaaa"],
            "reserved_persona_ids": ["bbbb"],
        },
    )
    values = root / runner.VALUES
    values.parent.mkdir(parents=True, exist_ok=True)
    values.write_text(
        "values:\n  Benevolence:\n    user_phrase: Care for others\n"
        "    definition: Preserve the welfare of close contacts.\n"
        "  Security:\n    user_phrase: Keep life steady\n"
        "    definition: Preserve safety and stability.\n"
    )
    for name in ("wrangled", "synthetic_data"):
        path = root / f"logs/{name}/persona_aaaa.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(row["initial_entry"] for row in rows))
    visited = []

    def history(actual_root, persona):
        assert actual_root == root
        assert persona == "aaaa", "Reserved history must never be opened"
        visited.append(persona)
        return deepcopy(rows)

    monkeypatch.setattr(runner, "_history", history)
    policy = json.loads((runner.ROOT / runner.POLICY).read_text())
    runner.write_json(root / runner.POLICY, policy)
    code = root / "executable.py"
    code.write_text("# frozen source\n")
    monkeypatch.setattr(runner, "EXECUTION_SOURCES", ("executable.py",))
    return root, rows, visited, policy


def test_inputs_deduplicate_per_value_and_keep_reserved_writing_unopened(frozen):
    root, _, visited, _ = frozen
    data = runner.build_inputs(root)
    assert visited == ["aaaa"]
    assert len(data["cases"]) == 4
    assert len(data["source_groups"]) == 3
    assert data["reserved_persona_ids"] == ["bbbb"]
    assert {c["persona_id"] for c in data["cases"]} == {"aaaa"}
    assert len(data["reference_batches"]) == 2
    case1, case2, empty, case3 = data["cases"]
    assert case1["group_ids"][0] in case2["group_ids"]
    assert case1["group_ids"][0] != case3["group_ids"][0]
    assert empty["sources"] == [] and empty["runtime_request"] is None
    assert [s["entry_id"] for s in case2["sources"]] == ["aaaa:entry:1", "aaaa:entry:0"]
    requests = [c["runtime_request"] for c in data["cases"] if c["runtime_request"]]
    requests.extend(b["request"] for b in data["reference_batches"])
    for request in requests:
        assert "NUDGE_WITH_UNKNOWN_AVAILABILITY" not in request["prompt"]
        assert "GENERATION_LABEL" not in request["prompt"]
        assert set(json.loads(request["prompt"])) == {
            "core_value",
            "approved_definition",
            "user_phrase",
            "sources",
        }
    for source in data["source_groups"]:
        assert source["source"]["nudge_response"] is None
    assert all("reports/" not in name for name in data["input_hashes"])


@pytest.mark.parametrize(
    "changed", ["cohort_overlap", "episode_hash", "date", "provenance"]
)
def test_changed_or_unverifiable_inputs_fail_before_requests(frozen, changed):
    root, rows, _, _ = frozen
    if changed == "cohort_overlap":
        path = root / runner.COHORT
        cohort = json.loads(path.read_text())
        cohort["reserved_persona_ids"].append("aaaa")
        runner.write_json(path, cohort)
    elif changed == "episode_hash":
        path = root / runner.EPISODES
        path.write_bytes(path.read_bytes() + b"changed")
    elif changed == "date":
        rows[1]["date"] = "2026-01-31"
    else:
        (root / "logs/synthetic_data/persona_aaaa.md").write_text("No source writing")
    with pytest.raises(ValueError):
        runner.build_inputs(root)


def test_prepare_needs_no_old_results_and_freezes_sources_policy(frozen, tmp_path):
    root, _, _, _ = frozen
    directory = tmp_path / "fresh"
    manifest = runner.prepare(directory, root)
    assert runner.verify(directory, root) == manifest
    assert manifest["budget_preflight"]["maximum_cumulative_usd"] <= 20
    assert not (root / "logs/experiments/reports").exists()
    with pytest.raises(ValueError, match="overwrite"):
        runner.prepare(directory, root)
    (root / "executable.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        runner.verify(directory, root)


def test_source_roles_share_exact_prompts_schema_but_different_cache_keys():
    value = {
        "core_value": "benevolence",
        "user_phrase": "Care for others",
        "approved_definition": "Preserve the welfare of close contacts.",
    }
    sources = [SourceEntry(entry_id="a:0", journal_entry="I cooked for my friend.")]
    runtime = runner.source_request(value, sources, "runtime")
    reference = runner.source_request(value, sources, "reference")
    for field in ("system", "prompt", "schema", "provider"):
        assert runtime[field] == reference[field]
    assert runtime["role"] == "runtime" and reference["role"] == "reference"
    assert stable_hash(runtime) != stable_hash(reference)


def source_raw(request, reason="observable_choice"):
    payload = json.loads(request["prompt"])
    return json.dumps(
        {
            "schema_version": assessment.SOURCE_SCHEMA_VERSION,
            "core_value": payload["core_value"],
            "results": [
                {
                    "entry_id": source["entry_id"],
                    "action_assessment": "The writer cooked dinner.",
                    "value_assessment": "Toy factual assessment.",
                    "conflict_assessment": "No opposing behavior reported.",
                    "reason_code": reason,
                    "quote_source": "journal_entry"
                    if reason == "observable_choice"
                    else None,
                    "evidence_quote": source["journal_entry"]
                    if reason == "observable_choice"
                    else "",
                }
                for source in payload["sources"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def candidate_raw(request, reason="observable_choice", quote_reason=None):
    payload = json.loads(request["prompt"])
    quote = payload["proposed_quote"]
    return json.dumps(
        {
            "schema_version": assessment.CANDIDATE_SCHEMA_VERSION,
            "core_value": payload["core_value"],
            "entry_id": quote["entry_id"],
            "action_assessment": "The writer cooked dinner.",
            "value_assessment": "Toy factual assessment.",
            "conflict_assessment": "No opposing behavior reported.",
            "quote_assessment": "The proposed quote reports cooking.",
            "source_reason": reason,
            "quote_reason": quote_reason
            or (
                "supported_action"
                if reason == "observable_choice"
                else "same_value_conflict"
                if reason == "same_value_conflict"
                else "source_not_supportive"
            ),
            "quote_source": quote["quote_source"],
            "evaluated_quote": quote["evidence_quote"],
        },
        ensure_ascii=False,
        indent=2,
    )


@pytest.fixture
def harness(frozen, tmp_path, monkeypatch):
    root, _, _, policy = frozen
    directory = tmp_path / "fresh"
    manifest = runner.prepare(directory, root)
    monkeypatch.setattr(runner, "verify", lambda path: manifest)
    controls = {
        "runtime_reason": "observable_choice",
        "source_reason": "observable_choice",
        "candidate_reason": "observable_choice",
        "count": 100,
    }
    calls = []
    real = BudgetedProvider

    class ToyProvider(real):
        async def complete(self, *, retry=False, **request):
            calls.append((deepcopy(request), retry))
            envelope = {**request, "policy_hash": stable_hash(self.ledger.policy)}
            attempt = self.ledger.reserve(envelope, retry=retry)
            if attempt.reused:
                return attempt
            raw = (
                candidate_raw(request, controls["candidate_reason"])
                if request["purpose"] == "nsm-fresh-quote"
                else source_raw(
                    request,
                    controls["runtime_reason"]
                    if request["role"] == "runtime"
                    else controls["source_reason"],
                )
            )
            if controls.get("invalid_outputs", 0):
                controls["invalid_outputs"] -= 1
                raw = "{}"
            attempt.status = "completed"
            attempt.raw_text = raw
            attempt.calculated_cost_usd = 0.001
            return self.ledger.finish(attempt)

    async def measure(requests, actual_policy, output):
        state = (
            json.loads(output.read_text())
            if output.exists()
            else {"schema_version": input_budget.SCHEMA_VERSION, "counts": {}}
        )
        for request in requests:
            payload = input_budget.count_payload(request, actual_policy)
            state["counts"][stable_hash(request)] = {
                "request_hash": stable_hash(request),
                "payload_hash": stable_hash(payload),
                "model": payload["model"],
                "input_tokens": controls["count"],
            }
        runner.write_json(output, state)
        return state

    monkeypatch.setattr(runner, "BudgetedProvider", ToyProvider)
    monkeypatch.setattr(input_budget, "measure_requests", measure)
    return directory, manifest, policy, controls, calls


def test_fresh_run_retains_raw_receipts_and_replay_never_calls_transport(
    harness, monkeypatch
):
    directory, manifest, _, _, calls = harness
    report = asyncio.run(runner.run(directory, allow_paid=True, concurrency=1))
    assert report["summary"]["selected"] == 3
    assert report["summary"]["accepted"] == 3
    budget = (directory / "budget.json").read_bytes()
    ledger = json.loads(budget)
    assert len(ledger["attempts"]) == len(calls)
    assert all(
        a["raw_text"].startswith('{\n  "schema_version"') for a in ledger["attempts"]
    )
    assert {a["role"] for a in ledger["attempts"]} == {"runtime", "reference"}
    blinded = json.loads((directory / "independent_review_inputs.json").read_text())
    assert len(blinded["source_groups"]) == len(manifest["source_groups"])
    assert len(blinded["selected_quotes"]) == 3
    serialized = json.dumps(blinded)
    for forbidden in (
        "action_assessment",
        "source_reason",
        "reason_code",
        "quote_reason",
    ):
        assert forbidden not in serialized
    forbidden = Mock(side_effect=AssertionError("Replay must never call transport"))
    monkeypatch.setattr(runner.BudgetedProvider, "complete", forbidden)
    monkeypatch.setattr(input_budget, "measure_requests", forbidden)
    assert asyncio.run(runner.run(directory)) == report
    assert (directory / "budget.json").read_bytes() == budget
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "primary,candidate,expected",
    [
        ("observable_choice", "wrong_value", "unresolved"),
        ("wrong_value", "observable_choice", "unresolved"),
        ("ambiguous", "ambiguous", "unresolved"),
        ("wrong_value", "same_value_conflict", "rejected"),
        ("wrong_value", "wrong_value", "rejected"),
    ],
)
def test_source_quote_disagreement_and_abstention_are_not_confirmed_errors(
    harness,
    primary,
    candidate,
    expected,
):
    directory, _, _, controls, _ = harness
    controls.update(source_reason=primary, candidate_reason=candidate)
    report = asyncio.run(runner.run(directory, allow_paid=True, concurrency=1))
    assert {c["grade"] for c in report["cases"] if c["selected"]} == {expected}
    if primary == "wrong_value" and candidate == "same_value_conflict":
        assert all(
            c["source_reason_disagreement"] for c in report["cases"] if c["selected"]
        )
    assert not report["summary"]["gate_passed"]


def test_abstention_cannot_establish_no_example_or_correct_omission(harness):
    directory, _, _, controls, _ = harness
    controls.update(runtime_reason="ambiguous", source_reason="ambiguous")
    report = asyncio.run(runner.run(directory, allow_paid=True, concurrency=1))
    assert {c["reference_state"] for c in report["cases"] if c["eligible_sources"]} == {
        "unresolved"
    }
    assert report["summary"]["correct_omission"]["denominator"] == 0
    assert report["summary"]["unresolved_no_positive_histories"] == 3
    assert not report["summary"]["gate_passed"]


def test_over_budget_complete_input_blocks_every_generation_request(harness):
    directory, _, _, controls, calls = harness
    controls["count"] = 16_001
    with pytest.raises(ValueError, match="16,000"):
        asyncio.run(runner.run(directory, allow_paid=True))
    assert calls == []
    assert not (directory / "budget.json").exists()


def test_completed_cached_request_with_other_role_does_not_satisfy_runtime(harness):
    directory, manifest, _, _, _ = harness
    reference = manifest["reference_batches"][0]["request"]
    request = {**reference, "role": "runtime", "purpose": "nsm-fresh-runtime"}
    ledger = BudgetLedger(directory / "budget.json", directory / "policy.json")
    attempt = ledger.reserve(
        {**reference, "policy_hash": stable_hash(ledger.policy)}, retry=False
    )
    attempt.status = "completed"
    attempt.raw_text = source_raw(reference)
    attempt.calculated_cost_usd = 0.001
    ledger.finish(attempt)
    provider = BudgetedProvider(ledger)
    assert (
        asyncio.run(
            runner.execute_request(request, provider, directory, allow_paid=False)
        )
        is None
    )


def test_invalid_response_retains_cost_and_raw_before_one_retry(harness):
    directory, _, policy, controls, calls = harness
    controls["invalid_outputs"] = 1
    report = asyncio.run(runner.run(directory, allow_paid=True, concurrency=1))
    ledger = json.loads((directory / "budget.json").read_text())
    invalid = [a for a in ledger["attempts"] if a["status"] == "invalid"]
    assert len(invalid) == 1
    assert invalid[0]["raw_text"] == "{}"
    assert invalid[0]["calculated_cost_usd"] == 0.001
    retried = [
        a for a in ledger["attempts"] if a["request_hash"] == invalid[0]["request_hash"]
    ]
    assert [a["attempt_number"] for a in retried] == [1, 2]
    assert retried[1]["status"] == "completed"
    assert sum(retry for _, retry in calls) == 1
    assert report["summary"]["code_invalid_attempts"] == 1
    assert report["summary"]["cumulative_spent_or_reserved_usd"] == pytest.approx(
        policy["prior_spend_usd"] + len(ledger["attempts"]) * 0.001
    )
    assert not report["summary"]["gate_passed"]
