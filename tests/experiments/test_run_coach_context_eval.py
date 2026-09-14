"""Bounded calls, evidence isolation, and immutable context-evaluation receipts."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from scripts.experiments import run_coach_context_eval as runner
from src.coach.schemas import LLMCallMetrics


@pytest.fixture
def plan():
    return runner.prepare()


def without_excerpts(value):
    if isinstance(value, dict):
        return {k: without_excerpts(v) for k, v in value.items() if k != "excerpt"}
    if isinstance(value, list):
        return [without_excerpts(v) for v in value]
    return value


def test_context_only_changes_selected_excerpts(plan):
    old, full = plan["digests"]["old"], plan["digests"]["full"]
    assert without_excerpts(old) == without_excerpts(full)
    assert len(full["evidence"]) == len(old["evidence"]) == 3
    assert "Brought it up in a 1:1 with my lead" in full["evidence"][1]["excerpt"]
    assert "Brought it up in a 1:1 with my lead" not in old["evidence"][1]["excerpt"]
    assert "Response:" in full["evidence"][1]["excerpt"]
    assert "I said I didn't know" in full["evidence"][2]["excerpt"]
    assert full["state_comparisons"][0]["current_evidence"][0] == full["evidence"][1]
    assert (
        plan["generation"]["old"]["request"]["instructions"]
        == plan["generation"]["full"]["request"]["instructions"]
    )
    assert [row["arm"] for row in plan["generation_order"]] == [
        "old",
        "full",
        "full",
        "old",
        "old",
        "full",
    ]


class FakeProvider:
    def __init__(self, *, transport_failure=False, malformed_generation=False):
        self.calls = []
        self.transport_failure = transport_failure
        self.malformed_generation = malformed_generation

    def factory(self, **kwargs):
        self.settings = kwargs

        async def complete(prompt, response_format=None, instructions=None):
            self.calls.append((prompt, response_format, instructions))
            error = self.transport_failure
            kwargs["call_metrics"].append(
                LLMCallMetrics(
                    provider="openai",
                    model=runner.DEFAULT_OPENAI_MODEL,
                    status="error" if error else "completed",
                    latency_seconds=0.01,
                    input_tokens=10,
                    output_tokens=10,
                    calculated_cost_usd=0.001,
                    error_type="APIConnectionError" if error else None,
                )
            )
            if error:
                return None
            if response_format["name"] == "coach_narrative_judge":
                return json.dumps(
                    {
                        "correctness": 2,
                        "specificity": 4,
                        "non_prescriptive_tone": 5,
                        "tension_honesty": 2,
                        "question_is_open_and_relevant": True,
                        "justification": "Synthetic stub judgment.",
                    }
                )
            if self.malformed_generation:
                return "invalid JSON"
            return runner.GOOD_CONTROL.model_dump_json()

        return complete


def test_dry_run_never_builds_provider(tmp_path, monkeypatch):
    def fail(**kwargs):
        pytest.fail("Dry run must not construct a provider")

    monkeypatch.setattr(runner, "build_llm_complete", fail)
    assert runner.main(["--out", str(tmp_path / "experiment")]) == 0
    assert not list((tmp_path / "experiment").glob("*.call.json"))
    assert (
        tmp_path / "experiment/source_snapshot/prompts/weekly_digest_coach.yaml"
    ).exists()


def test_bounded_calls_blind_full_context_judging_and_unchanged_resume(tmp_path, plan):
    runner.freeze(tmp_path, plan)
    provider = FakeProvider()
    summary = asyncio.run(runner.run(tmp_path, plan, provider_factory=provider.factory))
    assert len(provider.calls) == summary["provider_attempts"] == 15
    assert len(summary["generation"]) == 6
    assert len(summary["evaluations"]) == 9
    assert all(row["correctness_below_3"] for row in summary["evaluations"].values())
    for prompt, _, instructions in provider.calls[6:]:
        assert instructions is None
        assert all(
            label not in prompt
            for label in (
                "known_bad",
                "constructed_good",
                "generation_",
                "expected_correctness",
            )
        )
    # All fresh outputs see the same full evidence; only the old control omits it.
    for index in [*range(6, 12), 13, 14]:
        assert "Brought it up in a 1:1 with my lead" in provider.calls[index][0]
    assert "Brought it up in a 1:1 with my lead" not in provider.calls[12][0]
    receipts = {path.name: path.read_bytes() for path in tmp_path.glob("*.call.json")}
    resumed = asyncio.run(runner.run(tmp_path, plan, provider_factory=provider.factory))
    assert resumed == summary and len(provider.calls) == 15
    assert receipts == {
        path.name: path.read_bytes() for path in tmp_path.glob("*.call.json")
    }
    judge = json.loads((tmp_path / "judge_known_bad_full.call.json").read_text())
    assert json.loads(judge["raw_output"])["correctness"] == 2
    assert judge["request"]["settings"]["sdk_retries"] == 0
    assert summary["self_evaluation"] is True


def test_changed_manifest_pending_and_tampered_receipts_block_resume(tmp_path, plan):
    runner.freeze(tmp_path, plan)
    changed = copy.deepcopy(plan)
    changed["settings"]["seed"] = 42
    with pytest.raises(ValueError, match="fingerprint changed"):
        runner.freeze(tmp_path, changed)
    pending = tmp_path / "generation_1_old.pending.json"
    pending.write_text("{}")
    with pytest.raises(ValueError, match="Unresolved provider"):
        asyncio.run(runner.run(tmp_path, plan, provider_factory=FakeProvider().factory))
    pending.unlink()
    provider = FakeProvider()
    asyncio.run(runner.run(tmp_path, plan, provider_factory=provider.factory))
    saved = tmp_path / "generation_1_old.call.json"
    record = json.loads(saved.read_text())
    record["raw_output"] = "changed"
    saved.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integrity failure"):
        asyncio.run(runner.run(tmp_path, plan, provider_factory=provider.factory))
    assert len(provider.calls) == 15


def test_transport_error_stops_calls_and_resume_skips_failed_key(tmp_path, plan):
    runner.freeze(tmp_path, plan)
    broken = FakeProvider(transport_failure=True)
    with pytest.raises(RuntimeError, match="Provider failure retained"):
        asyncio.run(runner.run(tmp_path, plan, provider_factory=broken.factory))
    assert len(broken.calls) == 1
    assert not list(tmp_path.glob("*.pending.json"))
    receipt = (tmp_path / "generation_1_old.call.json").read_bytes()
    recovered = FakeProvider()
    summary = asyncio.run(
        runner.run(tmp_path, plan, provider_factory=recovered.factory)
    )
    assert len(recovered.calls) == 13
    assert summary["provider_attempts"] == 14
    assert (
        summary["evaluations"]["judge_generation_1_old"]["status"]
        == "skipped_no_narrative"
    )
    assert (tmp_path / "generation_1_old.call.json").read_bytes() == receipt


def test_malformed_outputs_are_reported_without_retries_or_favorable_filtering(
    tmp_path, plan
):
    runner.freeze(tmp_path, plan)
    provider = FakeProvider(malformed_generation=True)
    summary = asyncio.run(runner.run(tmp_path, plan, provider_factory=provider.factory))
    assert len(provider.calls) == 9  # Six failures plus the three control judgments.
    assert len(summary["generation"]) == 6
    assert all(row["diagnostic"]["accepted"] is False for row in summary["generation"])
    assert (
        sum(
            row["status"] == "skipped_no_narrative"
            for row in summary["evaluations"].values()
        )
        == 6
    )
