"""Saved-frontend coverage checks without provider calls or baseline rewrites."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from scripts.experiments import north_star_saved_checks as runner


@pytest.fixture
def baseline_inputs():
    directory = runner.baseline.DEFAULT_DIRECTORY
    return (
        json.loads((directory / "manifest.json").read_text()),
        json.loads((directory / "report.json").read_text()),
    )


def test_all_saved_weeks_include_meera_as_a_no_trigger_control(baseline_inputs):
    data = runner.coverage(runner.ROOT, *baseline_inputs)
    assert len(data["weeks"]) == 36
    assert {w["persona_id"] for w in data["weeks"]} == {
        "23d101f8",
        "8f83c818",
        "988d1a65",
        "02fb94f3",
        "11de77e8",
    }
    meera = [w for w in data["weeks"] if w["persona_id"] == "23d101f8"]
    assert len(meera) == 7
    for week in meera:
        assert week["delivery_state"] == "no_active_drift"
        assert week["core_value_states"] == {
            "achievement": "no_active_drift",
            "security": "no_active_drift",
        }
        assert week["expected_nsm_action"] == "no_request_no_card"
        assert week["source"] == "no_trigger_control"
        assert week["case_id"] is None
    assert all(c["persona_id"] != "23d101f8" for c in data["missing_cases"])


def test_reuse_requires_exact_original_sources_and_keeps_omission(baseline_inputs):
    original, report = baseline_inputs
    data = runner.coverage(runner.ROOT, original, report)
    reused = {
        w["case_id"]: w["frozen_grade"]
        for w in data["weeks"]
        if w["source"] == "matching_frozen_case"
    }
    assert reused == {
        "8f83c818:universalism:episode_01": "accepted",
        "988d1a65:power:episode_01": "accepted",
        "02fb94f3:tradition:episode_01": "not_selected",
    }
    changed = deepcopy(original)
    case = next(c for c in changed["cases"] if c["case_id"] == next(iter(reused)))
    case["sources"][0]["journal_entry"] += " Changed source text."
    with pytest.raises(ValueError, match="Source lacks original synthetic provenance"):
        runner.coverage(runner.ROOT, changed, report)
    changed = deepcopy(original)
    case = next(c for c in changed["cases"] if c["case_id"] == next(iter(reused)))
    case["episode"]["onset_t_index"] += 1
    changed_data = runner.coverage(runner.ROOT, changed, report)
    assert len(changed_data["missing_cases"]) == 2
    assert not any(w["case_id"] == case["case_id"] for w in changed_data["weeks"])


def test_lukas_additional_active_value_uses_only_pre_drift_writing(baseline_inputs):
    data = runner.coverage(runner.ROOT, *baseline_inputs)
    assert len(data["missing_cases"]) == 1
    case = data["missing_cases"][0]
    assert case["case_id"] == "two-values-lukas:week:5:conformity"
    assert case["persona_id"] == "11de77e8"
    assert case["value"]["core_value"] == "conformity"
    assert case["onset_t_index"] == 4
    assert [s["entry_id"] for s in case["sources"]] == [
        f"11de77e8:entry:{index}" for index in (3, 2, 1, 0)
    ]
    assert all(s["nudge_response"] is None for s in case["sources"])
    for role in ("runtime_request", "reference_request"):
        prompt = json.loads(case[role]["prompt"])
        assert prompt["sources"] == case["sources"]
        assert "11de77e8:entry:4" not in case[role]["prompt"]


def test_offline_supplement_preserves_original_results_and_precision(
    tmp_path, monkeypatch
):
    baseline_directory = runner.baseline.DEFAULT_DIRECTORY
    before = {p: p.read_bytes() for p in baseline_directory.iterdir() if p.is_file()}
    paid = Mock(side_effect=AssertionError("Offline checks must not call a provider"))
    monkeypatch.setattr(runner.BudgetedProvider, "complete", paid)
    supplement = tmp_path / "supplement"
    manifest = runner.prepare(supplement)
    result = asyncio.run(runner.run(supplement, allow_paid=False))
    assert result["original_selection_precision_unchanged"] is True
    assert result["generation_attempts"] == 0
    assert result["new_cost_usd"] == 0
    assert len(result["supplemental_cases"]) == 1
    assert result["supplemental_cases"][0]["failed"] is True
    assert manifest["original_report_sha256"] == runner.baseline.sha256(
        baseline_directory / "report.json"
    )
    assert all(path.read_bytes() == content for path, content in before.items())
    paid.assert_not_called()


@pytest.mark.parametrize("changed", ["frontend", "original_report"])
def test_frozen_input_changes_stop_before_provider_use(tmp_path, monkeypatch, changed):
    supplement = tmp_path / "supplement"
    manifest = runner.prepare(supplement)
    path = (
        runner.ROOT / next(iter(manifest["frontend_hashes"]))
        if changed == "frontend"
        else runner.baseline.DEFAULT_DIRECTORY / "report.json"
    )
    real_hash = runner.baseline.sha256
    monkeypatch.setattr(
        runner.baseline, "sha256", lambda p: "0" * 64 if p == path else real_hash(p)
    )
    provider = Mock(side_effect=AssertionError("Changed inputs must stop execution"))
    monkeypatch.setattr(runner, "BudgetedProvider", provider)
    expected = (
        "Frozen frontend bundle changed"
        if changed == "frontend"
        else "Original report changed"
    )
    with pytest.raises(ValueError, match=expected):
        asyncio.run(runner.run(supplement, allow_paid=False))
    provider.assert_not_called()
