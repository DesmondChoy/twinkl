"""Reporting repairs retain the original generation freeze and completion time."""

import asyncio
import copy
import hashlib
import json
import sys

import pytest

from scripts.experiments import nsm_experiment as experiment
from src.north_star.provider import stable_hash

RUNNER = "scripts/experiments/nsm_experiment.py"
EVALUATOR = "scripts/experiments/nsm_evaluation.py"
PROVIDER = "scripts/experiments/nsm_provider.py"


@pytest.fixture
def corrected_record(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    original = "# Frozen generation-time source.\n"
    corrected = "# Corrected deterministic reporting source.\n"
    original_hash = hashlib.sha256(original.encode()).hexdigest()
    for name in (RUNNER, EVALUATOR, PROVIDER):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(corrected if name in (RUNNER, EVALUATOR) else original)
    methodology = tmp_path / "methodology.md"
    methodology.write_text("Frozen methodology.\n")
    execution = {
        "completed_at": "2026-09-07T01:00:00+00:00",
        "cases": [{"case_id": "case"}],
        "retrieval": {},
        "provider_policy": {},
        "consistency_case_ids": [],
        "case_provenance": {"source_hashes": {}},
        "variants": {
            variant: {"case": {"measurement": {}}} for variant in experiment.VARIANTS
        },
        "references": {"case": {"source_reviews": {}}},
        "quote_reviews": {"case": {"reviews": {}}},
        "rechecks": {"case": {"reviews": {}}},
        "repeats": {
            repeat: {"references": {}, "quotes": {}}
            for repeat in ("repeat_2", "repeat_3")
        },
        "requests": {},
        "stages": {
            name: {"completed_at": "2026-09-07T00:59:00+00:00"}
            for name in (
                "runtime:nomic",
                "runtime:full_history",
                "reference:primary",
                "quotation:primary",
                "recheck:primary",
                "reference:repeat_2",
                "quotation:repeat_2",
                "reference:repeat_3",
                "quotation:repeat_3",
            )
        },
    }
    record = {
        "status": "completed_ai_evaluation",
        "methodology": "methodology.md",
        "frozen_settings": {},
        "execution": execution,
        "reporting": {
            "code_corrections": {
                name: {
                    "original_source_text": original,
                    "corrected_sha256": experiment.file_hash(tmp_path / name),
                    "reason": "Correct a deterministic reporting error.",
                }
                for name in (RUNNER, EVALUATOR)
            }
        },
    }
    execution["freeze"] = {
        "code_sha256": {name: original_hash for name in (RUNNER, EVALUATOR, PROVIDER)},
        "methodology_text": methodology.read_text(),
        "methodology_sha256": experiment.file_hash(methodology),
        **{
            f"{key}_sha256": stable_hash(value)
            for key, value in (
                ("cases", execution["cases"]),
                ("retrieval", execution["retrieval"]),
                ("policy", execution["provider_policy"]),
                ("settings", record["frozen_settings"]),
                ("consistency_sample", execution["consistency_case_ids"]),
            )
        },
    }
    return record


def test_default_generation_verification_still_rejects_corrected_code(corrected_record):
    with pytest.raises(ValueError, match="Frozen experiment code changed"):
        experiment.verify(corrected_record)


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_generation_entrypoints_cannot_use_reporting_corrections(
    corrected_record, tmp_path, command
):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(corrected_record))
    with pytest.raises(ValueError, match="Frozen experiment code changed"):
        if command == "prepare":
            experiment.prepare(path)
        else:
            asyncio.run(experiment.run(path))


def test_reporting_verification_accepts_documented_correction_without_rewriting_freeze(
    corrected_record,
):
    before = copy.deepcopy(corrected_record)
    experiment.verify(corrected_record, allow_reporting_corrections=True)
    assert corrected_record == before


def test_cli_verify_accepts_reporting_corrections_without_rewriting_record(
    corrected_record, tmp_path, monkeypatch, capsys
):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(corrected_record))
    before = path.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["nsm_experiment.py", "verify", "--record", str(path)]
    )
    experiment.main()
    assert "FROZEN_EXPERIMENT_VERIFIED" in capsys.readouterr().out
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", [RUNNER, EVALUATOR])
def test_changed_original_archive_cannot_authorize_a_correction(corrected_record, name):
    corrected_record["reporting"]["code_corrections"][name]["original_source_text"] += (
        "# Tampered.\n"
    )
    with pytest.raises(ValueError, match="Unverified original reporting code"):
        experiment.verify(corrected_record, allow_reporting_corrections=True)


def test_unrecorded_further_code_change_is_rejected(corrected_record):
    path = experiment.ROOT / EVALUATOR
    path.write_text(path.read_text() + "# Unrecorded change.\n")
    with pytest.raises(ValueError, match="Corrected reporting code changed"):
        experiment.verify(corrected_record, allow_reporting_corrections=True)


def test_provider_cannot_be_whitelisted_as_a_reporting_correction(corrected_record):
    correction = copy.deepcopy(
        corrected_record["reporting"]["code_corrections"][RUNNER]
    )
    corrected_record["reporting"]["code_corrections"][PROVIDER] = correction
    with pytest.raises(ValueError, match="Unsupported reporting code correction"):
        experiment.verify(corrected_record, allow_reporting_corrections=True)


@pytest.mark.parametrize(
    "incomplete",
    ["status", "completion_time", "stage", "case", "repeat", "pending_attempt"],
)
def test_reporting_correction_cannot_bypass_unfinished_execution(
    corrected_record, incomplete
):
    execution = corrected_record["execution"]
    if incomplete == "status":
        corrected_record["status"] = "running"
    elif incomplete == "completion_time":
        del execution["completed_at"]
    elif incomplete == "stage":
        del execution["stages"]["quotation:repeat_3"]["completed_at"]
    elif incomplete == "case":
        execution["variants"]["full_history"].clear()
    elif incomplete == "repeat":
        execution["repeats"]["repeat_3"]["quotes"]["unexpected"] = {}
    else:
        execution["requests"]["pending"] = {"attempts": [{"status": "pending"}]}
    with pytest.raises(ValueError, match="require completed execution"):
        experiment.verify(corrected_record, allow_reporting_corrections=True)


def test_reporting_only_permission_cannot_hide_changed_cases_or_provider_policy(
    corrected_record,
):
    corrected_record["execution"]["provider_policy"]["new_setting"] = True
    with pytest.raises(ValueError, match="Frozen policy changed"):
        experiment.verify(corrected_record, allow_reporting_corrections=True)


def test_rereport_preserves_generation_completion_and_records_later_report_time(
    corrected_record, monkeypatch
):
    execution = corrected_record["execution"]
    frozen = copy.deepcopy(execution["freeze"])
    completed_at = execution["completed_at"]
    monkeypatch.setattr(experiment, "now", lambda: "2026-09-07T03:00:00+00:00")
    monkeypatch.setattr(
        experiment.nsm_evaluation,
        "grade_case",
        lambda case, *args: {"case_id": case["case_id"]},
    )
    monkeypatch.setattr(experiment.nsm_evaluation, "summarize", lambda *args: {})
    monkeypatch.setattr(
        experiment.nsm_evaluation, "consistency_report", lambda *args: {"cases": 0}
    )
    experiment.report(corrected_record)
    assert execution["completed_at"] == completed_at
    assert execution["reported_at"] == "2026-09-07T03:00:00+00:00"
    assert execution["freeze"] == frozen
