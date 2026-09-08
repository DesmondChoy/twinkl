"""Completed experiment replay retains exact source and provider provenance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.demo.north_star_replay import (
    EXPERIMENT_PATH,
    EXPERIMENT_SHA256,
    EXPORT_PATH,
    ExperimentEvidence,
    SavedExperimentRecord,
    export_records,
    project_record,
    validate_saved_record,
    verify_experiment_source,
)
from src.demo.scenarios import (
    SELECTIONS,
    build_saved_north_star_request,
    build_scenario_fixture,
)
from src.drift_detector import DriftDetectorResult
from src.north_star import runtime
from src.north_star.provider import BudgetedProvider
from src.wrangling.parse_wrangled_data import parse_wrangled_file

ROOT = Path(__file__).resolve().parents[2]


def test_completed_case_without_pre_onset_writing_preserves_ineligible_outcome():
    raw = (ROOT / EXPERIMENT_PATH).read_bytes()
    execution = json.loads(raw)["execution"]
    case = next(
        c for c in execution["cases"] if c["context_reason"] == "no_eligible_writing"
    )
    _, entries, _ = parse_wrangled_file(
        ROOT / f"logs/wrangled/persona_{case['persona_id']}.md"
    )
    by_index = {entry["t_index"]: entry for entry in entries}
    writing = []
    for metadata in case["source_metadata"].values():
        entry = by_index[metadata["t_index"]]
        writing.append(
            runtime.SourceWriting(
                **{
                    key: metadata[key]
                    for key in (
                        "owner_id",
                        "entry_id",
                        "t_index",
                        "date",
                        "available_at",
                        "response_available_at",
                    )
                },
                journal_entry=entry["initial_entry"],
                nudge_response=entry["response_text"] or None,
            )
        )
    request = runtime.build_north_star_request(
        session_id="saved:no-earlier-writing",
        owner_id=case["persona_id"],
        profile_ref=case["profile_ref"],
        core_values=case["core_values"],
        week_start=case["week_start"],
        week_end=case["week_end"],
        cutoff_at=case["cutoff_at"],
        drift_result=DriftDetectorResult.model_validate(case["drift_result"]),
        writing=writing,
    )
    evidence = ExperimentEvidence(
        source_path=EXPERIMENT_PATH.as_posix(),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        case_id=case["case_id"],
        case=case,
        output=execution["variants"]["full_history"][case["case_id"]],
        receipts=[],
        policy=execution["provider_policy"],
    )
    record = project_record(request, evidence, created_at=execution["completed_at"])
    assert record.status == "not_eligible"
    assert record.reason == "no_eligible_writing"
    assert record.selected is None


@pytest.fixture(scope="module")
def replay_records():
    report = json.loads((ROOT / EXPORT_PATH).read_text())
    by_week = {row["week_id"]: row["record"] for row in report["cases"]}
    rows = []
    for selection in SELECTIONS:
        fixture = build_scenario_fixture(ROOT, selection, include_north_star=False)
        for week in fixture.scenario.weeks:
            rows.append(
                (
                    build_saved_north_star_request(fixture, week.week_id),
                    SavedExperimentRecord.model_validate(by_week[week.week_id]),
                )
            )
    return rows


def test_every_export_matches_completed_experiment_without_provider_calls(
    replay_records, monkeypatch
):
    async def forbidden(*args, **kwargs):
        pytest.fail("Saved replay cannot call a provider")

    monkeypatch.setattr(BudgetedProvider, "complete", forbidden)
    raw = (ROOT / EXPERIMENT_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPERIMENT_SHA256
    original = json.loads(raw)["execution"]
    assert len(replay_records) == 27
    assert len({request.owner_id for request, _ in replay_records}) == 5
    cards = 0
    for request, record in replay_records:
        assert validate_saved_record(record, request) == record
        evidence = record.experiment
        assert evidence.source_sha256 == hashlib.sha256(raw).hexdigest()
        output = original["variants"]["full_history"][evidence.case_id]
        assert evidence.output == output
        assert evidence.receipts == [
            original["requests"][key] for key in output["request_hashes"]
        ]
        assert (
            record.selected.model_dump(mode="json") if record.selected else None
        ) == output["selected"]
        cards += record.selected is not None
    assert cards > 0


@pytest.mark.parametrize(
    ("owner_id", "first_status", "first_reason"),
    [
        ("a24b8d8f", "complete", "no_supportive_source"),
        ("961a4e3f", "not_eligible", "no_eligible_writing"),
    ],
)
def test_replacement_replays_preserve_first_week_omissions_and_later_cards(
    replay_records, owner_id, first_status, first_reason
):
    weeks = sorted(
        (
            (request, record)
            for request, record in replay_records
            if request.owner_id == owner_id
        ),
        key=lambda row: row[0].week_start,
    )
    assert len(weeks) == 5
    first = weeks[0][1]
    assert (first.status, first.reason, first.selected) == (
        first_status,
        first_reason,
        None,
    )
    assert len(first.experiment.receipts) == (1 if owner_id == "a24b8d8f" else 0)
    for request, record in weeks[1:]:
        assert record.status == "complete"
        assert record.selected is not None
        assert validate_saved_record(record, request) == record


@pytest.mark.parametrize(
    "mutation",
    ["quote", "source", "receipt", "state", "model", "source_hash", "source_path"],
)
def test_replay_rejects_changed_selection_sources_or_provenance(
    replay_records, mutation
):
    request, record = next(row for row in replay_records if row[1].selected)
    payload = copy.deepcopy(record.model_dump(mode="json"))
    if mutation == "quote":
        payload["selected"]["evidence_quote"] = "A quotation that was never written."
    elif mutation == "source":
        payload["experiment"]["case"]["values"][0]["sources"][0]["journal_entry"] += (
            "changed"
        )
    elif mutation == "receipt":
        payload["experiment"]["receipts"][0]["request_hash"] = "0" * 64
    elif mutation == "state":
        payload["experiment"]["case"]["weekly_state"] = "insufficient_evidence"
    elif mutation == "model":
        payload["experiment"]["policy"]["runtime"]["reasoning_effort"] = "xhigh"
    elif mutation == "source_hash":
        payload["experiment"]["source_sha256"] = "0" * 64
    else:
        payload["experiment"]["source_path"] = "superseded-experiment.json"
    with pytest.raises(ValueError):
        validate_saved_record(SavedExperimentRecord.model_validate(payload), request)


def test_missing_cohort_case_remains_explicitly_unassessed(replay_records):
    request, record = replay_records[0]
    evidence = record.experiment.model_dump(mode="json")
    evidence.update(case=None, output=None, receipts=[])
    projected = project_record(
        request,
        ExperimentEvidence.model_validate(evidence),
        created_at=record.created_at,
    )
    assert projected.status == "pending"
    assert projected.reason == "not_in_completed_experiment"
    assert projected.selected is None


def test_replay_record_cannot_be_rebound_to_other_profile(replay_records):
    request, record = replay_records[0]
    changed = record.model_copy(update={"profile_ref": "another-profile"})
    with pytest.raises(ValueError, match="projection differs"):
        validate_saved_record(changed, request)


def test_experiment_source_verification_supports_compact_production(tmp_path):
    verify_experiment_source(ROOT)
    verify_experiment_source(tmp_path)


def test_changed_local_experiment_invalidates_cached_source_hash(tmp_path, monkeypatch):
    import src.demo.north_star_replay as replay

    source = tmp_path / EXPERIMENT_PATH
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original")
    monkeypatch.setattr(
        replay, "EXPERIMENT_SHA256", hashlib.sha256(b"original").hexdigest()
    )
    verify_experiment_source(tmp_path)
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source hash differs"):
        verify_experiment_source(tmp_path)


def test_export_rejects_changed_source_before_writing_records(tmp_path):
    source = tmp_path / EXPERIMENT_PATH
    source.parent.mkdir(parents=True)
    source.write_text("{}")
    with pytest.raises(ValueError, match="source hash differs"):
        export_records(tmp_path)
    assert not (tmp_path / EXPORT_PATH).exists()
