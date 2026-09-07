"""Offline diagnostic denominators and evidence identities remain explicit."""

import json
from copy import deepcopy

import pytest

from scripts.experiments.north_star_phase0b_diagnostics import (
    HISTORICAL,
    PHASE0A,
    SCRIPT,
    derive_diagnostics,
    digest_bytes,
    load_verified_inputs,
    validate_cases,
    write_diagnostics,
)


def fixture_case(case_id, decisions, *, text="I repaired my brakes.", shown=False):
    sources = [
        {
            "entry_id": f"persona:entry:{i}",
            "journal_entry": text,
            "nudge_response": None,
        }
        for i in range(len(decisions))
    ]
    case = {
        "case_id": case_id,
        "core_value": "security",
        "value": {"user_phrase": "Being safe", "definition": "Safety and stability."},
        "all_eligible_sources_in_retrieval_order": sources,
        "runtime_entry_ids": [s["entry_id"] for s in sources[:3]],
    }
    rows = [
        {
            "entry_id": s["entry_id"],
            "decision": decision,
            "reason_code": {
                "supportive": "observable_choice",
                "abstain": "ambiguous",
                "not_supportive": "wrong_value",
            }[decision],
        }
        for s, decision in zip(sources, decisions, strict=True)
    ]
    result = {
        "case_id": case_id,
        "reference": {"results": rows} if sources else None,
        "selected": {"entry_id": sources[0]["entry_id"]} if shown else None,
        "attempts": [],
    }
    return case, result


def analyze(pairs):
    return derive_diagnostics(
        {"cases": [c for c, _ in pairs]}, {"cases": [r for _, r in pairs]}
    )


def test_omission_strata_exclude_empty_and_missing_reference():
    missing = fixture_case("missing", ["not_supportive"])
    missing[1]["reference"] = None
    result = analyze(
        [
            fixture_case("empty", []),
            missing,
            fixture_case("rejected", ["not_supportive"]),
            fixture_case("all_ambiguous", ["abstain"], shown=True),
            fixture_case("mixed", ["abstain", "not_supportive"]),
            fixture_case("positive", ["supportive"]),
        ]
    )
    assert result["counts"]["structurally_empty_histories"] == 1
    assert result["counts"]["nonempty_histories_without_reference"] == 1
    assert result["counts"]["reference_positive_histories"] == 1
    strata = result["no_reference_supportive_strata"]
    assert strata["all_not_supportive"]["correct_omission"]["denominator"] == 1
    assert strata["all_abstain"]["correct_omission"]["numerator"] == 0
    assert strata["includes_abstain"]["correct_omission"]["numerator"] == 1
    assert len(result["no_reference_supportive_histories"]) == 3


def test_retrieval_workload_counts_sources_but_recall_counts_histories():
    result = analyze(
        [
            fixture_case("empty", []),
            fixture_case("short_negative", ["not_supportive"]),
            fixture_case(
                "positive_at_five", ["not_supportive"] * 4 + ["supportive"] * 2
            ),
        ]
    )
    workloads = {r["k"]: r for r in result["retrieval_workloads"]}
    assert workloads[3]["source_decisions"] == 4
    assert workloads[3]["task_reference_retrieval_recall"]["numerator"] == 0
    assert workloads[5]["source_decisions"] == 6
    assert workloads[5]["task_reference_retrieval_recall"]["denominator"] == 1
    assert workloads[7]["task_reference_retrieval_recall"]["numerator"] == 1
    assert result["retrieval_misses_at_3"][0]["reference_supportive_source_ranks"] == [
        {"entry_id": "persona:entry:4", "rank": 5},
        {"entry_id": "persona:entry:5", "rank": 6},
    ]


@pytest.mark.parametrize("changed", [None, "text", "definition", "phrase", "value"])
def test_reference_consistency_requires_identical_source_and_requested_value(changed):
    first = fixture_case("first", ["supportive"])
    second = fixture_case("second", ["not_supportive"])
    if changed == "text":
        second[0]["all_eligible_sources_in_retrieval_order"][0]["journal_entry"] += (
            " Later."
        )
    elif changed == "definition":
        second[0]["value"]["definition"] = "A different definition."
    elif changed == "phrase":
        second[0]["value"]["user_phrase"] = "Feeling calm"
    elif changed == "value":
        second[0]["core_value"] = "tradition"
    conflicts = analyze([first, second])[
        "repeated_identical_source_reference_conflicts"
    ]
    assert len(conflicts) == (1 if changed is None else 0)
    if conflicts:
        assert conflicts[0]["changes_supportive_status"]
        assert conflicts[0]["identity"]["source"]["entry_id"] == "persona:entry:0"


def test_reason_only_change_is_separate_from_supportive_status_change():
    first = fixture_case("first", ["not_supportive"])
    second = fixture_case("second", ["abstain"])
    conflicts = analyze([first, second])[
        "repeated_identical_source_reference_conflicts"
    ]
    assert len(conflicts) == 1
    assert not conflicts[0]["changes_supportive_status"]


def valid_inputs():
    case, result = fixture_case("persona:security:episode_01", ["supportive"])
    episode = {
        "episode_id": case["case_id"],
        "persona_id": "persona",
        "dimension": "security",
    }
    case["episode"] = episode
    source = case["all_eligible_sources_in_retrieval_order"][0]
    result.update(
        {
            "core_value": "security",
            "eligible_sources": 1,
            "reference_valid_ids": [source["entry_id"]],
            "reference_no_example": False,
        }
    )
    result["reference"].update(
        {
            "schema_version": "north-star-moment-review-v1",
            "core_value": "security",
        }
    )
    result["reference"]["results"][0].update(
        {
            "quote_source": "journal_entry",
            "evidence_quote": source["journal_entry"],
        }
    )
    retrieval = {
        "selected_k": 3,
        "cohort": {
            "development_persona_ids": ["persona"],
            "reserved_persona_ids": ["reserved"],
        },
        "config": {"queries": {"security": case["value"]}},
        "cases": [
            {
                "episode": episode,
                "ranking": [
                    {
                        "entry_id": source["entry_id"],
                        "t_index": 0,
                        "source_sha256": digest_bytes(source["journal_entry"].encode()),
                    }
                ],
            }
        ],
    }
    return (
        {"case_count": 1, "cases": [case]},
        {"cases": [result]},
        {"retrieval": retrieval},
    )


@pytest.mark.parametrize("change", ["duplicate", "reserved", "source", "membership"])
def test_validation_rejects_changed_or_wrong_membership_evidence(change):
    manifest, report, retrieval = valid_inputs()
    validate_cases(manifest, report, retrieval)
    if change == "duplicate":
        report["cases"].append(deepcopy(report["cases"][0]))
    elif change == "reserved":
        retrieval["retrieval"]["cohort"]["reserved_persona_ids"].append("persona")
    elif change == "source":
        manifest["cases"][0]["all_eligible_sources_in_retrieval_order"][0][
            "journal_entry"
        ] += " Changed."
    else:
        report["cases"][0]["reference"]["results"][0]["entry_id"] = "outsider:entry:0"
    with pytest.raises(ValueError):
        validate_cases(manifest, report, retrieval)


def frozen_files(root):
    def save(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return digest_bytes(path.read_bytes())

    source_hashes = {
        path: save(path, {})
        for path in [
            "src/north_star/provider.py",
            "scripts/experiments/north_star_phase0b.py",
        ]
    }
    retrieval_path = PHASE0A / "retrieval.json"
    manifest_hash = save(
        HISTORICAL / "manifest.json",
        {
            "source_hashes": {str(retrieval_path): save(retrieval_path, {})},
        },
    )
    source_hashes[str(HISTORICAL / "manifest.json")] = manifest_hash
    save(HISTORICAL / "execution_freeze.json", {"source_hashes": source_hashes})
    report_path = HISTORICAL / "report.json"
    report_hash = save(
        report_path,
        {
            "manifest_sha256": manifest_hash,
            "provider_sha256": source_hashes["src/north_star/provider.py"],
            "runner_sha256": source_hashes["scripts/experiments/north_star_phase0b.py"],
        },
    )
    save(HISTORICAL / "validation.json", {"hashes": {str(report_path): report_hash}})
    save(SCRIPT, {})


@pytest.mark.parametrize(
    "relative",
    [
        HISTORICAL / "manifest.json",
        HISTORICAL / "report.json",
        PHASE0A / "retrieval.json",
    ],
)
def test_input_hash_chain_rejects_mutation_before_analysis(tmp_path, relative):
    frozen_files(tmp_path)
    load_verified_inputs(tmp_path)
    (tmp_path / relative).write_text("{}")
    # Retrieval was an empty object already; change bytes even if the object is equal.
    if relative.parent == PHASE0A:
        (tmp_path / relative).write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_inputs(tmp_path)


def test_output_cannot_target_historical_directory_or_redirect_there(tmp_path):
    with pytest.raises(ValueError, match="historical evidence"):
        write_diagnostics(tmp_path / HISTORICAL, tmp_path)
    output = tmp_path / "derived"
    output.mkdir()
    (output / "diagnostics.json").symlink_to(tmp_path / HISTORICAL / "report.json")
    with pytest.raises(ValueError, match="historical evidence"):
        write_diagnostics(output, tmp_path)
