"""Reference-priority, adjudication, and paired cluster-metric regressions."""

from copy import deepcopy

import pytest

from scripts.experiments import nsm_evaluation as evaluation

VALUE = "benevolence"
ACTION = "I called Mum."
OUTCOME = "I felt relieved."


def case(*, case_id="case", persona="person", state="no_active_drift"):
    sources = [
        {
            "entry_id": "new",
            "journal_entry": f"{ACTION} {OUTCOME}",
            "nudge_response": None,
        },
        {"entry_id": "old", "journal_entry": ACTION, "nudge_response": None},
    ]
    metadata = {
        "new": {"t_index": 2, "date": "2026-01-06"},
        "old": {"t_index": 1, "date": "2026-01-04"},
    }
    return {
        "case_id": case_id,
        "persona_id": persona,
        "split": "development",
        "weekly_state": state,
        "week_start": "2026-01-05",
        "week_end": "2026-01-11",
        "values": [
            {"core_value": VALUE, "sources": sources, "source_metadata": metadata}
        ]
        if state != "insufficient_evidence"
        else [],
    }


def batch(new="observable_choice", old="wrong_value"):
    return {
        VALUE: {
            "core_value": VALUE,
            "results": [
                {
                    "entry_id": entry,
                    "reason_code": reason,
                    "quote_source": "journal_entry"
                    if reason == "observable_choice"
                    else None,
                    "evidence_quote": ACTION if reason == "observable_choice" else "",
                }
                for entry, reason in (("new", new), ("old", old))
            ],
        }
    }


def quote(output, source_reason="observable_choice", quote_reason="supported_action"):
    selected = output["selected"]
    return {
        "schema_version": "north-star-candidate-assessment-v1",
        "core_value": output["core_value"],
        "entry_id": selected["entry_id"],
        "quote_source": selected["quote_source"],
        "source_reason": source_reason,
        "quote_reason": quote_reason,
        "action_assessment": "The writer called.",
        "value_assessment": "A supportive call.",
        "conflict_assessment": "None reported.",
        "quote_assessment": "Describes the call.",
        "evaluated_quote": selected["evidence_quote"],
    }


def outputs(item, reference=None):
    reference = reference or batch()
    output = evaluation.select_card(item, reference)
    return {variant: deepcopy(output) for variant in evaluation.VARIANTS}


def grade(item, *, card=True, opportunity=True, incorrect=False):
    reference = batch() if opportunity else batch("wrong_value")
    runtime = outputs(item)
    quotes = {}
    for variant, output in runtime.items():
        if not card:
            output["selected"] = None
            output["core_value"] = None
            output["mode"] = None
            continue
        if not opportunity:
            quotes[variant] = quote(output, "wrong_value", "source_not_supportive")
        elif incorrect:
            output["selected"]["evidence_quote"] = OUTCOME
            quotes[variant] = quote(output, quote_reason="missing_action")
        else:
            quotes[variant] = quote(output)
    return evaluation.grade_case(item, runtime, reference, quotes)


def test_documented_ten_week_selection_error_matrix():
    scenarios = (
        [(True, True, False)] * 3
        + [(True, True, True)]
        + [(False, True, False)] * 2
        + [(True, False, True)]
        + [(False, False, False)] * 3
    )
    grades = [
        grade(
            case(case_id=str(i), persona=str(i)),
            card=card,
            opportunity=opportunity,
            incorrect=incorrect,
        )
        for i, (card, opportunity, incorrect) in enumerate(scenarios)
    ]
    result = evaluation.summarize(grades, n_resamples=100)["splits"]["development"]
    for variant in evaluation.VARIANTS:
        row = result["variants"][variant]
        assert row["selection_errors"] == {"included": 10, "tp": 3, "fp": 2, "fn": 3}
        assert row["metrics"]["card_precision"]["value"] == 0.6
        assert row["metrics"]["opportunity_recall"]["value"] == 0.5
        assert row["metrics"]["correct_omission"]["value"] == 0.75


def test_shortlist_supportive_hit_can_miss_full_history_priority():
    item = case()
    reference = batch(old="observable_choice")
    runtime = outputs(item, reference)
    runtime["nomic"] = evaluation.select_card(item, reference, {VALUE: ["old"]})
    assert runtime["nomic"]["mode"] == "reminder"
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    result = evaluation.grade_case(item, runtime, reference, quotes)
    nomic = result["variants"]["nomic"]
    assert nomic["metrics"]["retrieval_hit_rate_at_3"]["numerator"] == 1
    assert nomic["correct_card"] is False
    assert nomic["selection_errors"] == {"included": True, "tp": 0, "fp": 1, "fn": 1}
    assert result["variants"]["full_history"]["correct_card"] is True


def test_current_week_then_profile_then_source_priority():
    item = case()
    first = item["values"][0]
    second = deepcopy(first)
    second["core_value"] = "security"
    item["values"].append(second)
    reference = batch(new="wrong_value", old="observable_choice")
    second_batch = deepcopy(batch()[VALUE])
    second_batch["core_value"] = "security"
    reference["security"] = second_batch
    assert evaluation.select_card(item, reference)["core_value"] == "security"
    reference[VALUE]["results"][0].update(
        reason_code="observable_choice",
        quote_source="journal_entry",
        evidence_quote=ACTION,
    )
    selected = evaluation.select_card(item, reference)
    assert selected["core_value"] == VALUE
    assert selected["selected"]["entry_id"] == "new"


def test_active_drift_uses_supplied_selected_value_and_reflection():
    item = case(state="active_drift")
    assert evaluation.select_card(item, batch())["mode"] == "reflection"
    assert evaluation.select_card(item, batch("wrong_value"))["selected"] is None


def test_runtime_failure_is_a_missed_known_opportunity_without_fp():
    item = case()
    runtime = outputs(item)
    runtime["nomic"].update(status="failed", reason="provider_timeout", selected=None)
    result = evaluation.grade_case(
        item, runtime, batch(), {"full_history": quote(runtime["full_history"])}
    )
    nomic = result["variants"]["nomic"]
    assert nomic["metrics"]["opportunity_recall"]["denominator"] == 1
    assert nomic["selection_errors"] == {"included": True, "tp": 0, "fp": 0, "fn": 1}
    assert nomic["operational_failure"] == "provider_timeout"


def test_insufficient_evidence_is_correct_omission_not_recall_opportunity():
    item = case(state="insufficient_evidence")
    result = evaluation.grade_case(item, outputs(item), {}, {})
    assert result["reference"]["opportunity"] is False
    for row in result["variants"].values():
        assert row["metrics"]["opportunity_recall"]["denominator"] == 0
        assert row["metrics"]["correct_omission"]["numerator"] == 1
    summary = evaluation.summarize([result], n_resamples=10)["splits"]["development"]
    assert summary["variants"]["nomic"]["metrics"]["card_precision"]["display"] == "N/A"
    assert (
        summary["uncertainty"]["intervals"]["nomic"]["card_precision"][
            "undefined_resamples"
        ]
        == 10
    )


def test_supportive_source_with_bad_quote_does_not_request_recheck():
    item = case()
    runtime = outputs(item)
    quotes = {
        variant: quote(output, quote_reason="missing_action")
        for variant, output in runtime.items()
    }
    assert evaluation.contradictions(item, runtime, batch(), quotes) == []


def test_shared_source_recheck_changes_both_reference_and_grades():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    reference = batch("wrong_value")
    result = evaluation.grade_case(item, runtime, reference, quotes, quotes)
    assert result["reference"]["opportunity"] is True
    assert len(result["contradictions"]) == 2
    assert all(row["correct_card"] for row in result["variants"].values())


def test_conflicting_shared_source_rechecks_exclude_both_variants():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"]),
        "full_history": quote(
            runtime["full_history"], "same_value_conflict", "same_value_conflict"
        ),
    }
    result = evaluation.grade_case(item, runtime, batch("wrong_value"), quotes, quotes)
    assert result["reference"]["opportunity"] is None
    assert (
        result["reference"]["unresolved_sources"][0]["unresolved_reason"]
        == "conflicting_shared_source_rechecks"
    )
    for row in result["variants"].values():
        assert row["metrics"]["card_precision"]["excluded"]
        assert row["metrics"]["opportunity_recall"]["excluded"]
        assert not row["selection_errors"]["included"]


def test_failed_recheck_does_not_keep_favourable_original_assessment():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"], "wrong_value", "source_not_supportive"),
        "full_history": quote(runtime["full_history"]),
    }
    result = evaluation.grade_case(item, runtime, batch(), quotes)
    assert result["reference"]["opportunity"] is None
    assert all(
        row["metrics"]["opportunity_recall"]["excluded"]
        for row in result["variants"].values()
    )


def test_unknown_lower_priority_source_does_not_hide_known_card():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    result = evaluation.grade_case(item, runtime, batch(old="ambiguous"), quotes)
    assert result["reference"]["opportunity"] is True
    assert result["reference"]["priority_resolved"] is True
    assert all(row["correct_card"] for row in result["variants"].values())


def test_missing_quote_assessment_has_identical_paired_exclusion():
    item = case()
    runtime = outputs(item)
    result = evaluation.grade_case(
        item, runtime, batch(), {"full_history": quote(runtime["full_history"])}
    )
    for row in result["variants"].values():
        assert row["metrics"]["card_precision"]["excluded"]
        assert row["metrics"]["opportunity_recall"]["excluded"]
        assert not row["metrics"]["selection_rule_correctness"]["excluded"]


def test_whole_persona_bootstrap_retains_correlated_weeks_and_pairing():
    first = grade(case(case_id="a1", persona="a"))
    second = grade(case(case_id="b1", persona="b"), card=False)
    once = evaluation.summarize([first, second], n_resamples=1000)
    duplicated = [first, second, deepcopy(first), deepcopy(second)]
    duplicated[2]["case_id"] = "a2"
    duplicated[3]["case_id"] = "b2"
    twice = evaluation.summarize(duplicated, n_resamples=1000)
    a = once["splits"]["development"]["uncertainty"]
    b = twice["splits"]["development"]["uncertainty"]
    assert a == b
    difference = a["intervals"]["nomic_minus_full_history"]["opportunity_recall"]
    assert difference["low"] == difference["high"] == 0


def test_runtime_measurements_include_all_cases_and_report_differences():
    item = grade(case())
    measurements = {
        "case": {
            "nomic": {"cost_usd": 0.2, "latency_seconds": 3},
            "full_history": {"cost_usd": 0.3, "latency_seconds": 4},
        }
    }
    result = evaluation.summarize([item], measurements, n_resamples=10)
    differences = result["splits"]["development"]["nomic_minus_full_history"]
    assert differences["cost_usd"] == pytest.approx(-0.1)
    assert differences["mean_latency_seconds"] == -1
    with pytest.raises(ValueError, match="cover every case"):
        evaluation.summarize([item], {}, n_resamples=10)


def test_consistency_retains_disagreement_and_failed_reviews():
    item = case()
    runtime = outputs(item)
    primary = {
        "source_reviews": batch(),
        "quote_reviews": {"nomic": quote(runtime["nomic"])},
        "grade": grade(item),
    }
    repeat = deepcopy(primary)
    repeat["source_reviews"] = batch("wrong_value")
    result = evaluation.consistency_report(
        [{"case_id": "case", "persona_id": "person", "reviews": [primary, repeat, {}]}]
    )
    reasons = result["metrics"]["source_reason"]
    assert reasons["incomplete"] == 2
    assert reasons["pairwise_numerator"] == 1
    assert reasons["pairwise_denominator"] == 2


def test_ambiguous_exact_quote_recheck_keeps_opportunity_but_excludes_card_metrics():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    rechecks = deepcopy(quotes)
    rechecks["nomic"]["quote_reason"] = "insufficient_context"
    result = evaluation.grade_case(
        item, runtime, batch("wrong_value"), quotes, rechecks
    )
    assert result["reference"]["opportunity"] is True
    assert result["reference"]["priority_resolved"] is True
    assert result["variants"]["nomic"]["correct_card"] is None
    for row in result["variants"].values():
        assert row["metrics"]["card_precision"]["excluded"]
        assert row["metrics"]["opportunity_recall"]["excluded"]
        assert not row["metrics"]["selection_rule_correctness"]["excluded"]


def test_invalid_recheck_cannot_revise_shared_source_to_supportive():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    rechecks = deepcopy(quotes)
    rechecks["nomic"]["evaluated_quote"] = "A different quote."
    result = evaluation.grade_case(
        item, runtime, batch("wrong_value"), quotes, rechecks
    )
    assert result["reference"]["opportunity"] is None
    assert (
        result["reference"]["unresolved_sources"][0]["unresolved_reason"]
        == "failed_contradiction_recheck"
    )


def test_nonexact_quote_is_incorrect_even_if_quotation_review_failed():
    item = case()
    runtime = outputs(item)
    runtime["nomic"]["selected"]["evidence_quote"] = "I never wrote this."
    result = evaluation.grade_case(
        item, runtime, batch(), {"full_history": quote(runtime["full_history"])}
    )
    assert result["variants"]["nomic"]["correct_card"] is False
    assert not result["variants"]["nomic"]["metrics"]["opportunity_recall"]["excluded"]


def test_consistency_reports_expected_coordinates_when_all_passes_fail():
    item = case()
    result = evaluation.consistency_report(
        [
            {
                "case_id": item["case_id"],
                "persona_id": item["persona_id"],
                "case": item,
                "displayed_variants": ["nomic"],
                "reviews": [{}, {}, {}],
            }
        ]
    )
    assert result["metrics"]["source_reason"]["coordinates"] == 2
    assert result["metrics"]["source_reason"]["incomplete"] == 2
    assert result["metrics"]["quotation_acceptance"]["incomplete"] == 1
    assert result["failures"]["empty_review_passes"] == 3


def test_missing_primary_source_cannot_authorize_or_adopt_contradiction_recheck():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    assert evaluation.contradictions(item, runtime, {}, quotes) == []
    result = evaluation.grade_case(item, runtime, {}, quotes, quotes)
    assert result["contradictions"] == []
    assert result["reference"]["opportunity"] is None
    assert all(
        row["origin"] == "primary_source_review"
        for row in result["reference"]["rulings"]
    )
    assert all(
        row["metrics"]["opportunity_recall"]["excluded"]
        for row in result["variants"].values()
    )


def test_present_ambiguous_source_judgment_can_still_be_rechecked():
    item = case()
    runtime = outputs(item)
    quotes = {variant: quote(output) for variant, output in runtime.items()}
    source = batch("ambiguous")
    assert len(evaluation.contradictions(item, runtime, source, quotes)) == 2
    result = evaluation.grade_case(item, runtime, source, quotes, quotes)
    assert result["reference"]["opportunity"] is True
    assert all(row["correct_card"] is True for row in result["variants"].values())


@pytest.mark.parametrize(
    "old_source,old_quote",
    [
        ("wrong_value", "source_not_supportive"),
        ("same_value_conflict", "same_value_conflict"),
    ],
)
def test_shared_source_reversal_cannot_keep_superseded_quote_exclusion(
    old_source, old_quote
):
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"]),
        "full_history": quote(runtime["full_history"], old_source, old_quote),
    }
    result = evaluation.grade_case(
        item, runtime, batch(old_source), quotes, {"nomic": quote(runtime["nomic"])}
    )
    assert result["reference"]["opportunity"] is True
    assert result["reference"]["priority_resolved"] is True
    assert result["variants"]["nomic"]["correct_card"] is True
    full = result["variants"]["full_history"]
    assert full["correct_card"] is None
    assert full["quotation_errors"] == [
        "quotation_source_exclusion_unresolved_after_shared_recheck"
    ]
    for row in result["variants"].values():
        for name in ("card_precision", "opportunity_recall"):
            assert row["metrics"][name]["excluded"]
            assert (
                row["metrics"][name]["exclusion_reason"] == full["quotation_errors"][0]
            )
        assert not row["metrics"]["selection_rule_correctness"]["excluded"]
    assert not result["variants"]["nomic"]["metrics"]["retrieval_hit_rate_at_3"][
        "excluded"
    ]


def test_shared_recheck_preserves_an_independent_quote_only_rejection():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"], "wrong_value", "source_not_supportive"),
        "full_history": quote(runtime["full_history"], quote_reason="missing_action"),
    }
    result = evaluation.grade_case(
        item, runtime, batch(), quotes, {"nomic": quote(runtime["nomic"])}
    )
    assert result["reference"]["opportunity"] is True
    assert result["variants"]["full_history"]["correct_card"] is False
    assert result["variants"]["full_history"]["quotation_errors"] == []
    assert all(
        not row["metrics"]["card_precision"]["excluded"]
        for row in result["variants"].values()
    )


def test_final_unambiguous_source_exclusion_remains_decisive_for_both_cards():
    item = case()
    runtime = outputs(item)
    rejected = quote(runtime["nomic"], "wrong_value", "source_not_supportive")
    quotes = {"nomic": rejected, "full_history": quote(runtime["full_history"])}
    result = evaluation.grade_case(item, runtime, batch(), quotes, {"nomic": rejected})
    assert result["reference"]["opportunity"] is False
    assert all(row["correct_card"] is False for row in result["variants"].values())
    assert all(
        not row["metrics"]["card_precision"]["excluded"]
        for row in result["variants"].values()
    )


def test_failed_shared_recheck_cannot_leave_a_source_gated_quote_decisive():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"]),
        "full_history": quote(
            runtime["full_history"], "wrong_value", "source_not_supportive"
        ),
    }
    result = evaluation.grade_case(item, runtime, batch("wrong_value"), quotes)
    assert result["reference"]["opportunity"] is None
    full = result["variants"]["full_history"]
    assert full["correct_card"] is None
    assert full["quotation_errors"] == [
        "quotation_source_exclusion_unresolved_after_shared_recheck"
    ]


def test_stale_source_gate_does_not_erase_a_mechanically_nonexact_quote_failure():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"]),
        "full_history": quote(
            runtime["full_history"], "wrong_value", "source_not_supportive"
        ),
    }
    runtime["full_history"]["selected"]["evidence_quote"] = "Never written."
    result = evaluation.grade_case(
        item, runtime, batch("wrong_value"), quotes, {"nomic": quote(runtime["nomic"])}
    )
    assert result["variants"]["full_history"]["correct_card"] is False
    assert result["variants"]["full_history"]["quotation_errors"] == [
        "nonexact_or_forbidden_quotation"
    ]


def test_unfavourable_recheck_does_not_supply_another_variants_quote_grade():
    item = case()
    runtime = outputs(item)
    quotes = {
        "nomic": quote(runtime["nomic"]),
        "full_history": quote(
            runtime["full_history"], "wrong_value", "source_not_supportive"
        ),
    }
    result = evaluation.grade_case(
        item,
        runtime,
        batch("wrong_value"),
        quotes,
        {"nomic": quote(runtime["nomic"], quote_reason="missing_action")},
    )
    assert result["reference"]["opportunity"] is True
    assert result["variants"]["nomic"]["correct_card"] is False
    assert result["variants"]["full_history"]["correct_card"] is None
    assert all(
        row["metrics"]["card_precision"]["excluded"]
        for row in result["variants"].values()
    )
