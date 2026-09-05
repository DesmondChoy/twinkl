"""Feasibility criteria remain meaningful for omissions and failed references."""

from scripts.experiments.north_star_phase0b import summarize


def case(case_id, *, shown=False, correct=True, no_example=False, sources=1):
    return {
        "case_id": case_id,
        "status": "completed",
        "eligible_sources": sources,
        "selected": {"entry_id": "source"} if shown else None,
        "incorrect_displayed": shown and not correct,
        "reference_no_example": no_example,
        "reference_valid_ids": [] if no_example else ["source"],
        "task_retrieval_hit": not no_example,
        "retrieval_only_selected": bool(sources),
        "retrieval_only_correct": not no_example,
    }


def attempts(failed=0):
    return [
        {
            "provider": provider,
            "status": "failed" if i < failed else "completed",
            "calculated_cost_usd": 0.001,
        }
        for provider in ("openai", "gemini")
        for i in range(20)
    ]


def test_gate_requires_true_no_example_case_not_only_empty_history():
    results = [
        case("8f83c818:universalism:one", shown=True),
        case("empty", sources=0, no_example=True),
    ]
    assert not summarize(results, attempts())["gate_passed"]
    results.append(case("related_phrase_only", no_example=True))
    assert summarize(results, attempts())["gate_passed"]


def test_zero_cards_is_undefined_precision_and_fails_gate():
    summary = summarize([case("omitted", no_example=True)], attempts())
    assert summary["precision"]["rate"] is None
    assert not summary["gate_passed"]


def test_one_incorrect_quote_fails_even_when_saved_demo_is_correct():
    results = [
        case("8f83c818:universalism:one", shown=True),
        case("wrong_quote", shown=True, correct=False),
        case("no_example", no_example=True),
    ]
    assert not summarize(results, attempts())["gate_passed"]


def test_failures_before_successful_retries_count_toward_provider_rate():
    results = [
        case("8f83c818:universalism:one", shown=True),
        case("no_example", no_example=True),
    ]
    assert summarize(results, attempts(failed=1))["gate_passed"]
    assert not summarize(results, attempts(failed=2))["gate_passed"]


def test_missing_reference_does_not_count_as_correct_omission():
    result = case("8f83c818:universalism:one", shown=True)
    failed = {
        "case_id": "failure",
        "status": "failed",
        "eligible_sources": 1,
        "selected": None,
    }
    summary = summarize([result, failed], attempts())
    assert summary["correct_no_card"]["denominator"] == 0
    assert summary["failed_cases"] == 1
    assert not summary["gate_passed"]
