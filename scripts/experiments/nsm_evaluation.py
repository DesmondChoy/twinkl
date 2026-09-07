"""Deterministic grading for the frozen two-variant North Star Moment study.

All semantic inputs are fresh AI judgments supplied by the caller. This module
makes no provider calls, and never infers support from cohort or Drift labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.north_star import assessment
from src.north_star.review import ReviewValidationError, SourceEntry

SEED = 20260906
VARIANTS = ("nomic", "full_history")
METRICS = (
    "card_precision",
    "opportunity_recall",
    "correct_omission",
    "selection_rule_correctness",
    "retrieval_hit_rate_at_3",
)
AMBIGUOUS_REASONS = {"ambiguous", "insufficient_text"}


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value) if isinstance(value, Mapping) else {}


def _sources(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for value in case["values"]:
        for source in value["sources"]:
            row = _dict(source)
            metadata = value.get("source_metadata", case.get("source_metadata", {}))
            row.update(metadata.get(row["entry_id"], {}))
            row["core_value"] = value["core_value"]
            rows.append(row)
    return rows


def _key(source: Mapping[str, Any]) -> tuple[str, str]:
    return source["core_value"], source["entry_id"]


def _ordered(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _sources(case)
    if case["weekly_state"] == "insufficient_evidence":
        return []
    if case["weekly_state"] != "active_drift":
        current = [row for row in rows if row["date"] >= case["week_start"]]
        history = [row for row in rows if row["date"] < case["week_start"]]
        return current + history
    return rows


def _source_rulings(
    case: dict[str, Any], source_reviews: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    rulings = {}
    for source in _sources(case):
        value, entry = _key(source)
        batch = _dict(source_reviews.get(value))
        found = [
            _dict(row)
            for row in batch.get("results", [])
            if _dict(row).get("entry_id") == entry
        ]
        row = found[0] if len(found) == 1 else {}
        reason = row.get("reason_code")
        rulings[value, entry] = {
            "core_value": value,
            "entry_id": entry,
            "reason_code": reason,
            "supportive": None
            if not reason or reason in AMBIGUOUS_REASONS
            else reason == "observable_choice",
            "origin": "primary_source_review",
            "unresolved_reason": "missing_source_assessment"
            if not reason
            else "ambiguous_source_assessment"
            if reason in AMBIGUOUS_REASONS
            else None,
            "assessment": row,
        }
    return rulings


def select_card(
    case: dict[str, Any],
    source_reviews: Mapping[str, Any],
    candidate_ids_by_value: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Select using product order; source-review abstentions simply omit support.

    Failed runtime calls should be represented by the runner as failed outputs,
    rather than passed here as missing assessments.
    """
    candidates = {
        value["core_value"]: [row["entry_id"] for row in value["sources"]]
        for value in case["values"]
    }
    if candidate_ids_by_value is not None:
        candidates = {value: list(ids) for value, ids in candidate_ids_by_value.items()}
    output: dict[str, Any] = {
        "status": "complete",
        "reason": "no_supportive_source",
        "selected": None,
        "core_value": None,
        "mode": None,
        "candidate_ids_by_value": candidates,
    }
    if case["weekly_state"] == "insufficient_evidence":
        output.update(status="not_eligible", reason="insufficient_evidence")
        return output
    rulings = _source_rulings(case, source_reviews)
    for source in _ordered(case):
        value, entry = _key(source)
        ruling = rulings[value, entry]
        if entry not in candidates.get(value, []) or not ruling["supportive"]:
            continue
        selected = {key: source[key] for key in ("entry_id", "t_index", "date")}
        selected.update(
            {
                key: ruling["assessment"][key]
                for key in ("quote_source", "evidence_quote")
            }
        )
        output.update(
            reason="supportive_action_selected",
            selected=selected,
            core_value=value,
            mode=_mode(case, source),
        )
        return output
    if not _sources(case):
        output.update(status="not_eligible", reason="no_eligible_writing")
    return output


def _mode(case: dict[str, Any], source: dict[str, Any]) -> str:
    if case["weekly_state"] == "active_drift":
        return "reflection"
    return "encouragement" if source["date"] >= case["week_start"] else "reminder"


def contradictions(
    case: dict[str, Any],
    outputs: Mapping[str, Any],
    source_reviews: Mapping[str, Any],
    quote_reviews: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Locate source-reason contradictions, not legitimate quote-only rejection."""
    rulings = _source_rulings(case, source_reviews)
    result = []
    for variant, raw_output in outputs.items():
        output = _dict(raw_output)
        selected = _dict(output.get("selected"))
        quote = _dict(quote_reviews.get(variant))
        if not selected or not quote:
            continue
        key = output.get("core_value", ""), selected["entry_id"]
        source = rulings.get(key)
        if (
            source
            and source["reason_code"] is not None
            and quote.get("source_reason") is not None
            and source["reason_code"] != quote["source_reason"]
        ):
            result.append(
                {
                    "variant": variant,
                    "core_value": key[0],
                    "entry_id": key[1],
                    "source_reason": source["reason_code"],
                    "quotation_source_reason": quote.get("source_reason"),
                }
            )
    return result


def _adjudicate(
    case: dict[str, Any],
    outputs: Mapping[str, Any],
    source_reviews: Mapping[str, Any],
    quote_reviews: Mapping[str, Any],
    rechecks: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], list]:
    rulings = _source_rulings(case, source_reviews)
    quotes = {variant: _dict(review) for variant, review in quote_reviews.items()}
    conflicts = contradictions(case, outputs, source_reviews, quote_reviews)
    changes: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for conflict in conflicts:
        variant = conflict["variant"]
        key = conflict["core_value"], conflict["entry_id"]
        fresh = _dict(rechecks.get(variant))
        _, errors = _quotation(case, _dict(outputs[variant]), fresh)
        if errors and not all(error.startswith("ambiguous_") for error in errors):
            fresh = {}
        changes[key].append((variant, fresh))
        quotes[variant] = fresh
    for key, proposals in changes.items():
        reasons = {row.get("source_reason") for _, row in proposals}
        unresolved = None
        if None in reasons:
            unresolved = "failed_contradiction_recheck"
        elif reasons & AMBIGUOUS_REASONS:
            unresolved = "ambiguous_contradiction_recheck"
        elif len(reasons) != 1:
            unresolved = "conflicting_shared_source_rechecks"
        reason = next(iter(reasons)) if len(reasons) == 1 else None
        rulings[key].update(
            reason_code=reason,
            supportive=None if unresolved else reason == "observable_choice",
            origin="contradiction_recheck",
            unresolved_reason=unresolved,
            rechecks=[
                {"variant": variant, "assessment": row} for variant, row in proposals
            ],
        )
    return rulings, quotes, conflicts


def _reference(
    case: dict[str, Any], rulings: Mapping[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    ordered = _ordered(case)
    supportive = [row for row in ordered if rulings[_key(row)]["supportive"] is True]
    unknown = [row for row in ordered if rulings[_key(row)]["supportive"] is None]
    opportunity = True if supportive else None if unknown else False
    preferred = None
    priority_resolved = True
    for row in ordered:
        support = rulings[_key(row)]["supportive"]
        if support is None:
            priority_resolved = False
            break
        if support:
            preferred = {"core_value": row["core_value"], "entry_id": row["entry_id"]}
            break
    return {
        "opportunity": opportunity,
        "preferred_source": preferred,
        "priority_resolved": priority_resolved,
        "supportive_source_count": len(supportive),
        "unresolved_sources": [rulings[_key(row)] for row in unknown],
        "rulings": list(rulings.values()),
    }


def _quotation(
    case: dict[str, Any], output: dict[str, Any], quote: dict[str, Any]
) -> tuple[bool | None, list[str]]:
    selected = _dict(output.get("selected"))
    if not selected:
        return False, []
    value = output.get("core_value", "")
    matches = [
        row for row in _sources(case) if _key(row) == (value, selected.get("entry_id"))
    ]
    if len(matches) != 1:
        return False, ["ineligible_source_or_core_value"]
    source = matches[0]
    evidence_source = SourceEntry(
        entry_id=source["entry_id"],
        journal_entry=source["journal_entry"],
        nudge_response=source.get("nudge_response"),
    )
    quote_source = selected.get("quote_source")
    if quote_source not in ("journal_entry", "nudge_response"):
        return False, ["invalid_quote_source"]
    try:
        assessment._validate_proposed_quote(
            core_value=value,
            source=evidence_source,
            quote_source=quote_source,
            evidence_quote=selected.get("evidence_quote", ""),
        )
    except (ReviewValidationError, ValueError, TypeError):
        return False, ["nonexact_or_forbidden_quotation"]
    if not quote:
        return None, ["missing_quotation_assessment"]
    try:
        validated = assessment.validate_candidate_review(
            quote,
            core_value=value,
            source=evidence_source,
            quote_source=quote_source,
            evidence_quote=selected.get("evidence_quote", ""),
        )
    except (ReviewValidationError, ValueError, TypeError):
        return None, ["invalid_quotation_assessment"]
    if validated.source_reason in AMBIGUOUS_REASONS:
        return None, ["ambiguous_quotation_assessment"]
    if validated.quote_reason == "insufficient_context":
        return None, ["ambiguous_exact_quotation"]
    return validated.accepted, []


def _rules(
    case: dict[str, Any], output: dict[str, Any], reference: dict[str, Any]
) -> tuple[bool | None, dict[str, bool | None]]:
    selected = _dict(output.get("selected"))
    if not selected:
        return False, {}
    matches = [
        source
        for source in _sources(case)
        if _key(source) == (output.get("core_value"), selected.get("entry_id"))
    ]
    checks: dict[str, bool | None] = {
        "eligible_core_value_and_source": len(matches) == 1,
        "eligible_weekly_state": case["weekly_state"] != "insufficient_evidence",
    }
    if matches:
        source = matches[0]
        checks.update(
            source_metadata=(
                selected.get("t_index") == source["t_index"]
                and selected.get("date") == source["date"]
            ),
            framing=output.get("mode") == _mode(case, source),
        )
    preferred = reference["preferred_source"]
    checks["full_history_priority"] = (
        {"core_value": output.get("core_value"), "entry_id": selected["entry_id"]}
        == preferred
        if reference["priority_resolved"]
        else None
    )
    valid = (
        False if False in checks.values() else None if None in checks.values() else True
    )
    return valid, checks


def _count(numerator: int = 0, denominator: int = 0, reason: str | None = None) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "excluded": reason is not None,
        "exclusion_reason": reason,
    }


def grade_case(
    case: dict[str, Any],
    runtime_outputs_by_variant: Mapping[str, Any],
    source_reviews: Mapping[str, Any],
    quote_reviews: Mapping[str, Any],
    rechecks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply shared-source rulings and identical paired metric exclusions."""
    if set(runtime_outputs_by_variant) != set(VARIANTS):
        raise ValueError(f"Expected variants {VARIANTS}")
    rulings, quotes, conflicts = _adjudicate(
        case, runtime_outputs_by_variant, source_reviews, quote_reviews, rechecks or {}
    )
    reference = _reference(case, rulings)
    opportunity = reference["opportunity"]
    graded: dict[str, Any] = {}
    for variant in VARIANTS:
        output = _dict(runtime_outputs_by_variant[variant])
        selected = _dict(output.get("selected"))
        displayed = bool(selected)
        quote_valid, quote_errors = _quotation(case, output, quotes.get(variant, {}))
        rule_valid, checks = _rules(case, output, reference)
        ruling = rulings.get(
            (output.get("core_value", ""), selected.get("entry_id", "")), {}
        )
        source_support = ruling.get("supportive") if displayed else False
        superseded_quote_exclusion = (
            quote_valid is False
            and not quote_errors
            and ruling.get("origin") == "contradiction_recheck"
            and source_support is not False
            and quotes.get(variant, {}).get("quote_reason")
            in {"source_not_supportive", "same_value_conflict"}
        )
        exclusion_reason = "quotation_source_exclusion_unresolved_after_shared_recheck"
        if superseded_quote_exclusion:
            # This source-gated rejection contains no independent quote-only ruling.
            quote_valid = None
            quote_errors = [exclusion_reason]
        validity = (quote_valid, rule_valid, source_support)
        correct = (
            False
            if False in validity or not displayed
            else (None if None in validity else True)
        )
        counts = {
            "card_precision": _count(int(correct is True), int(displayed))
            if correct is not None
            else _count(
                reason=exclusion_reason
                if superseded_quote_exclusion
                else "unresolved_displayed_card"
            ),
            "opportunity_recall": _count(int(correct is True), int(opportunity is True))
            if opportunity is not None and (not opportunity or correct is not None)
            else _count(
                reason=exclusion_reason
                if superseded_quote_exclusion
                else "unresolved_opportunity_or_card"
            ),
            "correct_omission": _count(
                int(not displayed and opportunity is False), int(opportunity is False)
            )
            if opportunity is not None
            else _count(reason="unresolved_opportunity"),
            "selection_rule_correctness": _count(
                int(rule_valid is True), int(displayed)
            )
            if rule_valid is not None
            else _count(reason="unresolved_reference_priority"),
        }
        if variant == "nomic":
            candidates = output.get("candidate_ids_by_value", {})
            included = [
                ruling["supportive"]
                for (value, entry), ruling in rulings.items()
                if entry in candidates.get(value, [])
            ]
            hit = True if True in included else None if None in included else False
            counts["retrieval_hit_rate_at_3"] = (
                _count(
                    int(hit is True and opportunity is True), int(opportunity is True)
                )
                if opportunity is not None and (not opportunity or hit is not None)
                else _count(reason="unresolved_opportunity_or_shortlist")
            )
        graded[variant] = {
            "displayed": displayed,
            "correct_card": correct,
            "quotation_valid": quote_valid,
            "quotation_errors": quote_errors,
            "selection_rules_valid": rule_valid,
            "rule_checks": checks,
            "operational_failure": output.get("reason")
            if output.get("status") == "failed"
            else None,
            "metrics": counts,
        }
    for metric in METRICS[:-1]:
        excluded = [
            graded[v]["metrics"][metric]["exclusion_reason"]
            for v in VARIANTS
            if graded[v]["metrics"][metric]["excluded"]
        ]
        if excluded:
            reason = ";".join(sorted(set(excluded)))
            for variant in VARIANTS:
                graded[variant]["metrics"][metric] = _count(reason=reason)
    for variant in VARIANTS:
        row = graded[variant]
        resolved = opportunity is not None and all(
            not row["metrics"][metric]["excluded"]
            for metric in ("card_precision", "opportunity_recall")
        )
        row["selection_errors"] = {
            "included": resolved,
            "tp": int(resolved and row["correct_card"] is True),
            "fp": int(resolved and row["displayed"] and row["correct_card"] is False),
            "fn": int(
                resolved and opportunity is True and row["correct_card"] is False
            ),
        }
    return {
        **{
            key: case[key] for key in ("case_id", "persona_id", "split", "weekly_state")
        },
        "reference": reference,
        "contradictions": conflicts,
        "variants": graded,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(grades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        rows = [grade["variants"][variant] for grade in grades]
        metrics = {}
        for metric in METRICS:
            values = [
                row["metrics"][metric] for row in rows if metric in row["metrics"]
            ]
            if not values:
                continue
            numerator = sum(row["numerator"] for row in values)
            denominator = sum(row["denominator"] for row in values)
            exclusions = Counter(
                row["exclusion_reason"] for row in values if row["excluded"]
            )
            metrics[metric] = {
                "numerator": numerator,
                "denominator": denominator,
                "value": _ratio(numerator, denominator),
                "display": f"{numerator / denominator:.2%}" if denominator else "N/A",
                "excluded_cases": sum(exclusions.values()),
                "exclusion_reasons": dict(exclusions),
                "excluded_case_ids": [
                    grade["case_id"]
                    for grade in grades
                    if grade["variants"][variant]["metrics"][metric]["excluded"]
                ],
            }
            if metric in ("card_precision", "opportunity_recall"):
                error = "fp" if metric == "card_precision" else "fn"
                metrics[metric]["selection_errors"] = {
                    "tp": numerator,
                    error: denominator - numerator,
                }
        variants[variant] = {
            "metrics": metrics,
            "displayed_cards": sum(row["displayed"] for row in rows),
            "operational_failures": dict(
                Counter(
                    row["operational_failure"]
                    for row in rows
                    if row["operational_failure"]
                )
            ),
            "selection_errors": {
                key: sum(row["selection_errors"][key] for row in rows)
                for key in ("included", "tp", "fp", "fn")
            },
            "selection_error_sample": (
                "Jointly resolved paired precision and recall cases; each metric "
                "also reports counts for its own affected-case exclusions."
            ),
        }
    return {
        "cases": len(grades),
        "personas": len({row["persona_id"] for row in grades}),
        "weekly_states": dict(Counter(row["weekly_state"] for row in grades)),
        "reference_opportunities": sum(
            row["reference"]["opportunity"] is True for row in grades
        ),
        "unresolved_opportunities": sum(
            row["reference"]["opportunity"] is None for row in grades
        ),
        "variants": variants,
    }


def _interval(values: np.ndarray, resamples: int) -> dict[str, Any]:
    defined = values[np.isfinite(values)]
    return {
        "low": float(np.percentile(defined, 2.5)) if len(defined) else None,
        "high": float(np.percentile(defined, 97.5)) if len(defined) else None,
        "defined_resamples": len(defined),
        "undefined_resamples": resamples - len(defined),
    }


def _bootstrap(
    grades: Sequence[dict[str, Any]],
    measurements: Mapping[str, Any],
    n_resamples: int,
) -> dict[str, Any]:
    personas = sorted({grade["persona_id"] for grade in grades})
    if not personas:
        return {"personas": 0, "intervals": {}}
    indices = {persona: i for i, persona in enumerate(personas)}
    metric_names = [metric for metric in METRICS[:-1]]
    # Each cell stores per-Persona totals; both variants share one sampled row.
    totals = np.zeros((len(personas), len(VARIANTS), len(metric_names), 2))
    runtime = np.zeros((len(personas), len(VARIANTS), 3))
    retrieval = np.zeros((len(personas), 2))
    for grade in grades:
        persona = indices[grade["persona_id"]]
        for j, variant in enumerate(VARIANTS):
            for k, metric in enumerate(metric_names):
                value = grade["variants"][variant]["metrics"][metric]
                totals[persona, j, k] += [value["numerator"], value["denominator"]]
            measured = measurements.get(grade["case_id"], {}).get(variant, {})
            runtime[persona, j] += [
                measured.get("cost_usd", 0.0),
                measured.get("latency_seconds", 0.0),
                1,
            ]
        hit = grade["variants"]["nomic"]["metrics"]["retrieval_hit_rate_at_3"]
        retrieval[persona] += [hit["numerator"], hit["denominator"]]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(personas), size=(n_resamples, len(personas)))
    sampled = totals[draws].sum(axis=1)
    sampled_hit = retrieval[draws].sum(axis=1)
    sampled_runtime = runtime[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = sampled[..., 0] / sampled[..., 1]
        hits = sampled_hit[..., 0] / sampled_hit[..., 1]
        time_means = sampled_runtime[..., 1] / sampled_runtime[..., 2]
        cost_means = sampled_runtime[..., 0] / sampled_runtime[..., 2]
    intervals: dict[str, Any] = {variant: {} for variant in VARIANTS}
    intervals["nomic_minus_full_history"] = {}
    for k, metric in enumerate(metric_names):
        for j, variant in enumerate(VARIANTS):
            intervals[variant][metric] = _interval(ratios[:, j, k], n_resamples)
        intervals["nomic_minus_full_history"][metric] = _interval(
            ratios[:, 0, k] - ratios[:, 1, k], n_resamples
        )
    intervals["nomic"]["retrieval_hit_rate_at_3"] = _interval(hits, n_resamples)
    for name, values in (
        ("mean_latency_seconds", time_means),
        ("mean_cost_usd", cost_means),
        ("latency_seconds", sampled_runtime[..., 1]),
        ("cost_usd", sampled_runtime[..., 0]),
    ):
        for j, variant in enumerate(VARIANTS):
            intervals[variant][name] = _interval(values[:, j], n_resamples)
        intervals["nomic_minus_full_history"][name] = _interval(
            values[:, 0] - values[:, 1], n_resamples
        )
    return {"personas": len(personas), "intervals": intervals}


def summarize(
    grades: Sequence[dict[str, Any]],
    measurements: Mapping[str, Any] | None = None,
    *,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Aggregate separately by split, with whole-Persona paired bootstrap.

    measurements[case_id][variant] supplies cost_usd and latency_seconds,
    including that case's explicitly allocated embedding overhead. Caller must
    document allocation and report shared evaluation charges separately.
    Missing measurements omit runtime reporting instead of claiming zero cost.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    ids = [row["case_id"] for row in grades]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id")
    splits_by_persona: dict[str, set[str]] = defaultdict(set)
    for row in grades:
        splits_by_persona[row["persona_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in splits_by_persona.values()):
        raise ValueError("A Persona cannot cross splits")
    supplied = measurements is not None
    measurements = measurements or {}
    if supplied and any(
        metric not in measurements.get(case_id, {}).get(variant, {})
        for case_id in ids
        for variant in VARIANTS
        for metric in ("cost_usd", "latency_seconds")
    ):
        raise ValueError("Runtime measurements must cover every case and variant")
    result: dict[str, Any] = {
        "ai_evaluation_only": True,
        "bootstrap": {
            "unit": "whole_persona",
            "paired_variants": True,
            "within_split": True,
            "resamples": n_resamples,
            "seed": SEED,
            "interval": "95% percentile",
        },
        "splits": {},
    }
    for split in sorted({grade["split"] for grade in grades}):
        selected = [grade for grade in grades if grade["split"] == split]
        report = _aggregate(selected)
        intervals = _bootstrap(selected, measurements, n_resamples)
        differences: dict[str, Any] = {}
        for metric in METRICS[:-1]:
            left = report["variants"]["nomic"]["metrics"][metric]["value"]
            right = report["variants"]["full_history"]["metrics"][metric]["value"]
            differences[metric] = (
                left - right if left is not None and right is not None else None
            )
        for variant in VARIANTS:
            if supplied:
                rows = [measurements[grade["case_id"]][variant] for grade in selected]
                cost = sum(row["cost_usd"] for row in rows)
                latency = sum(row["latency_seconds"] for row in rows)
                report["variants"][variant]["runtime"] = {
                    "cost_usd": cost,
                    "latency_seconds": latency,
                    "mean_cost_usd": cost / len(rows),
                    "mean_latency_seconds": latency / len(rows),
                    "unknown_cost_attempts": sum(
                        row.get("unknown_cost_attempts", 0) for row in rows
                    ),
                    "cost_complete": not any(
                        row.get("unknown_cost_attempts", 0) for row in rows
                    ),
                }
            else:
                for name in (
                    "mean_cost_usd",
                    "mean_latency_seconds",
                    "cost_usd",
                    "latency_seconds",
                ):
                    intervals["intervals"][variant].pop(name)
        if supplied:
            for metric in (
                "cost_usd",
                "latency_seconds",
                "mean_cost_usd",
                "mean_latency_seconds",
            ):
                differences[metric] = (
                    report["variants"]["nomic"]["runtime"][metric]
                    - report["variants"]["full_history"]["runtime"][metric]
                )
        else:
            for name in (
                "mean_cost_usd",
                "mean_latency_seconds",
                "cost_usd",
                "latency_seconds",
            ):
                intervals["intervals"]["nomic_minus_full_history"].pop(name)
        if supplied:
            report["runtime_cost_interpretation"] = (
                "Costs and intervals sum known attempt costs; where cost_complete "
                "is false, totals are lower bounds and cost differences may be biased."
            )
        report.update(nomic_minus_full_history=differences, uncertainty=intervals)
        result["splits"][split] = report
    return result


def _agreement(counter: Counter, values: list[Any]) -> None:
    counter["coordinates"] += 1
    if any(value is None for value in values):
        counter["incomplete"] += 1
    elif len(set(values)) == 1:
        counter["all_three_agree"] += 1
    else:
        counter["disagreements"] += 1
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if values[i] is not None and values[j] is not None:
            counter["pairwise_denominator"] += 1
            counter["pairwise_numerator"] += int(values[i] == values[j])


def consistency_report(
    assessments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare three blinded assessments per case without favourable-pass choice.

    Each record has case_id/persona_id and reviews, a list of three dicts with
    source_reviews, quote_reviews and optional grade. Failed reviews remain empty.
    Optional case and displayed_variants preserve expected coordinates even when
    all three assessments fail. Raw reason/decision agreement precedes adjudication.
    """
    metrics: dict[str, Counter] = defaultdict(Counter)
    failures: Counter = Counter()
    for record in assessments:
        reviews = record["reviews"]
        if len(reviews) != 3:
            raise ValueError("Consistency requires primary and two additional reviews")
        source_maps = []
        quote_maps = []
        for review in reviews:
            source_maps.append(
                {
                    (value, row["entry_id"]): row.get("reason_code")
                    for value, raw in review.get("source_reviews", {}).items()
                    for row in _dict(raw).get("results", [])
                }
            )
            quote_maps.append(
                {
                    variant: (raw.get("source_reason"), raw.get("quote_reason"))
                    for variant, data in review.get("quote_reviews", {}).items()
                    if (raw := _dict(data))
                }
            )
            failures["source_review_batches"] += sum(
                not _dict(raw).get("results")
                for raw in review.get("source_reviews", {}).values()
            )
            failures["empty_review_passes"] += int(not review)
        expected_sources = (
            {_key(source) for source in _sources(record["case"])}
            if "case" in record
            else set()
        )
        for label, maps, expected in (
            ("source_reason", source_maps, expected_sources),
            ("quotation_reason", quote_maps, set(record.get("displayed_variants", []))),
        ):
            keys = expected.union(*(mapping.keys() for mapping in maps))
            for key in keys:
                values: list[Any] = [mapping.get(key) for mapping in maps]
                _agreement(metrics[label], values)
                if label == "source_reason":
                    decisions = [
                        assessment.DECISION_BY_REASON.get(value) if value else None
                        for value in values
                    ]
                    _agreement(metrics["source_decision"], decisions)
                else:
                    accepted = [
                        value == ("observable_choice", "supported_action")
                        if value
                        else None
                        for value in values
                    ]
                    _agreement(metrics["quotation_acceptance"], accepted)
        for label in ("reference_opportunity", "reference_priority"):
            values = []
            for review in reviews:
                reference = review.get("grade", {}).get("reference", {})
                if label == "reference_opportunity":
                    values.append(reference.get("opportunity"))
                else:
                    value = reference.get("preferred_source")
                    values.append(
                        tuple(sorted(value.items()))
                        if value
                        else (
                            "no_opportunity"
                            if reference.get("priority_resolved")
                            else None
                        )
                    )
            _agreement(metrics[label], values)
        for variant in VARIANTS:
            values = [
                review.get("grade", {})
                .get("variants", {})
                .get(variant, {})
                .get("correct_card")
                for review in reviews
            ]
            _agreement(metrics[f"{variant}_correct_card"], values)
    return {
        "cases": len(assessments),
        "distinct_personas": len({row["persona_id"] for row in assessments}),
        "failures": dict(failures),
        "metrics": {
            label: {
                **dict(counts),
                "pairwise_agreement": _ratio(
                    counts["pairwise_numerator"], counts["pairwise_denominator"]
                ),
            }
            for label, counts in metrics.items()
        },
        "interpretation": "AI evaluator consistency does not establish human validity.",
    }
