"""Batch report for Coach Digest Validations.

Runs the existing ``validate_weekly_digest_narrative`` checks
(groundedness, non_circularity, value_leakage, state_claims, and length) over a
set of persisted Weekly Drift Detection output records. It reports per-check
pass rates against the targets in ``docs/evals/explanation_quality_eval.md``.

These are mechanical code checks, not human validation. They verify surface
properties (quotes trace to evidence, no raw scoring or Schwartz-label
terminology, length in range). They do not assess correctness, tone, or
usefulness. Coach Digest Evals assess those properties. Future human
calibration of the AI review remains separate work.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from src.coach.schemas import (
    CoachNarrative,
    DigestValidation,
    EvidenceSnippet,
    WeeklyDigest,
)
from src.coach.weekly_digest import validate_weekly_digest_narrative

DEFAULT_PARQUET = Path("logs/exports/weekly_digests/weekly_digests.parquet")

# The approved runtime tags its digests with this signal_source. By default the
# report measures only these rows, excluding deprecated vif_runtime leftovers.
APPROVED_SIGNAL_SOURCE = "weekly_drift_reviewer"
ValidationPolicy = Literal["recorded", "current"]

# Pass-rate targets from docs/evals/explanation_quality_eval.md.
# value_leakage has no published target; treated as informational (target None).
CHECK_TARGETS: dict[str, float | None] = {
    "groundedness": 0.70,
    "non_circularity": 0.95,
    "value_leakage": None,
    "state_claims": None,
    "length": 0.90,
}


@dataclass
class CheckSummary:
    """Aggregated pass rate for one Coach Digest Validation."""

    name: str
    passed: int
    total: int
    target: float | None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def meets_target(self) -> bool | None:
        if self.target is None:
            return None
        return self.pass_rate >= self.target

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "target": self.target,
            "meets_target": self.meets_target,
        }


@dataclass
class CoachDigestValidationReport:
    """Coach Digest Validations report for a Weekly Drift Detection output set."""

    input_source: str
    input_kind: str
    n_rows: int
    n_with_narrative: int
    n_evaluated: int
    validation_policy: ValidationPolicy = "recorded"
    signal_source_filter: str | None = None
    n_rows_after_filter: int = 0
    checks: dict[str, CheckSummary] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    sample_results: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval": "coach_digest_validations",
            "source": "mechanical_code_checks",
            "note": (
                "Coach Digest Validations, not human validation. Surface "
                "properties only; correctness/tone are out of scope."
            ),
            "input_source": self.input_source,
            "input_kind": self.input_kind,
            "validation_policy": self.validation_policy,
            "signal_source_filter": self.signal_source_filter,
            "n_rows": self.n_rows,
            "n_rows_after_filter": self.n_rows_after_filter,
            "n_with_narrative": self.n_with_narrative,
            "n_evaluated": self.n_evaluated,
            "checks": {
                name: summary.to_dict() for name, summary in self.checks.items()
            },
            "skipped_persona_weeks": self.skipped,
            "sample_results": self.sample_results,
            "api_usage": {
                "n_calls": 0,
                "calculated_cost_usd": 0.0,
                "total_latency_seconds": 0.0,
            },
        }


def _reconstruct_digest(row: dict[str, object]) -> WeeklyDigest:
    """Rebuild a WeeklyDigest from a persisted parquet row (narrative aside)."""
    evidence_raw = json.loads(str(row.get("evidence_json") or "[]"))
    evidence = [EvidenceSnippet.model_validate(item) for item in evidence_raw]
    persona_name = row.get("persona_name")
    overall_mean = row.get("overall_mean")
    overall_uncertainty = row.get("overall_uncertainty")
    return WeeklyDigest(
        persona_id=str(row["persona_id"]),
        persona_name=None if persona_name is None else str(persona_name),
        week_start=str(row["week_start"]),
        week_end=str(row["week_end"]),
        response_mode=str(row["response_mode"]),  # type: ignore[arg-type]
        mode_source=str(row["mode_source"]),
        mode_rationale=str(row["mode_rationale"]),
        signal_source=str(row.get("signal_source") or "judge_labels"),
        n_entries=int(str(row["n_entries"])),
        overall_mean=None if overall_mean is None else float(str(overall_mean)),
        overall_uncertainty=(
            None if overall_uncertainty is None else float(str(overall_uncertainty))
        ),
        core_values=json.loads(str(row.get("core_values_json") or "[]")),
        goal_context=(
            None if row.get("goal_context") is None else str(row["goal_context"])
        ),
        drift_states=json.loads(str(row.get("drift_states_json") or "{}")),
        drift_details=json.loads(str(row.get("drift_details_json") or "{}")),
        state_comparisons=json.loads(str(row.get("state_comparisons_json") or "[]")),
        drift_reasons=json.loads(str(row.get("drift_reasons_json") or "[]")),
        top_tensions=json.loads(str(row.get("top_tensions_json") or "[]")),
        top_strengths=json.loads(str(row.get("top_strengths_json") or "[]")),
        dimensions=json.loads(str(row.get("dimensions_json") or "[]")),
        evidence=evidence,
        validation=(
            DigestValidation.model_validate_json(str(row["validation_json"]))
            if row.get("validation_json")
            else None
        ),
    )


def _validate_for_policy(
    digest: WeeklyDigest,
    narrative: CoachNarrative,
    policy: ValidationPolicy,
    *,
    prompt_version: str | None = None,
) -> tuple[DigestValidation, str]:
    """Recompute checks without changing the response's saved validation receipt."""
    if policy not in {"recorded", "current"}:
        raise ValueError(f"Unknown validation policy: {policy}")
    names = (
        {check.name for check in digest.validation.checks}
        if digest.validation
        else set()
    )
    current = policy == "current" or "reflective_question_form" in names
    voice = policy == "current" or (
        bool(names & {"conversational_voice", "natural_reflection_voice"})
        if digest.validation is not None
        else prompt_version in {"4.4", "4.5"}
    )
    voice_version: Literal["4.4", "4.5"] = (
        "4.5"
        if policy == "current"
        or "natural_reflection_voice" in names
        or (digest.validation is None and prompt_version == "4.5")
        else "4.4"
    )
    resolved = (
        "current"
        if current and voice and voice_version == "4.5"
        else f"current_voice_{voice_version}"
        if current and voice
        else "current_base"
        if current
        else f"historical_voice_{voice_version}"
        if voice
        else "historical_base"
    )
    if policy == "recorded" and not names and prompt_version is None:
        resolved += "_legacy_default"
    return validate_weekly_digest_narrative(
        digest,
        narrative,
        validate_voice=voice,
        voice_version=voice_version,
        validation_policy="current" if current else "historical",
    ), resolved


def _count_checks(counts: dict[str, list[int]], validation: DigestValidation) -> None:
    for check in validation.checks:
        count = counts.setdefault(check.name, [0, 0])
        count[1] += 1
        count[0] += int(check.passed)


def evaluate_rows(
    rows: list[dict[str, object]],
    input_source: str,
    signal_source: str | None = APPROVED_SIGNAL_SOURCE,
    *,
    validation_policy: ValidationPolicy = "recorded",
) -> CoachDigestValidationReport:
    """Run Coach Digest Validations over parquet-shaped digest rows.

    By default only rows whose ``signal_source`` matches the approved runtime
    (``weekly_drift_reviewer``) are evaluated. Pass ``signal_source=None`` to
    include every row regardless of source.
    """
    counts: dict[str, list[int]] = {name: [0, 0] for name in CHECK_TARGETS}
    n_with_narrative = 0
    n_evaluated = 0
    skipped: list[str] = []
    sample_results: list[dict[str, object]] = []

    total_rows = len(rows)
    if signal_source is not None:
        rows = [row for row in rows if row.get("signal_source") == signal_source]

    for row in rows:
        narrative_json = row.get("coach_narrative_json")
        if not narrative_json:
            continue
        n_with_narrative += 1
        label = f"{row.get('persona_id')}:{row.get('week_end')}"
        try:
            narrative = CoachNarrative.model_validate(json.loads(str(narrative_json)))
            digest = _reconstruct_digest(row)
        except Exception:
            skipped.append(label)
            sample_results.append({"sample_id": label, "status": "skipped"})
            continue

        validation, resolved_policy = _validate_for_policy(
            digest, narrative, validation_policy
        )
        n_evaluated += 1
        _count_checks(counts, validation)
        sample_results.append(
            {
                "sample_id": label,
                "status": "evaluated",
                "validation_policy": resolved_policy,
                "all_passed": validation.all_passed,
                "checks": {check.name: check.passed for check in validation.checks},
                "failed_checks": [
                    check.name for check in validation.checks if not check.passed
                ],
            }
        )

    checks = {
        name: CheckSummary(
            name=name,
            passed=counts[name][0],
            total=counts[name][1],
            target=CHECK_TARGETS.get(name),
        )
        for name in counts
    }
    return CoachDigestValidationReport(
        input_source=input_source,
        input_kind="parquet",
        validation_policy=validation_policy,
        n_rows=total_rows,
        n_rows_after_filter=len(rows),
        signal_source_filter=signal_source,
        n_with_narrative=n_with_narrative,
        n_evaluated=n_evaluated,
        checks=checks,
        skipped=skipped,
        sample_results=sample_results,
    )


def evaluate_parquet(
    parquet_path: Path,
    signal_source: str | None = APPROVED_SIGNAL_SOURCE,
    *,
    validation_policy: ValidationPolicy = "recorded",
) -> CoachDigestValidationReport:
    """Run Coach Digest Validations on persisted Weekly Drift Detection output."""
    frame = pl.read_parquet(parquet_path)
    rows = frame.to_dicts()
    return evaluate_rows(
        rows,
        input_source=str(parquet_path),
        signal_source=signal_source,
        validation_policy=validation_policy,
    )


def evaluate_manifest(
    manifest_path: Path,
    signal_source: str | None = APPROVED_SIGNAL_SOURCE,
    *,
    validation_policy: ValidationPolicy = "recorded",
) -> CoachDigestValidationReport:
    """Run Coach Digest Validations on exact manifest digest-response pairs."""
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, list[int]] = {name: [0, 0] for name in CHECK_TARGETS}
    skipped: list[str] = []
    sample_results: list[dict[str, object]] = []
    filtered_items = [
        item
        for item in items
        if signal_source is None
        or item.get("digest", {}).get("signal_source") == signal_source
    ]
    n_evaluated = 0
    for item in filtered_items:
        digest_data = item.get("digest", {})
        label = f"{digest_data.get('persona_id')}:{digest_data.get('week_end')}"
        try:
            digest = WeeklyDigest.model_validate(digest_data)
            narrative = CoachNarrative.model_validate(item["narrative"])
        except Exception:
            skipped.append(label)
            sample_results.append({"sample_id": label, "status": "skipped"})
            continue
        provenance = item.get("provenance", {})
        validation, resolved_policy = _validate_for_policy(
            digest,
            narrative,
            validation_policy,
            prompt_version=provenance.get("generation", {}).get("prompt_version"),
        )
        n_evaluated += 1
        _count_checks(counts, validation)
        sample_results.append(
            {
                "sample_id": label,
                "scenario_id": provenance.get("scenario_id"),
                "status": "evaluated",
                "validation_policy": resolved_policy,
                "all_passed": validation.all_passed,
                "weekly_drift_input_sha256": provenance.get(
                    "weekly_drift_input_sha256"
                ),
                "coach_response_sha256": provenance.get("coach_response_sha256"),
                "checks": {check.name: check.passed for check in validation.checks},
                "failed_checks": [
                    check.name for check in validation.checks if not check.passed
                ],
            }
        )
    checks = {
        name: CheckSummary(
            name=name,
            passed=counts[name][0],
            total=counts[name][1],
            target=CHECK_TARGETS.get(name),
        )
        for name in counts
    }
    return CoachDigestValidationReport(
        input_source=str(manifest_path),
        input_kind="public_scenario_manifest",
        validation_policy=validation_policy,
        n_rows=len(items),
        n_rows_after_filter=len(filtered_items),
        signal_source_filter=signal_source,
        n_with_narrative=len(filtered_items),
        n_evaluated=n_evaluated,
        checks=checks,
        skipped=skipped,
        sample_results=sample_results,
    )


def render_markdown(report: CoachDigestValidationReport) -> str:
    """Render a short Coach Digest Validations report."""
    lines = [
        "# Coach Digest Validations — Batch Report",
        "",
        "**Source:** mechanical code checks (not human validation). Surface "
        "properties only.",
        "",
        f"- Input: `{report.input_source}`",
        f"- Input kind: `{report.input_kind}`",
        f"- Validation policy: `{report.validation_policy}`",
        "- Recorded mode uses saved checks or generation metadata; records without "
        "either use the explicitly labelled historical base policy.",
        (
            f"- Signal source filter: `{report.signal_source_filter}` "
            f"({report.n_rows_after_filter} of {report.n_rows} rows)"
            if report.signal_source_filter is not None
            else f"- Signal source filter: none (all {report.n_rows} rows)"
        ),
        f"- With narrative: {report.n_with_narrative}",
        f"- Evaluated: {report.n_evaluated}",
        "- API calls: 0",
        "- Provider cost: $0.00",
        "- Provider request latency: not applicable",
    ]
    if report.skipped:
        lines.append(f"- Skipped (unparseable): {', '.join(report.skipped)}")
    lines += [
        "",
        "| Check | Passed | Total | Pass rate | Target | Meets target |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary in report.checks.items():
        target = "—" if summary.target is None else f"> {summary.target:.0%}"
        meets = (
            "—"
            if summary.meets_target is None
            else ("✅" if summary.meets_target else "❌")
        )
        lines.append(
            f"| {name} | {summary.passed} | {summary.total} | "
            f"{summary.pass_rate:.0%} | {target} | {meets} |"
        )
    if report.sample_results:
        lines += [
            "",
            "## Per-Response Results",
            "",
            "| Response | Scenario | Policy | Result | Failed checks |",
            "| --- | --- | --- | --- | --- |",
        ]
        for result in report.sample_results:
            failed_checks_value = result.get("failed_checks")
            failed_checks = (
                [str(check) for check in failed_checks_value]
                if isinstance(failed_checks_value, list)
                else []
            )
            lines.append(
                f"| {result['sample_id']} | {result.get('scenario_id') or '—'} | "
                f"{result.get('validation_policy') or '—'} | "
                f"{result['status']} | {', '.join(failed_checks) or 'none'} |"
            )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Coach Digest Validations over Coach Digest responses."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--parquet", type=Path, default=None)
    source_group.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Evaluate exact digest-response pairs from a judge sample manifest.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write metrics.json and report.md.",
    )
    parser.add_argument(
        "--signal-source",
        default=APPROVED_SIGNAL_SOURCE,
        help=(
            "Only evaluate rows with this signal_source "
            f"(default: {APPROVED_SIGNAL_SOURCE}, the approved runtime)."
        ),
    )
    parser.add_argument(
        "--validation-policy",
        choices=["recorded", "current"],
        default="recorded",
        help="Reproduce recorded checks (default), or apply every current live check.",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Evaluate every row regardless of signal_source.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    signal_source = None if args.all_sources else args.signal_source
    report = (
        evaluate_manifest(
            args.manifest,
            signal_source=signal_source,
            validation_policy=args.validation_policy,
        )
        if args.manifest is not None
        else evaluate_parquet(
            args.parquet or DEFAULT_PARQUET,
            signal_source=signal_source,
            validation_policy=args.validation_policy,
        )
    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n"
        )
        (args.out / "report.md").write_text(render_markdown(report))

    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
