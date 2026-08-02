"""Batch report for Coach Digest Validations.

Runs the existing ``validate_weekly_digest_narrative`` checks
(groundedness, non_circularity, value_leakage, length) over a set of persisted
Weekly Drift Detection output records and reports per-check pass rates against
the targets in ``docs/evals/explanation_quality_eval.md``.

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

import polars as pl

from src.coach.schemas import (
    CoachNarrative,
    EvidenceSnippet,
    WeeklyDigest,
)
from src.coach.weekly_digest import validate_weekly_digest_narrative

DEFAULT_PARQUET = Path("logs/exports/weekly_digests/weekly_digests.parquet")

# The approved runtime tags its digests with this signal_source. By default the
# report measures only these rows, excluding deprecated vif_runtime leftovers.
APPROVED_SIGNAL_SOURCE = "weekly_drift_reviewer"

# Pass-rate targets from docs/evals/explanation_quality_eval.md.
# value_leakage has no published target; treated as informational (target None).
CHECK_TARGETS: dict[str, float | None] = {
    "groundedness": 0.70,
    "non_circularity": 0.95,
    "value_leakage": None,
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

    parquet_source: str
    n_rows: int
    n_with_narrative: int
    n_evaluated: int
    signal_source_filter: str | None = None
    n_rows_after_filter: int = 0
    checks: dict[str, CheckSummary] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval": "coach_digest_validations",
            "source": "mechanical_code_checks",
            "note": (
                "Coach Digest Validations, not human validation. Surface "
                "properties only; correctness/tone are out of scope."
            ),
            "parquet_source": self.parquet_source,
            "signal_source_filter": self.signal_source_filter,
            "n_rows": self.n_rows,
            "n_rows_after_filter": self.n_rows_after_filter,
            "n_with_narrative": self.n_with_narrative,
            "n_evaluated": self.n_evaluated,
            "checks": {
                name: summary.to_dict() for name, summary in self.checks.items()
            },
            "skipped_persona_weeks": self.skipped,
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
            None
            if row.get("goal_context") is None
            else str(row["goal_context"])
        ),
        drift_states=json.loads(str(row.get("drift_states_json") or "{}")),
        drift_reasons=json.loads(str(row.get("drift_reasons_json") or "[]")),
        top_tensions=json.loads(str(row.get("top_tensions_json") or "[]")),
        top_strengths=json.loads(str(row.get("top_strengths_json") or "[]")),
        dimensions=json.loads(str(row.get("dimensions_json") or "[]")),
        evidence=evidence,
    )


def evaluate_rows(
    rows: list[dict[str, object]],
    parquet_source: str,
    signal_source: str | None = APPROVED_SIGNAL_SOURCE,
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
            continue

        validation = validate_weekly_digest_narrative(digest, narrative)
        n_evaluated += 1
        for check in validation.checks:
            if check.name in counts:
                counts[check.name][1] += 1
                if check.passed:
                    counts[check.name][0] += 1

    checks = {
        name: CheckSummary(
            name=name,
            passed=counts[name][0],
            total=counts[name][1],
            target=CHECK_TARGETS[name],
        )
        for name in CHECK_TARGETS
    }
    return CoachDigestValidationReport(
        parquet_source=parquet_source,
        n_rows=total_rows,
        n_rows_after_filter=len(rows),
        signal_source_filter=signal_source,
        n_with_narrative=n_with_narrative,
        n_evaluated=n_evaluated,
        checks=checks,
        skipped=skipped,
    )


def evaluate_parquet(
    parquet_path: Path,
    signal_source: str | None = APPROVED_SIGNAL_SOURCE,
) -> CoachDigestValidationReport:
    """Run Coach Digest Validations on persisted Weekly Drift Detection output."""
    frame = pl.read_parquet(parquet_path)
    rows = frame.to_dicts()
    return evaluate_rows(
        rows, parquet_source=str(parquet_path), signal_source=signal_source
    )


def render_markdown(report: CoachDigestValidationReport) -> str:
    """Render a short Coach Digest Validations report."""
    lines = [
        "# Coach Digest Validations — Batch Report",
        "",
        "**Source:** mechanical code checks (not human validation). Surface "
        "properties only.",
        "",
        f"- Parquet: `{report.parquet_source}`",
        (
            f"- Signal source filter: `{report.signal_source_filter}` "
            f"({report.n_rows_after_filter} of {report.n_rows} rows)"
            if report.signal_source_filter is not None
            else f"- Signal source filter: none (all {report.n_rows} rows)"
        ),
        f"- With narrative: {report.n_with_narrative}",
        f"- Evaluated: {report.n_evaluated}",
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
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Coach Digest Validations over Coach Digest responses."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
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
        "--all-sources",
        action="store_true",
        help="Evaluate every row regardless of signal_source.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    signal_source = None if args.all_sources else args.signal_source
    report = evaluate_parquet(args.parquet, signal_source=signal_source)

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
