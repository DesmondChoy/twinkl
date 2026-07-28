"""Tier-1 batch evaluation for Weekly Coach narratives.

Runs the existing ``validate_weekly_digest_narrative`` Tier-1 checks
(groundedness, non_circularity, value_leakage, length) over a set of persisted
Weekly Digest records and reports per-check pass rates against the targets in
``docs/evals/explanation_quality_eval.md``.

These are mechanical code checks, not human validation. They verify surface
properties (quotes trace to evidence, no raw scoring or Schwartz-label
terminology, length in range); they do not assess correctness, tone, or
usefulness. Those belong to the Tier-2 LLM-as-judge eval and Tier-3 human
calibration.
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

# Pass-rate targets from docs/evals/explanation_quality_eval.md (Tier 1 table).
# value_leakage has no published target; treated as informational (target None).
CHECK_TARGETS: dict[str, float | None] = {
    "groundedness": 0.70,
    "non_circularity": 0.95,
    "value_leakage": None,
    "length": 0.90,
}


@dataclass
class CheckSummary:
    """Aggregated pass rate for one Tier-1 check across the sample."""

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
class Tier1Report:
    """Full Tier-1 batch report over a Weekly Digest set."""

    parquet_source: str
    n_rows: int
    n_with_narrative: int
    n_evaluated: int
    checks: dict[str, CheckSummary] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "eval": "coach_narrative_tier1",
            "source": "mechanical_code_checks",
            "note": (
                "Tier-1 automated checks, not human validation. Surface "
                "properties only; correctness/tone are out of scope."
            ),
            "parquet_source": self.parquet_source,
            "n_rows": self.n_rows,
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
        drift_states=json.loads(str(row.get("drift_states_json") or "{}")),
        drift_reasons=json.loads(str(row.get("drift_reasons_json") or "[]")),
        top_tensions=json.loads(str(row.get("top_tensions_json") or "[]")),
        top_strengths=json.loads(str(row.get("top_strengths_json") or "[]")),
        dimensions=json.loads(str(row.get("dimensions_json") or "[]")),
        evidence=evidence,
    )


def evaluate_rows(rows: list[dict[str, object]], parquet_source: str) -> Tier1Report:
    """Run Tier-1 validation over parquet-shaped digest rows."""
    counts: dict[str, list[int]] = {name: [0, 0] for name in CHECK_TARGETS}
    n_with_narrative = 0
    n_evaluated = 0
    skipped: list[str] = []

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
    return Tier1Report(
        parquet_source=parquet_source,
        n_rows=len(rows),
        n_with_narrative=n_with_narrative,
        n_evaluated=n_evaluated,
        checks=checks,
        skipped=skipped,
    )


def evaluate_parquet(parquet_path: Path) -> Tier1Report:
    """Load a persisted Weekly Digest parquet and run Tier-1 evaluation."""
    frame = pl.read_parquet(parquet_path)
    rows = frame.to_dicts()
    return evaluate_rows(rows, parquet_source=str(parquet_path))


def render_markdown(report: Tier1Report) -> str:
    """Render a short markdown summary of a Tier-1 batch report."""
    lines = [
        "# Weekly Coach Narrative — Tier-1 Batch Report",
        "",
        "**Source:** mechanical code checks (not human validation). Surface "
        "properties only.",
        "",
        f"- Parquet: `{report.parquet_source}`",
        f"- Rows: {report.n_rows}",
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
        description="Run Tier-1 batch checks over Weekly Coach narratives."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write metrics.json and report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = evaluate_parquet(args.parquet)

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
