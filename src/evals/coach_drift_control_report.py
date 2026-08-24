"""Compare Coach Digest response quality on Drift weeks against control weeks.

The Coach Digest Validations and the AI evaluation each report one pooled
number. Pooled numbers cannot answer the question this evaluation exists for:
does the Weekly Drift Coach behave differently when Drift is present?

This module splits both result sets by evaluation arm and by delivery state,
and reports a confidence interval with every rate. At about 42 responses per
arm a difference of 15 points is not separable from chance, so a rate without
an interval invites a wrong conclusion.

Read the output with three limits in mind:

  1. The scores come from mechanical checks and an AI evaluator. Neither is
     human validation.
  2. ``tension_honesty`` and ``specificity`` are expected to differ by arm. A
     control week holds no tension to describe, so a lower control score can be
     correct behavior rather than a defect.
  3. Control weeks are weeks with no detected Drift, not weeks with no Drift.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECK_NAMES = (
    "groundedness",
    "non_circularity",
    "value_leakage",
    "length",
    "state_claims",
)
SCORE_DIMENSIONS = (
    "correctness",
    "specificity",
    "non_prescriptive_tone",
    "tension_honesty",
)
ARMS = ("drift", "control")


def wilson_interval(
    passed: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Return a Wilson score interval for a pass rate.

    The Wilson interval stays inside 0 to 1 at small counts, where the normal
    approximation does not.
    """
    if total == 0:
        return (0.0, 0.0)
    p = passed / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    ) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class RateCell:
    """One pass rate with its counts and interval."""

    passed: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        low, high = wilson_interval(self.passed, self.total)
        return {
            "passed": self.passed,
            "total": self.total,
            "rate": round(self.rate, 3),
            "ci95": [round(low, 3), round(high, 3)],
        }


@dataclass
class ScoreCell:
    """Mean of one AI evaluation dimension over a group."""

    values: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n": len(self.values),
            "mean": round(self.mean, 3),
        }


def _rows_from_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten manifest entries into one row per Coach Digest response."""
    rows: list[dict[str, Any]] = []
    for item in manifest:
        digest = item.get("digest") or {}
        target = item.get("target") or {}
        validation = item.get("validation") or {}
        checks = {
            check["name"]: bool(check["passed"])
            for check in validation.get("checks", [])
        }
        rows.append(
            {
                "key": f"{digest.get('persona_id')}:{digest.get('week_end')}",
                "persona_id": digest.get("persona_id"),
                "week_end": digest.get("week_end"),
                "arm": target.get("arm"),
                "delivery_state": target.get("delivery_state"),
                "match_quality": target.get("match_quality"),
                "n_truncated_weeks": target.get("n_truncated_weeks"),
                "response_mode": digest.get("response_mode"),
                "n_entries": digest.get("n_entries"),
                "n_evidence": len(digest.get("evidence") or []),
                "checks": checks,
                "all_passed": bool(validation.get("all_passed"))
                if validation
                else None,
            }
        )
    return rows


def _attach_verdicts(
    rows: list[dict[str, Any]], verdicts: list[dict[str, Any]]
) -> None:
    by_key = {record["key"]: record.get("verdict") for record in verdicts}
    for row in rows:
        row["verdict"] = by_key.get(row["key"])


def _rate_by_group(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, dict[str, dict[str, object]]]:
    """Pass rate for every validation check, split by one grouping field."""
    cells: dict[str, dict[str, RateCell]] = defaultdict(
        lambda: defaultdict(RateCell)
    )
    for row in rows:
        group = row.get(group_key)
        if group is None:
            continue
        for name in CHECK_NAMES:
            if name not in row["checks"]:
                continue
            cell = cells[str(group)][name]
            cell.total += 1
            cell.passed += int(row["checks"][name])
    return {
        group: {name: cell.to_dict() for name, cell in checks.items()}
        for group, checks in cells.items()
    }


def _scores_by_group(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, dict[str, dict[str, object]]]:
    """AI evaluation means for every dimension, split by one grouping field."""
    cells: dict[str, dict[str, ScoreCell]] = defaultdict(
        lambda: defaultdict(ScoreCell)
    )
    for row in rows:
        group = row.get(group_key)
        verdict = row.get("verdict")
        if group is None or not verdict:
            continue
        for dim in SCORE_DIMENSIONS:
            if dim in verdict:
                cells[str(group)][dim].values.append(int(verdict[dim]))
    return {
        group: {dim: cell.to_dict() for dim, cell in dims.items()}
        for group, dims in cells.items()
    }


def _counts_by_group(
    rows: list[dict[str, Any]], group_key: str, value_key: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = row.get(group_key)
        if group is None:
            continue
        counts[str(group)][str(row.get(value_key))] += 1
    return {group: dict(values) for group, values in counts.items()}


def _history_length_check(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Report whether groundedness tracks history length within each arm.

    If the control arm ended up systematically shorter, an apparent arm
    difference would really be a history length difference.
    """
    summary: dict[str, object] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        grounded = [r for r in arm_rows if r["checks"].get("groundedness")]
        ungrounded = [r for r in arm_rows if r["checks"].get("groundedness") is False]

        def mean(items: list[dict[str, Any]], key: str) -> float | None:
            values = [i[key] for i in items if i.get(key) is not None]
            return round(sum(values) / len(values), 2) if values else None

        summary[arm] = {
            "n": len(arm_rows),
            "mean_entries": mean(arm_rows, "n_entries"),
            "mean_reviewed_weeks": mean(arm_rows, "n_truncated_weeks"),
            "mean_evidence": mean(arm_rows, "n_evidence"),
            "mean_entries_grounded": mean(grounded, "n_entries"),
            "mean_entries_ungrounded": mean(ungrounded, "n_entries"),
        }
    return summary


def build_report(
    manifest: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    *,
    manifest_source: str = "",
) -> dict[str, Any]:
    """Build the Drift against control comparison."""
    rows = _rows_from_manifest(manifest)
    _attach_verdicts(rows, verdicts)

    generator_models = {
        item.get("generator_model") for item in manifest if item.get("generator_model")
    }
    generator_model = (
        next(iter(generator_models)) if len(generator_models) == 1 else None
    )

    return {
        "eval": "coach_drift_control_comparison",
        "source": "mechanical_code_checks + ai_review",
        "note": (
            "Coach Digest Validations are mechanical checks and the dimension "
            "scores come from an AI evaluator. Neither is human validation."
        ),
        "manifest_source": manifest_source,
        "generator_model": generator_model,
        "n_rows": len(rows),
        "n_by_arm": {
            arm: sum(1 for row in rows if row["arm"] == arm) for arm in ARMS
        },
        "validations_by_arm": _rate_by_group(rows, "arm"),
        "validations_by_delivery_state": _rate_by_group(rows, "delivery_state"),
        "scores_by_arm": _scores_by_group(rows, "arm"),
        "scores_by_delivery_state": _scores_by_group(rows, "delivery_state"),
        "response_mode_by_arm": _counts_by_group(rows, "arm", "response_mode"),
        "match_quality": _counts_by_group(rows, "arm", "match_quality").get(
            "control", {}
        ),
        "history_length_check": _history_length_check(rows),
        "n_without_verdict": sum(1 for row in rows if not row.get("verdict")),
        "rows": rows,
        "limits": [
            "Scores are mechanical checks and AI evaluation, not human "
            "validation.",
            "tension_honesty and specificity are expected to differ by arm. A "
            "control week holds no tension to describe.",
            "Control weeks are weeks with no detected Drift, not weeks with no "
            "Drift.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the comparison as a short report."""
    lines = [
        "# Coach Digest Response — Drift against Control",
        "",
        "**Source:** Coach Digest Validations (mechanical checks) and AI "
        "evaluation scores. Neither is human validation.",
        "",
        f"- Coach Digest responses: {report['n_rows']}",
        f"- Drift arm: {report['n_by_arm'].get('drift', 0)}, "
        f"control arm: {report['n_by_arm'].get('control', 0)}",
        f"- Generator model: `{report.get('generator_model') or 'unrecorded'}`",
        f"- Responses with no AI verdict: {report['n_without_verdict']}",
        "",
        "## Validation pass rate by arm",
        "",
        "| Check | Drift | Control |",
        "| --- | --- | --- |",
    ]
    by_arm = report["validations_by_arm"]
    for name in CHECK_NAMES:
        cells = []
        for arm in ARMS:
            cell = by_arm.get(arm, {}).get(name)
            if cell is None:
                cells.append("—")
                continue
            low, high = cell["ci95"]
            cells.append(
                f"{cell['rate']:.0%} ({cell['passed']}/{cell['total']}, "
                f"95% CI {low:.0%}-{high:.0%})"
            )
        if cells != ["—", "—"]:
            lines.append(f"| {name} | {cells[0]} | {cells[1]} |")

    lines += ["", "## AI evaluation mean by arm", "", "| Dimension | Drift | Control |",
              "| --- | --- | --- |"]
    scores = report["scores_by_arm"]
    for dim in SCORE_DIMENSIONS:
        cells = []
        for arm in ARMS:
            cell = scores.get(arm, {}).get(dim)
            cells.append(f"{cell['mean']:.2f} (n={cell['n']})" if cell else "—")
        lines.append(f"| {dim} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Response mode by arm", ""]
    for arm in ARMS:
        modes = report["response_mode_by_arm"].get(arm, {})
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(modes.items())) or "none"
        lines.append(f"- {arm}: {rendered}")

    lines += ["", "## Limits", ""]
    lines += [f"- {limit}" for limit in report["limits"]]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Coach Digest response quality on Drift weeks against "
            "control weeks."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Judge sample manifest written by run_coach_drift_control_eval.py.",
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=None,
        help="verdicts.json written by the AI evaluator. Without it the "
        "report holds validation results only.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    verdicts = json.loads(args.verdicts.read_text()) if args.verdicts else []
    report = build_report(
        manifest, verdicts, manifest_source=str(args.manifest)
    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "report.md").write_text(render_markdown(report))

    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
