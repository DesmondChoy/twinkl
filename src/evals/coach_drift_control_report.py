"""Compare Coach Digest results for known Drift and matched control targets."""

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
    "state_claims",
    "length",
)
SCORE_DIMENSIONS = (
    "correctness",
    "specificity",
    "non_prescriptive_tone",
    "tension_honesty",
)
GROUPS = ("drift", "control")


def wilson_interval(
    passed: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson score interval that stays between zero and one."""
    if total == 0:
        return (0.0, 0.0)
    rate = passed / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
    ) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class RateCell:
    passed: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, object]:
        low, high = wilson_interval(self.passed, self.total)
        rate = self.passed / self.total if self.total else 0.0
        return {
            "passed": self.passed,
            "total": self.total,
            "rate": round(rate, 3),
            "ci95": [round(low, 3), round(high, 3)],
        }


@dataclass
class ScoreCell:
    values: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        mean = sum(self.values) / len(self.values) if self.values else 0.0
        return {"n": len(self.values), "mean": round(mean, 3)}


def _manifest_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in manifest:
        digest = item.get("digest") or {}
        target = item.get("target") or {}
        validation = item.get("validation") or {}
        target_id = str(
            target.get("target_id")
            or f"{digest.get('persona_id')}:{digest.get('week_end')}"
        )
        if target_id in seen:
            raise ValueError(f"Duplicate manifest target: {target_id}")
        seen.add(target_id)
        checks = {
            str(check["name"]): bool(check["passed"])
            for check in validation.get("checks", [])
        }
        rows.append(
            {
                "sample_id": target_id,
                "persona_id": digest.get("persona_id"),
                "week_end": digest.get("week_end"),
                "group": target.get("group"),
                "known_delivery_state": target.get("delivery_state"),
                "match_quality": target.get("match_quality"),
                "reviewed_week_count": target.get("reviewed_week_count"),
                "response_mode": digest.get("response_mode"),
                "n_entries": digest.get("n_entries"),
                "n_evidence": len(digest.get("evidence") or []),
                "checks": checks,
                "all_passed": bool(checks)
                and all(checks.get(name) is True for name in CHECK_NAMES),
            }
        )
    return rows


def _attach_eval_results(
    rows: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
) -> None:
    sample_results = eval_metrics.get("sample_results") or []
    by_sample: dict[str, dict[str, Any]] = {}
    for result in sample_results:
        sample_id = str(result["sample_id"])
        if sample_id in by_sample:
            raise ValueError(f"Duplicate AI evaluation result: {sample_id}")
        by_sample[sample_id] = result
    row_ids = {str(row["sample_id"]) for row in rows}
    unmatched = sorted(set(by_sample) - row_ids)
    if unmatched:
        raise ValueError(
            "AI evaluation results do not match the manifest: "
            + ", ".join(unmatched)
        )
    for row in rows:
        result = by_sample.get(row["sample_id"])
        row["eval_result"] = (
            result if result and result.get("status") == "scored" else None
        )


def _rates_by(
    rows: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, dict[str, object]]]:
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


def _scores_by(
    rows: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, dict[str, object]]]:
    cells: dict[str, dict[str, ScoreCell]] = defaultdict(
        lambda: defaultdict(ScoreCell)
    )
    for row in rows:
        group = row.get(group_key)
        result = row.get("eval_result")
        if group is None or result is None:
            continue
        for dimension in SCORE_DIMENSIONS:
            if dimension in result:
                cells[str(group)][dimension].values.append(int(result[dimension]))
    return {
        group: {name: cell.to_dict() for name, cell in scores.items()}
        for group, scores in cells.items()
    }


def _counts_by(
    rows: list[dict[str, Any]],
    group_key: str,
    value_key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        group = row.get(group_key)
        if group is not None:
            counts[str(group)][str(row.get(value_key))] += 1
    return {group: dict(values) for group, values in counts.items()}


def _history_summary(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Summarize inputs that can cause a false difference between groups."""
    result: dict[str, dict[str, float | int]] = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row.get("group") == group]
        if not group_rows:
            continue
        result[group] = {"n": len(group_rows)}
        for key in ("n_entries", "n_evidence", "reviewed_week_count"):
            values = [
                float(row[key])
                for row in group_rows
                if row.get(key) is not None
            ]
            result[group][f"mean_{key}"] = (
                round(sum(values) / len(values), 3) if values else 0.0
            )
    return result


def build_report(
    manifest: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
    *,
    manifest_source: str = "",
    eval_source: str = "",
) -> dict[str, Any]:
    """Build the complete Drift/control comparison."""
    rows = _manifest_rows(manifest)
    _attach_eval_results(rows, eval_metrics)
    generator_models = {
        str(item["generator_model"])
        for item in manifest
        if item.get("generator_model")
    }
    if len(generator_models) > 1:
        raise ValueError("Manifest uses multiple Coach Digest generator models.")
    generator_model = (
        next(iter(generator_models)) if len(generator_models) == 1 else None
    )
    evaluator_model = eval_metrics.get("judge_model")
    eval_generator_model = eval_metrics.get("generator_model")
    if (
        generator_model is not None
        and eval_generator_model is not None
        and generator_model != eval_generator_model
    ):
        raise ValueError("Evaluator metrics use a different generator model.")
    return {
        "eval": "coach_digest_drift_control_comparison",
        "source": "mechanical_code_checks_and_ai_review",
        "note": (
            "Coach Digest Validations are code checks. Coach Digest Evals are AI "
            "review. Neither result is human validation."
        ),
        "manifest_source": manifest_source,
        "eval_source": eval_source,
        "generator_model": generator_model,
        "evaluator_model": evaluator_model,
        "cross_provider": (
            generator_model.split(":", 1)[0]
            != str(evaluator_model).split(":", 1)[0]
            if generator_model is not None and evaluator_model is not None
            else None
        ),
        "self_evaluation": (
            generator_model == evaluator_model
            if generator_model is not None and evaluator_model is not None
            else None
        ),
        "n_rows": len(rows),
        "n_by_group": {
            group: sum(row.get("group") == group for row in rows)
            for group in GROUPS
        },
        "validations_by_group": _rates_by(rows, "group"),
        "validations_by_known_delivery_state": _rates_by(
            rows,
            "known_delivery_state",
        ),
        "scores_by_group": _scores_by(rows, "group"),
        "scores_by_known_delivery_state": _scores_by(
            rows,
            "known_delivery_state",
        ),
        "response_mode_by_group": _counts_by(rows, "group", "response_mode"),
        "history_summary": _history_summary(rows),
        "match_quality": _counts_by(rows, "group", "match_quality").get(
            "control",
            {},
        ),
        "n_without_eval_result": sum(
            row.get("eval_result") is None for row in rows
        ),
        "rows": rows,
        "limits": [
            "The known Drift records are AI-reviewed synthetic development data.",
            (
                "A control target has no known Drift for its Persona. This is not "
                "human ground truth."
            ),
            (
                "Active and ended Drift records can require different Coach Digest "
                "responses."
            ),
            (
                "Specificity and tension honesty can differ by group because a "
                "control response has no known Drift to describe."
            ),
        ],
    }


def _render_score_table(
    lines: list[str],
    scores: dict[str, dict[str, dict[str, object]]],
    groups: tuple[str, ...],
) -> None:
    lines += [
        "| Dimension | " + " | ".join(groups) + " |",
        "| --- | " + " | ".join("---:" for _ in groups) + " |",
    ]
    for dimension in SCORE_DIMENSIONS:
        cells: list[str] = []
        for group in groups:
            cell = scores.get(group, {}).get(dimension)
            cells.append(
                f"{cell['mean']:.2f} (n={cell['n']})" if cell is not None else "—"
            )
        lines.append(f"| {dimension} | " + " | ".join(cells) + " |")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Coach Digest Drift and Control Comparison",
        "",
        "**Source:** Coach Digest Validations and Coach Digest Evals. These "
        "results are not human validation.",
        "",
        f"- Coach Digest responses: {report['n_rows']}",
        f"- Drift group: {report['n_by_group'].get('drift', 0)}",
        f"- Control group: {report['n_by_group'].get('control', 0)}",
        f"- Generator model: `{report.get('generator_model') or 'unrecorded'}`",
        f"- Evaluator model: `{report.get('evaluator_model') or 'unrecorded'}`",
        "- Cross-provider AI review: "
        + (
            "yes"
            if report.get("cross_provider") is True
            else "no"
            if report.get("cross_provider") is False
            else "unrecorded"
        ),
        f"- Responses without AI review: {report['n_without_eval_result']}",
        "",
        "## Coach Digest Validation pass rate",
        "",
        "| Check | Drift group | Control group |",
        "| --- | --- | --- |",
    ]
    rates = report["validations_by_group"]
    for name in CHECK_NAMES:
        cells: list[str] = []
        for group in GROUPS:
            cell = rates.get(group, {}).get(name)
            if cell is None:
                cells.append("—")
                continue
            low, high = cell["ci95"]
            cells.append(
                f"{cell['rate']:.0%} ({cell['passed']}/{cell['total']}; "
                f"95% interval {low:.0%}-{high:.0%})"
            )
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Coach Digest Eval mean by group", ""]
    _render_score_table(lines, report["scores_by_group"], GROUPS)
    state_groups = tuple(sorted(report["scores_by_known_delivery_state"]))
    if state_groups:
        lines += ["", "## Coach Digest Eval mean by known Drift state", ""]
        _render_score_table(
            lines,
            report["scores_by_known_delivery_state"],
            state_groups,
        )

    lines += [
        "",
        "## Input history by group",
        "",
        "| Group | Responses | Mean Journal Entries | Mean evidence items | "
        "Mean reviewed weeks |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    history = report["history_summary"]
    for group in GROUPS:
        row = history.get(group)
        if row is None:
            continue
        lines.append(
            f"| {group} | {row['n']} | {row['mean_n_entries']:.2f} | "
            f"{row['mean_n_evidence']:.2f} | "
            f"{row['mean_reviewed_week_count']:.2f} |"
        )

    lines += ["", "## Response mode by group", ""]
    for group in GROUPS:
        modes = report["response_mode_by_group"].get(group, {})
        value = ", ".join(
            f"{mode}={count}" for mode, count in sorted(modes.items())
        )
        lines.append(f"- {group}: {value or 'none'}")

    match_quality = report.get("match_quality") or {}
    if match_quality:
        value = ", ".join(
            f"{quality}={count}"
            for quality, count in sorted(match_quality.items())
        )
        lines += ["", "## Control match quality", "", value]

    lines += ["", "## Limits", ""]
    if report.get("self_evaluation") is True:
        lines += [
            "- One provider model generated and evaluated the responses. "
            "Correlated errors can make this AI review too favorable."
        ]
    lines += [f"- {limit}" for limit in report["limits"]]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Coach Digest results for Drift and control targets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    eval_metrics = json.loads(args.eval_metrics.read_text())
    report = build_report(
        manifest,
        eval_metrics,
        manifest_source=str(args.manifest),
        eval_source=str(args.eval_metrics),
    )
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "report.md").write_text(render_markdown(report))
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
