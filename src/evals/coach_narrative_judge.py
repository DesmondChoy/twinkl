"""Coach Digest Evals for Coach Digest responses.

Scores a Coach Digest response against its Weekly Drift Detection evidence.
The four dimensions are correctness, specificity, non-prescriptive tone, and
tension honesty. It also checks whether the reflective question is open-ended
and relevant.

These are **AI evaluation scores, not human validation**. They are a low-cost,
repeatable proxy for response quality. Future human calibration of the AI
review is required before treating these scores as ground truth.

The evaluator LLM is an injected ``LLMCompleteFn``. Coach Digest generation uses
the same contract. This module is provider-agnostic and testable. An empty,
malformed, or invalid evaluator response yields a ``None`` verdict that is
skipped in aggregation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, ValidationError

from prompts import load_prompt
from src.coach.llm_client import build_llm_complete
from src.coach.schemas import CoachNarrative, WeeklyDigest
from src.coach.weekly_digest import (
    LLMCompleteFn,
    build_coach_digest_prompt_inputs,
)

# Flag any dimension scoring below this for human review.
REVIEW_THRESHOLD = 3
# Target mean per dimension from the evaluation guide.
MEAN_TARGET = 3.5

COACH_NARRATIVE_JUDGE_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "name": "coach_narrative_judge",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
            "non_prescriptive_tone": {"type": "integer", "minimum": 1, "maximum": 5},
            "tension_honesty": {"type": "integer", "minimum": 1, "maximum": 5},
            "question_is_open_and_relevant": {"type": "boolean"},
            "justification": {"type": "string"},
        },
        "required": [
            "correctness",
            "specificity",
            "non_prescriptive_tone",
            "tension_honesty",
            "question_is_open_and_relevant",
            "justification",
        ],
    },
}

SCORE_DIMENSIONS = (
    "correctness",
    "specificity",
    "non_prescriptive_tone",
    "tension_honesty",
)


class JudgeVerdict(BaseModel):
    """One AI evaluation verdict for a single Coach Digest response."""

    correctness: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    non_prescriptive_tone: int = Field(ge=1, le=5)
    tension_honesty: int = Field(ge=1, le=5)
    question_is_open_and_relevant: bool
    justification: str

    @property
    def needs_review(self) -> bool:
        return any(getattr(self, dim) < REVIEW_THRESHOLD for dim in SCORE_DIMENSIONS)


def render_judge_prompt(digest: WeeklyDigest, narrative: CoachNarrative) -> str:
    """Render the AI evaluation prompt for one response and its facts."""
    template = load_prompt("coach_narrative_judge")
    factual_inputs = build_coach_digest_prompt_inputs(digest)
    return cast(
        str,
        template.render(
            **factual_inputs,
            weekly_mirror=narrative.weekly_mirror,
            tension_explanation=narrative.tension_explanation,
            reflective_question=narrative.reflective_question,
        ),
    )


async def judge_narrative(
    digest: WeeklyDigest,
    narrative: CoachNarrative,
    llm_complete: LLMCompleteFn,
) -> JudgeVerdict | None:
    """Score one response with the evaluator LLM; return None on failure."""
    prompt = render_judge_prompt(digest, narrative)
    raw = await llm_complete(prompt, COACH_NARRATIVE_JUDGE_RESPONSE_FORMAT)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return cast(JudgeVerdict, JudgeVerdict.model_validate(payload))
    except ValidationError:
        return None


@dataclass
class JudgeReport:
    """Aggregated AI evaluation results for Coach Digest responses."""

    judge_model: str
    n_scored: int
    means: dict[str, float] = field(default_factory=dict)
    pct_ge_4: dict[str, float] = field(default_factory=dict)
    question_open_rate: float = 0.0
    n_flagged: int = 0
    n_failed: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "eval": "coach_digest_evals",
            "source": "ai_review",
            "note": (
                "AI evaluation scores, NOT human validation. Future human "
                "calibration of the AI review is required before treating "
                "these as ground truth."
            ),
            "judge_model": self.judge_model,
            "n_scored": self.n_scored,
            "n_failed": self.n_failed,
            "means": {k: round(v, 3) for k, v in self.means.items()},
            "mean_target": MEAN_TARGET,
            "meets_mean_target": {
                dim: (self.means.get(dim, 0.0) >= MEAN_TARGET)
                for dim in SCORE_DIMENSIONS
            },
            "pct_ge_4": {k: round(v, 3) for k, v in self.pct_ge_4.items()},
            "question_open_rate": round(self.question_open_rate, 3),
            "n_flagged_for_review": self.n_flagged,
        }


def aggregate_verdicts(
    verdicts: list[JudgeVerdict | None], judge_model: str
) -> JudgeReport:
    """Aggregate per-narrative verdicts into a report; None verdicts count as failed."""
    scored = [v for v in verdicts if v is not None]
    n = len(scored)
    means: dict[str, float] = {}
    pct_ge_4: dict[str, float] = {}
    for dim in SCORE_DIMENSIONS:
        values = [getattr(v, dim) for v in scored]
        means[dim] = sum(values) / n if n else 0.0
        pct_ge_4[dim] = (sum(1 for x in values if x >= 4) / n) if n else 0.0
    question_open_rate = (
        sum(1 for v in scored if v.question_is_open_and_relevant) / n if n else 0.0
    )
    return JudgeReport(
        judge_model=judge_model,
        n_scored=n,
        means=means,
        pct_ge_4=pct_ge_4,
        question_open_rate=question_open_rate,
        n_flagged=sum(1 for v in scored if v.needs_review),
        n_failed=sum(1 for v in verdicts if v is None),
    )


def render_markdown(report: JudgeReport) -> str:
    """Render a short markdown summary of an AI evaluation report."""
    lines = [
        "# Coach Digest Evals Report",
        "",
        "**Source:** AI evaluation scores, NOT human validation. Future human "
        "calibration of the AI review remains separate work.",
        "",
        f"- Evaluator model: `{report.judge_model}`",
        f"- Scored: {report.n_scored}",
        f"- Failed (no valid verdict): {report.n_failed}",
        f"- Flagged for human review (any dimension < {REVIEW_THRESHOLD}): "
        f"{report.n_flagged}",
        f"- Reflective question open & relevant: {report.question_open_rate:.0%}",
        "",
        "| Dimension | Mean | Target | Meets | % ≥ 4 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dim in SCORE_DIMENSIONS:
        mean = report.means.get(dim, 0.0)
        meets = "✅" if mean >= MEAN_TARGET else "❌"
        lines.append(
            f"| {dim} | {mean:.2f} | ≥ {MEAN_TARGET} | {meets} | "
            f"{report.pct_ge_4.get(dim, 0.0):.0%} |"
        )
    return "\n".join(lines) + "\n"


async def _run_sample(
    pairs: list[tuple[WeeklyDigest, CoachNarrative]],
    llm_complete: LLMCompleteFn,
) -> list[JudgeVerdict | None]:
    return [await judge_narrative(d, n, llm_complete) for d, n in pairs]


def _load_manifest(manifest_path: Path) -> list[tuple[WeeklyDigest, CoachNarrative]]:
    """Load digest + narrative pairs from a committed sample manifest.

    The manifest is a JSON list of objects, each with a ``digest`` object and a
    ``narrative`` object matching the WeeklyDigest / CoachNarrative schemas.
    """
    raw = json.loads(manifest_path.read_text())
    pairs: list[tuple[WeeklyDigest, CoachNarrative]] = []
    for item in raw:
        digest = WeeklyDigest.model_validate(item["digest"])
        narrative = CoachNarrative.model_validate(item["narrative"])
        pairs.append((digest, narrative))
    return pairs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Coach Digest Evals with an AI evaluator."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON file of {digest, narrative} sample pairs to evaluate.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize paid evaluator LLM calls. Without it, this prints the plan "
        "and makes no calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pairs = _load_manifest(args.manifest)

    if not args.execute:
        print(
            f"[dry run] Would evaluate {len(pairs)} Coach Digest response(s) "
            "with the "
            "configured provider. Re-run with --execute to make paid calls."
        )
        return 0

    llm_complete = build_llm_complete()
    if llm_complete is None:
        print("No evaluator provider available (missing API key). Aborting.")
        return 1

    judge_model = "unknown"  # provider resolves its own model internally
    verdicts = asyncio.run(_run_sample(pairs, llm_complete))
    report = aggregate_verdicts(verdicts, judge_model=judge_model)

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
