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

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from prompts import load_prompt
from src.coach.llm_client import (
    OPENAI_LUNA_PRICING_SOURCE,
    build_llm_complete,
    resolve_coach_model,
    summarize_llm_call_metrics,
)
from src.coach.schemas import CoachNarrative, LLMCallMetrics, WeeklyDigest
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
    generator_model: str | None = None
    call_metrics: list[LLMCallMetrics] = field(default_factory=list)
    sample_results: list[dict[str, object]] = field(default_factory=list)

    @property
    def self_evaluation(self) -> bool:
        """True when the same model wrote and scored the Coach Narratives."""
        return (
            self.generator_model is not None
            and self.generator_model == self.judge_model
        )

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
            "generator_model": self.generator_model,
            "self_evaluation": self.self_evaluation,
            "self_evaluation_note": (
                "The same model wrote and scored these Coach Narratives. "
                "Treat the scores as self evaluation and expect them to be "
                "too high."
                if self.self_evaluation
                else None
            ),
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
            "score_distributions": {
                dim: {
                    str(score): sum(
                        result.get(dim) == score
                        for result in self.sample_results
                        if result.get("status") == "scored"
                    )
                    for score in range(1, 6)
                }
                for dim in SCORE_DIMENSIONS
            },
            "sample_results": self.sample_results,
            "api_usage": summarize_llm_call_metrics(self.call_metrics),
            "api_calls": [
                metric.model_dump(mode="json") for metric in self.call_metrics
            ],
            "pricing_source": OPENAI_LUNA_PRICING_SOURCE,
            "cost_note": (
                "Calculated from response token usage and published standard-tier "
                "rates. This is not an OpenAI billing export."
            ),
        }


def aggregate_verdicts(
    verdicts: list[JudgeVerdict | None],
    judge_model: str,
    generator_model: str | None = None,
    call_metrics: list[LLMCallMetrics] | None = None,
    sample_labels: list[str] | None = None,
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
    labels = sample_labels or [
        f"sample_{index}" for index in range(1, len(verdicts) + 1)
    ]
    if len(labels) != len(verdicts):
        raise ValueError("Sample labels must match the number of verdicts.")
    sample_results: list[dict[str, object]] = []
    for label, verdict in zip(labels, verdicts, strict=True):
        if verdict is None:
            sample_results.append(
                {"sample_id": label, "status": "failed", "needs_review": True}
            )
            continue
        sample_results.append(
            {
                "sample_id": label,
                "status": "scored",
                **verdict.model_dump(mode="json"),
                "needs_review": verdict.needs_review,
            }
        )
    return JudgeReport(
        judge_model=judge_model,
        generator_model=generator_model,
        n_scored=n,
        means=means,
        pct_ge_4=pct_ge_4,
        question_open_rate=question_open_rate,
        n_flagged=sum(1 for v in scored if v.needs_review),
        n_failed=sum(1 for v in verdicts if v is None),
        call_metrics=list(call_metrics or []),
        sample_results=sample_results,
    )


def render_markdown(report: JudgeReport) -> str:
    """Render a short markdown summary of an AI evaluation report."""
    usage = summarize_llm_call_metrics(report.call_metrics)
    cost = usage["calculated_cost_usd"]
    lines = [
        "# Coach Digest Evals Report",
        "",
        "**Source:** AI evaluation scores, NOT human validation. Future human "
        "calibration of the AI review remains separate work.",
        "",
    ]
    if report.self_evaluation:
        lines += [
            "**Self evaluation:** the same model wrote and scored these Coach "
            "Narratives. The scores are too high. Do not read them as an "
            "independent measurement.",
            "",
        ]
    lines += [
        f"- Evaluator model: `{report.judge_model}`",
        f"- Generator model: `{report.generator_model or 'unrecorded'}`",
        f"- Scored: {report.n_scored}",
        f"- Failed (no valid verdict): {report.n_failed}",
        f"- Flagged for human review (any dimension < {REVIEW_THRESHOLD}): "
        f"{report.n_flagged}",
        f"- Reflective question open & relevant: {report.question_open_rate:.0%}",
        f"- Paid API calls recorded: {usage['n_calls']}",
        f"- Input tokens: {usage['input_tokens']}",
        f"- Cached input tokens: {usage['cached_input_tokens']}",
        f"- Output tokens: {usage['output_tokens']}",
        (
            f"- Calculated published-rate cost: `${float(cost):.8f}`"
            if cost is not None
            else "- Calculated published-rate cost: unavailable"
        ),
        f"- Total request latency: "
        f"{float(usage['total_latency_seconds'] or 0):.3f}s",
        (
            "- Mean / median / maximum request latency: "
            f"{float(usage['mean_latency_seconds'] or 0):.3f}s / "
            f"{float(usage['median_latency_seconds'] or 0):.3f}s / "
            f"{float(usage['max_latency_seconds'] or 0):.3f}s"
        ),
        "- Cost basis: response token usage and published standard-tier Luna "
        f"rates ([source]({OPENAI_LUNA_PRICING_SOURCE})); not a billing export",
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
    if report.sample_results:
        lines += [
            "",
            "## Per-Response Scores",
            "",
            "| Response | Correctness | Specificity | Non-prescriptive tone | "
            "Tension honesty | Question | Review flag |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for result in report.sample_results:
            if result.get("status") != "scored":
                lines.append(
                    f"| {result['sample_id']} | — | — | — | — | failed | yes |"
                )
                continue
            question = (
                "pass" if result["question_is_open_and_relevant"] else "fail"
            )
            review = "yes" if result["needs_review"] else "no"
            lines.append(
                f"| {result['sample_id']} | {result['correctness']} | "
                f"{result['specificity']} | "
                f"{result['non_prescriptive_tone']} | "
                f"{result['tension_honesty']} | {question} | {review} |"
            )

        lines += ["", "## Evaluator Justifications"]
        for result in report.sample_results:
            lines += [
                "",
                f"### {result['sample_id']}",
                "",
                str(result.get("justification") or "No valid verdict was returned."),
            ]
    if report.call_metrics:
        lines += [
            "",
            "## Per-Call API Metrics",
            "",
            "| Call | Input | Cached | Output | Latency | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for metric in report.call_metrics:
            metric_cost = (
                f"${metric.calculated_cost_usd:.8f}"
                if metric.calculated_cost_usd is not None
                else "—"
            )
            lines.append(
                f"| {metric.call_label or 'unlabeled'} | "
                f"{metric.input_tokens or 0} | "
                f"{metric.cached_input_tokens or 0} | "
                f"{metric.output_tokens or 0} | "
                f"{metric.latency_seconds:.3f}s | {metric_cost} |"
            )
    return "\n".join(lines) + "\n"


async def _run_sample(
    pairs: list[tuple[WeeklyDigest, CoachNarrative]],
    llm_complete: LLMCompleteFn,
    call_metrics: list[LLMCallMetrics] | None = None,
) -> list[JudgeVerdict | None]:
    verdicts: list[JudgeVerdict | None] = []
    for digest, narrative in pairs:
        metrics_before_call = len(call_metrics or [])
        verdicts.append(await judge_narrative(digest, narrative, llm_complete))
        if call_metrics is not None and len(call_metrics) > metrics_before_call:
            call_metrics[-1].call_label = (
                f"coach_eval:{digest.persona_id}:{digest.week_end}"
            )
    return verdicts


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


def _load_generator_model(manifest_path: Path) -> str | None:
    """Read the generator model id recorded on the manifest entries.

    The driver script records ``generator_model`` on every entry. Returns
    ``None`` when the manifest predates that field or the entries disagree,
    so the report says ``unrecorded`` instead of claiming a wrong model.
    """
    raw = json.loads(manifest_path.read_text())
    recorded = {
        item.get("generator_model")
        for item in raw
        if isinstance(item, dict) and item.get("generator_model")
    }
    if len(recorded) != 1:
        return None
    return str(next(iter(recorded)))


def _verdict_records(
    pairs: list[tuple[WeeklyDigest, CoachNarrative]],
    verdicts: list[JudgeVerdict | None],
) -> list[dict[str, object]]:
    """Pair each verdict with its Persona and week, for the comparison report."""
    records: list[dict[str, object]] = []
    for (digest, _narrative), verdict in zip(pairs, verdicts, strict=True):
        record: dict[str, object] = {
            "key": f"{digest.persona_id}:{digest.week_end}",
            "persona_id": digest.persona_id,
            "week_end": digest.week_end,
            "response_mode": digest.response_mode,
        }
        if verdict is None:
            record["verdict"] = None
        else:
            record["verdict"] = {
                **{dim: getattr(verdict, dim) for dim in SCORE_DIMENSIONS},
                "question_is_open_and_relevant": (
                    verdict.question_is_open_and_relevant
                ),
                "needs_review": verdict.needs_review,
                "justification": verdict.justification,
            }
        records.append(record)
    return records


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
    parser.add_argument(
        "--judge-provider",
        default=None,
        help="Evaluator provider (openai or gemini). Defaults to "
        "TWINKL_COACH_PROVIDER. Set a provider other than the generator's to "
        "avoid self evaluation.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Evaluator model id. Defaults to TWINKL_COACH_MODEL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    pairs = _load_manifest(args.manifest)

    if not args.execute:
        print(
            f"[dry run] Would evaluate {len(pairs)} Coach Digest response(s) "
            "with the "
            "configured provider. Re-run with --execute to make paid calls."
        )
        return 0

    call_metrics: list[LLMCallMetrics] = []
    llm_complete = build_llm_complete(
        provider=args.judge_provider,
        model=args.judge_model,
        call_metrics=call_metrics,
    )
    if llm_complete is None:
        print("No evaluator provider available (missing API key). Aborting.")
        return 1

    judge_model = resolve_coach_model(
        provider=args.judge_provider,
        model=args.judge_model,
    )
    generator_model = _load_generator_model(args.manifest)
    verdicts = asyncio.run(_run_sample(pairs, llm_complete, call_metrics))
    sample_labels = [
        f"{digest.persona_id}:{digest.week_end}" for digest, _narrative in pairs
    ]
    report = aggregate_verdicts(
        verdicts,
        judge_model=judge_model,
        generator_model=generator_model,
        call_metrics=call_metrics,
        sample_labels=sample_labels,
    )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n"
        )
        (args.out / "report.md").write_text(render_markdown(report))
        (args.out / "verdicts.json").write_text(
            json.dumps(_verdict_records(pairs, verdicts), indent=2) + "\n"
        )

    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
