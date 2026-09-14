"""Bounded diagnostic rerun of the two prompt-audit inputs; never applies fixtures.

From the repository root, activate the environment, then run:
PYTHONPATH=. python logs/experiments/reports/coach_prompt_audit_20260913/rerun.py
Add --execute for the user-authorized generation (at most four provider calls).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from prompts import get_prompt_metadata
from scripts.coach.complete_scenario_coach import collect_cases
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    build_llm_complete,
    summarize_llm_call_metrics,
)
from src.coach.schemas import CoachNarrative, LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import (
    generate_weekly_digest_coach_diagnostic,
    render_digest_messages,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)
from tests.coach.test_generate_approved_judge_sample import _digest

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "prompts/weekly_digest_coach.yaml",
    "prompts/coach_narrative_judge.yaml",
    "config/schwartz_values.yaml",
    "src/coach/weekly_digest.py",
    "src/coach/llm_client.py",
    "src/coach/schemas.py",
    "src/model_guardrails.py",
    "src/prompt_boundary.py",
    "tests/coach/test_generate_approved_judge_sample.py",
)


def write(name, value):
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def prepare():
    cases = collect_cases(ROOT)
    wei_jun = cases["uncertain-wei-jun::2025-06-30"]
    old_path = (
        ROOT / "logs/experiments/reports/coach_voice_last_20260909"
        "/inputs/uncertain-wei-jun_2025-06-30.json"
    )
    digest = WeeklyDigest.model_validate(wei_jun["digest"])
    old_input = json.loads(old_path.read_text())
    assert digest == WeeklyDigest.model_validate(old_input["digest"])
    inputs = {"wei_jun": digest, "casey": _digest("casey")}
    manifest = {
        "issue": "twinkl-u3sd",
        "purpose": "User-authorized rerun of the three selected prompt-audit fixes",
        "prompt": get_prompt_metadata("weekly_digest_coach"),
        "provider": "openai",
        "model": DEFAULT_OPENAI_MODEL,
        "reasoning_effort": DEFAULT_OPENAI_REASONING_EFFORT,
        "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "sdk_retries": 0,
        "maximum_attempts_per_input": 2,
        "seed": None,
        "initial_editorial_repairs": [],
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in SOURCE_FILES
        },
        "wei_jun_source": wei_jun,
        "casey_source": "Synthetic audit fixture shared by quotation and length cases",
        "requests": {},
    }
    for name, item in inputs.items():
        instructions, data = render_digest_messages(item)
        request = {
            "digest": item.model_dump(mode="json"),
            "instructions": instructions,
            "input": data,
            "prompt_sha256": hashlib.sha256(
                render_digest_prompt(item).encode()
            ).hexdigest(),
        }
        manifest["requests"][name] = request
    return manifest


async def execute(manifest, *, editorial_repair=False):
    load_dotenv(ROOT / ".env")
    results = {}
    metrics = []
    complete = build_llm_complete(
        provider="openai", model=DEFAULT_OPENAI_MODEL,
        timeout=60, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        call_metrics=metrics,
    )
    if complete is None:
        raise RuntimeError("OpenAI provider credentials are unavailable")
    for name, request in manifest["requests"].items():
        digest = WeeklyDigest.model_validate(request["digest"])
        prompt_hash = hashlib.sha256(render_digest_prompt(digest).encode()).hexdigest()
        assert prompt_hash == request["prompt_sha256"]
        repairs, attempts = [], []
        if editorial_repair:
            prior = json.loads((OUTPUT / f"{name}.attempt_1.json").read_text())
            review = json.loads((OUTPUT / "editorial_review.json").read_text())[name]
            raw_hash = hashlib.sha256(
                prior["diagnostic"]["raw_output"].encode()
            ).hexdigest()
            assert raw_hash == review["response_sha256"]
            repairs = review["requirements"]
            attempts.append(prior)
            metrics.extend(LLMCallMetrics.model_validate(m) for m in prior["metrics"])
        for attempt in range(2 if editorial_repair else 1, 3):
            start = len(metrics)
            diagnostic, prompt = await generate_weekly_digest_coach_diagnostic(
                digest, complete, repair_requirements=repairs,
            )
            result = {
                "generated_at": datetime.now(UTC).isoformat(),
                "attempt": attempt,
                "repair_requirements": list(repairs),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "diagnostic": diagnostic.model_dump(mode="json"),
                "metrics": [m.model_dump(mode="json") for m in metrics[start:]],
            }
            write(f"{name}.attempt_{attempt}.json", result)
            attempts.append(result)
            if diagnostic.accepted or diagnostic.failure_stage != "coach_validation":
                break
            repairs = diagnostic.failure_details
        results[name] = {
            "attempts": len(attempts),
            "accepted": diagnostic.accepted,
            "response": (
                diagnostic.narrative.model_dump() if diagnostic.narrative else None
            ),
        }
    short = CoachNarrative(
        weekly_mirror='You "called my mom".',
        tension_explanation="You also helped a colleague debug.",
        reflective_question="What stayed with you after the call?",
    )
    write("short_response_check.json", {
        "source": "Constructed audit response; not provider output",
        "response": short.model_dump(),
        "current": validate_weekly_digest_narrative(
            _digest("casey"), short
        ).model_dump(),
        "historical": validate_weekly_digest_narrative(
            _digest("casey"), short, validation_policy="historical"
        ).model_dump(),
    })
    summary = {"results": results, "usage": summarize_llm_call_metrics(metrics)}
    write("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--editorial-repair", action="store_true")
    args = parser.parse_args()
    if args.editorial_repair:
        if not args.execute or any(OUTPUT.glob("*.attempt_2.json")):
            raise SystemExit(
                "Editorial repair requires execution and no second attempt"
            )
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        assert manifest == prepare(), "Inputs or source changed since initial requests"
    elif any(OUTPUT.glob("*.attempt_*.json")):
        raise SystemExit("Existing provider receipts must not be overwritten")
    else:
        manifest = prepare()
        write("manifest.json", manifest)
    if args.execute:
        asyncio.run(execute(manifest, editorial_repair=args.editorial_repair))
    else:
        print("Prepared two frozen inputs; no provider calls made.")
