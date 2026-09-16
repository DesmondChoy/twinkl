"""Recheck three captured onboarding cases; --execute makes three Coach calls.

Run from the repository root with PYTHONPATH=. after activating .venv. Without
--execute, revalidate the original outputs and any existing receipts.json.
No evaluator, retry, application session, or saved Persona output is changed.
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
from src.coach.llm_client import build_llm_complete
from src.coach.schemas import CoachNarrative, LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import (
    generate_weekly_digest_coach_diagnostic,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


async def run(execute: bool, output: Path) -> None:
    cases = json.loads((HERE / "inputs.json").read_text())["cases"]
    assert len(cases) == 3
    if execute and output.exists():
        raise SystemExit("Choose a new output path; existing receipts are immutable.")
    input_hash = hashlib.sha256((HERE / "inputs.json").read_bytes()).hexdigest()
    recorded = {}
    receipts = HERE / "receipts.json"
    if not execute and receipts.exists():
        previous = json.loads(receipts.read_text())
        if previous["input_file_sha256"] != input_hash:
            raise SystemExit("The captured generation inputs have changed.")
        recorded = {row["case_id"]: row for row in previous["cases"]}
    metadata = get_prompt_metadata("weekly_digest_coach")
    if metadata["version"] != "4.7":
        raise SystemExit("This check requires Coach prompt 4.7.")
    metrics: list[LLMCallMetrics] = []
    load_dotenv(ROOT / ".env")
    llm = build_llm_complete(
        provider="openai", model="gpt-5.6-luna", call_metrics=metrics,
    ) if execute else None
    if execute and llm is None:
        raise SystemExit("The configured provider is unavailable.")
    rows: list[dict[str, object]] = []
    result = {
        "started_at": datetime.now(UTC).isoformat(),
        "executed": execute,
        "prompt_metadata": metadata,
        "requested_model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "seed": None,
        "maximum_calls": 3,
        "input_file_sha256": input_hash,
        "source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/coach/weekly_digest.py", "src/coach/llm_client.py",
                "src/coach/schemas.py", "src/model_guardrails.py",
                "prompts/weekly_digest_coach.yaml",
            )
        },
        "cases": rows,
    }
    for case in cases:
        digest = WeeklyDigest.model_validate(case["digest"])
        original = CoachNarrative.model_validate(case["original_narrative"])
        validation = validate_weekly_digest_narrative(
            digest, original, validate_voice=True,
        )
        row: dict[str, object] = {
            "case_id": case["case_id"],
            "original_revalidation": validation.model_dump(mode="json"),
            "prompt": render_digest_prompt(digest),
        }
        if case["case_id"] in recorded:
            captured = recorded[case["case_id"]]["diagnostic"]["narrative"]
            row["recorded_response_validation"] = (
                validate_weekly_digest_narrative(
                    digest, CoachNarrative.model_validate(captured),
                    validate_voice=True,
                ).model_dump(mode="json") if captured is not None else None
            )
        if llm is not None:
            diagnostic, prompt = await generate_weekly_digest_coach_diagnostic(
                digest, llm,
            )
            row.update({
                "prompt": prompt,
                "diagnostic": diagnostic.model_dump(mode="json"),
                "metrics": metrics[-1].model_dump(mode="json"),
            })
        rows.append(row)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(case["case_id"], "recorded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "verification.json")
    arguments = parser.parse_args()
    asyncio.run(run(arguments.execute, arguments.output))
