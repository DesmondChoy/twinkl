#!/usr/bin/env python3
"""Estimate and run isolated exact-state reviews for the a30f full corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import AlignmentScores  # noqa: E402
from src.vif.security_target import read_jsonl  # noqa: E402

DEFAULT_MODEL = "gpt-5.4-mini"
MODEL_PRICING_PER_1M = {"input": 0.75, "output": 4.50}
DEFAULT_MANIFEST = Path(
    "logs/exports/twinkl_a30f_active_critic_state_full_v1/"
    "active_critic_state_manifest.jsonl"
)
DEFAULT_RESULTS_DIR = Path(
    "logs/exports/twinkl_a30f_active_critic_state_full_v1/results"
)


class ExactStateReview(BaseModel):
    scores: AlignmentScores
    security_rationale: str
    security_confidence: Literal["high", "medium", "low"]


def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def estimate(manifest: list[dict[str, Any]], *, calls_per_case: int = 3) -> dict:
    input_tokens = sum(approx_token_count(row["prompt"]) for row in manifest)
    output_per_call = 220
    return {
        "case_count": len(manifest),
        "planned_calls": len(manifest) * calls_per_case,
        "estimated_input_tokens": input_tokens * calls_per_case,
        "estimated_output_tokens": len(manifest) * calls_per_case * output_per_call,
        "estimated_cost_usd": (
            input_tokens * calls_per_case * MODEL_PRICING_PER_1M["input"]
            + len(manifest)
            * calls_per_case
            * output_per_call
            * MODEL_PRICING_PER_1M["output"]
        )
        / 1_000_000,
        "tiebreak_calls_excluded": True,
    }


def _result_path(results_dir: Path, pass_index: int) -> Path:
    name = (
        "tiebreak_results.jsonl"
        if pass_index == 4
        else f"pass_{pass_index}_results.jsonl"
    )
    return results_dir / name


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def response_usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        key: getattr(usage, key, None) if usage is not None else None
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def call_openai(
    prompt: str, *, model: str, reasoning_effort: str, timeout: float
) -> tuple[ExactStateReview, Any]:
    """Make one API request containing exactly one canonical prompt."""
    from openai import OpenAI

    response = OpenAI().responses.parse(
        model=model,
        input=prompt,
        text_format=ExactStateReview,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=1200,
        store=False,
        timeout=timeout,
    )
    parsed = getattr(response, "output_parsed", None)
    if not isinstance(parsed, ExactStateReview):
        raise ValueError("OpenAI response did not contain an exact-state review.")
    return parsed, response


def tiebreak_case_ids(
    results_dir: Path, *, expected_case_ids: set[str]
) -> set[str]:
    passes = [read_jsonl(_result_path(results_dir, index)) for index in (1, 2, 3)]
    labels: dict[str, list[int]] = {}
    for rows in passes:
        observed = {str(row["case_id"]) for row in rows}
        if observed != expected_case_ids or len(observed) != len(rows):
            raise ValueError(
                "Passes 1-3 must be complete and unique before tie-break review."
            )
        for row in rows:
            if row.get("status") == "ok":
                labels.setdefault(str(row["case_id"]), []).append(
                    int(row["scores"]["security"])
                )
    return {
        case_id
        for case_id, votes in labels.items()
        if len(votes) == 3 and set(votes) == {-1, 0, 1}
    }


def run_pass(
    manifest: list[dict[str, Any]],
    *,
    results_dir: Path,
    pass_index: int,
    execute: bool,
    model: str,
    reasoning_effort: str,
    timeout: float,
    max_attempts: int,
    workers: int = 1,
) -> dict:
    if pass_index not in {1, 2, 3, 4}:
        raise ValueError("pass_index must be 1, 2, 3, or 4.")
    eligible = (
        tiebreak_case_ids(
            results_dir,
            expected_case_ids={str(row["case_id"]) for row in manifest},
        )
        if pass_index == 4
        else {row["case_id"] for row in manifest}
    )
    path = _result_path(results_dir, pass_index)
    existing = read_jsonl(path) if path.exists() else []
    by_case = {str(row["case_id"]): row for row in existing}
    if len(by_case) != len(existing):
        raise ValueError(f"Duplicate case IDs in resumable result file: {path}")
    manifest_by_case = {str(row["case_id"]): row for row in manifest}
    for case_id, row in by_case.items():
        source = manifest_by_case.get(case_id)
        if source is None or case_id not in eligible:
            raise ValueError(f"Unexpected case in resumable result file: {case_id}")
        if int(row.get("pass_index", -1)) != pass_index or row.get("status") != "ok":
            raise ValueError(f"Invalid resumable result state for {case_id}")
        for field in (
            "state_contract_version",
            "state_input_sha256",
            "prompt_sha256",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(
                    f"Stale resumable result receipt for {case_id}: {field}"
                )
    pending = [
        row
        for row in manifest
        if row["case_id"] in eligible and row["case_id"] not in by_case
    ]
    if not execute:
        return {
            "pass_index": pass_index,
            "eligible": len(eligible),
            "complete": len(existing),
            "pending": len(pending),
        }

    def review_one(manifest_row: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                parsed, response = call_openai(
                    manifest_row["prompt"],
                    model=model,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                )
                scores = {
                    key: int(value) for key, value in parsed.scores.model_dump().items()
                }
                rationale = parsed.security_rationale.strip()
                if scores["security"] != 0 and not rationale:
                    raise ValueError("Non-neutral Security review has no rationale.")
                usage = response_usage(response)
                actual_input = int(
                    usage["input_tokens"] or approx_token_count(manifest_row["prompt"])
                )
                actual_output = int(usage["output_tokens"] or 0)
                record = {
                    "case_id": manifest_row["case_id"],
                    "state_contract_version": manifest_row["state_contract_version"],
                    "state_input_sha256": manifest_row["state_input_sha256"],
                    "prompt_sha256": manifest_row["prompt_sha256"],
                    "reviewer": f"openai:{model}:pass_{pass_index}",
                    "reviewed_at": datetime.now(UTC).isoformat(),
                    "confidence": parsed.security_confidence,
                    "rationale_status": "not_applicable_neutral"
                    if scores["security"] == 0
                    else "provided",
                    "scores": scores,
                    "rationales": {"security": rationale} if rationale else {},
                    "pass_index": pass_index,
                    "model_requested": model,
                    "response_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "reasoning_effort": reasoning_effort,
                    "usage": usage,
                    "estimated_cost_usd": (
                        actual_input * MODEL_PRICING_PER_1M["input"]
                        + actual_output * MODEL_PRICING_PER_1M["output"]
                    )
                    / 1_000_000,
                    "latency_seconds": time.perf_counter() - started,
                    "attempts": attempt,
                    "status": "ok",
                }
                return record
            except Exception as exc:  # pragma: no cover - API edge handling
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 5))
        raise RuntimeError(
            f"Review failed for {manifest_row['case_id']}: {last_error}"
        ) from last_error

    if workers < 1:
        raise ValueError("workers must be at least 1.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(review_one, row): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            by_case[record["case_id"]] = record
            _atomic_write_jsonl(path, [by_case[key] for key in sorted(by_case)])
    return {
        "pass_index": pass_index,
        "eligible": len(eligible),
        "complete": len(by_case),
        "pending": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("estimate", "dry-run", "execute"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--pass-index", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=("none", "low"), default="none")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        help="Restrict the manifest for a smoke test; resumable results remain valid.",
    )
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be at least 1.")
        manifest = manifest[: args.limit]
    if args.mode == "estimate":
        payload = estimate(manifest)
    else:
        if args.mode == "execute":
            load_dotenv(ROOT / ".env")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    f"OPENAI_API_KEY is not set in the environment or {ROOT / '.env'}."
                )
        payload = run_pass(
            manifest,
            results_dir=args.results_dir,
            pass_index=args.pass_index,
            execute=args.mode == "execute",
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            workers=args.workers,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
