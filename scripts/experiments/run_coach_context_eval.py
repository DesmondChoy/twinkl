"""Compare frozen and full selected Coach evidence; dry-run unless --execute.

This development experiment makes at most six generation and nine evaluator
calls. It never repairs responses or applies them to saved Persona fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from prompts import get_prompt_metadata
from scripts.experiments.coach_context_sources import freeze_sources, verify_sources
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    DEFAULT_TIMEOUT_SECONDS,
    build_llm_complete,
    summarize_llm_call_metrics,
)
from src.coach.schemas import CoachNarrative, LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import (
    WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
    _evidence_from_decisions,
    generate_weekly_digest_coach_diagnostic,
    render_digest_messages,
    render_digest_prompt,
)
from src.evals.coach_narrative_judge import (
    COACH_NARRATIVE_JUDGE_RESPONSE_FORMAT,
    judge_narrative,
    render_judge_prompt,
)
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path("logs/experiments/reports/coach_prompt_audit_20260913")
SOURCE_PATHS = (
    "scripts/experiments/run_coach_context_eval.py",
    "scripts/experiments/coach_context_sources.py",
    "tests/experiments/test_run_coach_context_eval.py",
    "prompts/weekly_digest_coach.yaml",
    "prompts/coach_narrative_judge.yaml",
    "config/schwartz_values.yaml",
    "src/coach/weekly_digest.py",
    "src/coach/llm_client.py",
    "src/coach/schemas.py",
    "src/evals/coach_narrative_judge.py",
    "src/model_guardrails.py",
    "src/prompt_boundary.py",
)
SETTINGS = {
    "provider": "openai",
    "model": DEFAULT_OPENAI_MODEL,
    "reasoning_effort": DEFAULT_OPENAI_REASONING_EFFORT,
    "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
    "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
    "sdk_retries": 0,
    "attempts_per_response": 1,
    "seed": None,
    "store": False,
}
GOOD_CONTROL = CoachNarrative(
    weekly_mirror=(
        "When your lead said the planned sprint could not fit a fix for the failed "
        'transfers, you wrote, "I nodded. Again." You had already raised the '
        "problem with him after seeing the support tickets."
    ),
    tension_explanation=(
        "Later, Xiao Yu asked whether you help everyone. You told her you try to, "
        "and later told Mei Ling you did not know what you were going to do "
        "about the remittance tickets."
    ),
    reflective_question=(
        "What did Xiao Yu’s question about helping everyone bring up for you?"
    ),
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def expand_selected_evidence(old: WeeklyDigest, bundle: dict) -> WeeklyDigest:
    """Replace excerpts only, using the same decision coordinates and source receipt."""
    requests = [
        event["details"]["request"]
        for event in bundle["trace_events"]
        if event["event_type"] == "weekly_review_requested"
        and event["details"]["request"]["week_start"] == old.week_start
        and event["details"]["request"]["week_end"] == old.week_end
    ]
    if len(requests) != 1:
        raise ValueError("Expected one frozen review request for target week")
    history = requests[0]["history"]
    texts = {(row["date"], row["t_index"]): row["text"] for row in history}
    if len(texts) != len(history) or any(day > old.week_end for day, _ in texts):
        raise ValueError("Duplicate or future source history")
    decisions = {
        (
            row["date"],
            row["t_index"],
            row["core_value"],
        ): WeeklyDriftReviewerDecision.model_validate(row)
        for row in bundle["scenario"]["weekly_reviewer_decisions"]
    }

    def expand(snippet):
        key = (snippet.date, snippet.t_index)
        if key not in texts or not texts[key].strip():
            raise ValueError(f"Missing full text for selected coordinate: {key}")
        decision = decisions[(*key, snippet.dimensions[0])]
        generated = _evidence_from_decisions([decision], entry_texts=texts)
        if len(generated) != 1 or generated[0].excerpt != texts[key]:
            raise ValueError("Current evidence builder did not preserve full text")
        return snippet.model_copy(update={"excerpt": generated[0].excerpt})

    comparisons = [
        comparison.model_copy(
            update={
                "previous_evidence": [
                    expand(row) for row in comparison.previous_evidence
                ],
                "current_evidence": [
                    expand(row) for row in comparison.current_evidence
                ],
            }
        )
        for comparison in old.state_comparisons
    ]
    return old.model_copy(
        update={
            "evidence": [expand(row) for row in old.evidence],
            "state_comparisons": comparisons,
        }
    )


def _request(prompt: str, response_format: dict, instructions: str | None) -> dict:
    return {
        "input": prompt,
        "instructions": instructions,
        "response_format": response_format,
        "settings": SETTINGS,
        "prompt_sha256": fingerprint({"input": prompt, "instructions": instructions}),
    }


def prepare(root: Path = ROOT) -> dict:
    baseline = _read(root / BASELINE / "manifest.json")
    source = baseline["wei_jun_source"]
    bundle_path = root / source["source_bundle_path"]
    if _file_hash(bundle_path) != source["source_bundle_content_sha256"]:
        raise ValueError("Frozen scenario bundle hash changed")
    if get_prompt_metadata("weekly_digest_coach")["version"] != "4.6":
        raise ValueError("Experiment requires Coach prompt 4.6")
    if get_prompt_metadata("coach_narrative_judge")["version"] != "3.1":
        raise ValueError("Experiment requires evaluator prompt 3.1")
    old = WeeklyDigest.model_validate(baseline["requests"]["wei_jun"]["digest"])
    full = expand_selected_evidence(old, _read(bundle_path))
    if old == full:
        raise ValueError("Context arms must differ")
    bad = CoachNarrative.model_validate(
        _read(root / BASELINE / "receipts.json")["records"]["wei_jun.attempt_2.json"][
            "diagnostic"
        ]["narrative"]
    )
    paths = [
        *SOURCE_PATHS,
        str(BASELINE / "manifest.json"),
        str(BASELINE / "receipts.json"),
        source["source_bundle_path"],
    ]
    source_hashes = {path: _file_hash(root / path) for path in paths}
    digests = {"old": old, "full": full}
    generation: dict[str, dict[str, Any]] = {}
    for arm, digest in digests.items():
        instructions, data = render_digest_messages(digest)
        generation[arm] = {
            "request": _request(
                data, WEEKLY_DIGEST_COACH_RESPONSE_FORMAT, instructions
            ),
            "receipt": render_digest_prompt(digest),
        }
    if (
        generation["old"]["request"]["instructions"]
        != generation["full"]["request"]["instructions"]
    ):
        raise ValueError(
            "Context comparison must use identical generation instructions"
        )
    order = [
        {"id": f"generation_{repeat}_{arm}", "arm": arm, "repeat": repeat}
        for repeat in range(1, 4)
        for arm in (("old", "full") if repeat % 2 else ("full", "old"))
    ]
    plan = {
        "schema_version": "coach-context-eval-v1",
        "issue": "twinkl-7wo2",
        "source": "synthetic_development",
        "source_hashes": source_hashes,
        "digests": {
            arm: digest.model_dump(mode="json") for arm, digest in digests.items()
        },
        "generation": generation,
        "generation_order": order,
        "controls": {
            "known_bad_old": {"evidence_arm": "old", "narrative": bad.model_dump()},
            "known_bad_full": {"evidence_arm": "full", "narrative": bad.model_dump()},
            "constructed_good_full": {
                "evidence_arm": "full",
                "narrative": GOOD_CONTROL.model_dump(),
            },
        },
        "control_provenance": (
            "Known-bad is saved provider output; good control is agent-authored "
            "factual text, not provider output or human validation."
        ),
        "settings": SETTINGS,
        "maximum_provider_calls": 15,
        "fresh_evaluator_evidence": "full",
        "self_evaluation": True,
        "limitations": (
            "One synthetic case, three repeats per arm; same-model AI review "
            "can share errors. No human validation."
        ),
    }
    return {**plan, "fingerprint": fingerprint(plan)}


def freeze(out: Path, plan: dict, root: Path = ROOT) -> None:
    path = out / "manifest.json"
    if path.exists():
        if _read(path) != plan:
            raise ValueError(
                "Experiment fingerprint changed; use a new output directory"
            )
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output directory is nonempty without its manifest")
        _write_new(path, plan)
    freeze_sources(out, plan["source_hashes"], root=root)


def _validate_record(value: Any, plan: dict, origin: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Malformed saved call: {origin}")
    record = dict(value)
    checksum = record.pop("record_sha256", None)
    if (
        fingerprint(record) != checksum
        or record.get("experiment_fingerprint") != plan["fingerprint"]
    ):
        raise ValueError(f"Saved call integrity failure: {origin}")
    return record


def _saved_call(path: Path, plan: dict) -> dict:
    """Read a legacy single-call receipt without changing its original object."""
    return _validate_record(_read(path), plan, path.name)


def _load_calls(out: Path, plan: dict) -> dict[str, dict]:
    """Validate both formats completely before constructing a provider."""
    allowed = {row["id"] for row in plan["generation_order"]}
    allowed.update("judge_" + row["id"] for row in plan["generation_order"])
    allowed.update("judge_" + key for key in plan["controls"])
    records: dict[str, dict] = {}

    def add(record, origin, legacy_name=None):
        call_id = record.get("call_id")
        if not isinstance(call_id, str) or call_id not in allowed or (
            legacy_name is not None and legacy_name != call_id + ".call.json"
        ):
            raise ValueError(f"Unexpected saved call: {origin}")
        if call_id in records:
            raise ValueError(f"Duplicate call ID across receipts: {call_id}")
        records[call_id] = record

    for path in sorted(out.glob("*.call.json")):
        add(_saved_call(path, plan), path.name, path.name)
    consolidated = out / "receipts.jsonl"
    if consolidated.exists():
        raw = consolidated.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("Truncated receipts.jsonl: missing final newline")
        for index, line in enumerate(raw.splitlines(), start=1):
            origin = f"receipts.jsonl:{index}"
            try:
                value = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Malformed receipt line: {origin}") from exc
            add(_validate_record(value, plan, origin), origin)
    return records


def _append_call(out: Path, record: dict, plan: dict) -> None:
    existing = _load_calls(out, plan)
    if record["call_id"] in existing:
        raise ValueError("Refusing to append a duplicate call ID")
    value = {**record, "record_sha256": fingerprint(record)}
    with (out / "receipts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def verify_resume(out: Path, plan: dict) -> None:
    if list(out.glob("*.pending.json")):
        raise ValueError("Unresolved provider attempt; inspect before resuming")
    _load_calls(out, plan)


async def run(out: Path, plan: dict, *, provider_factory=build_llm_complete) -> dict:
    """Run each planned call once; completed and failed calls are immutable."""
    verify_resume(out, plan)
    verify_sources(out, plan["source_hashes"], root=ROOT)
    metrics: list[LLMCallMetrics] = []
    provider = None
    digests = {
        arm: WeeklyDigest.model_validate(value)
        for arm, value in plan["digests"].items()
    }

    async def call(call_id, request, operation):
        nonlocal provider
        saved_calls = _load_calls(out, plan)
        if call_id in saved_calls:
            saved = saved_calls[call_id]
            if saved["request"] != request:
                raise ValueError("Saved provider request changed")
            return saved
        if provider is None:
            provider = provider_factory(
                provider="openai",
                model=DEFAULT_OPENAI_MODEL,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                call_metrics=metrics,
            )
            if provider is None:
                raise ValueError("Evaluator/generator provider is unavailable")
        pending = out / f"{call_id}.pending.json"
        started_at = datetime.now(UTC).isoformat()
        _write_new(
            pending,
            {
                "experiment_fingerprint": plan["fingerprint"],
                "request": request,
                "started_at": started_at,
            },
        )
        captured = []
        before = len(metrics)

        async def capture(prompt, response_format=None, instructions=None):
            actual = _request(prompt, response_format, instructions)
            if captured or actual != request:
                raise ValueError("Unexpected or repeated provider request")
            captured.append({"raw_output": None})
            raw = await provider(prompt, response_format, instructions)
            captured[0]["raw_output"] = raw
            return raw

        try:
            result = await operation(capture)
            error = None
        except Exception as exc:
            result, error = None, type(exc).__name__
        for metric in metrics[before:]:
            metric.call_label = call_id
        record = {
            "call_id": call_id,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "experiment_fingerprint": plan["fingerprint"],
            "request": request,
            "raw_output": captured[0]["raw_output"] if captured else None,
            "result": result,
            "error_type": error,
            "metrics": [row.model_dump(mode="json") for row in metrics[before:]],
            "provider_attempted": bool(captured),
        }
        _append_call(out, record, plan)
        pending.unlink()
        if error or any(row.status == "error" for row in metrics[before:]):
            raise RuntimeError(
                "Provider failure retained; stopped before further calls. "
                "Explicit resume skips this failed call; use a new directory to retry."
            )
        return record

    rows = []
    for sample in plan["generation_order"]:
        arm, call_id = sample["arm"], sample["id"]

        async def generate(complete, digest=digests[arm]):
            diagnostic, receipt = await generate_weekly_digest_coach_diagnostic(
                digest, complete
            )
            return {
                "diagnostic": diagnostic.model_dump(mode="json"),
                "receipt": receipt,
            }

        record = await call(call_id, plan["generation"][arm]["request"], generate)
        result = record.get("result") or {}
        diagnostic = result.get("diagnostic") or {}
        rows.append(
            {**sample, "diagnostic": diagnostic, "error_type": record["error_type"]}
        )

    targets = [
        ("judge_" + row["id"], "full", row["diagnostic"].get("narrative"))
        for row in rows
    ] + [
        ("judge_" + key, value["evidence_arm"], value["narrative"])
        for key, value in plan["controls"].items()
    ]
    evaluations: dict[str, dict[str, Any]] = {}
    for call_id, evidence_arm, payload in targets:
        if payload is None:
            evaluations[call_id] = {"status": "skipped_no_narrative"}
            continue
        narrative = CoachNarrative.model_validate(payload)
        digest = digests[evidence_arm]
        request = _request(
            render_judge_prompt(digest, narrative),
            COACH_NARRATIVE_JUDGE_RESPONSE_FORMAT,
            None,
        )

        async def evaluate(complete, digest=digest, narrative=narrative):
            verdict = await judge_narrative(digest, narrative, complete)
            return verdict.model_dump() if verdict else None

        record = await call(call_id, request, evaluate)
        verdict = record["result"]
        evaluations[call_id] = {
            "status": "scored" if verdict else "failed",
            "verdict": verdict,
            "correctness_below_3": verdict["correctness"] < 3 if verdict else None,
            "error_type": record["error_type"],
        }
    saved_calls = _load_calls(out, plan)
    records = [saved_calls[key] for key in sorted(saved_calls)]
    all_metrics = [
        LLMCallMetrics.model_validate(metric)
        for record in records
        for metric in record["metrics"]
    ]
    summary = {
        "experiment_fingerprint": plan["fingerprint"],
        "source": "ai_review",
        "self_evaluation": True,
        "generation": rows,
        "evaluations": evaluations,
        "provider_attempts": sum(record["provider_attempted"] for record in records),
        "maximum_provider_calls": 15,
        "usage": summarize_llm_call_metrics(all_metrics),
        "limitations": plan["limitations"],
    }
    summary_path = out / "summary.json"
    if summary_path.exists() and _read(summary_path) != summary:
        raise ValueError("Completed summary differs; inspect saved calls")
    if not summary_path.exists():
        _write_new(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare()
    freeze(args.out, plan)
    verify_resume(args.out, plan)
    if not args.execute:
        print(
            "Dry run: frozen six generation and nine evaluator calls; "
            "no provider calls."
        )
        return 0
    load_dotenv(ROOT / ".env")
    print(json.dumps(asyncio.run(run(args.out, plan)), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
