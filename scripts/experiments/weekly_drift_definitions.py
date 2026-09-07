"""Compare the frozen current Weekly Drift Reviewer with definitions added.

The baseline snapshot is captured before the runtime edit. Preparation freezes
both complete provider requests. Execution appends durable attempt events, and
reporting derives unique historical Drifts from complete Persona histories.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from prompts import get_prompt_metadata
from src import weekly_drift_reviewer as reviewer
from src.drift_detector import detect_drift
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.prompt_boundary import UNTRUSTED_DATA_RULE

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / (
    "logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907"
)
VARIANTS = ("baseline", "definitions")
SETTINGS: dict[str, Any] = {
    "model": "gpt-5.6-luna",
    "reasoning_effort": "low",
    "repeats": 3,
    "timeout_seconds": 60,
    "max_output_tokens": 2000,
    "max_attempts": 2,
    "sdk_retries": 0,
    "concurrency": 8,
    "schedule_seed": 20260907,
    "service_tier": "default",
    "store": False,
}
FROZEN_CODE = (
    "scripts/experiments/weekly_drift_definitions.py",
    "src/weekly_drift_reviewer.py",
    "src/drift_detector.py",
    "src/drift_rules.py",
    "src/prompt_boundary.py",
    "prompts/weekly_vif_verifier.yaml",
    "prompts/__init__.py",
    "config/schwartz_values.yaml",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    # Refuse a partial journal instead of silently dropping a paid attempt.
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(output: Path) -> dict:
    if (output / "manifest.json").exists():
        return verify(output)[0]
    baseline_path = output / "baseline_snapshot.json"
    baseline = json.loads(baseline_path.read_text())
    schema = reviewer.WeeklyVerifierResponse.model_json_schema()
    if schema != baseline["schema"]:
        raise ValueError("Output schema changed; this must be definitions-only")
    mutable = {"src/weekly_drift_reviewer.py", "prompts/weekly_vif_verifier.yaml"}
    for name, expected in baseline["source_hashes"].items():
        if name not in mutable and file_hash(ROOT / name) != expected:
            raise ValueError(f"Baseline source changed: {name}")
    rows = []
    for original in baseline["requests"]:
        request = reviewer.build_weekly_drift_reviewer_request(**original["arguments"])
        if request.input_data != original["input_data"]:
            raise ValueError("The experimental Journal Entry input changed")
        if request.runtime_text_sha256 != original["request"]["runtime_text_sha256"]:
            raise ValueError("The experimental source history changed")
        existing_rules, marker, value_context = request.instructions.partition(
            "\n\nAPPROVED CORE VALUE DEFINITIONS\n"
        )
        if (
            not marker
            or existing_rules + "\n\n" + UNTRUSTED_DATA_RULE != original["instructions"]
            or value_context
            != reviewer._render_core_value_definitions(request.core_values)
            + "\n\n"
            + UNTRUSTED_DATA_RULE
        ):
            raise ValueError(
                "Treatment changed more than the approved definitions block"
            )
        variants = {}
        for variant, instructions, version in (
            (
                "baseline",
                original["instructions"],
                baseline["prompt_metadata"]["version"],
            ),
            (
                "definitions",
                request.instructions,
                get_prompt_metadata(reviewer.WEEKLY_DRIFT_REVIEWER_PROMPT)["version"],
            ),
        ):
            payload = {
                "instructions": instructions,
                "input_data": request.input_data,
                "schema": schema,
                "prompt_version": str(version),
            }
            payload["request_sha256"] = digest(payload)
            variants[variant] = payload
        if (
            variants["baseline"]["instructions"]
            == variants["definitions"]["instructions"]
        ):
            raise ValueError("Treatment does not add definitions")
        rows.append(
            {
                "case_id": original["case_id"],
                "request": original["request"],
                "variants": variants,
            }
        )
    requests_path = output / "requests.jsonl"
    with requests_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    import importlib.metadata

    manifest = {
        "schema_version": "weekly-drift-definitions-experiment-v1",
        "issue": "twinkl-j3k7",
        "frozen_at": now(),
        "settings": SETTINGS,
        "baseline_prompt_version": baseline["prompt_metadata"]["version"],
        "treatment_prompt_version": get_prompt_metadata(
            reviewer.WEEKLY_DRIFT_REVIEWER_PROMPT
        )["version"],
        "personas": baseline["personas"],
        "weeks": len(rows),
        "expected_terminal_requests": len(rows) * 2 * SETTINGS["repeats"],
        "baseline_snapshot_sha256": file_hash(baseline_path),
        "requests_sha256": file_hash(requests_path),
        "code_sha256": {name: file_hash(ROOT / name) for name in FROZEN_CODE},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("openai", "pydantic", "pyyaml")
        },
        "comparison": {
            "population": "Existing Weekly Drift study: 204 Personas and 951 weeks",
            "baseline": "Fresh current v3; historical v2 receipts are context only",
            "intervention": "Selected approved definitions and core motivations only",
            "schedule": "Seeded shuffle of all case/repeat/variant requests",
            "counting": "Unique historical Drifts at each Persona's final cutoff",
            "identity": "Persona, Core Value, onset, confirmation, supporting indices",
            "boundaries": "Also compare termination reason, index and verdict",
            "overlap": "Overlapping unmatched spans form boundary/split/merge groups",
            "failure": "Invalid or refused output is terminal and becomes Abstain",
            "scope": "Detected counts and identity changes; no NSM rerun or relabeling",
            "interpretation": "Synthetic development study; counts are not accuracy",
        },
        "authorization": "User authorized the paid definitions study on 2026-09-07",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify(output: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((output / "manifest.json").read_text())
    for filename, key in (
        ("baseline_snapshot.json", "baseline_snapshot_sha256"),
        ("requests.jsonl", "requests_sha256"),
    ):
        if file_hash(output / filename) != manifest[key]:
            raise ValueError(f"Frozen experiment file changed: {filename}")
    for name, expected in manifest["code_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen experiment code changed: {name}")
    rows = read_rows(output / "requests.jsonl")
    if len(rows) != manifest["weeks"] or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicated experimental cases")
    for row in rows:
        for payload in row["variants"].values():
            if (
                digest({k: v for k, v in payload.items() if k != "request_sha256"})
                != (payload["request_sha256"])
            ):
                raise ValueError("Request hash mismatch")
    return manifest, rows


def request_key(case_id: str, variant: str, repeat: int) -> str:
    return f"{variant}:{repeat}:{case_id}"


def jobs(rows: list[dict], settings: dict) -> list[tuple[dict, str, int]]:
    result = [
        (row, variant, repeat)
        for row in rows
        for repeat in range(1, settings["repeats"] + 1)
        for variant in VARIANTS
    ]
    random.Random(settings["schedule_seed"]).shuffle(result)
    return result


def jsonable(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def prior_attempts(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for event in events:
        key, number = event["request_key"], event["attempt_number"]
        if event["event"] == "started":
            if number in grouped[key]:
                raise ValueError("Repeated attempt reservation")
            grouped[key][number] = event
        elif event["event"] == "finished":
            if number not in grouped[key] or grouped[key][number]["event"] != "started":
                raise ValueError("Attempt completion has no unique reservation")
            grouped[key][number] = event
        else:
            raise ValueError("Unknown attempt event")
    return {key: [items[n] for n in sorted(items)] for key, items in grouped.items()}


async def execute_request(
    client: Any,
    row: dict,
    variant: str,
    repeat: int,
    settings: dict,
    attempts: list[dict],
    append: Any,
) -> dict:
    key = request_key(row["case_id"], variant, repeat)
    payload = row["variants"][variant]
    request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
    if attempts and attempts[-1]["event"] == "started":
        # An interrupted provider call may have been billed. Do not resend it.
        return terminal_result(row, variant, repeat, attempts)
    if attempts and not attempts[-1].get("retryable", False):
        return terminal_result(row, variant, repeat, attempts)
    for number in range(len(attempts) + 1, settings["max_attempts"] + 1):
        event = {
            "event": "started",
            "request_key": key,
            "request_sha256": payload["request_sha256"],
            "attempt_number": number,
            "started_at": now(),
        }
        append(event)
        started = time.monotonic()
        attempt = dict(event, event="finished", retryable=False)
        try:
            response = await client.responses.parse(
                model=settings["model"],
                instructions=payload["instructions"],
                input=payload["input_data"],
                text_format=reviewer.WeeklyVerifierResponse,
                reasoning={"effort": settings["reasoning_effort"]},
                max_output_tokens=settings["max_output_tokens"],
                store=settings["store"],
                service_tier=settings["service_tier"],
                timeout=settings["timeout_seconds"],
            )
            parsed = getattr(response, "output_parsed", None)
            attempt.update(
                resolved_model=getattr(response, "model", None),
                response_id=getattr(response, "id", None),
                provider_request_id=getattr(response, "_request_id", None),
                provider_status=getattr(response, "status", None),
                raw_text=getattr(response, "output_text", None),
                usage=jsonable(getattr(response, "usage", None)),
            )
            if not isinstance(parsed, reviewer.WeeklyVerifierResponse):
                attempt.update(
                    status="refusal", refusal=reviewer._response_refusal(response)
                )
            else:
                attempt["parsed"] = parsed.model_dump(mode="json")
                try:
                    reviewer.validate_weekly_drift_reviewer_response(parsed, request)
                except ValueError as error:
                    attempt.update(status="invalid", validation_error=str(error))
                else:
                    attempt["status"] = "ok"
        except (
            Exception
        ) as error:  # Provider failures become recorded Abstain decisions.
            attempt.update(
                status="error",
                error_type=type(error).__name__,
                retryable=reviewer._is_transient_error(error),
            )
        attempt.update(completed_at=now(), latency_seconds=time.monotonic() - started)
        append(attempt)
        attempts.append(attempt)
        if not attempt["retryable"]:
            break
        if number < settings["max_attempts"]:
            await asyncio.sleep(2 ** (number - 1))
    return terminal_result(row, variant, repeat, attempts)


def terminal_result(row: dict, variant: str, repeat: int, attempts: list[dict]) -> dict:
    final = attempts[-1]
    unknown = final["event"] == "started"
    status = "error" if unknown else final["status"]
    if status not in {"ok", "refusal", "invalid", "error"}:
        raise ValueError("Invalid terminal review status")
    response = (
        reviewer.WeeklyVerifierResponse.model_validate(final["parsed"])
        if status == "ok"
        else None
    )
    request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
    decisions = reviewer._effective_decisions(
        request, status=cast(reviewer.ReviewStatus, status), response=response
    )
    return {
        "request_key": request_key(row["case_id"], variant, repeat),
        "case_id": row["case_id"],
        "variant": variant,
        "repeat": repeat,
        "status": status,
        "unknown_interrupted_attempt": unknown,
        "attempts": len(attempts),
        "request_sha256": row["variants"][variant]["request_sha256"],
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
    }


async def run(output: Path, *, limit: int | None = None) -> dict:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(ROOT / ".env", override=False)
    manifest, rows = verify(output)
    settings = manifest["settings"]
    journal_path, results_path = output / "attempts.jsonl", output / "responses.jsonl"
    attempted = prior_attempts(read_rows(journal_path))
    completed_rows = read_rows(results_path)
    complete = {r["request_key"]: r for r in completed_rows}
    if len(complete) != len(completed_rows):
        raise ValueError("Duplicated terminal response")
    pending = [
        job
        for job in jobs(rows, settings)
        if request_key(job[0]["case_id"], job[1], job[2]) not in complete
    ]
    if limit is not None:
        pending = pending[:limit]
    queue: asyncio.Queue = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    started_at = now()
    completed_here = 0
    statuses: Counter = Counter()
    with journal_path.open("a") as journal, results_path.open("a") as results:

        def append(event: dict) -> None:
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()

        async with AsyncOpenAI(max_retries=0) as client:

            async def worker() -> None:
                nonlocal completed_here
                while not queue.empty():
                    row, variant, repeat = queue.get_nowait()
                    key = request_key(row["case_id"], variant, repeat)
                    result = await execute_request(
                        client,
                        row,
                        variant,
                        repeat,
                        settings,
                        attempted.get(key, []),
                        append,
                    )
                    results.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.flush()
                    completed_here += 1
                    statuses[result["status"]] += 1
                    if completed_here % 25 == 0 or completed_here == len(pending):
                        print(
                            json.dumps(
                                {
                                    "completed": len(complete) + completed_here,
                                    "expected": manifest["expected_terminal_requests"],
                                    "new_statuses": dict(statuses),
                                }
                            ),
                            flush=True,
                        )
                    queue.task_done()

            await asyncio.gather(*(worker() for _ in range(settings["concurrency"])))
    return {
        "started_at": started_at,
        "completed_at": now(),
        "processed": completed_here,
        "total_completed": len(complete) + completed_here,
        "statuses": dict(statuses),
    }


def drift_signature(drift: dict) -> tuple:
    return (
        drift["persona_id"],
        drift["core_value"],
        drift["onset_t_index"],
        drift["confirmation_t_index"],
        tuple(drift["supporting_t_indices"]),
        drift["termination_reason"],
        drift["termination_t_index"],
        drift["termination_verdict"],
    )


def compare_drifts(before: list[dict], after: list[dict]) -> dict:
    left = {drift_signature(d): d for d in before}
    right = {drift_signature(d): d for d in after}
    common = left.keys() & right.keys()
    unmatched = {
        ("before", n): drift
        for n, drift in enumerate(before)
        if drift_signature(drift) not in common
    } | {
        ("after", n): drift
        for n, drift in enumerate(after)
        if drift_signature(drift) not in common
    }
    edges: dict[tuple, set] = {key: set() for key in unmatched}
    for key_a, a in unmatched.items():
        for key_b, b in unmatched.items():
            if (
                key_a[0] != key_b[0]
                and a["persona_id"] == b["persona_id"]
                and a["core_value"] == b["core_value"]
                and max(a["onset_t_index"], b["onset_t_index"])
                <= min(a["end_t_index"], b["end_t_index"])
            ):
                edges[key_a].add(key_b)
    overlap, added, removed, changed, termination = [], [], [], [], []
    unseen = set(unmatched)
    while unseen:
        stack, component = [min(unseen)], set()
        while stack:
            key = stack.pop()
            if key in component:
                continue
            component.add(key)
            stack.extend(edges[key] - component)
        unseen -= component
        old = [unmatched[key] for key in sorted(component) if key[0] == "before"]
        new = [unmatched[key] for key in sorted(component) if key[0] == "after"]
        if not old:
            added.extend(new)
        elif not new:
            removed.extend(old)
        else:
            kind = (
                "boundary_change"
                if len(old) == len(new) == 1
                else "split"
                if len(old) == 1
                else "merge"
                if len(new) == 1
                else "resegmented"
            )
            if len(old) == len(new) == 1:
                pair = {"before": old[0], "after": new[0]}
                if drift_signature(old[0])[:5] == drift_signature(new[0])[:5]:
                    kind = "termination_change"
                    termination.append(pair)
                else:
                    changed.append(pair)
            overlap.append({"kind": kind, "before": old, "after": new})
    before_counts, after_counts = (
        Counter(d["core_value"] for d in before),
        Counter(d["core_value"] for d in after),
    )
    return {
        "before_total": len(before),
        "after_total": len(after),
        "delta": len(after) - len(before),
        "per_core_value": {
            value: {
                "before": before_counts[value],
                "after": after_counts[value],
                "delta": after_counts[value] - before_counts[value],
            }
            for value in SCHWARTZ_VALUE_ORDER
        },
        "unchanged": len(common),
        "added": added,
        "removed": removed,
        "changed_boundaries": changed,
        "changed_termination": termination,
        "overlap_changes": overlap,
        "identity_note": "Overlapping unmatched spans form connected groups",
    }


def score_rows(rows: list[dict], responses: list[dict], repeats: int) -> dict:
    expected = {
        request_key(row["case_id"], variant, repeat)
        for row in rows
        for variant in VARIANTS
        for repeat in range(1, repeats + 1)
    }
    if (
        len(responses) != len(expected)
        or {r["request_key"] for r in responses} != expected
    ):
        raise ValueError(
            "Scoring requires exactly one terminal result for every request"
        )
    by_case = {row["case_id"]: row for row in rows}
    grouped: dict[tuple, list] = defaultdict(list)
    status_counts: dict[tuple, Counter] = defaultdict(Counter)
    for response in responses:
        row = by_case[response["case_id"]]
        variant, repeat = response["variant"], response["repeat"]
        if response["request_key"] != request_key(row["case_id"], variant, repeat):
            raise ValueError("Response identity differs from its request key")
        if response["request_sha256"] != row["variants"][variant]["request_sha256"]:
            raise ValueError("Response belongs to a different request")
        decisions = [
            reviewer.WeeklyDriftReviewerDecision.model_validate(d)
            for d in response["decisions"]
        ]
        actual = {(d.t_index, d.core_value) for d in decisions}
        request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
        if len(decisions) != len(actual) or actual != request.expected_coordinates:
            raise ValueError("Response has incomplete effective decisions")
        entry_dates = {entry.t_index: entry.date for entry in request.history}
        if any(
            decision.persona_id != request.persona_id
            or decision.week_start != request.week_start
            or decision.week_end != request.week_end
            or decision.date != entry_dates[decision.t_index]
            or decision.review_status != response["status"]
            for decision in decisions
        ):
            raise ValueError("Effective decision provenance differs from its request")
        grouped[variant, repeat, request.persona_id].extend(decisions)
        status_counts[variant, repeat][response["status"]] += 1
    populations: dict[tuple, list] = defaultdict(list)
    decisions_by_run: dict[tuple, dict] = defaultdict(dict)
    for (variant, repeat, pid), decisions in sorted(grouped.items()):
        result = detect_drift(decisions, persona_id=pid)
        populations[variant, repeat].extend(
            d.model_dump(mode="json") for d in result.drifts
        )
        decisions_by_run[variant, repeat].update(
            {(pid, d.t_index, d.core_value): d.verdict for d in decisions}
        )
    runs = {}
    for repeat in range(1, repeats + 1):
        comparison = compare_drifts(
            populations["baseline", repeat], populations["definitions", repeat]
        )
        before, after = (
            decisions_by_run["baseline", repeat],
            decisions_by_run["definitions", repeat],
        )
        comparison["changed_entry_decisions"] = [
            {
                "persona_id": key[0],
                "t_index": key[1],
                "core_value": key[2],
                "before": before[key],
                "after": after[key],
            }
            for key in sorted(before)
            if before[key] != after[key]
        ]
        comparison["request_statuses"] = {
            variant: dict(status_counts[variant, repeat]) for variant in VARIANTS
        }
        comparison["drifts"] = {
            variant: populations[variant, repeat] for variant in VARIANTS
        }
        runs[str(repeat)] = comparison
    summaries = {}
    for value in ("all", *SCHWARTZ_VALUE_ORDER):
        counts = {
            variant: [
                len(populations[variant, repeat])
                if value == "all"
                else sum(d["core_value"] == value for d in populations[variant, repeat])
                for repeat in range(1, repeats + 1)
            ]
            for variant in VARIANTS
        }
        summaries[value] = {
            variant: {
                "runs": values,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            for variant, values in counts.items()
        }
    repeat_variation = {
        variant: [
            {
                "repeats": [a, b],
                **compare_drifts(populations[variant, a], populations[variant, b]),
            }
            for a in range(1, repeats + 1)
            for b in range(a + 1, repeats + 1)
        ]
        for variant in VARIANTS
    }
    return {
        "runs": runs,
        "counts": summaries,
        "within_variant_variation": repeat_variation,
    }


def report(output: Path) -> dict:
    manifest, rows = verify(output)
    responses = read_rows(output / "responses.jsonl")
    scores = score_rows(rows, responses, manifest["settings"]["repeats"])
    events = read_rows(output / "attempts.jsonl")
    attempts = [e for e in events if e["event"] == "finished"]
    scores.update(
        schema_version="weekly-drift-definitions-results-v1",
        generated_at=now(),
        manifest_sha256=file_hash(output / "manifest.json"),
        responses_sha256=file_hash(output / "responses.jsonl"),
        attempts_sha256=file_hash(output / "attempts.jsonl"),
        completed_requests=len(responses),
        completed_attempts=len(attempts),
        actual_models=dict(
            Counter(a["resolved_model"] for a in attempts if a.get("resolved_model"))
        ),
        usage={
            "input_tokens": sum(
                (a.get("usage") or {}).get("input_tokens", 0) for a in attempts
            ),
            "output_tokens": sum(
                (a.get("usage") or {}).get("output_tokens", 0) for a in attempts
            ),
            "attempts_without_usage": sum(a.get("usage") is None for a in attempts),
        },
        limitations=[
            "Synthetic development study; no fresh final test or human validation.",
            "More detected Drifts does not establish better accuracy.",
            "Fresh v3 versus v4 isolates added context; historical v2 is different.",
            "Same model and low reasoning; independent calls, no provider seed.",
            "Overlap groups cover boundary shifts, splits, merges and terminations.",
            "No NSM inputs, historical receipts, or reference labels were replaced.",
        ],
    )
    write_json(output / "results.json", scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most this many new requests; retain them for the full run",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = asyncio.run(run(args.output, limit=args.limit))
    elif args.command == "report":
        scores = report(args.output)
        result = {
            "counts": scores["counts"],
            "completed_requests": scores["completed_requests"],
        }
    else:
        manifest, rows = verify(args.output)
        result = {
            "status": "verified",
            "weeks": len(rows),
            "settings": manifest["settings"],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
