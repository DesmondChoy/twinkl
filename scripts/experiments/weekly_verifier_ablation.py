#!/usr/bin/env python3
"""Prepare, run, and score the bounded twinkl-752.1 verifier ablation.

The paid API path is deliberately opt-in. ``prepare`` and ``estimate`` are
offline; ``run`` refuses to call OpenAI unless ``--execute`` is present.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from prompts import load_prompt
from src.vif.state_encoder import concatenate_entry_text
from src.wrangling.parse_wrangled_data import parse_wrangled_file

DEFAULT_CONFIG_PATH = Path("config/evals/twinkl_752_1_weekly_verifier_ablation_v1.yaml")
ARMS = ("without_critic", "with_critic")
TERMINAL_RESPONSE_STATUSES = {"ok", "refusal", "invalid"}
Verdict = Literal["conflict", "not_conflict", "abstain"]
Confidence = Literal["low", "medium", "high"]
ReasonCode = Literal[
    "ambiguous",
    "direct_aligned_or_neutral_behavior",
    "direct_behavior_or_choice",
    "external_constraint",
    "feeling_or_intent_only",
    "missing_text",
    "needs_hidden_context",
]


class VerifierAssessment(BaseModel):
    """One current-week entry/value decision."""

    model_config = ConfigDict(extra="forbid")

    t_index: int = Field(ge=0)
    dimension: str
    verdict: Verdict
    confidence: Confidence
    reason_code: ReasonCode
    evidence_quote: str = ""


class WeeklyVerifierResponse(BaseModel):
    """Strict structured response for one persona-week."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[VerifierAssessment]


@dataclass(frozen=True)
class TargetCell:
    """Resolved target for one persona/value entry cell."""

    persona_id: str
    dimension: str
    t_index: int
    conflict: bool


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return loaded


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rooted(path: str | Path, root: Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else root / resolved


def _normalize_value(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _week_start(entry_date: str) -> date:
    parsed = date.fromisoformat(entry_date)
    return parsed - timedelta(days=parsed.weekday())


def _approx_tokens(text: str) -> int:
    """Conservative preflight estimate matching the existing baseline script."""
    return max(1, math.ceil(len(text) / 4))


def _estimated_output_tokens(expected_assessments: int) -> int:
    """Estimate compact structured output, including short evidence quotes."""
    return 55 + 45 * expected_assessments


def _request_cost_usd(
    *, input_tokens: int, output_tokens: int, pricing: dict[str, float]
) -> float:
    return (
        input_tokens * float(pricing["input"])
        + output_tokens * float(pricing["output"])
    ) / 1_000_000


def _case_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Review document is missing cases")
    mapped = {str(case["review_case_id"]): case for case in cases}
    if len(mapped) != len(cases):
        raise ValueError("Review document contains duplicate case IDs")
    return mapped


def _load_population(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Merge value-specific review cases into unique persona timelines."""
    population = config["population"]
    packet_path = _rooted(population["packet_path"], root)
    key_path = _rooted(population["reconciliation_key_path"], root)
    packet = _read_json(packet_path)
    key = _read_json(key_path)
    if packet.get("split") != "development" or key.get("target_version") != packet.get(
        "target_version"
    ):
        raise ValueError(
            "Configured review bundle is not the active development target"
        )

    packet_cases = _case_map(packet)
    key_cases = _case_map(key)
    if set(packet_cases) != set(key_cases):
        raise ValueError("Blind packet and reconciliation key case IDs differ")

    wrangled_dir = _rooted(population["wrangled_dir"], root)
    personas: dict[str, dict[str, Any]] = {}
    cases: dict[str, dict[str, Any]] = {}
    for review_case_id in sorted(packet_cases):
        packet_case = packet_cases[review_case_id]
        key_case = key_cases[review_case_id]
        persona_id = str(key_case["persona_id"])
        dimension = str(key_case["dimension"])
        if _normalize_value(str(packet_case["declared_core_value"])) != dimension:
            raise ValueError(f"Declared value mismatch for {review_case_id}")

        packet_entries = packet_case["entries"]
        key_entries = key_case["entries"]
        if len(packet_entries) != len(key_entries):
            raise ValueError(f"Entry count mismatch for {review_case_id}")

        wrangled_path = wrangled_dir / f"persona_{persona_id}.md"
        _profile, wrangled_entries, _warnings = parse_wrangled_file(wrangled_path)
        wrangled_by_index = {int(entry["t_index"]): entry for entry in wrangled_entries}

        case_entries: list[dict[str, Any]] = []
        for packet_entry, key_entry in zip(packet_entries, key_entries, strict=True):
            if int(packet_entry["position"]) != len(case_entries) + 1:
                raise ValueError(
                    f"Non-contiguous packet positions for {review_case_id}"
                )
            t_index = int(key_entry["t_index"])
            entry_date = str(key_entry["date"])
            text = str(packet_entry["journal_entry"]).strip()
            wrangled = wrangled_by_index.get(t_index)
            if wrangled is None:
                raise ValueError(f"Missing wrangled entry {persona_id}:{t_index}")
            expected_text = concatenate_entry_text(
                wrangled.get("initial_entry"),
                wrangled.get("nudge_text"),
                wrangled.get("response_text"),
            ).strip()
            if text != expected_text or entry_date != str(wrangled["date"]):
                raise ValueError(
                    f"Runtime text contract mismatch for {persona_id}:{t_index}"
                )
            case_entries.append({"t_index": t_index, "date": entry_date, "text": text})

        case_record = {
            "review_case_id": review_case_id,
            "persona_id": persona_id,
            "dimension": dimension,
            "entries": case_entries,
        }
        cases[review_case_id] = case_record
        persona = personas.setdefault(
            persona_id, {"persona_id": persona_id, "values": set(), "entries": {}}
        )
        persona["values"].add(dimension)
        for entry in case_entries:
            coordinate = int(entry["t_index"])
            previous = persona["entries"].get(coordinate)
            if previous is not None and previous != entry:
                raise ValueError(
                    f"Conflicting duplicate entry for {persona_id}:{coordinate}"
                )
            persona["entries"][coordinate] = entry

    for persona in personas.values():
        persona["values"] = sorted(persona["values"])
        persona["entries"] = [
            persona["entries"][index] for index in sorted(persona["entries"])
        ]

    expected_personas = int(population["expected_personas"])
    expected_cases = int(population["expected_cases"])
    expected_entries = int(population["expected_entries"])
    unique_entries = sum(len(persona["entries"]) for persona in personas.values())
    if (len(personas), len(cases), unique_entries) != (
        expected_personas,
        expected_cases,
        expected_entries,
    ):
        raise ValueError(
            "Development population drift: "
            f"got {len(personas)} personas/{len(cases)} cases/{unique_entries} entries"
        )
    return personas, cases


def _load_critic_evidence(
    config: dict[str, Any], root: Path
) -> dict[tuple[str, str, int], dict[str, float]]:
    path = _rooted(config["population"]["run_020_evidence_path"], root)
    frame = pl.read_parquet(path)
    required = {
        "source",
        "persona_id",
        "dimension",
        "t_index",
        "p_minus1",
        "uncertainty",
    }
    if not required.issubset(frame.columns) or set(frame["source"].unique()) != {
        "run_020"
    }:
        raise ValueError("Configured Critic evidence is not the fixed run_020 surface")
    evidence: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in frame.to_dicts():
        key = (str(row["persona_id"]), str(row["dimension"]), int(row["t_index"]))
        if key in evidence:
            raise ValueError(f"Duplicate run_020 evidence coordinate: {key}")
        evidence[key] = {
            "p_minus1": float(row["p_minus1"]),
            "uncertainty": float(row["uncertainty"]),
        }
    return evidence


def _mlp_seed_input_divergence(
    config: dict[str, Any], root: Path, cases: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Summarize incumbent-family P(-1) spread on the development cells."""
    coordinates = {
        (case["persona_id"], case["dimension"], int(entry["t_index"]))
        for case in cases.values()
        for entry in case["entries"]
    }
    by_coordinate: dict[tuple[str, str, int], list[float]] = {
        coordinate: [] for coordinate in coordinates
    }
    run_ids = []
    for configured_path in config["population"]["mlp_run_paths"]:
        run = _read_yaml(_rooted(configured_path, root))
        run_id = str(run["metadata"]["run_id"])
        run_ids.append(run_id)
        outputs_path = _rooted(run["artifacts"]["validation_outputs"], root)
        frame = pl.read_parquet(outputs_path)
        observed = 0
        for row in frame.to_dicts():
            coordinate = (
                str(row["persona_id"]),
                str(row["dimension"]),
                int(row["t_index"]),
            )
            if coordinate not in by_coordinate:
                continue
            probabilities = row["class_probabilities"]
            if not isinstance(probabilities, list) or len(probabilities) != 3:
                raise ValueError(f"Invalid class probabilities in {run_id}")
            by_coordinate[coordinate].append(float(probabilities[0]))
            observed += 1
        if observed != len(coordinates):
            raise ValueError(
                f"{run_id} covers {observed}/{len(coordinates)} development cells"
            )
    if any(len(values) != len(run_ids) for values in by_coordinate.values()):
        raise ValueError("Incomplete incumbent-family seed coverage")
    ranges = sorted(max(values) - min(values) for values in by_coordinate.values())
    return {
        "run_ids": run_ids,
        "cell_count": len(ranges),
        "mean_p_minus1_range": statistics.fmean(ranges),
        "median_p_minus1_range": statistics.median(ranges),
        "p90_p_minus1_range": ranges[math.ceil(0.90 * len(ranges)) - 1],
        "max_p_minus1_range": max(ranges),
        "fraction_range_at_least_0_20": sum(value >= 0.20 for value in ranges)
        / len(ranges),
        "note": "Local input-sensitivity annex only; no family-aggregate API arm.",
    }


def _format_history(entries: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[ENTRY t_index={entry['t_index']}]\n{entry['text']}" for entry in entries
    )


def _format_current_entries(entries: list[dict[str, Any]]) -> str:
    return "\n".join(f"- t_index={entry['t_index']}" for entry in entries)


def _critic_block(
    *,
    persona_id: str,
    values: list[str],
    current_entries: list[dict[str, Any]],
    evidence: dict[tuple[str, str, int], dict[str, float]],
) -> str:
    lines = [
        "",
        "CRITIC SIGNALS FOR CURRENT-WEEK ENTRIES",
        "These numeric signals are fallible hints, not labels or targets.",
        "t_index | value | P(-1) | MC uncertainty",
    ]
    for entry in current_entries:
        for value in values:
            key = (persona_id, value, int(entry["t_index"]))
            signal = evidence.get(key)
            if signal is None:
                raise ValueError(f"Missing run_020 signal for {key}")
            lines.append(
                f"{entry['t_index']} | {value} | "
                f"{signal['p_minus1']:.6f} | {signal['uncertainty']:.6f}"
            )
    return "\n".join(lines)


def build_prompt_records(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the two-arm, deduplicated persona-week prompt surface."""
    personas, cases = _load_population(config, root)
    critic_evidence = _load_critic_evidence(config, root)
    template = load_prompt("weekly_vif_verifier")
    records: list[dict[str, Any]] = []
    runtime_contract: list[dict[str, Any]] = []

    for persona_id in sorted(personas):
        persona = personas[persona_id]
        entries = persona["entries"]
        week_starts = sorted({_week_start(entry["date"]) for entry in entries})
        for start in week_starts:
            end = start + timedelta(days=6)
            history = [
                entry for entry in entries if date.fromisoformat(entry["date"]) <= end
            ]
            current = [
                entry
                for entry in entries
                if start <= date.fromisoformat(entry["date"]) <= end
            ]
            if not current:
                raise ValueError(f"Empty current week for {persona_id}:{start}")
            runtime_text_payload = [
                {"t_index": entry["t_index"], "text": entry["text"]}
                for entry in history
            ]
            runtime_text_sha = _sha256_text(_canonical_json(runtime_text_payload))
            runtime_contract.append(
                {
                    "persona_id": persona_id,
                    "week_start": start.isoformat(),
                    "runtime_text_sha256": runtime_text_sha,
                }
            )
            expected_coordinates = [
                {"t_index": int(entry["t_index"]), "dimension": value}
                for entry in current
                for value in persona["values"]
            ]
            shared = {
                "declared_values": "\n".join(
                    f"- {value}" for value in persona["values"]
                ),
                "cumulative_history": _format_history(history),
                "current_week_entries": _format_current_entries(current),
            }
            critic_text = _critic_block(
                persona_id=persona_id,
                values=persona["values"],
                current_entries=current,
                evidence=critic_evidence,
            )
            for arm in ARMS:
                inserted = critic_text if arm == "with_critic" else ""
                prompt = template.render(**shared, critic_block=inserted).strip() + "\n"
                record = {
                    "persona_id": persona_id,
                    "week_start": start.isoformat(),
                    "week_end": end.isoformat(),
                    "arm": arm,
                    "declared_values": persona["values"],
                    "current_t_indices": [int(entry["t_index"]) for entry in current],
                    "expected_coordinates": expected_coordinates,
                    "runtime_text_sha256": runtime_text_sha,
                    "critic_block_sha256": _sha256_text(inserted),
                    "prompt": prompt,
                    "prompt_sha256": _sha256_text(prompt),
                }
                records.append(record)

    expected_weeks = int(config["population"]["expected_persona_weeks"])
    observed_weeks = len(records) // len(ARMS)
    if len(records) != expected_weeks * len(ARMS) or observed_weeks != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} persona-weeks, built {observed_weeks}"
        )

    packet_path = _rooted(config["population"]["packet_path"], root)
    key_path = _rooted(config["population"]["reconciliation_key_path"], root)
    manifest = {
        "study_id": config["study_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "repo_head": _git_head(root),
        "target_split": config["population"]["split"],
        "week_definition": config["population"]["week_definition"],
        "packet_sha256": _sha256_file(packet_path),
        "reconciliation_key_sha256": _sha256_file(key_path),
        "reviewer_a_sha256": _sha256_file(
            _rooted(config["population"]["reviewer_a_path"], root)
        ),
        "reviewer_b_sha256": _sha256_file(
            _rooted(config["population"]["reviewer_b_path"], root)
        ),
        "run_020_evidence_sha256": _sha256_file(
            _rooted(config["population"]["run_020_evidence_path"], root)
        ),
        "prompt_template_sha256": _sha256_file(
            root / "prompts/weekly_vif_verifier.yaml"
        ),
        "config_payload_sha256": _sha256_text(_canonical_json(config)),
        "runtime_contract_sha256": _sha256_text(_canonical_json(runtime_contract)),
        "prompt_manifest_sha256": _sha256_text(
            _canonical_json(
                [
                    {
                        "persona_id": record["persona_id"],
                        "week_start": record["week_start"],
                        "arm": record["arm"],
                        "prompt_sha256": record["prompt_sha256"],
                    }
                    for record in records
                ]
            )
        ),
        "mlp_seed_input_divergence": _mlp_seed_input_divergence(config, root, cases),
        "persona_count": len(personas),
        "case_count": len(cases),
        "entry_count": sum(len(persona["entries"]) for persona in personas.values()),
        "persona_week_count": observed_weeks,
        "prompt_count": len(records),
        "planned_successful_calls": len(records) * int(config["study"]["repeats"]),
        "arms": list(ARMS),
        "repeats": int(config["study"]["repeats"]),
        "model": config["api"]["model"],
        "reasoning_effort": config["api"]["reasoning_effort"],
        "store": bool(config["api"]["store"]),
        "locked_promotion_used": False,
        "retired_benchmark_used": False,
    }
    return records, manifest


def _git_head(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def estimate_plan(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    repeats = int(config["study"]["repeats"])
    pricing = config["api"]["pricing_usd_per_million_tokens"]
    input_tokens_one_pass = sum(_approx_tokens(record["prompt"]) for record in records)
    output_tokens_one_pass = sum(
        _estimated_output_tokens(len(record["expected_coordinates"]))
        for record in records
    )
    return {
        "successful_calls": len(records) * repeats,
        "unique_prompts": len(records),
        "repeats": repeats,
        "estimated_input_tokens": input_tokens_one_pass * repeats,
        "estimated_output_tokens": output_tokens_one_pass * repeats,
        "estimated_cost_usd": _request_cost_usd(
            input_tokens=input_tokens_one_pass * repeats,
            output_tokens=output_tokens_one_pass * repeats,
            pricing=pricing,
        ),
        "max_budget_usd": float(config["api"]["max_budget_usd"]),
        "token_estimator": "ceil(utf8_characters/4); output=55+45*assessment_count",
    }


def validate_parsed_response(
    *, parsed: WeeklyVerifierResponse, record: dict[str, Any]
) -> None:
    expected = {
        (int(item["t_index"]), str(item["dimension"]))
        for item in record["expected_coordinates"]
    }
    observed = {
        (assessment.t_index, assessment.dimension) for assessment in parsed.assessments
    }
    if len(observed) != len(parsed.assessments) or observed != expected:
        raise ValueError(
            f"Response coordinate mismatch: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    entry_text = {
        int(match.group(1)): match.group(2)
        for match in _entry_sections(record["prompt"])
    }
    for assessment in parsed.assessments:
        quote = assessment.evidence_quote.strip()
        if assessment.verdict == "conflict" and not quote:
            raise ValueError("Conflict assessment is missing an evidence quote")
        if assessment.verdict == "conflict" and quote not in entry_text.get(
            assessment.t_index, ""
        ):
            raise ValueError(
                f"Evidence quote is not present in t_index={assessment.t_index}"
            )


def _entry_sections(prompt: str) -> list[Any]:
    import re

    history = prompt.split("CURRENT-WEEK ENTRIES TO ASSESS", maxsplit=1)[0]
    pattern = re.compile(
        r"\[ENTRY t_index=(\d+)\]\n(.*?)(?=\n\n\[ENTRY t_index=|\Z)",
        re.DOTALL,
    )
    return list(pattern.finditer(history))


def _response_key(record: dict[str, Any], repeat: int) -> tuple[str, str, str, int]:
    return (
        str(record["persona_id"]),
        str(record["week_start"]),
        str(record["arm"]),
        repeat,
    )


def _completed_keys(
    rows: list[dict[str, Any]],
    records_by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    repeats: int | None = None,
    requested_model: str | None = None,
) -> set[tuple[str, str, str, int]]:
    completed: set[tuple[str, str, str, int]] = set()
    for row in rows:
        if row.get("status") not in TERMINAL_RESPONSE_STATUSES:
            continue
        base_key = (str(row["persona_id"]), str(row["week_start"]), str(row["arm"]))
        record = records_by_key.get(base_key)
        if (
            record is None
            or row.get("prompt_sha256") != record["prompt_sha256"]
            or row.get("runtime_text_sha256") != record["runtime_text_sha256"]
            or row.get("week_end") != record["week_end"]
        ):
            raise ValueError(
                f"Stale or unknown successful response receipt: {base_key}"
            )
        repeat = int(row["repeat"])
        if repeats is not None and repeat not in range(1, repeats + 1):
            raise ValueError(f"Out-of-range response repeat: {repeat}")
        if (
            requested_model is not None
            and row.get("requested_model") != requested_model
        ):
            raise ValueError(f"Unexpected requested model for response: {base_key}")
        if row["status"] == "ok":
            parsed = WeeklyVerifierResponse.model_validate(row.get("parsed"))
            validate_parsed_response(parsed=parsed, record=record)
        key = (*base_key, repeat)
        if key in completed:
            raise ValueError(f"Duplicate successful response receipt: {key}")
        completed.add(key)
    return completed


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0)), int(
        getattr(usage, "output_tokens", 0)
    )


def _is_transient_error(error: Exception) -> bool:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError),
    )


async def _call_openai(
    *,
    client: Any,
    record: dict[str, Any],
    repeat: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    api = config["api"]
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, int(api["max_attempts"]) + 1):
        try:
            response = await client.responses.parse(
                model=api["model"],
                input=record["prompt"],
                text_format=WeeklyVerifierResponse,
                reasoning={"effort": api["reasoning_effort"]},
                max_output_tokens=int(api["max_output_tokens"]),
                store=bool(api["store"]),
                service_tier=api["service_tier"],
                timeout=float(api["timeout_seconds"]),
            )
            parsed = getattr(response, "output_parsed", None)
            if not isinstance(parsed, WeeklyVerifierResponse):
                input_tokens, output_tokens = _usage_tokens(response)
                return {
                    "status": "refusal",
                    "persona_id": record["persona_id"],
                    "week_start": record["week_start"],
                    "week_end": record["week_end"],
                    "arm": record["arm"],
                    "repeat": repeat,
                    "prompt_sha256": record["prompt_sha256"],
                    "runtime_text_sha256": record["runtime_text_sha256"],
                    "requested_model": api["model"],
                    "resolved_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "attempts": attempt,
                    "latency_seconds": time.monotonic() - started,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "refusal": _response_refusal(response),
                }
            try:
                validate_parsed_response(parsed=parsed, record=record)
            except ValueError as error:
                input_tokens, output_tokens = _usage_tokens(response)
                return {
                    "status": "invalid",
                    "persona_id": record["persona_id"],
                    "week_start": record["week_start"],
                    "week_end": record["week_end"],
                    "arm": record["arm"],
                    "repeat": repeat,
                    "prompt_sha256": record["prompt_sha256"],
                    "runtime_text_sha256": record["runtime_text_sha256"],
                    "requested_model": api["model"],
                    "resolved_model": getattr(response, "model", None),
                    "response_id": getattr(response, "id", None),
                    "attempts": attempt,
                    "latency_seconds": time.monotonic() - started,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "validation_error": str(error),
                    "parsed": parsed.model_dump(mode="json"),
                }
            input_tokens, output_tokens = _usage_tokens(response)
            return {
                "status": "ok",
                "persona_id": record["persona_id"],
                "week_start": record["week_start"],
                "week_end": record["week_end"],
                "arm": record["arm"],
                "repeat": repeat,
                "prompt_sha256": record["prompt_sha256"],
                "runtime_text_sha256": record["runtime_text_sha256"],
                "requested_model": api["model"],
                "resolved_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None),
                "attempts": attempt,
                "latency_seconds": time.monotonic() - started,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "parsed": parsed.model_dump(mode="json"),
            }
        except Exception as error:  # noqa: BLE001 - API boundary must persist failures
            last_error = error
            if attempt >= int(api["max_attempts"]) or not _is_transient_error(error):
                break
            await asyncio.sleep((2 ** (attempt - 1)) + random.random())
    return {
        "status": "error",
        "persona_id": record["persona_id"],
        "week_start": record["week_start"],
        "week_end": record["week_end"],
        "arm": record["arm"],
        "repeat": repeat,
        "prompt_sha256": record["prompt_sha256"],
        "runtime_text_sha256": record["runtime_text_sha256"],
        "requested_model": api["model"],
        "attempts": attempt,
        "latency_seconds": time.monotonic() - started,
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": str(last_error) if last_error else "Unknown API error",
    }


def _response_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None


async def execute_calls(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Execute pending calls with bounded concurrency and spend reservation."""
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --execute")
    existing = _load_jsonl(output_path)
    records_by_key = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    completed = _completed_keys(
        existing,
        records_by_key,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(config["api"]["model"]),
    )
    pricing = config["api"]["pricing_usd_per_million_tokens"]
    actual_spend = sum(
        _request_cost_usd(
            input_tokens=int(row.get("usage", {}).get("input_tokens", 0)),
            output_tokens=int(row.get("usage", {}).get("output_tokens", 0)),
            pricing=pricing,
        )
        for row in existing
        if row.get("status") in TERMINAL_RESPONSE_STATUSES
    )
    pending = [
        (record, repeat)
        for repeat in range(1, int(config["study"]["repeats"]) + 1)
        for record in records
        if _response_key(record, repeat) not in completed
    ]
    random.Random(20260712).shuffle(pending)

    client = AsyncOpenAI()
    concurrency = int(config["api"]["concurrency"])
    max_budget = float(config["api"]["max_budget_usd"])
    completed_now = 0
    stopped_for_budget = False
    for offset in range(0, len(pending), concurrency):
        batch = pending[offset : offset + concurrency]
        worst_case_reservation = sum(
            _request_cost_usd(
                input_tokens=_approx_tokens(record["prompt"]),
                output_tokens=int(config["api"]["max_output_tokens"]),
                pricing=pricing,
            )
            for record, _repeat in batch
        )
        if actual_spend + worst_case_reservation > max_budget:
            stopped_for_budget = True
            break
        results = await asyncio.gather(
            *[
                _call_openai(
                    client=client,
                    record=record,
                    repeat=repeat,
                    config=config,
                )
                for record, repeat in batch
            ]
        )
        _append_jsonl(output_path, results)
        for result in results:
            if result["status"] in TERMINAL_RESPONSE_STATUSES:
                completed_now += 1
                actual_spend += _request_cost_usd(
                    input_tokens=int(result.get("usage", {}).get("input_tokens", 0)),
                    output_tokens=int(result.get("usage", {}).get("output_tokens", 0)),
                    pricing=pricing,
                )
    return {
        "completed_before": len(completed),
        "completed_now": completed_now,
        "remaining": len(pending) - completed_now,
        "actual_spend_usd": actual_spend,
        "stopped_for_budget": stopped_for_budget,
    }


def _load_targets(
    config: dict[str, Any], root: Path, cases: dict[str, dict[str, Any]]
) -> tuple[list[TargetCell], dict[str, bool]]:
    population = config["population"]
    reviewer_a = _case_map(_read_json(_rooted(population["reviewer_a_path"], root)))
    reviewer_b = _case_map(_read_json(_rooted(population["reviewer_b_path"], root)))
    unresolved_cases = set(population["unresolved_case_ids"])
    cells: list[TargetCell] = []
    episode_targets: dict[str, bool] = {}
    for review_case_id, case in cases.items():
        first = reviewer_a[review_case_id]
        second = reviewer_b[review_case_id]
        if review_case_id in unresolved_cases:
            continue
        if first["sustained_conflict"] != second["sustained_conflict"]:
            raise ValueError(f"Unexpected unresolved case: {review_case_id}")
        episode_targets[review_case_id] = first["sustained_conflict"] == "yes"
        first_entries = {
            int(row["position"]): row["observable_negative"]
            for row in first["entry_assessments"]
        }
        second_entries = {
            int(row["position"]): row["observable_negative"]
            for row in second["entry_assessments"]
        }
        for position, entry in enumerate(case["entries"], start=1):
            if first_entries[position] != second_entries[position]:
                continue
            label = first_entries[position]
            if label not in {"yes", "no"}:
                continue
            cells.append(
                TargetCell(
                    persona_id=case["persona_id"],
                    dimension=case["dimension"],
                    t_index=int(entry["t_index"]),
                    conflict=label == "yes",
                )
            )
    expected_cells = int(population["expected_resolved_entry_cells"])
    expected_cases = int(population["expected_resolved_trajectories"])
    expected_positive = int(population["expected_reference_episodes"])
    if len(cells) != expected_cells or len(episode_targets) != expected_cases:
        raise ValueError(
            f"Resolved target drift: {len(cells)} cells/{len(episode_targets)} cases"
        )
    if sum(episode_targets.values()) != expected_positive:
        raise ValueError("Reference episode count drift")
    return cells, episode_targets


def _flatten_predictions(
    responses: list[dict[str, Any]],
) -> dict[tuple[str, int, str, int, str], VerifierAssessment]:
    flattened: dict[tuple[str, int, str, int, str], VerifierAssessment] = {}
    for response in responses:
        if response.get("status") != "ok":
            continue
        parsed = WeeklyVerifierResponse.model_validate(response["parsed"])
        for assessment in parsed.assessments:
            key = (
                str(response["arm"]),
                int(response["repeat"]),
                str(response["persona_id"]),
                assessment.t_index,
                assessment.dimension,
            )
            if key in flattened:
                raise ValueError(f"Duplicate prediction coordinate: {key}")
            flattened[key] = assessment
    return flattened


def _recover_prompt_contract_valid_responses(
    responses: list[dict[str, Any]],
    record_map: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Recover raw invalid receipts that satisfy the corrected prompt contract."""
    effective = []
    recovered = 0
    for response in responses:
        if response.get("status") != "invalid" or not response.get("parsed"):
            effective.append(response)
            continue
        key = (
            str(response["persona_id"]),
            str(response["week_start"]),
            str(response["arm"]),
        )
        record = record_map[key]
        try:
            parsed = WeeklyVerifierResponse.model_validate(response["parsed"])
            validate_parsed_response(parsed=parsed, record=record)
        except (ValueError, TypeError):
            effective.append(response)
            continue
        effective.append({**response, "status": "ok", "recovered_from": "invalid"})
        recovered += 1
    return effective, recovered


def _binary_metrics(
    targets: list[bool], predictions: list[bool | None]
) -> dict[str, float]:
    covered = [
        (target, prediction)
        for target, prediction in zip(targets, predictions, strict=True)
        if prediction is not None
    ]
    tp = sum(target and prediction for target, prediction in covered)
    fp = sum((not target) and prediction for target, prediction in covered)
    fn = sum(
        target and prediction is not True
        for target, prediction in zip(targets, predictions, strict=True)
    )
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "predicted_conflict_rate": sum(prediction is True for prediction in predictions)
        / len(predictions)
        if predictions
        else 0.0,
        "coverage": len(covered) / len(targets) if targets else 0.0,
        "abstention_rate": 1.0 - (len(covered) / len(targets)) if targets else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "n": float(len(targets)),
    }


def _entry_metric_bundle(
    cells: list[TargetCell], predictions: list[bool | None]
) -> dict[str, Any]:
    pooled = _binary_metrics([cell.conflict for cell in cells], predictions)
    per_dimension = {}
    recalls = []
    for dimension in sorted({cell.dimension for cell in cells}):
        indices = [
            index for index, cell in enumerate(cells) if cell.dimension == dimension
        ]
        metrics = _binary_metrics(
            [cells[index].conflict for index in indices],
            [predictions[index] for index in indices],
        )
        negative_support = int(metrics["tp"] + metrics["fn"])
        metrics["negative_support"] = negative_support
        metrics["conclusion_eligible"] = len(indices) >= 10
        if negative_support:
            recalls.append(metrics["recall"])
        per_dimension[dimension] = metrics
    return {
        **pooled,
        "recall_micro": pooled["recall"],
        "recall_macro": statistics.fmean(recalls) if recalls else 0.0,
        "per_dimension": per_dimension,
    }


def _entry_metrics_for_cells(
    *,
    cells: list[TargetCell],
    predictions: dict[tuple[str, int, str, int, str], VerifierAssessment],
    repeats: int,
) -> list[dict[str, Any]]:
    results = []
    for arm in ARMS:
        for repeat in range(1, repeats + 1):
            assessments = [
                predictions.get(
                    (arm, repeat, cell.persona_id, cell.t_index, cell.dimension)
                )
                for cell in cells
            ]
            entry_predictions = _confidence_predictions(assessments, "low")
            results.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    **_entry_metric_bundle(cells, entry_predictions),
                }
            )
    return results


def _mlp_family_baseline(
    *, config: dict[str, Any], root: Path, cells: list[TargetCell]
) -> dict[str, Any]:
    """Score the incumbent MLP seeds on the exact resolved entry cells."""
    results = []
    for configured_path in config["population"]["mlp_run_paths"]:
        run = _read_yaml(_rooted(configured_path, root))
        frame = pl.read_parquet(_rooted(run["artifacts"]["validation_outputs"], root))
        predicted_by_coordinate = {
            (str(row["persona_id"]), str(row["dimension"]), int(row["t_index"])): (
                int(row["predicted_class"]) == -1
            )
            for row in frame.to_dicts()
        }
        predictions = [
            predicted_by_coordinate.get((cell.persona_id, cell.dimension, cell.t_index))
            for cell in cells
        ]
        if any(prediction is None for prediction in predictions):
            raise ValueError("MLP family baseline is missing a resolved entry cell")
        results.append(
            {
                "run_id": str(run["metadata"]["run_id"]),
                **_entry_metric_bundle(cells, predictions),
            }
        )
    return {
        "surface": "same 316 reviewer-resolved development entry cells",
        "results": results,
        "median_recall_macro": statistics.median(
            row["recall_macro"] for row in results
        ),
        "median_recall_micro": statistics.median(
            row["recall_micro"] for row in results
        ),
        "median_precision": statistics.median(row["precision"] for row in results),
        "median_predicted_conflict_rate": statistics.median(
            row["predicted_conflict_rate"] for row in results
        ),
    }


def _consensus_replay(
    *,
    config: dict[str, Any],
    root: Path,
    reviewer_cells: list[TargetCell],
    predictions: dict[tuple[str, int, str, int, str], VerifierAssessment],
) -> dict[str, Any]:
    """Replay the same outputs against the frozen twinkl-754 consensus labels."""
    path = _rooted(config["study"]["baselines"]["consensus_replay"], root)
    frame = pl.read_parquet(path)
    consensus = {
        (str(row["persona_id"]), int(row["t_index"]), dimension): (
            int(row[f"alignment_{dimension}"]) == -1
        )
        for row in frame.to_dicts()
        for dimension in {cell.dimension for cell in reviewer_cells}
    }
    cells = []
    agreement = []
    for reviewer_cell in reviewer_cells:
        key = (
            reviewer_cell.persona_id,
            reviewer_cell.t_index,
            reviewer_cell.dimension,
        )
        if key not in consensus:
            raise ValueError(f"Consensus replay is missing {key}")
        conflict = consensus[key]
        cells.append(
            TargetCell(
                persona_id=reviewer_cell.persona_id,
                dimension=reviewer_cell.dimension,
                t_index=reviewer_cell.t_index,
                conflict=conflict,
            )
        )
        agreement.append(conflict == reviewer_cell.conflict)
    return {
        "status": "available",
        "surface": (
            "same 316 reviewer-resolved development coordinates; target swapped only"
        ),
        "target_agreement": sum(agreement) / len(agreement),
        "reviewer_negative_cells": sum(cell.conflict for cell in reviewer_cells),
        "consensus_negative_cells": sum(cell.conflict for cell in cells),
        "verifier_results": _entry_metrics_for_cells(
            cells=cells,
            predictions=predictions,
            repeats=int(config["study"]["repeats"]),
        ),
        "mlp_family": _mlp_family_baseline(config=config, root=root, cells=cells),
    }


def _human_anchor_availability(
    *, root: Path, reviewer_cells: list[TargetCell]
) -> dict[str, Any]:
    """Check whether the three-way human anchor overlaps the study population."""
    annotation_paths = sorted((root / "logs/annotations").glob("*.parquet"))
    persona_ids = {cell.persona_id for cell in reviewer_cells}
    entry_sets = []
    per_annotator_entries = {}
    for path in annotation_paths:
        frame = pl.read_parquet(path).filter(pl.col("persona_id").is_in(persona_ids))
        entries = {
            (str(row["persona_id"]), int(row["t_index"]))
            for row in frame.select("persona_id", "t_index").unique().to_dicts()
        }
        entry_sets.append(entries)
        per_annotator_entries[path.stem] = len(entries)
    strict_overlap = set.intersection(*entry_sets) if entry_sets else set()
    return {
        "status": (
            "available"
            if strict_overlap
            else "unavailable_on_matched_development_inputs"
        ),
        "matched_strict_overlap_entries": len(strict_overlap),
        "matched_entries_by_annotator": per_annotator_entries,
        "reason": (
            None
            if strict_overlap
            else (
                "The existing three-annotator anchor has no strict overlap with the "
                "28-persona development surface; no new annotation was authorized."
            )
        ),
    }


def _historical_entry_llm_baseline(root: Path) -> dict[str, Any]:
    """Load the frozen-test entry LLM baseline without making API calls."""
    from scripts.experiments.llm_critic_baseline import score_records

    relative = Path(
        "logs/experiments/artifacts/llm_critic_baseline/"
        "20260702_test_main_context_arms_none/"
        "test_student_visible_gpt-5.4-mini_none_shots0.jsonl"
    )
    metrics = score_records(_load_jsonl(root / relative))
    return {
        "surface": (
            "separate frozen 221-entry test split; context only, not matched "
            "causal evidence"
        ),
        "path": str(relative),
        "n": metrics["n_ok"],
        "qwk_mean": metrics["qwk_mean"],
        "recall_minus1": metrics["recall_minus1"],
        "minority_recall_mean": metrics["minority_recall_mean"],
        "hedging_mean": metrics["hedging_mean"],
    }


def _response_summary(
    responses: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    by_arm: dict[str, dict[str, int]] = {}
    input_tokens = 0
    output_tokens = 0
    for row in responses:
        status = str(row.get("status"))
        arm = str(row.get("arm"))
        statuses[status] = statuses.get(status, 0) + 1
        arm_statuses = by_arm.setdefault(arm, {})
        arm_statuses[status] = arm_statuses.get(status, 0) + 1
        usage = row.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
    return {
        "total": len(responses),
        "statuses": statuses,
        "by_arm": by_arm,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_spend_usd": _request_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=config["api"]["pricing_usd_per_million_tokens"],
        ),
    }


def _predicted_episode(
    *,
    case: dict[str, Any],
    arm: str,
    repeat: int,
    predictions: dict[tuple[str, int, str, int, str], VerifierAssessment],
) -> tuple[bool, bool, int | None]:
    verdicts: list[VerifierAssessment | None] = [
        predictions.get(
            (arm, repeat, case["persona_id"], int(entry["t_index"]), case["dimension"])
        )
        for entry in case["entries"]
    ]
    covered = all(
        verdict is not None and verdict.verdict != "abstain" for verdict in verdicts
    )
    confirmation_t_index = None
    for index, (first, second) in enumerate(zip(verdicts, verdicts[1:], strict=False)):
        adjacent = (
            int(case["entries"][index + 1]["t_index"])
            == int(case["entries"][index]["t_index"]) + 1
        )
        if (
            adjacent
            and first is not None
            and second is not None
            and first.verdict == "conflict"
            and second.verdict == "conflict"
        ):
            confirmation_t_index = int(case["entries"][index + 1]["t_index"])
            break
    return confirmation_t_index is not None, covered, confirmation_t_index


def _confidence_predictions(
    assessments: list[VerifierAssessment | None], minimum: Confidence
) -> list[bool | None]:
    rank = {"low": 1, "medium": 2, "high": 3}
    minimum_rank = rank[minimum]
    return [
        None
        if assessment is None
        or assessment.verdict == "abstain"
        or rank[assessment.confidence] < minimum_rank
        else assessment.verdict == "conflict"
        for assessment in assessments
    ]


def _target_confirmation_t_index(
    case: dict[str, Any], conflict_by_coordinate: dict[tuple[str, str, int], bool]
) -> int | None:
    labels = [
        conflict_by_coordinate.get(
            (case["persona_id"], case["dimension"], int(entry["t_index"]))
        )
        for entry in case["entries"]
    ]
    for index, (first, second) in enumerate(zip(labels, labels[1:], strict=False)):
        if (
            first is True
            and second is True
            and int(case["entries"][index + 1]["t_index"])
            == int(case["entries"][index]["t_index"]) + 1
        ):
            return int(case["entries"][index + 1]["t_index"])
    return None


def _episode_rows(
    *,
    case: dict[str, Any],
    labels: list[bool | None],
    source: str,
) -> list[dict[str, Any]]:
    """Collapse consecutive conflict labels into sustained-conflict episodes."""
    episodes = []
    run: list[dict[str, Any]] = []

    def finish() -> None:
        if len(run) < 2:
            return
        episodes.append(
            {
                "episode_id": (
                    f"{source}:{case['review_case_id']}:{run[0]['t_index']}:"
                    f"{run[-1]['t_index']}"
                ),
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "onset_t_index": int(run[0]["t_index"]),
                "confirmation_t_index": int(run[1]["t_index"]),
                "end_t_index": int(run[-1]["t_index"]),
                "delivery_state": "active",
            }
        )

    for entry, label in zip(case["entries"], labels, strict=True):
        adjacent = not run or int(entry["t_index"]) == int(run[-1]["t_index"]) + 1
        if label is True and adjacent:
            run.append(entry)
            continue
        finish()
        run = [entry] if label is True else []
    finish()
    return episodes


def _episode_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "episode_id": pl.String,
        "persona_id": pl.String,
        "dimension": pl.String,
        "onset_t_index": pl.Int64,
        "confirmation_t_index": pl.Int64,
        "end_t_index": pl.Int64,
        "delivery_state": pl.String,
    }
    return (
        pl.DataFrame(rows, schema=schema, strict=False)
        if rows
        else pl.DataFrame(schema=schema)
    )


def _trajectory_covered(
    labels: list[bool | None], entries: list[dict[str, Any]]
) -> bool:
    adjacent_pairs = [
        (first, second)
        for index, (first, second) in enumerate(zip(labels, labels[1:], strict=False))
        if int(entries[index + 1]["t_index"]) == int(entries[index]["t_index"]) + 1
    ]
    if any(first is True and second is True for first, second in adjacent_pairs):
        return True
    return bool(adjacent_pairs) and all(
        first is False or second is False for first, second in adjacent_pairs
    )


def _score_episode_surface(
    *,
    cases: dict[str, dict[str, Any]],
    episode_targets: dict[str, bool],
    conflict_by_coordinate: dict[tuple[str, str, int], bool],
    arm: str,
    repeat: int,
    predictions: dict[tuple[str, int, str, int, str], VerifierAssessment],
) -> tuple[dict[str, float], dict[str, bool | None], list[dict[str, Any]]]:
    from src.vif.drift_benchmark import match_episodes

    reference_rows = []
    predicted_rows = []
    case_predictions = {}
    reference_case_by_episode = {}
    for review_case_id, case in cases.items():
        case = {**case, "review_case_id": review_case_id}
        target_labels = [
            conflict_by_coordinate.get(
                (case["persona_id"], case["dimension"], int(entry["t_index"]))
            )
            for entry in case["entries"]
        ]
        case_reference = _episode_rows(
            case=case, labels=target_labels, source="reference"
        )
        reference_rows.extend(case_reference)
        reference_case_by_episode.update(
            {row["episode_id"]: review_case_id for row in case_reference}
        )

        assessments = [
            predictions.get(
                (
                    arm,
                    repeat,
                    case["persona_id"],
                    int(entry["t_index"]),
                    case["dimension"],
                )
            )
            for entry in case["entries"]
        ]
        predicted_labels = [
            None
            if assessment is None or assessment.verdict == "abstain"
            else assessment.verdict == "conflict"
            for assessment in assessments
        ]
        case_predicted = _episode_rows(
            case=case, labels=predicted_labels, source=f"{arm}:{repeat}"
        )
        predicted_rows.extend(case_predicted)
        if case_predicted:
            case_predictions[review_case_id] = True
        elif _trajectory_covered(predicted_labels, case["entries"]):
            case_predictions[review_case_id] = False
        else:
            case_predictions[review_case_id] = None

    if len(reference_rows) != sum(episode_targets.values()):
        raise ValueError("Reference episode extraction drifted from reviewer decisions")
    matches = match_episodes(
        _episode_frame(reference_rows),
        _episode_frame(predicted_rows),
        max_confirmation_lag=2,
    )
    true_positive = matches.height
    false_positive = len(predicted_rows) - true_positive
    false_negative = len(reference_rows) - true_positive
    covered = sum(value is not None for value in case_predictions.values())
    metrics = {
        "recall": true_positive / len(reference_rows) if reference_rows else 0.0,
        "precision": true_positive / len(predicted_rows) if predicted_rows else 0.0,
        "predicted_conflict_rate": sum(
            value is True for value in case_predictions.values()
        )
        / len(case_predictions),
        "coverage": covered / len(case_predictions),
        "abstention_rate": 1.0 - (covered / len(case_predictions)),
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "n": float(len(case_predictions)),
    }
    match_by_reference = {
        str(row["reference_episode_id"]): row for row in matches.to_dicts()
    }
    timing = []
    for reference in reference_rows:
        match = match_by_reference.get(reference["episode_id"])
        timing.append(
            {
                "review_case_id": reference_case_by_episode[reference["episode_id"]],
                "target_confirmation_t_index": reference["confirmation_t_index"],
                "predicted_confirmation_t_index": (
                    int(match["predicted_confirmation_t_index"]) if match else None
                ),
                "delay_entries": int(match["latency_entries"]) if match else None,
            }
        )
    return metrics, case_predictions, timing


def score_responses(
    *,
    config: dict[str, Any],
    root: Path,
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    personas, cases = _load_population(config, root)
    del personas
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    _completed_keys(
        responses,
        record_map,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(config["api"]["model"]),
    )
    cells, episode_targets = _load_targets(config, root, cases)
    effective_responses, recovered_invalid = _recover_prompt_contract_valid_responses(
        responses, record_map
    )
    predictions = _flatten_predictions(effective_responses)
    results: list[dict[str, Any]] = []
    unresolved = set(config["population"]["unresolved_case_ids"])
    scored_cases = {
        review_case_id: case
        for review_case_id, case in cases.items()
        if review_case_id not in unresolved
    }
    conflict_by_coordinate = {
        (cell.persona_id, cell.dimension, cell.t_index): cell.conflict for cell in cells
    }

    for arm in ARMS:
        for repeat in range(1, int(config["study"]["repeats"]) + 1):
            entry_targets = [cell.conflict for cell in cells]
            entry_assessments: list[VerifierAssessment | None] = []
            for cell in cells:
                assessment = predictions.get(
                    (arm, repeat, cell.persona_id, cell.t_index, cell.dimension)
                )
                entry_assessments.append(assessment)
            entry_predictions = _confidence_predictions(entry_assessments, "low")
            entry_metrics = _entry_metric_bundle(cells, entry_predictions)
            confidence_operating_points = {
                name: _binary_metrics(
                    entry_targets,
                    _confidence_predictions(entry_assessments, minimum),
                )
                for name, minimum in (
                    ("high_only", "high"),
                    ("medium_plus", "medium"),
                    ("all_confident", "low"),
                )
            }
            per_dimension = entry_metrics.pop("per_dimension")

            episode_metrics, case_predictions, detection_timing = (
                _score_episode_surface(
                    cases=scored_cases,
                    episode_targets=episode_targets,
                    conflict_by_coordinate=conflict_by_coordinate,
                    arm=arm,
                    repeat=repeat,
                    predictions=predictions,
                )
            )
            results.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "entry": entry_metrics,
                    "entry_confidence_operating_points": confidence_operating_points,
                    "entry_per_dimension": per_dimension,
                    "episode": episode_metrics,
                    "case_predictions": case_predictions,
                    "detection_timing": detection_timing,
                }
            )

    return {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "resolved_entry_cells": len(cells),
        "resolved_trajectories": len(episode_targets),
        "positive_episodes": sum(episode_targets.values()),
        "negative_trajectories": len(episode_targets) - sum(episode_targets.values()),
        "response_summary": {
            **_response_summary(responses, config),
            "recovered_prompt_contract_valid": recovered_invalid,
            "effective_ok": sum(
                response.get("status") == "ok" for response in effective_responses
            ),
            "effective_invalid": sum(
                response.get("status") == "invalid" for response in effective_responses
            ),
        },
        "baselines": {
            "mlp_family": _mlp_family_baseline(config=config, root=root, cells=cells),
            "same_surface_llm": "without_critic verifier arm",
            "historical_entry_llm": _historical_entry_llm_baseline(root),
        },
        "consensus_replay": _consensus_replay(
            config=config,
            root=root,
            reviewer_cells=cells,
            predictions=predictions,
        ),
        "human_anchor": _human_anchor_availability(root=root, reviewer_cells=cells),
        "results": results,
        "paired_case_changes": _paired_case_changes(results, episode_targets),
        "decision": _paired_decision(results),
    }


def _paired_case_changes(
    results: list[dict[str, Any]], episode_targets: dict[str, bool]
) -> list[dict[str, Any]]:
    indexed = {(row["arm"], row["repeat"]): row for row in results}
    paired = []
    for repeat in sorted({int(row["repeat"]) for row in results}):
        without = indexed[("without_critic", repeat)]["case_predictions"]
        with_critic = indexed[("with_critic", repeat)]["case_predictions"]
        helped = []
        harmed = []
        unchanged = []
        for review_case_id, target in episode_targets.items():
            without_correct = without[review_case_id] is not None and (
                without[review_case_id] == target
            )
            with_correct = with_critic[review_case_id] is not None and (
                with_critic[review_case_id] == target
            )
            if with_correct and not without_correct:
                helped.append(review_case_id)
            elif without_correct and not with_correct:
                harmed.append(review_case_id)
            else:
                unchanged.append(review_case_id)
        paired.append(
            {
                "repeat": repeat,
                "helped": sorted(helped),
                "harmed": sorted(harmed),
                "unchanged": sorted(unchanged),
            }
        )
    return paired


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _paired_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm: [row for row in results if row["arm"] == arm] for arm in ARMS}
    if any(not rows for rows in by_arm.values()):
        return {"verdict": "incomplete", "reason": "Both arms require scored results."}
    expected_repeats = {1, 2, 3}
    if any(
        {int(row["repeat"]) for row in rows} != expected_repeats
        for rows in by_arm.values()
    ):
        return {
            "verdict": "incomplete",
            "reason": "Both arms require exactly repeats 1, 2, and 3.",
        }
    summary = {
        arm: {
            "median_episode_recall": _median(
                [row["episode"]["recall"] for row in rows]
            ),
            "median_false_alerts": _median([row["episode"]["fp"] for row in rows]),
            "median_coverage": _median([row["episode"]["coverage"] for row in rows]),
        }
        for arm, rows in by_arm.items()
    }
    without = summary["without_critic"]
    with_critic = summary["with_critic"]
    recall_delta = (
        with_critic["median_episode_recall"] - without["median_episode_recall"]
    )
    false_alert_delta = (
        with_critic["median_false_alerts"] - without["median_false_alerts"]
    )
    coverage_delta = with_critic["median_coverage"] - without["median_coverage"]
    if recall_delta > 0 and false_alert_delta <= 0 and coverage_delta >= 0:
        verdict = "positive"
    elif recall_delta <= 0 and (
        false_alert_delta >= 0
        or with_critic["median_episode_recall"] <= without["median_episode_recall"]
    ):
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "summary": summary,
        "median_deltas": {
            "episode_recall": recall_delta,
            "false_alerts": false_alert_delta,
            "coverage": coverage_delta,
        },
        "note": (
            "Development-only conditional evidence; no architecture adoption "
            "or promotion."
        ),
    }


def _artifact_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    artifacts = config["artifacts"]
    output_dir = _rooted(artifacts["output_dir"], root)
    return {
        "output_dir": output_dir,
        "prompts": output_dir / artifacts["prompts_filename"],
        "manifest": output_dir / artifacts["manifest_filename"],
        "responses": output_dir / artifacts["responses_filename"],
        "metrics": output_dir / artifacts["metrics_filename"],
    }


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = _read_yaml(_rooted(args.config, root))
    records, manifest = build_prompt_records(config, root)
    paths = _artifact_paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    _write_jsonl(paths["prompts"], records)
    estimate = estimate_plan(records, config)
    manifest["estimate"] = estimate
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"paths": {k: str(v) for k, v in paths.items()}, **estimate}, indent=2
        )
    )


def _load_prepared(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    paths = _artifact_paths(config, root)
    records = _load_jsonl(paths["prompts"])
    if not records or not paths["manifest"].exists():
        raise FileNotFoundError("Run prepare before this command")
    manifest = _read_json(paths["manifest"])
    for record in records:
        if _sha256_text(record["prompt"]) != record["prompt_sha256"]:
            raise ValueError("Prepared prompt content hash mismatch")
    rebuilt_records, rebuilt_manifest = build_prompt_records(config, root)
    if _canonical_json(records) != _canonical_json(rebuilt_records):
        raise ValueError("Prepared prompts no longer match the registered sources")
    immutable_manifest_keys = (
        "config_payload_sha256",
        "packet_sha256",
        "prompt_manifest_sha256",
        "prompt_template_sha256",
        "reconciliation_key_sha256",
        "reviewer_a_sha256",
        "reviewer_b_sha256",
        "run_020_evidence_sha256",
        "runtime_contract_sha256",
    )
    for key in immutable_manifest_keys:
        if manifest.get(key) != rebuilt_manifest.get(key):
            raise ValueError(f"Prepared manifest source mismatch: {key}")
    observed_hash = _sha256_text(
        _canonical_json(
            [
                {
                    "persona_id": record["persona_id"],
                    "week_start": record["week_start"],
                    "arm": record["arm"],
                    "prompt_sha256": record["prompt_sha256"],
                }
                for record in records
            ]
        )
    )
    if observed_hash != manifest["prompt_manifest_sha256"]:
        raise ValueError("Prepared prompt manifest hash mismatch")
    return records, paths


def command_estimate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = _read_yaml(_rooted(args.config, root))
    records, _paths = _load_prepared(config, root)
    print(json.dumps(estimate_plan(records, config), indent=2))


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = _read_yaml(_rooted(args.config, root))
    records, paths = _load_prepared(config, root)
    result = asyncio.run(
        execute_calls(records=records, config=config, output_path=paths["responses"])
    )
    print(json.dumps(result, indent=2))


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = _read_yaml(_rooted(args.config, root))
    records, paths = _load_prepared(config, root)
    responses = _load_jsonl(paths["responses"])
    metrics = score_responses(
        config=config,
        root=root,
        records=records,
        responses=responses,
    )
    metrics["artifact_provenance"] = {
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "prompts_sha256": _sha256_file(paths["prompts"]),
        "responses_sha256": _sha256_file(paths["responses"]),
        "reviewer_a_sha256": _sha256_file(
            _rooted(config["population"]["reviewer_a_path"], root)
        ),
        "reviewer_b_sha256": _sha256_file(
            _rooted(config["population"]["reviewer_b_path"], root)
        ),
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["decision"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("estimate").set_defaults(func=command_estimate)
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true")
    run.set_defaults(func=command_run)
    subparsers.add_parser("score").set_defaults(func=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
