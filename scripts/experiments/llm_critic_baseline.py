#!/usr/bin/env python3
"""Run small OpenAI LLM critic baselines for twinkl-w2mu.

The experiment compares product-plausible OpenAI small models against the
frozen VIF Critic holdout under explicit context arms: student-visible,
human-context history, and optional full judge context.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge.labeling import build_session_content
from src.models.judge import SCHWARTZ_VALUE_ORDER, AlignmentScores
from src.vif.dataset import load_all_data, split_by_persona
from src.vif.eval import (
    _nanmean_or_nan,
    compute_accuracy_per_dimension,
    compute_hedging_per_dimension,
    compute_mae_per_dimension,
    compute_qwk_nan_dims_count,
    compute_qwk_per_dimension,
    compute_recall_per_class,
)
from src.vif.holdout import load_fixed_holdout_ids
from src.vif.state_encoder import VALUE_NAME_TO_KEY
from src.wrangling.parse_wrangled_data import parse_wrangled_file

DEFAULT_MODELS = (
    "gpt-5.4-nano",
    "gpt-5.4-mini",
)
DEFAULT_REASONING_EFFORTS = ("none", "low")
DEFAULT_CONTEXT_ARMS = ("student_visible",)
CONTEXT_ARMS = ("student_visible", "human_context", "full_judge_context")
DEFAULT_HOLDOUT_MANIFEST = Path("config/experiments/vif/twinkl_681_5_holdout.yaml")
DEFAULT_LABELS_PATH = Path("logs/judge_labels/judge_labels.parquet")
DEFAULT_WRANGLED_DIR = Path("logs/wrangled")
DEFAULT_OUTPUT_ROOT = Path("logs/experiments/artifacts/llm_critic_baseline")
DEFAULT_REPORTS_DIR = Path("logs/experiments/reports")
RUN_020_PATH = Path("logs/experiments/runs/run_020_BalancedSoftmax.yaml")
AGREEMENT_REPORT_PATH = Path("logs/exports/agreement_report_20260318_130642.md")
EXPECTED_FROZEN_TEST_ROWS = 221

MODEL_PRICING_PER_1M = {
    "gpt-5.4-nano-2026-03-17": {"input": 0.20, "output": 1.25},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}
PRICING_SOURCE = (
    "OpenAI GPT-5.4 mini/nano model docs, text-token prices per 1M tokens."
)


class LLMCriticResponse(BaseModel):
    """Structured output expected from the OpenAI response."""

    scores: AlignmentScores


@dataclass(frozen=True)
class ExperimentRow:
    split: str
    persona_id: str
    persona_name: str
    persona_age: str
    persona_profession: str
    persona_culture: str
    persona_bio: str
    t_index: int
    date: str
    session_content: str
    previous_entries: tuple[dict[str, str], ...]
    core_values: tuple[str, ...]
    profile_weights: tuple[float, ...]
    target_vector: tuple[int, ...]

    @property
    def entry_id(self) -> str:
        return f"{self.persona_id}__{self.t_index}"


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()


def approx_token_count(text: str) -> int:
    """Cheap token estimate; good enough for preflight cost budgeting."""
    return max(1, math.ceil(len(text) / 4))


def normalize_core_values(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(value.strip() for value in raw.split(",") if value.strip())
    return tuple(str(value).strip() for value in raw if str(value).strip())


def load_persona_profiles(wrangled_dir: Path = DEFAULT_WRANGLED_DIR) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    wrangled_path = wrangled_dir if wrangled_dir.is_absolute() else ROOT / wrangled_dir
    for path in sorted(wrangled_path.glob("persona_*.md")):
        profile, _entries, _warnings = parse_wrangled_file(path)
        profiles[str(profile["persona_id"])] = profile
    return profiles


def core_values_to_profile_weights(core_values: tuple[str, ...]) -> tuple[float, ...]:
    matched_indices = []
    for value in core_values:
        canonical_key = VALUE_NAME_TO_KEY.get(value) or VALUE_NAME_TO_KEY.get(
            value.strip()
        )
        if canonical_key is None:
            normalized = value.lower().replace("-", "_").replace(" ", "_").strip()
            if normalized in SCHWARTZ_VALUE_ORDER:
                canonical_key = normalized
        if canonical_key in SCHWARTZ_VALUE_ORDER:
            matched_indices.append(SCHWARTZ_VALUE_ORDER.index(canonical_key))

    matched_indices = list(dict.fromkeys(matched_indices))
    weights = np.zeros(len(SCHWARTZ_VALUE_ORDER), dtype=np.float32)
    if matched_indices:
        weight = 1.0 / len(matched_indices)
        for index in matched_indices:
            weights[index] = weight
    else:
        weights = np.full(
            len(SCHWARTZ_VALUE_ORDER),
            1.0 / len(SCHWARTZ_VALUE_ORDER),
            dtype=np.float32,
        )
    return tuple(float(value) for value in weights.tolist())


def load_split_frames(
    *,
    labels_path: Path = DEFAULT_LABELS_PATH,
    wrangled_dir: Path = DEFAULT_WRANGLED_DIR,
    holdout_manifest: Path = DEFAULT_HOLDOUT_MANIFEST,
) -> dict[str, pl.DataFrame]:
    labels_df, entries_df = load_all_data(labels_path, wrangled_dir)
    val_ids, test_ids = load_fixed_holdout_ids(holdout_manifest)
    train_df, val_df, test_df = split_by_persona(
        labels_df,
        entries_df,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=2025,
        fixed_val_persona_ids=val_ids,
        fixed_test_persona_ids=test_ids,
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def rows_from_frame(
    frame: pl.DataFrame,
    split: str,
    *,
    persona_profiles: dict[str, dict] | None = None,
) -> list[ExperimentRow]:
    rows = []
    previous_by_persona: dict[str, list[dict[str, str]]] = {}
    for raw in frame.sort(["persona_id", "t_index"]).to_dicts():
        persona_id = str(raw["persona_id"])
        core_values = normalize_core_values(raw.get("core_values"))
        session = build_session_content(
            raw.get("initial_entry"),
            raw.get("nudge_text"),
            raw.get("response_text"),
        )
        profile = (persona_profiles or {}).get(persona_id, {})
        rows.append(
            ExperimentRow(
                split=split,
                persona_id=persona_id,
                persona_name=str(profile.get("name") or raw.get("persona_name") or ""),
                persona_age=str(profile.get("age") or ""),
                persona_profession=str(profile.get("profession") or ""),
                persona_culture=str(profile.get("culture") or ""),
                persona_bio=str(profile.get("bio") or ""),
                t_index=int(raw["t_index"]),
                date=str(raw["date"]),
                session_content=session,
                previous_entries=tuple(previous_by_persona.get(persona_id, [])),
                core_values=core_values,
                profile_weights=core_values_to_profile_weights(core_values),
                target_vector=tuple(int(value) for value in raw["alignment_vector"]),
            )
        )
        previous_by_persona.setdefault(persona_id, []).append(
            {
                "date": str(raw["date"]),
                "t_index": str(raw["t_index"]),
                "content": session,
            }
        )
    return rows


def select_rows(
    rows: list[ExperimentRow],
    *,
    limit: int | None,
    seed: int,
) -> list[ExperimentRow]:
    if limit is None or limit >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=limit, replace=False).tolist())
    return [rows[index] for index in indices]


def scores_dict_from_vector(vector: tuple[int, ...]) -> dict[str, int]:
    return {
        dimension: int(vector[index])
        for index, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
    }


def profile_block(row: ExperimentRow) -> str:
    weight_lines = [
        f"- {dimension}: {row.profile_weights[index]:.3f}"
        for index, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
    ]
    core_values = ", ".join(row.core_values) or "none"
    return "\n".join(
        [
            f"Core Values list: {core_values}",
            "Normalized 10-dim value profile:",
            *weight_lines,
        ]
    )


def previous_entries_block(row: ExperimentRow) -> str:
    if not row.previous_entries:
        return "Previous entries: none"

    parts = ["Previous journal entries before the current session:"]
    for previous in row.previous_entries:
        parts.append(
            "\n".join(
                [
                    f"- Entry {previous['t_index']} ({previous['date']}):",
                    previous["content"],
                ]
            )
        )
    return "\n\n".join(parts)


def persona_profile_block(row: ExperimentRow) -> str:
    return "\n".join(
        [
            "Persona profile:",
            f"- Name: {row.persona_name or 'unknown'}",
            f"- Age: {row.persona_age or 'unknown'}",
            f"- Profession: {row.persona_profession or 'unknown'}",
            f"- Culture: {row.persona_culture or 'unknown'}",
            f"- Bio: {row.persona_bio or 'unknown'}",
        ]
    )


def context_block(row: ExperimentRow, *, context_arm: str) -> str:
    if context_arm == "student_visible":
        return "\n\n".join(
            [
                "Context arm: student_visible",
                "Use only the current journal session and value profile.",
                "Value profile:",
                profile_block(row),
                "Journal session:",
                row.session_content,
            ]
        )

    if context_arm == "human_context":
        return "\n\n".join(
            [
                "Context arm: human_context",
                "Use prior journal history, current journal session, and "
                "value profile.",
                "Do not use persona bio or demographics.",
                "Value profile:",
                profile_block(row),
                previous_entries_block(row),
                "Current journal session:",
                row.session_content,
            ]
        )

    if context_arm == "full_judge_context":
        return "\n\n".join(
            [
                "Context arm: full_judge_context",
                "Use persona profile, prior journal history, current journal session, "
                "and value profile.",
                "Value profile:",
                profile_block(row),
                persona_profile_block(row),
                previous_entries_block(row),
                "Current journal session:",
                row.session_content,
            ]
        )

    raise ValueError(f"Unknown context arm: {context_arm}")


def render_prompt(
    row: ExperimentRow,
    *,
    few_shot_rows: list[ExperimentRow] | None = None,
    context_arm: str = "student_visible",
) -> str:
    examples = []
    for index, example in enumerate(few_shot_rows or [], start=1):
        examples.append(
            "\n".join(
                [
                    f"Example {index}",
                    context_block(example, context_arm=context_arm),
                    "Scores:",
                    json.dumps(
                        {"scores": scores_dict_from_vector(example.target_vector)},
                        sort_keys=True,
                    ),
                ]
            )
        )

    examples_block = ""
    if examples:
        examples_block = "\n\nTraining examples:\n" + "\n\n".join(examples) + "\n"

    return "\n".join(
        [
            "Score this journal session against Schwartz value alignment.",
            "",
            "Use exactly these labels for each value:",
            "- -1: behavior, choices, or expressed attitudes conflict with the value.",
            "- 0: no clear relation, weak evidence, or neutral/status quo.",
            "- 1: behavior, choices, or expressed attitudes support the value.",
            "",
            "Rules:",
            "- Judge observable behavior and choices, not mood alone.",
            "- Trade-offs are allowed; score each value independently.",
            "- Use only the context provided for this arm.",
            "- Do not use future entries, target labels, rationales, or generation "
            "metadata.",
            examples_block.rstrip(),
            "",
            context_block(row, context_arm=context_arm),
            "",
            "Return JSON only.",
        ]
    ).strip()


def choose_few_shot_rows(
    train_rows: list[ExperimentRow],
    *,
    shots: int,
    seed: int,
) -> list[ExperimentRow]:
    if shots <= 0:
        return []
    active = [
        row
        for row in train_rows
        if -1 in row.target_vector or 1 in row.target_vector
    ]
    neutral = [row for row in train_rows if set(row.target_vector) == {0}]
    pool = active if len(active) >= shots else train_rows
    rng = np.random.default_rng(seed)
    selected: list[ExperimentRow] = []
    if neutral and shots >= 3:
        selected.append(neutral[int(rng.integers(0, len(neutral)))])
    selected_ids = {row.entry_id for row in selected}
    remaining = [row for row in pool if row.entry_id not in selected_ids]
    needed = shots - len(selected)
    if needed > 0:
        indices = rng.choice(
            len(remaining),
            size=min(needed, len(remaining)),
            replace=False,
        )
        selected.extend(remaining[int(index)] for index in indices.tolist())
    return selected[:shots]


def response_usage_to_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    result = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        result[key] = getattr(usage, key, None)
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        result["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return result


def estimate_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = MODEL_PRICING_PER_1M.get(model)
    if pricing is None:
        return float("nan")
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


def summarize_estimates(
    rows: list[ExperimentRow],
    *,
    models: list[str],
    few_shot_rows: list[ExperimentRow],
    context_arm: str,
) -> dict[str, Any]:
    prompts = [
        render_prompt(row, few_shot_rows=few_shot_rows, context_arm=context_arm)
        for row in rows
    ]
    input_estimates = [approx_token_count(prompt) for prompt in prompts]
    output_estimate = approx_token_count(
        json.dumps({"scores": {dimension: 0 for dimension in SCHWARTZ_VALUE_ORDER}})
    )
    total_input = int(sum(input_estimates))
    total_output = int(output_estimate * len(rows))

    by_model = {}
    for model in models:
        by_model[model] = {
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": estimate_cost(
                model=model,
                input_tokens=total_input,
                output_tokens=total_output,
            ),
        }

    return {
        "n_rows": len(rows),
        "context_arm": context_arm,
        "prompt_input_tokens": {
            "mean": float(statistics.mean(input_estimates)) if input_estimates else 0.0,
            "p50": (
                float(statistics.median(input_estimates)) if input_estimates else 0.0
            ),
            "p95": (
                float(np.percentile(input_estimates, 95)) if input_estimates else 0.0
            ),
            "max": int(max(input_estimates)) if input_estimates else 0,
        },
        "estimated_output_tokens_per_row": output_estimate,
        "pricing_source": PRICING_SOURCE,
        "by_model": by_model,
    }


def ensure_frozen_test_size(splits: dict[str, pl.DataFrame]) -> None:
    test_rows = int(splits["test"].height)
    if test_rows != EXPECTED_FROZEN_TEST_ROWS:
        raise ValueError(
            f"Expected frozen test split to contain {EXPECTED_FROZEN_TEST_ROWS} rows, "
            f"got {test_rows}. Check holdout manifest and labels path."
        )


def call_openai(
    *,
    model: str,
    reasoning_effort: str,
    prompt: str,
    timeout: float,
    max_output_tokens: int,
) -> tuple[LLMCriticResponse, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=prompt,
        text_format=LLMCriticResponse,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        timeout=timeout,
    )
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, LLMCriticResponse):
        return parsed, response
    raise ValueError("OpenAI response did not contain parsed LLMCriticResponse")


def run_model_rows(
    rows: list[ExperimentRow],
    *,
    context_arm: str,
    model: str,
    reasoning_effort: str,
    few_shot_rows: list[ExperimentRow],
    execute: bool,
    timeout: float,
    max_attempts: int,
    max_output_tokens: int,
    start_ordinal: int = 1,
) -> list[dict[str, Any]]:
    output_rows = []
    for ordinal, row in enumerate(rows, start=start_ordinal):
        prompt = render_prompt(
            row,
            few_shot_rows=few_shot_rows,
            context_arm=context_arm,
        )
        base = {
            "entry_id": row.entry_id,
            "persona_id": row.persona_id,
            "t_index": row.t_index,
            "date": row.date,
            "split": row.split,
            "model": model,
            "context_arm": context_arm,
            "reasoning_effort": reasoning_effort,
            "shots": len(few_shot_rows),
            "ordinal": ordinal,
            "target": scores_dict_from_vector(row.target_vector),
            "core_values": list(row.core_values),
            "profile_weights": {
                dimension: row.profile_weights[index]
                for index, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
            },
            "prompt_chars": len(prompt),
            "estimated_input_tokens": approx_token_count(prompt),
        }
        if not execute:
            output_rows.append({**base, "status": "dry_run", "scores": None})
            continue

        row_started = time.perf_counter()
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                parsed, response = call_openai(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    prompt=prompt,
                    timeout=timeout,
                    max_output_tokens=max_output_tokens,
                )
                latency = time.perf_counter() - row_started
                scores = parsed.scores.model_dump()
                usage = response_usage_to_dict(response)
                output_rows.append(
                    {
                        **base,
                        "status": "ok",
                        "scores": {key: int(value) for key, value in scores.items()},
                        "latency_seconds": latency,
                        "response_id": getattr(response, "id", None),
                        "response_model": getattr(response, "model", None),
                        "usage": usage,
                        "attempts": attempt,
                    }
                )
                break
            except Exception as exc:  # pragma: no cover - network/API edge handling
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 5))
        else:
            latency = time.perf_counter() - row_started
            output_rows.append(
                {
                    **base,
                    "status": "error",
                    "scores": None,
                    "latency_seconds": latency,
                    "error": f"{type(last_error).__name__}: {last_error}",
                    "attempts": max_attempts,
                }
            )
    return output_rows


def records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    for record in records:
        scores = record.get("scores")
        target = record.get("target")
        if record.get("status") != "ok" or not isinstance(scores, dict):
            continue
        predictions.append(
            [int(scores[dimension]) for dimension in SCHWARTZ_VALUE_ORDER]
        )
        targets.append([int(target[dimension]) for dimension in SCHWARTZ_VALUE_ORDER])
    if not predictions:
        raise ValueError("No successful scored rows found")
    return (
        np.asarray(predictions, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions, targets = records_to_arrays(records)
    qwk_per_dim = compute_qwk_per_dimension(predictions, targets)
    recall = compute_recall_per_class(predictions, targets)
    hedging = compute_hedging_per_dimension(predictions)
    mae = compute_mae_per_dimension(predictions, targets)
    accuracy = compute_accuracy_per_dimension(predictions, targets)

    ok_records = [record for record in records if record.get("status") == "ok"]
    latencies = [
        float(record["latency_seconds"])
        for record in ok_records
        if record.get("latency_seconds") is not None
    ]
    input_tokens = 0
    output_tokens = 0
    for record in ok_records:
        usage = record.get("usage") or {}
        input_tokens += int(
            usage.get("input_tokens") or record["estimated_input_tokens"]
        )
        output_tokens += int(usage.get("output_tokens") or 0)

    model = str(records[0].get("model", "")) if records else ""
    context_arm = (
        str(records[0].get("context_arm", "student_visible")) if records else ""
    )
    return {
        "model": model,
        "context_arm": context_arm,
        "reasoning_effort": records[0].get("reasoning_effort") if records else None,
        "shots": records[0].get("shots") if records else None,
        "n_rows": len(records),
        "n_ok": len(ok_records),
        "n_error": sum(1 for record in records if record.get("status") == "error"),
        "qwk_per_dim": qwk_per_dim,
        "qwk_mean": _nanmean_or_nan(list(qwk_per_dim.values())),
        "qwk_nan_dims_count": compute_qwk_nan_dims_count(qwk_per_dim),
        "mae_per_dim": mae,
        "mae_mean": float(np.mean(list(mae.values()))),
        "accuracy_per_dim": accuracy,
        "accuracy_mean": float(np.mean(list(accuracy.values()))),
        "recall_per_class": recall,
        "recall_minus1": recall["mean"]["minus1"],
        "recall_zero": recall["mean"]["zero"],
        "recall_plus1": recall["mean"]["plus1"],
        "minority_recall_mean": _nanmean_or_nan(
            [recall["mean"]["minus1"], recall["mean"]["plus1"]]
        ),
        "hedging_per_dim": hedging,
        "hedging_mean": float(np.mean(list(hedging.values()))),
        "latency_seconds": {
            "p50": float(np.percentile(latencies, 50)) if latencies else None,
            "p90": float(np.percentile(latencies, 90)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "max": float(max(latencies)) if latencies else None,
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        },
    }


def metrics_from_run_020() -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / RUN_020_PATH).read_text(encoding="utf-8"))
    return {
        "qwk_mean": payload["evaluation"]["qwk_mean"],
        "recall_minus1": payload["evaluation"]["recall_minus1"],
        "minority_recall_mean": payload["evaluation"]["minority_recall_mean"],
        "hedging_mean": payload["evaluation"]["hedging_mean"],
        "qwk_per_dim": {
            dimension: payload["per_dimension"][dimension]["qwk"]
            for dimension in SCHWARTZ_VALUE_ORDER
        },
    }


def human_fleiss_from_report(path: Path = AGREEMENT_REPORT_PATH) -> dict[str, float]:
    text = (ROOT / path).read_text(encoding="utf-8")
    mapping = {
        "Self Direction": "self_direction",
        "Stimulation": "stimulation",
        "Hedonism": "hedonism",
        "Achievement": "achievement",
        "Power": "power",
        "Security": "security",
        "Conformity": "conformity",
        "Tradition": "tradition",
        "Benevolence": "benevolence",
        "Universalism": "universalism",
    }
    values: dict[str, float] = {}
    in_section = False
    for line in text.splitlines():
        if line.startswith("## Fleiss"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("|"):
            continue
        parts = [part.strip().strip("*") for part in line.strip().strip("|").split("|")]
        if len(parts) < 2 or parts[0] not in mapping:
            continue
        values[mapping[parts[0]]] = float(parts[1])
    return values


def write_metrics_outputs(
    *,
    results_path: Path,
    metrics_path: Path,
    long_path: Path,
) -> dict[str, Any]:
    records = read_jsonl(results_path)
    metrics = score_records(records)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    long_rows = []
    for record in records:
        if record.get("status") != "ok":
            continue
        for dimension in SCHWARTZ_VALUE_ORDER:
            long_rows.append(
                {
                    "entry_id": record["entry_id"],
                    "persona_id": record["persona_id"],
                    "t_index": int(record["t_index"]),
                    "date": record["date"],
                    "split": record["split"],
                    "model": record["model"],
                    "context_arm": record.get("context_arm", "student_visible"),
                    "reasoning_effort": record["reasoning_effort"],
                    "shots": int(record["shots"]),
                    "dimension": dimension,
                    "target": int(record["target"][dimension]),
                    "predicted_class": int(record["scores"][dimension]),
                }
            )
    pl.DataFrame(long_rows).write_parquet(long_path)
    return metrics


def render_report(metrics_files: list[Path]) -> str:
    run_020 = metrics_from_run_020()
    human = human_fleiss_from_report()
    metrics_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in metrics_files
    ]

    lines = [
        "# twinkl-w2mu LLM Critic Baseline",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Contract",
        "",
        "- `student_visible`: current journal session plus normalized 10-dim "
        "Core Values profile.",
        "- `human_context`: `student_visible` plus previous entries for the same "
        "persona where previous.t_index < current.t_index.",
        "- `full_judge_context`: `human_context` plus persona bio and demographics; "
        "upper-bound diagnostic only.",
        "- Always excluded: future entries, target labels, rationales, and "
        "generation metadata.",
        "- Output: structured JSON scores in {-1, 0, +1} for all 10 values.",
        "",
        "## Summary",
        "",
        "| Arm | Model | Effort | Shots | Rows | QWK | recall_-1 | MinR | Hedging | "
        "p95 latency | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for payload in metrics_payloads:
        p95 = payload["latency_seconds"]["p95"]
        p95_text = "N/A" if p95 is None else f"{p95:.3f}s"
        lines.append(
            "| {arm} | {model} | {effort} | {shots} | {rows} | {qwk:.3f} | "
            "{recall:.3f} | "
            "{minr:.3f} | {hedging:.3f} | {latency} | ${cost:.4f} |".format(
                arm=payload.get("context_arm", "student_visible"),
                model=payload["model"],
                effort=payload["reasoning_effort"],
                shots=payload["shots"],
                rows=payload["n_ok"],
                qwk=payload["qwk_mean"],
                recall=payload["recall_minus1"],
                minr=payload["minority_recall_mean"],
                hedging=payload["hedging_mean"],
                latency=p95_text,
                cost=payload["usage"]["estimated_cost_usd"],
            )
        )
    lines.extend(
        [
            "| current_vif | run_020_BalancedSoftmax | n/a | n/a | 221 | "
            f"{run_020['qwk_mean']:.3f} | {run_020['recall_minus1']:.3f} | "
            f"{run_020['minority_recall_mean']:.3f} | {run_020['hedging_mean']:.3f} | "
            "n/a | n/a |",
            "",
            "## Context Gap",
            "",
            "| Model | Effort | Shots | Metric | "
            "human_context - student_visible | Interpretation |",
            "|---|---:|---:|---|---:|---|",
        ]
    )

    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for payload in metrics_payloads:
        key = (
            payload["model"],
            str(payload["reasoning_effort"]),
            int(payload["shots"]),
        )
        grouped.setdefault(key, {})[
            payload.get("context_arm", "student_visible")
        ] = payload

    for (model, effort, shots), by_arm in grouped.items():
        student = by_arm.get("student_visible")
        human_context = by_arm.get("human_context")
        if not student or not human_context:
            continue
        for metric in ("qwk_mean", "recall_minus1", "minority_recall_mean"):
            gap = human_context[metric] - student[metric]
            if gap > 0.05:
                interpretation = "history likely helps"
            elif gap < -0.05:
                interpretation = "history did not help this arm"
            else:
                interpretation = "small context gap"
            lines.append(
                f"| {model} | {effort} | {shots} | {metric} | {gap:.3f} | "
                f"{interpretation} |"
            )

    def dimension_verdict(dimension: str) -> str:
        run_value = run_020["qwk_per_dim"][dimension]
        scored_payloads = [
            payload
            for payload in metrics_payloads
            if payload["qwk_per_dim"].get(dimension) is not None
        ]
        if not scored_payloads:
            return "No LLM score"

        best = max(
            scored_payloads,
            key=lambda payload: payload["qwk_per_dim"][dimension],
        )
        best_value = best["qwk_per_dim"][dimension]
        best_label = f"{best.get('context_arm', 'student_visible')}:{best['model']}"

        student_values = [
            payload["qwk_per_dim"][dimension]
            for payload in scored_payloads
            if payload.get("context_arm", "student_visible") == "student_visible"
        ]
        history_values = [
            payload["qwk_per_dim"][dimension]
            for payload in scored_payloads
            if payload.get("context_arm") == "human_context"
        ]

        if best_value > run_value + 0.05:
            main = "LLM stronger"
        elif run_value > best_value + 0.05:
            main = "run_020 stronger"
        else:
            main = "near tie"

        if student_values and history_values:
            history_gap = max(history_values) - max(student_values)
            if history_gap > 0.05:
                context = "history helps"
            elif history_gap < -0.05:
                context = "history hurts"
            else:
                context = "small history gap"
            return f"{main}; {context}; best {best_label}"

        return f"{main}; best {best_label}"

    lines.extend(
        [
            "",
            "## Per-Dimension QWK",
            "",
            "| Dimension | Human Fleiss k | run_020 | "
            + " | ".join(
                f"{payload.get('context_arm', 'student_visible')}:{payload['model']}"
                for payload in metrics_payloads
            )
            + " | Verdict |",
            "|---|---:|---:|"
            + "---:|" * len(metrics_payloads)
            + "---|",
        ]
    )
    for dimension in SCHWARTZ_VALUE_ORDER:
        model_cells = [
            f"{payload['qwk_per_dim'][dimension]:.3f}"
            if payload["qwk_per_dim"][dimension] is not None
            else "N/A"
            for payload in metrics_payloads
        ]
        lines.append(
            "| {dimension} | {human:.3f} | {student:.3f} | {models} | "
            "{verdict} |".format(
                dimension=dimension,
                human=human.get(dimension, float("nan")),
                student=run_020["qwk_per_dim"][dimension],
                models=" | ".join(model_cells),
                verdict=dimension_verdict(dimension),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Verdicts are heuristic: compare per-dimension QWK against run_020 "
            "and check whether history improves the best LLM arm.",
            "- Treat large-model comparisons, if any, as oracle diagnostics only.",
            "",
        ]
    )
    return "\n".join(lines)


def command_estimate(args: argparse.Namespace) -> None:
    splits = load_split_frames(
        labels_path=args.labels_path,
        wrangled_dir=args.wrangled_dir,
        holdout_manifest=args.holdout_manifest,
    )
    ensure_frozen_test_size(splits)
    persona_profiles = load_persona_profiles(args.wrangled_dir)
    train_rows = rows_from_frame(
        splits["train"],
        "train",
        persona_profiles=persona_profiles,
    )
    target_rows = select_rows(
        rows_from_frame(
            splits[args.split],
            args.split,
            persona_profiles=persona_profiles,
        ),
        limit=args.limit,
        seed=args.seed,
    )
    few_shot_rows = choose_few_shot_rows(train_rows, shots=args.shots, seed=args.seed)
    if len(args.context_arms) == 1:
        summary = summarize_estimates(
            target_rows,
            models=args.models,
            few_shot_rows=few_shot_rows,
            context_arm=args.context_arms[0],
        )
    else:
        summary = {
            "by_context_arm": {
                arm: summarize_estimates(
                    target_rows,
                    models=args.models,
                    few_shot_rows=few_shot_rows,
                    context_arm=arm,
                )
                for arm in args.context_arms
            }
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    if args.execute:
        load_dotenv(ROOT / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to "
                f"{ROOT / '.env'} before running with --execute."
            )

    splits = load_split_frames(
        labels_path=args.labels_path,
        wrangled_dir=args.wrangled_dir,
        holdout_manifest=args.holdout_manifest,
    )
    ensure_frozen_test_size(splits)
    persona_profiles = load_persona_profiles(args.wrangled_dir)
    train_rows = rows_from_frame(
        splits["train"],
        "train",
        persona_profiles=persona_profiles,
    )
    target_rows = select_rows(
        rows_from_frame(
            splits[args.split],
            args.split,
            persona_profiles=persona_profiles,
        ),
        limit=args.limit,
        seed=args.seed,
    )
    few_shot_rows = choose_few_shot_rows(train_rows, shots=args.shots, seed=args.seed)

    run_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "shots": args.shots,
        "execute": args.execute,
        "context_arms": args.context_arms,
        "models": args.models,
        "reasoning_efforts": args.reasoning_efforts,
        "max_attempts": args.max_attempts,
        "max_output_tokens": args.max_output_tokens,
        "n_rows": len(target_rows),
        "few_shot_entry_ids": [row.entry_id for row in few_shot_rows],
        "pricing_source": PRICING_SOURCE,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for context_arm in args.context_arms:
        for model in args.models:
            for effort in args.reasoning_efforts:
                safe_model = model.replace("/", "_")
                stem = (
                    f"{args.split}_{context_arm}_{safe_model}_{effort}_shots"
                    f"{args.shots}"
                )
                output_path = run_dir / f"{stem}.jsonl"
                if output_path.exists():
                    output_path.unlink()
                for ordinal, row in enumerate(target_rows, start=1):
                    records = run_model_rows(
                        [row],
                        context_arm=context_arm,
                        model=model,
                        reasoning_effort=effort,
                        few_shot_rows=few_shot_rows,
                        execute=args.execute,
                        timeout=args.timeout,
                        max_attempts=args.max_attempts,
                        max_output_tokens=args.max_output_tokens,
                        start_ordinal=ordinal,
                    )
                    append_jsonl(output_path, records)
                print(output_path)


def command_score(args: argparse.Namespace) -> None:
    for results_path in args.results:
        stem = results_path.with_suffix("").name
        metrics_path = results_path.parent / f"{stem}.metrics.json"
        long_path = results_path.parent / f"{stem}.long.parquet"
        metrics = write_metrics_outputs(
            results_path=results_path,
            metrics_path=metrics_path,
            long_path=long_path,
        )
        print(
            json.dumps(
                {
                    "results": str(results_path),
                    "metrics": str(metrics_path),
                    "long": str(long_path),
                    "context_arm": metrics["context_arm"],
                    "qwk_mean": metrics["qwk_mean"],
                    "recall_minus1": metrics["recall_minus1"],
                },
                indent=2,
                sort_keys=True,
            )
        )


def command_report(args: argparse.Namespace) -> None:
    report = render_report(args.metrics)
    output_path = args.output or (
        DEFAULT_REPORTS_DIR / f"experiment_review_{now_stamp()}_twinkl_w2mu.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
        target.add_argument("--wrangled-dir", type=Path, default=DEFAULT_WRANGLED_DIR)
        target.add_argument(
            "--holdout-manifest",
            type=Path,
            default=DEFAULT_HOLDOUT_MANIFEST,
        )

    def add_selection_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--split", choices=("val", "test"), default="test")
        target.add_argument("--limit", type=int, default=None)
        target.add_argument("--seed", type=int, default=2026)
        target.add_argument("--shots", type=int, default=0)
        target.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
        target.add_argument(
            "--context-arms",
            nargs="+",
            choices=CONTEXT_ARMS,
            default=list(DEFAULT_CONTEXT_ARMS),
        )

    estimate = subparsers.add_parser("estimate", help="Estimate token counts and cost.")
    add_data_args(estimate)
    add_selection_args(estimate)
    estimate.set_defaults(func=command_estimate)

    run = subparsers.add_parser("run", help="Dry-run or execute API calls.")
    add_data_args(run)
    add_selection_args(run)
    run.add_argument(
        "--reasoning-efforts",
        nargs="+",
        default=list(DEFAULT_REASONING_EFFORTS),
    )
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--max-output-tokens", type=int, default=1000)
    run.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the OpenAI API. Without this, writes dry-run records.",
    )
    run.set_defaults(func=command_run)

    score = subparsers.add_parser("score", help="Score one or more result JSONL files.")
    score.add_argument("results", nargs="+", type=Path)
    score.set_defaults(func=command_score)

    report = subparsers.add_parser("report", help="Write a markdown comparison report.")
    report.add_argument("metrics", nargs="+", type=Path)
    report.add_argument("--output", type=Path, default=None)
    report.set_defaults(func=command_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
