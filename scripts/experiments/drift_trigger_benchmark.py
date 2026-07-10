#!/usr/bin/env python3
"""Run the twinkl-wq9p decision-level sustained-conflict benchmark."""

from __future__ import annotations

# ruff: noqa: E402, E501
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.llm_critic_baseline import (
    ExperimentRow,
    core_values_to_profile_weights,
    read_jsonl,
    run_model_rows,
    write_jsonl,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift_benchmark import (
    build_detection_decisions,
    build_eligible_trajectories,
    build_reference_episodes,
    detect_sustained_conflict_episodes,
    episode_metrics,
    evidence_from_llm_records,
    evidence_from_ordinal_artifact,
    normalize_value_name,
    tune_detector_thresholds,
)
from src.vif.holdout import load_fixed_holdout_ids
from src.vif.runtime import load_runtime_bundle

DEFAULT_HOLDOUT = Path("config/evals/drift_v1_designed_holdout.yaml")
DEFAULT_HOLDOUT_MANIFEST = Path("config/experiments/vif/twinkl_681_5_holdout.yaml")
DEFAULT_CONSENSUS = Path("logs/judge_labels/consensus_labels.parquet")
DEFAULT_REGISTRY = Path("logs/registry/personas.parquet")
DEFAULT_OUTPUT_DIR = Path(
    "logs/experiments/artifacts/drift_trigger_benchmark_twinkl_wq9p_20260710"
)
DEFAULT_REPORT = Path(
    "logs/experiments/reports/experiment_review_2026-07-10_twinkl_wq9p.md"
)


@dataclass(frozen=True)
class MLPArm:
    arm_id: str
    run_yaml: Path
    artifact_kind: str = "selected"
    candidate_name: str | None = None
    score_designed_holdout: bool = True


MLP_ARMS = (
    MLPArm(
        "run_020_selected", Path("logs/experiments/runs/run_020_BalancedSoftmax.yaml")
    ),
    MLPArm(
        "run_025_same_budget_persisted",
        Path("logs/experiments/runs/run_025_BalancedSoftmax.yaml"),
        score_designed_holdout=False,
    ),
    MLPArm(
        "run_052_consensus_recall_0.02",
        Path("logs/experiments/runs/run_052_BalancedSoftmax.yaml"),
        artifact_kind="candidate",
        candidate_name="recall_qwk_window_0.02",
    ),
    MLPArm(
        "run_053_consensus_selected",
        Path("logs/experiments/runs/run_053_BalancedSoftmax.yaml"),
    ),
)

LLM_TEST_ARMS = {
    "llm_gpt-5.4-mini_student_visible": Path(
        "logs/experiments/artifacts/llm_critic_baseline/"
        "20260702_test_main_context_arms_none/"
        "test_student_visible_gpt-5.4-mini_none_shots0.jsonl"
    ),
    "llm_gpt-5.4-mini_human_context": Path(
        "logs/experiments/artifacts/llm_critic_baseline/"
        "20260702_test_main_context_arms_none/"
        "test_human_context_gpt-5.4-mini_none_shots0.jsonl"
    ),
}

PROBABILITY_GRID = tuple(round(value, 2) for value in np.arange(0.30, 0.81, 0.05))
TARGETS = {
    "recall": 0.80,
    "precision": 0.60,
    "f1": 0.50,
    "false_positive_rate": 0.20,
    "max_latency_entries": 2,
}


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(_rooted(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _artifact_paths(arm: MLPArm) -> tuple[Path, Path, Path]:
    run = _load_yaml(arm.run_yaml)
    artifacts = run["artifacts"]
    if arm.artifact_kind == "selected":
        return (
            _rooted(Path(artifacts["validation_outputs"])),
            _rooted(Path(artifacts["test_outputs"])),
            _rooted(Path(artifacts["checkpoint"])),
        )
    if arm.candidate_name is None:
        raise ValueError(f"candidate_name is required for {arm.arm_id}")
    return (
        _rooted(Path(artifacts["candidate_validation_outputs"][arm.candidate_name])),
        _rooted(Path(artifacts["candidate_test_outputs"][arm.candidate_name])),
        _rooted(Path(artifacts["candidate_checkpoints"][arm.candidate_name])),
    )


def _uncertainty_grid(evidence_df: pl.DataFrame) -> tuple[float, ...]:
    values = evidence_df["uncertainty"].drop_nulls()
    if values.is_empty():
        return (1.0,)
    quantiles = [
        float(values.quantile(quantile, interpolation="nearest"))
        for quantile in (0.25, 0.50, 0.75, 0.90, 1.0)
    ]
    return tuple(sorted(set(round(value, 6) for value in quantiles)))


def _per_dimension_metrics(
    reference_df: pl.DataFrame,
    predicted_df: pl.DataFrame,
    eligible_df: pl.DataFrame,
    decisions_df: pl.DataFrame,
) -> dict[str, dict[str, Any]]:
    result = {}
    for dimension in SCHWARTZ_VALUE_ORDER:
        eligible = eligible_df.filter(pl.col("dimension") == dimension)
        if eligible.is_empty():
            continue
        result[dimension] = episode_metrics(
            reference_df.filter(pl.col("dimension") == dimension),
            predicted_df.filter(pl.col("dimension") == dimension),
            eligible,
            decisions_df=decisions_df.filter(pl.col("dimension") == dimension),
        )
    return result


def _evaluate_soft_arm(
    *,
    arm_id: str,
    validation_evidence: pl.DataFrame,
    test_evidence: pl.DataFrame,
    validation_reference: pl.DataFrame,
    test_reference: pl.DataFrame,
    validation_eligible: pl.DataFrame,
    test_eligible: pl.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    thresholds, grid = tune_detector_thresholds(
        validation_evidence,
        validation_reference,
        validation_eligible,
        probability_thresholds=PROBABILITY_GRID,
        uncertainty_thresholds=_uncertainty_grid(validation_evidence),
    )
    predicted = detect_sustained_conflict_episodes(test_evidence, **thresholds)
    decisions = build_detection_decisions(test_evidence, **thresholds)
    metrics = episode_metrics(
        test_reference,
        predicted,
        test_eligible,
        decisions_df=decisions,
    )
    payload = {
        "arm_id": arm_id,
        "evidence_kind": "soft_probability",
        "thresholds": thresholds,
        "overall": metrics,
        "per_dimension": _per_dimension_metrics(
            test_reference, predicted, test_eligible, decisions
        ),
    }
    arm_dir = output_dir / "frozen" / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    grid.write_parquet(arm_dir / "validation_threshold_grid.parquet")
    test_evidence.write_parquet(arm_dir / "test_evidence.parquet")
    decisions.write_parquet(arm_dir / "test_decisions.parquet")
    predicted.write_parquet(arm_dir / "test_predicted_episodes.parquet")
    _write_json(arm_dir / "metrics.json", payload)
    return payload, thresholds


def _evaluate_hard_arm(
    *,
    arm_id: str,
    evidence: pl.DataFrame,
    reference: pl.DataFrame,
    eligible: pl.DataFrame,
    output_dir: Path,
    scope: str,
) -> dict[str, Any]:
    predicted = detect_sustained_conflict_episodes(
        evidence,
        probability_threshold=1.0,
        uncertainty_threshold=None,
    )
    decisions = build_detection_decisions(
        evidence,
        probability_threshold=1.0,
        uncertainty_threshold=None,
    )
    payload = {
        "arm_id": arm_id,
        "evidence_kind": "hard_class",
        "thresholds": {"probability_threshold": 1.0, "uncertainty_threshold": None},
        "overall": episode_metrics(
            reference, predicted, eligible, decisions_df=decisions
        ),
        "per_dimension": _per_dimension_metrics(
            reference, predicted, eligible, decisions
        ),
    }
    arm_dir = output_dir / scope / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    evidence.write_parquet(arm_dir / "evidence.parquet")
    decisions.write_parquet(arm_dir / "decisions.parquet")
    predicted.write_parquet(arm_dir / "predicted_episodes.parquet")
    _write_json(arm_dir / "metrics.json", payload)
    return payload


def load_designed_holdout(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], pl.DataFrame, pl.DataFrame]:
    payload = _load_yaml(path)
    if not payload.get("locked_before_scoring"):
        raise ValueError("Designed holdout must be locked before scoring")
    cases = payload.get("cases") or []
    if not cases:
        raise ValueError("Designed holdout contains no cases")

    profile_rows = []
    label_rows = []
    seen_personas = set()
    for case in cases:
        persona_id = str(case["persona_id"])
        if persona_id in seen_personas:
            raise ValueError(f"Duplicate designed persona_id: {persona_id}")
        seen_personas.add(persona_id)
        profile_rows.append(
            {
                "persona_id": persona_id,
                "core_values": list(case["core_values"]),
            }
        )
        for entry in case["entries"]:
            row = {
                "persona_id": persona_id,
                "t_index": int(entry["t_index"]),
                "date": str(entry["date"]),
            }
            core_labels = {
                normalize_value_name(str(key)): value
                for key, value in (entry.get("core_labels") or {}).items()
            }
            for dimension in SCHWARTZ_VALUE_ORDER:
                label = core_labels.get(dimension, 0)
                row[f"alignment_{dimension}"] = label
                row[f"confidence_{dimension}"] = (
                    "no_majority" if label is None else "designed"
                )
                row[f"consensus_agreement_{dimension}"] = 0.0 if label is None else 1.0
            label_rows.append(row)

    profiles = pl.DataFrame(profile_rows)
    labels = pl.DataFrame(
        label_rows,
        schema_overrides={
            f"alignment_{dimension}": pl.Int64 for dimension in SCHWARTZ_VALUE_ORDER
        },
    )
    fixture_hash = hashlib.sha256(_rooted(path).read_bytes()).hexdigest()
    payload["sha256"] = fixture_hash
    return payload, cases, profiles, labels


def _score_mlp_holdout(
    *,
    cases: list[dict[str, Any]],
    checkpoint_path: Path,
    arm_id: str,
    output_path: Path,
) -> pl.DataFrame:
    if output_path.exists():
        return pl.read_parquet(output_path)

    model, state_encoder, _config, _checkpoint, device = load_runtime_bundle(
        checkpoint_path
    )
    flattened: list[tuple[dict[str, Any], dict[str, Any]]] = []
    texts = []
    for case in cases:
        for entry in case["entries"]:
            flattened.append((case, entry))
            texts.append(str(entry["initial_entry"]))
    embeddings = state_encoder.text_encoder.encode_batch(texts)
    embedding_by_key = {
        (str(case["persona_id"]), int(entry["t_index"])): embeddings[index]
        for index, (case, entry) in enumerate(flattened)
    }

    states = []
    for case, entry in flattened:
        persona_id = str(case["persona_id"])
        t_index = int(entry["t_index"])
        case_entries = {int(row["t_index"]): row for row in case["entries"]}
        window_embeddings = []
        window_dates = []
        for offset in range(state_encoder.window_size):
            previous_index = t_index - offset
            if previous_index in case_entries:
                window_embeddings.append(embedding_by_key[(persona_id, previous_index)])
                window_dates.append(str(case_entries[previous_index]["date"]))
            else:
                window_embeddings.append(
                    np.zeros(state_encoder.text_encoder.embedding_dim, dtype=np.float32)
                )
                window_dates.append(None)
        states.append(
            state_encoder.build_state_vector_from_embeddings(
                embeddings=window_embeddings,
                dates=window_dates,
                core_values=list(case["core_values"]),
            )
        )

    state_tensor = torch.from_numpy(np.stack(states).astype(np.float32)).to(device)
    with torch.no_grad():
        _means, uncertainties = model.predict_with_uncertainty(
            state_tensor, n_samples=50
        )
        if not hasattr(model, "predict_probabilities"):
            raise ValueError(f"{arm_id} is not an ordinal probability model")
        probabilities = model.predict_probabilities(state_tensor)
    probabilities_np = probabilities.detach().cpu().numpy()
    uncertainties_np = uncertainties.detach().cpu().numpy()
    predicted_classes = probabilities_np.argmax(axis=-1) - 1

    rows = []
    for row_index, (case, entry) in enumerate(flattened):
        core_dimensions = {
            normalize_value_name(str(value)) for value in case["core_values"]
        }
        for dim_index, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
            if dimension not in core_dimensions:
                continue
            rows.append(
                {
                    "source": arm_id,
                    "persona_id": str(case["persona_id"]),
                    "dimension": dimension,
                    "t_index": int(entry["t_index"]),
                    "date": str(entry["date"]),
                    "p_minus1": float(probabilities_np[row_index, dim_index, 0]),
                    "uncertainty": float(uncertainties_np[row_index, dim_index]),
                    "predicted_class": int(predicted_classes[row_index, dim_index]),
                    "evidence_kind": "soft_probability",
                }
            )
    result = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output_path)
    return result


def _holdout_experiment_rows(cases: list[dict[str, Any]]) -> list[ExperimentRow]:
    rows = []
    for case in cases:
        previous_entries: list[dict[str, str]] = []
        core_values = tuple(str(value) for value in case["core_values"])
        for entry in sorted(case["entries"], key=lambda row: int(row["t_index"])):
            labels = {
                normalize_value_name(str(key)): value
                for key, value in (entry.get("core_labels") or {}).items()
            }
            target_vector = tuple(
                int(labels.get(dimension) or 0) for dimension in SCHWARTZ_VALUE_ORDER
            )
            session_content = str(entry["initial_entry"])
            rows.append(
                ExperimentRow(
                    split="designed_holdout",
                    persona_id=str(case["persona_id"]),
                    persona_name=str(case.get("persona_name") or ""),
                    persona_age=str(case.get("persona_age") or ""),
                    persona_profession=str(case.get("persona_profession") or ""),
                    persona_culture=str(case.get("persona_culture") or ""),
                    persona_bio="",
                    t_index=int(entry["t_index"]),
                    date=str(entry["date"]),
                    session_content=session_content,
                    previous_entries=tuple(previous_entries),
                    core_values=core_values,
                    profile_weights=core_values_to_profile_weights(core_values),
                    target_vector=target_vector,
                )
            )
            previous_entries.append(
                {
                    "date": str(entry["date"]),
                    "t_index": str(entry["t_index"]),
                    "content": session_content,
                }
            )
    return rows


def _score_llm_holdout(
    *,
    cases: list[dict[str, Any]],
    arm_id: str,
    context_arm: str,
    output_path: Path,
    execute: bool,
) -> pl.DataFrame | None:
    if output_path.exists():
        records = read_jsonl(output_path)
    elif not execute:
        return None
    else:
        load_dotenv(ROOT / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is required for designed holdout scoring"
            )
        records = run_model_rows(
            _holdout_experiment_rows(cases),
            context_arm=context_arm,
            model="gpt-5.4-mini",
            reasoning_effort="none",
            few_shot_rows=[],
            execute=True,
            timeout=60.0,
            max_attempts=2,
            max_output_tokens=1000,
        )
        write_jsonl(output_path, records)
    if any(record.get("status") != "ok" for record in records):
        raise RuntimeError(f"{arm_id} designed holdout contains failed LLM rows")
    return evidence_from_llm_records(records, source=arm_id)


def _passes_targets(metrics: dict[str, Any]) -> bool:
    false_positive_rate = metrics.get("false_positive_rate")
    max_latency = metrics.get("max_latency_entries")
    return (
        metrics["recall"] >= TARGETS["recall"]
        and metrics["precision"] > TARGETS["precision"]
        and metrics["f1"] > TARGETS["f1"]
        and false_positive_rate is not None
        and false_positive_rate < TARGETS["false_positive_rate"]
        and max_latency is not None
        and max_latency <= TARGETS["max_latency_entries"]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_report(
    *,
    fixture: dict[str, Any],
    frozen_results: list[dict[str, Any]],
    holdout_results: list[dict[str, Any]],
) -> str:
    lines = [
        "# Experiment Review — 2026-07-10 — `twinkl-wq9p` Drift-Trigger Benchmark",
        "",
        "## Decision",
        "",
    ]
    passing = [row for row in holdout_results if _passes_targets(row["overall"])]
    incumbent = next(
        (row for row in passing if row["arm_id"] == "run_020_selected"), None
    )
    consensus = next((row for row in passing if "consensus" in row["arm_id"]), None)
    llm = next((row for row in passing if row["arm_id"].startswith("llm_")), None)
    if incumbent:
        decision = (
            "The current run_020 MLP clears the designed POC gate. Keep the local MLP "
            "as the trigger scorer; do not add an LLM cascade unless later human review "
            "reverses this result."
        )
    elif consensus:
        decision = (
            "A consensus-trained MLP clears the designed POC gate while run_020 does "
            "not. Treat target regime as the active lever before production wiring."
        )
    elif llm:
        frozen_llm = next(
            (row for row in frozen_results if row["arm_id"] == llm["arm_id"]),
            None,
        )
        if frozen_llm and frozen_llm["overall"]["recall"] < TARGETS["recall"]:
            decision = (
                "No scorer is promotion-ready. The LLM arms clear the deliberately "
                "explicit designed holdout but miss the consensus-derived frozen "
                "episodes, while every MLP arm misses most designed episodes. Human "
                "review of the cross-set disagreement must precede production wiring "
                "or a cascade decision."
            )
        else:
            decision = (
                "Only an LLM arm clears both available decision checks. Do not replace "
                "the local MLP automatically; use this as evidence for a bounded "
                "verifier experiment."
            )
    else:
        decision = (
            "No evaluated arm clears the designed POC gate. Do not promote a production "
            "drift trigger yet; use the benchmark failures to choose the next target or "
            "context repair."
        )
    lines.extend([decision, "", "## Scope and evidence", ""])
    lines.extend(
        [
            "- Strict reference: two adjacent stored consensus `-1` labels on the same declared core value.",
            "- Soft detector: the mean `P(-1)` over an adjacent pair plus a maximum uncertainty gate.",
            "- Thresholds were selected on frozen validation personas only.",
            "- Frozen test results are diagnostic because that split has only five strict episodes.",
            f"- Designed holdout: `{fixture['holdout_id']}`, SHA-256 `{fixture['sha256']}`.",
            f"- Designed holdout review status: `{fixture['review_status']}`; it is not human ground truth.",
            "",
            "## Frozen test comparison",
            "",
            "| Arm | Evidence | Ref | Pred | Precision | Recall | F1 | Window FPR | Max latency | Recovery |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in frozen_results:
        metrics = result["overall"]
        lines.append(
            "| {arm} | {kind} | {ref} | {pred} | {precision} | {recall} | "
            "{f1} | {fpr} | {latency} | {recovery} |".format(
                arm=result["arm_id"],
                kind=result["evidence_kind"],
                ref=metrics["reference_episodes"],
                pred=metrics["predicted_episodes"],
                precision=_fmt(metrics["precision"]),
                recall=_fmt(metrics["recall"]),
                f1=_fmt(metrics["f1"]),
                fpr=_fmt(metrics["false_positive_rate"]),
                latency=_fmt(metrics["max_latency_entries"]),
                recovery=_fmt(metrics["recovery_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "## Isolated designed holdout",
            "",
            "POC targets: recall `>= 0.80`, precision `> 0.60`, F1 `> 0.50`, "
            "window false-positive rate `< 0.20`, and maximum confirmation-anchored latency `<= 2` entries.",
            "",
            "| Arm | Evidence | Ref | Pred | Precision | Recall | F1 | Window FPR | Max latency | Recovery | Pass |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for result in holdout_results:
        metrics = result["overall"]
        lines.append(
            "| {arm} | {kind} | {ref} | {pred} | {precision} | {recall} | "
            "{f1} | {fpr} | {latency} | {recovery} | {passed} |".format(
                arm=result["arm_id"],
                kind=result["evidence_kind"],
                ref=metrics["reference_episodes"],
                pred=metrics["predicted_episodes"],
                precision=_fmt(metrics["precision"]),
                recall=_fmt(metrics["recall"]),
                f1=_fmt(metrics["f1"]),
                fpr=_fmt(metrics["false_positive_rate"]),
                latency=_fmt(metrics["max_latency_entries"]),
                recovery=_fmt(metrics["recovery_accuracy"]),
                passed="yes" if _passes_targets(metrics) else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Cross-set interpretation",
            "",
            "The designed holdout and frozen consensus split disagree sharply. Both LLM arms detect all 10 deliberately explicit designed episodes with no false alarms, but neither detects any of the five consensus-derived frozen episodes. This is not evidence that the LLM is generally perfect; it is evidence that observable, explicit sustained conflict is within its capability while the consensus-derived cases may depend on subtler context, disputed labels, or a different target contract.",
            "",
            "The MLP conclusion is less ambiguous: run_020 detects 1/10 designed episodes, and the two consensus-trained variants detect only 2/10. Hard-consensus retraining therefore does not close the decision-level recall gap.",
            "",
            "Architecture consequence: keep the production trigger blocked. Human-review the five frozen reference episodes alongside the designed cases before deciding whether to repair labels/context, test an LLM verifier, or narrow the capstone claim to explicit conflict detection.",
            "",
            "## Promotion regime",
            "",
            "1. Primary: episode precision/recall/F1, window false-positive rate, and confirmation-anchored latency on an isolated holdout.",
            "2. Supporting: entry-level `recall_-1` at a declared precision floor.",
            "3. Diagnostic: QWK, hedging, calibration, and per-value slices.",
            "4. Production trigger schema and Coach delivery integration remain in `twinkl-a2w`.",
            "",
            "## Limitations",
            "",
            "- The designed holdout is intentionally small and author-designed; human review is the next validity upgrade.",
            "- LLM arms emit hard classes, not calibrated conflict probabilities, so their detector uses the strict two-label rule without uncertainty gating.",
            "- Per-value results are descriptive only; the benchmark does not fit per-value thresholds.",
            "- The consensus reference is more stable than one-pass labels but remains an AI-Judge reference rather than human ground truth.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    output_dir = _rooted(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _rooted(args.report)

    consensus = pl.read_parquet(_rooted(args.consensus_path))
    registry = pl.read_parquet(_rooted(args.registry_path))
    validation_ids, test_ids = load_fixed_holdout_ids(_rooted(args.holdout_manifest))
    validation_labels = consensus.filter(pl.col("persona_id").is_in(validation_ids))
    test_labels = consensus.filter(pl.col("persona_id").is_in(test_ids))
    all_reference = build_reference_episodes(consensus, registry)
    validation_reference = build_reference_episodes(validation_labels, registry)
    test_reference = build_reference_episodes(test_labels, registry)
    validation_eligible = build_eligible_trajectories(validation_labels, registry)
    test_eligible = build_eligible_trajectories(test_labels, registry)
    all_reference.write_parquet(output_dir / "reference_episodes_all.parquet")
    validation_reference.write_parquet(
        output_dir / "reference_episodes_validation.parquet"
    )
    test_reference.write_parquet(output_dir / "reference_episodes_test.parquet")

    frozen_results: list[dict[str, Any]] = []
    thresholds_by_arm: dict[str, dict[str, float]] = {}
    checkpoints_by_arm: dict[str, Path] = {}
    for arm in MLP_ARMS:
        validation_path, test_path, checkpoint_path = _artifact_paths(arm)
        checkpoints_by_arm[arm.arm_id] = checkpoint_path
        validation_evidence = evidence_from_ordinal_artifact(
            pl.read_parquet(validation_path), registry, source=arm.arm_id
        )
        test_evidence = evidence_from_ordinal_artifact(
            pl.read_parquet(test_path), registry, source=arm.arm_id
        )
        result, thresholds = _evaluate_soft_arm(
            arm_id=arm.arm_id,
            validation_evidence=validation_evidence,
            test_evidence=test_evidence,
            validation_reference=validation_reference,
            test_reference=test_reference,
            validation_eligible=validation_eligible,
            test_eligible=test_eligible,
            output_dir=output_dir,
        )
        frozen_results.append(result)
        thresholds_by_arm[arm.arm_id] = thresholds

    for arm_id, relative_path in LLM_TEST_ARMS.items():
        evidence = evidence_from_llm_records(
            read_jsonl(_rooted(relative_path)), source=arm_id
        )
        frozen_results.append(
            _evaluate_hard_arm(
                arm_id=arm_id,
                evidence=evidence,
                reference=test_reference,
                eligible=test_eligible,
                output_dir=output_dir,
                scope="frozen",
            )
        )

    fixture, cases, holdout_profiles, holdout_labels = load_designed_holdout(
        args.designed_holdout
    )
    overlap = set(holdout_profiles["persona_id"].to_list()) & set(
        registry["persona_id"].to_list()
    )
    if overlap:
        raise ValueError(
            f"Designed holdout overlaps existing personas: {sorted(overlap)}"
        )
    holdout_reference = build_reference_episodes(
        holdout_labels, holdout_profiles, source="designed_holdout"
    )
    holdout_eligible = build_eligible_trajectories(holdout_labels, holdout_profiles)
    if holdout_reference.height < 10:
        raise ValueError("Designed holdout must contain at least 10 reference episodes")
    holdout_dir = output_dir / "designed_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_reference.write_parquet(holdout_dir / "reference_episodes.parquet")
    holdout_eligible.write_parquet(holdout_dir / "eligible_trajectories.parquet")
    _write_json(holdout_dir / "fixture_manifest.json", fixture)

    holdout_results: list[dict[str, Any]] = []
    for arm in MLP_ARMS:
        if not arm.score_designed_holdout:
            continue
        evidence = _score_mlp_holdout(
            cases=cases,
            checkpoint_path=checkpoints_by_arm[arm.arm_id],
            arm_id=arm.arm_id,
            output_path=holdout_dir / arm.arm_id / "evidence.parquet",
        )
        thresholds = thresholds_by_arm[arm.arm_id]
        predicted = detect_sustained_conflict_episodes(evidence, **thresholds)
        decisions = build_detection_decisions(evidence, **thresholds)
        payload = {
            "arm_id": arm.arm_id,
            "evidence_kind": "soft_probability",
            "thresholds": thresholds,
            "overall": episode_metrics(
                holdout_reference,
                predicted,
                holdout_eligible,
                decisions_df=decisions,
            ),
            "per_dimension": _per_dimension_metrics(
                holdout_reference, predicted, holdout_eligible, decisions
            ),
        }
        arm_dir = holdout_dir / arm.arm_id
        decisions.write_parquet(arm_dir / "decisions.parquet")
        predicted.write_parquet(arm_dir / "predicted_episodes.parquet")
        _write_json(arm_dir / "metrics.json", payload)
        holdout_results.append(payload)

    for context_arm in ("student_visible", "human_context"):
        arm_id = f"llm_gpt-5.4-mini_{context_arm}"
        jsonl_path = holdout_dir / arm_id / "responses.jsonl"
        evidence = _score_llm_holdout(
            cases=cases,
            arm_id=arm_id,
            context_arm=context_arm,
            output_path=jsonl_path,
            execute=args.execute_llm,
        )
        if evidence is None:
            continue
        holdout_results.append(
            _evaluate_hard_arm(
                arm_id=arm_id,
                evidence=evidence,
                reference=holdout_reference,
                eligible=holdout_eligible,
                output_dir=output_dir,
                scope="designed_holdout",
            )
        )

    manifest = {
        "issue_id": "twinkl-wq9p",
        "reference_episode_count_all": all_reference.height,
        "reference_episode_count_validation": validation_reference.height,
        "reference_episode_count_test": test_reference.height,
        "designed_holdout_sha256": fixture["sha256"],
        "designed_reference_episode_count": holdout_reference.height,
        "thresholds_by_arm": thresholds_by_arm,
        "frozen_arms": [row["arm_id"] for row in frozen_results],
        "designed_holdout_arms": [row["arm_id"] for row in holdout_results],
        "targets": TARGETS,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "frozen_summary.json", frozen_results)
    _write_json(output_dir / "designed_holdout_summary.json", holdout_results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            fixture=fixture,
            frozen_results=frozen_results,
            holdout_results=holdout_results,
        ),
        encoding="utf-8",
    )
    print(report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-path", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST
    )
    parser.add_argument("--designed-holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--execute-llm",
        action="store_true",
        help="Call the LLM arms for the locked designed holdout.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
