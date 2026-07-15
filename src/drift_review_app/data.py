"""Load and verify frozen Weekly Drift Reviewer research inputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl
import yaml

from src.drift_rules import drift_spans, match_episodes, trajectory_covered
from src.wrangling.parse_wrangled_data import parse_wrangled_file

MODEL_DIR = Path("logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714")
LOW_DIR = Path("logs/experiments/artifacts/twinkl_52zz_luna_low_20260714")
REFERENCE_DIR = Path(
    "logs/experiments/artifacts/"
    "twinkl_qtwz_complete_development_review_20260714/results"
)

EXPECTED_COUNTS = {
    "personas": 204,
    "cases": 292,
    "entries": 1651,
    "entry_value_cells": 2377,
    "resolved_entry_value_cells": 2375,
    "uncertain_entry_value_cells": 2,
    "persona_weeks": 951,
    "reference_drifts": 42,
    "drift_trajectories": 36,
}

VALUE_LABELS = {
    "self_direction": "Self-Direction",
    "stimulation": "Stimulation",
    "hedonism": "Hedonism",
    "achievement": "Achievement",
    "power": "Power",
    "security": "Security",
    "conformity": "Conformity",
    "tradition": "Tradition",
    "benevolence": "Benevolence",
    "universalism": "Universalism",
}

QUEUE_LABELS = (
    "All persona/Core Value cases",
    "Reference Drift",
    "Uncertain LLM-Judge Conflict Label",
    "Missed Drift",
    "False Drift alert",
    "Model disagreement",
    "Run disagreement",
    "Unresolved because of Abstain",
    "Invalid response",
)


@dataclass(frozen=True)
class SetupSpec:
    """One frozen Weekly Drift Reviewer setup."""

    key: str
    label: str
    short_label: str
    model: str
    reasoning_effort: str
    response_path: Path
    expected_valid: int
    expected_invalid: int
    aggregate_key: str
    aggregate_path: Path


SETUP_SPECS = {
    "mini_none": SetupSpec(
        key="mini_none",
        label="GPT-5.4 mini — reasoning none",
        short_label="Mini none",
        model="gpt-5.4-mini-2026-03-17",
        reasoning_effort="none",
        response_path=MODEL_DIR / "responses_gpt_5_4_mini.jsonl",
        expected_valid=2815,
        expected_invalid=38,
        aggregate_key="gpt_5_4_mini",
        aggregate_path=MODEL_DIR / "metrics.json",
    ),
    "luna_none": SetupSpec(
        key="luna_none",
        label="Luna — reasoning none",
        short_label="Luna none",
        model="gpt-5.6-luna",
        reasoning_effort="none",
        response_path=MODEL_DIR / "responses_gpt_5_6_luna.jsonl",
        expected_valid=2837,
        expected_invalid=16,
        aggregate_key="gpt_5_6_luna",
        aggregate_path=MODEL_DIR / "metrics.json",
    ),
    "luna_low": SetupSpec(
        key="luna_low",
        label="Luna — reasoning low",
        short_label="Luna low",
        model="gpt-5.6-luna",
        reasoning_effort="low",
        response_path=LOW_DIR / "responses_gpt_5_6_luna_low.jsonl",
        expected_valid=2845,
        expected_invalid=8,
        aggregate_key="luna_low",
        aggregate_path=LOW_DIR / "metrics.json",
    ),
}


@dataclass(frozen=True)
class EntryRecord:
    """One Journal Entry/Core Value reference row plus display text."""

    case_id: str
    persona_id: str
    dimension: str
    position: int
    t_index: int
    date: str
    initial_entry: str
    nudge_text: str | None
    response_text: str | None
    runtime_text: str
    final_conflict: bool | None
    resolution_method: str
    resolution_status: str
    opus_resolved: bool


@dataclass(frozen=True)
class CaseRecord:
    """One evaluated persona/Core Value trajectory."""

    case_id: str
    persona_id: str
    dimension: str
    historical_split: str
    cohort_source: str
    cohort_role: str
    analysis_role: str
    entries: tuple[EntryRecord, ...]


@dataclass(frozen=True)
class Decision:
    """One Weekly Drift Reviewer decision or fail-closed receipt state."""

    setup_key: str
    run: int
    case_id: str
    persona_id: str
    dimension: str
    t_index: int
    week_start: str
    response_status: str
    verdict: str | None
    confidence: str | None
    reason_code: str | None
    evidence_quote: str
    validation_error: str | None

    @property
    def token(self) -> str:
        if self.response_status != "ok":
            return "invalid"
        return self.verdict or "missing"


@dataclass(frozen=True)
class DriftSpan:
    """One reference or predicted Drift span."""

    drift_id: str
    case_id: str
    persona_id: str
    dimension: str
    onset_t_index: int
    confirmation_t_index: int
    end_t_index: int
    onset_date: str
    confirmation_date: str
    end_date: str
    crosses_week: bool
    run: int | None = None
    detection_date: str | None = None
    result: str | None = None
    matched_reference_id: str | None = None
    delay_days: int | None = None
    delay_entries: int | None = None


@dataclass(frozen=True)
class PromptBoundary:
    """Exact Journal Entry boundary for one frozen weekly prompt."""

    persona_id: str
    week_start: str
    week_end: str
    review_at_date: str
    cutoff_t_index: int
    visible_t_indices: tuple[int, ...]
    current_t_indices: tuple[int, ...]
    declared_values: tuple[str, ...]
    prompt_sha256: str


@dataclass(frozen=True)
class ReviewData:
    """Verified in-memory view of the frozen comparison inputs."""

    profiles: dict[str, dict[str, Any]]
    cases: dict[str, CaseRecord]
    setup_specs: dict[str, SetupSpec]
    decisions: dict[tuple[str, int, str, int], Decision]
    receipt_statuses: dict[tuple[str, int, str, str], str]
    prompt_boundaries: dict[tuple[str, str], PromptBoundary]
    reference_drifts: dict[str, tuple[DriftSpan, ...]]
    predicted_drifts: dict[tuple[str, int, str], tuple[DriftSpan, ...]]
    case_metrics: dict[tuple[str, int, str], dict[str, Any]]
    aggregate_results: dict[str, tuple[dict[str, Any], ...]]
    integrity: dict[str, Any]

    def case(self, persona_id: str, dimension: str) -> CaseRecord:
        return self.cases[f"{persona_id}:{dimension}"]

    def cases_for_persona(self, persona_id: str) -> tuple[CaseRecord, ...]:
        return tuple(
            sorted(
                (case for case in self.cases.values() if case.persona_id == persona_id),
                key=lambda case: case.dimension,
            )
        )

    def decision(
        self, setup_key: str, run: int, case_id: str, t_index: int
    ) -> Decision:
        return self.decisions[(setup_key, run, case_id, t_index)]

    def boundaries_for_persona(self, persona_id: str) -> tuple[PromptBoundary, ...]:
        return tuple(
            sorted(
                (
                    boundary
                    for boundary in self.prompt_boundaries.values()
                    if boundary.persona_id == persona_id
                ),
                key=lambda boundary: boundary.week_start,
            )
        )

    def queue_case_ids(self, queue: str, setup_key: str) -> set[str]:
        if queue not in QUEUE_LABELS:
            raise ValueError(f"Unknown review queue: {queue}")
        if queue == "All persona/Core Value cases":
            return set(self.cases)
        if queue == "Reference Drift":
            return {case_id for case_id, rows in self.reference_drifts.items() if rows}
        if queue == "Uncertain LLM-Judge Conflict Label":
            return {
                case.case_id
                for case in self.cases.values()
                if any(entry.final_conflict is None for entry in case.entries)
            }
        if queue == "Model disagreement":
            return {
                case_id for case_id in self.cases if self.model_disagreement(case_id)
            }
        if queue == "Run disagreement":
            return {
                case_id
                for case_id in self.cases
                if self.run_disagreement(setup_key, case_id)
            }
        if queue == "Invalid response":
            return {
                case_id
                for case_id in self.cases
                if any(
                    self.decision(
                        setup_key, run, case_id, entry.t_index
                    ).response_status
                    != "ok"
                    for run in (1, 2, 3)
                    for entry in self.cases[case_id].entries
                )
            }
        if queue == "Unresolved because of Abstain":
            return {
                case_id
                for case_id in self.cases
                if any(
                    self.unresolved_pairs(setup_key, run, case_id) for run in (1, 2, 3)
                )
            }
        metric_key = {
            "Missed Drift": "missed_drifts",
            "False Drift alert": "false_drift_alerts",
        }[queue]
        return {
            case_id
            for case_id in self.cases
            if any(
                self.case_metrics[(setup_key, run, case_id)][metric_key] > 0
                for run in (1, 2, 3)
            )
        }

    def run_disagreement(self, setup_key: str, case_id: str) -> bool:
        case = self.cases[case_id]
        if any(
            len(
                {
                    self.decision(setup_key, run, case_id, entry.t_index).token
                    for run in (1, 2, 3)
                }
            )
            > 1
            for entry in case.entries
        ):
            return True
        run_spans = {
            tuple(
                (row.onset_t_index, row.end_t_index, row.result)
                for row in self.predicted_drifts[(setup_key, run, case_id)]
            )
            for run in (1, 2, 3)
        }
        return len(run_spans) > 1

    def model_disagreement(self, case_id: str) -> bool:
        case = self.cases[case_id]
        for entry in case.entries:
            groups = {
                preserved_group_signature(
                    [
                        self.decision(setup, run, case_id, entry.t_index).token
                        for run in (1, 2, 3)
                    ]
                )
                for setup in self.setup_specs
            }
            if len(groups) > 1:
                return True
        drift_groups = {
            tuple(
                sorted(
                    tuple(
                        (row.onset_t_index, row.end_t_index, row.result)
                        for row in self.predicted_drifts[(setup, run, case_id)]
                    )
                    for run in (1, 2, 3)
                )
            )
            for setup in self.setup_specs
        }
        return len(drift_groups) > 1

    def unresolved_pairs(
        self, setup_key: str, run: int, case_id: str
    ) -> tuple[tuple[int, int], ...]:
        case = self.cases[case_id]
        pairs = []
        for first, second in zip(case.entries, case.entries[1:], strict=False):
            if second.t_index != first.t_index + 1:
                continue
            decisions = (
                self.decision(setup_key, run, case_id, first.t_index),
                self.decision(setup_key, run, case_id, second.t_index),
            )
            verdicts = {decision.verdict for decision in decisions}
            if "abstain" in verdicts and "not_conflict" not in verdicts:
                pairs.append((first.t_index, second.t_index))
        return tuple(pairs)


def value_label(dimension: str) -> str:
    """Return the maintained display label for a Core Value."""
    return VALUE_LABELS.get(dimension, dimension.replace("_", " ").title())


def preserved_group_signature(values: list[str]) -> tuple[str, ...]:
    """Compare three preserved Run results without pairing or voting."""
    return tuple(sorted(values))


def _normalize_value(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _week_start(value: str) -> date:
    parsed = date.fromisoformat(value)
    return parsed - timedelta(days=parsed.weekday())


def _weekly_detection_date(value: str) -> str:
    return (_week_start(value) + timedelta(days=6)).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML object: {path}")
    return loaded


def _require_paths(root: Path) -> dict[str, Path]:
    paths = {
        "wrangled": root / "logs/wrangled",
        "prompts": root / MODEL_DIR / "prompts.jsonl",
        "base_manifest": root / MODEL_DIR / "manifest.json",
        "base_metrics": root / MODEL_DIR / "metrics.json",
        "low_manifest": root / LOW_DIR / "manifest.json",
        "low_metrics": root / LOW_DIR / "metrics.json",
        "base_config": root / "config/evals/twinkl_52zz_model_comparison_v1.yaml",
        "low_config": root / "config/evals/twinkl_52zz_luna_low_v1.yaml",
        "targets": root / REFERENCE_DIR / "complete_development_entry_target.parquet",
        "reference_drifts": root
        / REFERENCE_DIR
        / "complete_development_drift_episodes.parquet",
        "outcomes": root / REFERENCE_DIR / "complete_development_case_outcomes.parquet",
    }
    paths.update(
        {
            f"responses_{key}": root / spec.response_path
            for key, spec in SETUP_SPECS.items()
        }
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing review input(s): " + ", ".join(missing))
    return paths


def _validate_protocol_hashes(paths: dict[str, Path]) -> tuple[dict, dict, dict, dict]:
    base_config = _read_yaml(paths["base_config"])
    low_config = _read_yaml(paths["low_config"])
    base_manifest = _read_json(paths["base_manifest"])
    low_manifest = _read_json(paths["low_manifest"])
    base_metrics = _read_json(paths["base_metrics"])
    low_metrics = _read_json(paths["low_metrics"])

    checks = {
        "base config": (
            _sha256_text(_canonical_json(base_config)),
            str(base_manifest.get("config_sha256")),
        ),
        "low config": (
            _sha256_text(_canonical_json(low_config)),
            str(low_manifest.get("config_sha256")),
        ),
        "shared prompts": (
            _sha256_file(paths["prompts"]),
            str(base_manifest.get("prompts_sha256")),
        ),
        "base manifest": (
            _sha256_file(paths["base_manifest"]),
            str(base_metrics["provenance"]["manifest_sha256"]),
        ),
        "low manifest": (
            _sha256_file(paths["low_manifest"]),
            str(low_metrics["provenance"]["manifest_sha256"]),
        ),
    }
    frozen_baseline = low_manifest["frozen_baseline"]
    checks.update(
        {
            "low frozen shared prompts": (
                _sha256_file(paths["prompts"]),
                str(frozen_baseline["prompts_sha256"]),
            ),
            "low frozen base manifest": (
                _sha256_file(paths["base_manifest"]),
                str(frozen_baseline["manifest_sha256"]),
            ),
            "low frozen base metrics": (
                _sha256_file(paths["base_metrics"]),
                str(frozen_baseline["metrics_sha256"]),
            ),
            "low frozen Luna none responses": (
                _sha256_file(paths["responses_luna_none"]),
                str(frozen_baseline["luna_none_responses_sha256"]),
            ),
        }
    )
    source_keys = {
        "complete_entry_target_path": "targets",
        "complete_drift_episodes_path": "reference_drifts",
        "complete_case_outcomes_path": "outcomes",
    }
    for manifest_key, path_key in source_keys.items():
        checks[f"source {manifest_key}"] = (
            _sha256_file(paths[path_key]),
            str(base_manifest["source_sha256"][manifest_key]),
        )
    base_response_hashes = base_metrics["provenance"]["responses_sha256"]
    checks["mini responses"] = (
        _sha256_file(paths["responses_mini_none"]),
        str(base_response_hashes["gpt_5_4_mini"]),
    )
    checks["Luna none responses"] = (
        _sha256_file(paths["responses_luna_none"]),
        str(base_response_hashes["gpt_5_6_luna"]),
    )
    checks["Luna low responses"] = (
        _sha256_file(paths["responses_luna_low"]),
        str(low_metrics["provenance"]["luna_low_responses_sha256"]),
    )
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Frozen {label} hash mismatch: expected {expected}, got {actual}"
            )

    manifest_counts = {
        "personas": EXPECTED_COUNTS["personas"],
        "cases": EXPECTED_COUNTS["cases"],
        "entries": EXPECTED_COUNTS["entries"],
        "entry_value_cells": EXPECTED_COUNTS["entry_value_cells"],
        "resolved_entry_value_cells": EXPECTED_COUNTS["resolved_entry_value_cells"],
        "persona_weeks": EXPECTED_COUNTS["persona_weeks"],
        "drifts": EXPECTED_COUNTS["reference_drifts"],
        "drift_trajectories": EXPECTED_COUNTS["drift_trajectories"],
    }
    for label, manifest, study_id in (
        ("model comparison", base_manifest, "twinkl_52zz_model_comparison_v1"),
        ("Luna low", low_manifest, "twinkl_52zz_luna_low_v1"),
    ):
        identity = (
            manifest.get("study_id"),
            manifest.get("setup"),
            manifest.get("repeats"),
        )
        expected_identity = (study_id, "weekly_without_critic", 3)
        if identity != expected_identity:
            raise ValueError(
                f"Frozen {label} manifest identity mismatch: "
                f"{identity} != {expected_identity}"
            )
        if manifest.get("counts") != manifest_counts:
            raise ValueError(f"Frozen {label} manifest counts differ")

    expected_base_prompt_contract = {
        "development_labels_in_prompt": False,
        "fresh_final_test_inspected": False,
        "model_in_prompt": False,
        "store": False,
        "vif_critic_input": False,
    }
    expected_low_prompt_contract = {
        "development_labels_in_prompt": False,
        "fresh_final_test_inspected": False,
        "same_as_luna_none": True,
        "store": False,
        "vif_critic_input": False,
    }
    if base_manifest.get("prompt_contract") != expected_base_prompt_contract:
        raise ValueError("Frozen model comparison prompt contract differs")
    if low_manifest.get("prompt_contract") != expected_low_prompt_contract:
        raise ValueError("Frozen Luna low prompt contract differs")

    registered = {
        str(item["key"]): (str(item["model"]), str(item["reasoning_effort"]))
        for item in base_config["models"]
    }
    manifested = {
        str(item["key"]): (str(item["model"]), str(item["reasoning_effort"]))
        for item in base_manifest["models"]
    }
    for key, spec in SETUP_SPECS.items():
        source_key = {
            "mini_none": "gpt_5_4_mini",
            "luna_none": "gpt_5_6_luna",
        }.get(key)
        if source_key:
            expected_setup = (spec.model, spec.reasoning_effort)
            if (
                registered.get(source_key) != expected_setup
                or manifested.get(source_key) != expected_setup
            ):
                raise ValueError(f"Manifest/config identity mismatch for {spec.label}")
        else:
            expected_setup = (spec.model, spec.reasoning_effort)
            low_registered = (
                str(low_config["model"]["model"]),
                str(low_config["model"]["reasoning_effort"]),
            )
            low_manifested = (
                str(low_manifest["model"]["model"]),
                str(low_manifest["model"]["reasoning_effort"]),
            )
            if low_registered != expected_setup or low_manifested != expected_setup:
                raise ValueError(f"Manifest/config identity mismatch for {spec.label}")
    return base_config, base_manifest, base_metrics, low_metrics


def _displayed_entry_text(entry: dict[str, Any]) -> str:
    parts = []
    if entry.get("initial_entry"):
        parts.append(str(entry["initial_entry"]))
    if entry.get("nudge_text"):
        parts.append(f'Nudge: "{entry["nudge_text"]}"')
    if entry.get("response_text"):
        parts.append(f"Response: {entry['response_text']}")
    return "\n\n".join(parts)


def _load_profiles(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict]]]:
    profiles = {}
    entries = {}
    files = sorted((root / "logs/wrangled").glob("persona_*.md"))
    if len(files) != EXPECTED_COUNTS["personas"]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS['personas']} persona files, got {len(files)}"
        )
    for path in files:
        profile, parsed_entries, warnings = parse_wrangled_file(path)
        if warnings:
            messages = "; ".join(warning.message for warning in warnings)
            raise ValueError(
                f"Wrangled persona parse warning in {path.name}: {messages}"
            )
        persona_id = str(profile["persona_id"])
        profiles[persona_id] = profile
        entries[persona_id] = {int(entry["t_index"]): entry for entry in parsed_entries}
    return profiles, entries


def _load_cases(
    paths: dict[str, Path],
    profiles: dict[str, dict[str, Any]],
    wrangled_entries: dict[str, dict[int, dict]],
) -> tuple[
    dict[str, CaseRecord],
    dict[str, tuple[DriftSpan, ...]],
    pl.DataFrame,
    pl.DataFrame,
]:
    targets = pl.read_parquet(paths["targets"]).sort("canonical_case_id", "position")
    outcomes = pl.read_parquet(paths["outcomes"]).sort("canonical_case_id")
    reference_frame = pl.read_parquet(paths["reference_drifts"]).sort(
        "canonical_case_id", "onset_position"
    )
    if targets.height != EXPECTED_COUNTS["entry_value_cells"]:
        raise ValueError(f"Reference target count mismatch: {targets.height}")
    if outcomes.height != EXPECTED_COUNTS["cases"]:
        raise ValueError(f"Reference case count mismatch: {outcomes.height}")
    if reference_frame.height != EXPECTED_COUNTS["reference_drifts"]:
        raise ValueError(f"Reference Drift count mismatch: {reference_frame.height}")
    resolved = targets.filter(pl.col("final_conflict").is_not_null()).height
    uncertain = targets.height - resolved
    if (
        resolved != EXPECTED_COUNTS["resolved_entry_value_cells"]
        or uncertain != EXPECTED_COUNTS["uncertain_entry_value_cells"]
    ):
        raise ValueError(f"Reference resolution count mismatch: {resolved}/{uncertain}")

    targets_by_case = {
        str(key[0] if isinstance(key, tuple) else key): frame.to_dicts()
        for key, frame in targets.partition_by(
            "canonical_case_id", as_dict=True
        ).items()
    }
    cases = {}
    for outcome in outcomes.to_dicts():
        case_id = str(outcome["canonical_case_id"])
        persona_id = str(outcome["persona_id"])
        dimension = _normalize_value(str(outcome["dimension"]))
        if case_id != f"{persona_id}:{dimension}":
            raise ValueError(f"Unexpected case identifier: {case_id}")
        profile_values = {
            _normalize_value(value) for value in profiles[persona_id]["core_values"]
        }
        if dimension not in profile_values:
            raise ValueError(
                f"{value_label(dimension)} is not a Core Value for {persona_id}"
            )
        entry_rows = []
        for expected_position, target in enumerate(targets_by_case[case_id], start=1):
            if int(target["position"]) != expected_position:
                raise ValueError(f"Non-contiguous positions in {case_id}")
            t_index = int(target["t_index"])
            source = wrangled_entries[persona_id].get(t_index)
            if source is None:
                raise ValueError(f"Missing Journal Entry {persona_id}:{t_index}")
            runtime_text = _displayed_entry_text(source)
            if str(source["date"]) != str(target["date"]):
                raise ValueError(
                    f"Journal Entry date mismatch for {persona_id}:{t_index}"
                )
            if _sha256_text(runtime_text) != str(target["runtime_text_sha256"]):
                raise ValueError(
                    f"Journal Entry text hash mismatch for {persona_id}:{t_index}"
                )
            entry_rows.append(
                EntryRecord(
                    case_id=case_id,
                    persona_id=persona_id,
                    dimension=dimension,
                    position=expected_position,
                    t_index=t_index,
                    date=str(source["date"]),
                    initial_entry=str(source["initial_entry"]),
                    nudge_text=source.get("nudge_text"),
                    response_text=source.get("response_text"),
                    runtime_text=runtime_text,
                    final_conflict=target["final_conflict"],
                    resolution_method=str(target["resolution_method"]),
                    resolution_status=str(target["resolution_status"]),
                    opus_resolved=bool(target["opus_resolved"]),
                )
            )
        if len(entry_rows) != int(outcome["entry_count"]):
            raise ValueError(f"Journal Entry count mismatch in {case_id}")
        cases[case_id] = CaseRecord(
            case_id=case_id,
            persona_id=persona_id,
            dimension=dimension,
            historical_split=str(outcome["historical_split"]),
            cohort_source=str(outcome["cohort_source"]),
            cohort_role=str(outcome["cohort_role"]),
            analysis_role=str(outcome["analysis_role"]),
            entries=tuple(entry_rows),
        )

    unique_entries = {
        (entry.persona_id, entry.t_index)
        for case in cases.values()
        for entry in case.entries
    }
    if len(unique_entries) != EXPECTED_COUNTS["entries"]:
        raise ValueError(f"Unique Journal Entry count mismatch: {len(unique_entries)}")

    reference_drifts: dict[str, list[DriftSpan]] = {case_id: [] for case_id in cases}
    for row in reference_frame.to_dicts():
        case_id = str(row["canonical_case_id"])
        reference_drifts[case_id].append(
            DriftSpan(
                drift_id=str(row["episode_id"]),
                case_id=case_id,
                persona_id=str(row["persona_id"]),
                dimension=str(row["dimension"]),
                onset_t_index=int(row["onset_t_index"]),
                confirmation_t_index=int(row["confirmation_t_index"]),
                end_t_index=int(row["end_t_index"]),
                onset_date=str(row["onset_date"]),
                confirmation_date=str(row["confirmation_date"]),
                end_date=str(row["end_date"]),
                crosses_week=bool(row["crosses_week"]),
                result="reference",
            )
        )
    drift_trajectories = sum(bool(rows) for rows in reference_drifts.values())
    if drift_trajectories != EXPECTED_COUNTS["drift_trajectories"]:
        raise ValueError(
            f"Reference Drift trajectory count mismatch: {drift_trajectories}"
        )
    return (
        cases,
        {key: tuple(value) for key, value in reference_drifts.items()},
        targets,
        outcomes,
    )


def _load_prompts(
    paths: dict[str, Path], cases: dict[str, CaseRecord]
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, int], tuple[str, str]],
    dict[tuple[str, str], PromptBoundary],
]:
    records = _read_jsonl(paths["prompts"])
    if len(records) != EXPECTED_COUNTS["persona_weeks"]:
        raise ValueError(f"Prompt count mismatch: {len(records)}")
    prompt_map = {}
    coordinate_to_prompt = {}
    boundaries = {}
    case_coordinates = {
        (case.persona_id, case.dimension, entry.t_index)
        for case in cases.values()
        for entry in case.entries
    }
    entry_text = {
        (case.persona_id, entry.t_index): entry.runtime_text
        for case in cases.values()
        for entry in case.entries
    }
    entries_by_persona: dict[str, dict[int, EntryRecord]] = {}
    dimensions_by_persona: dict[str, set[str]] = {}
    for case in cases.values():
        entries_by_persona.setdefault(case.persona_id, {}).update(
            {entry.t_index: entry for entry in case.entries}
        )
        dimensions_by_persona.setdefault(case.persona_id, set()).add(case.dimension)
    for record in records:
        if _sha256_text(str(record["prompt"])) != str(record["prompt_sha256"]):
            raise ValueError(f"Prompt text hash mismatch: {record['review_event_id']}")
        if record.get("arm") != "weekly_without_critic":
            raise ValueError(
                f"Unexpected Weekly Drift Reviewer setup: {record.get('arm')}"
            )
        persona_id = str(record["persona_id"])
        week_start = str(record["week_start"])
        week_end = str(record["week_end"])
        key = (persona_id, week_start)
        if key in prompt_map:
            raise ValueError(f"Duplicate prompt key: {key}")
        prompt_map[key] = record
        visible_t_indices = tuple(
            sorted(int(value) for value in record["entry_text_by_t_index"])
        )
        current_t_indices = tuple(
            sorted(int(value) for value in record["current_t_indices"])
        )
        persona_entries = entries_by_persona[persona_id]
        expected_visible = tuple(
            sorted(
                t_index
                for t_index, entry in persona_entries.items()
                if entry.date <= week_end
            )
        )
        expected_current = tuple(
            sorted(
                t_index
                for t_index, entry in persona_entries.items()
                if week_start <= entry.date <= week_end
            )
        )
        if visible_t_indices != expected_visible:
            raise ValueError(f"Prompt history boundary mismatch: {key}")
        if current_t_indices != expected_current:
            raise ValueError(f"Prompt current-week boundary mismatch: {key}")
        if int(record["cutoff_t_index"]) != max(current_t_indices):
            raise ValueError(f"Prompt cutoff mismatch: {key}")
        declared_values = tuple(
            _normalize_value(str(value)) for value in record["declared_values"]
        )
        if set(declared_values) != dimensions_by_persona[persona_id]:
            raise ValueError(f"Prompt Core Values differ: {key}")
        for t_index in visible_t_indices:
            if (
                record["entry_text_by_t_index"][str(t_index)]
                != entry_text[(persona_id, t_index)]
            ):
                raise ValueError(f"Prompt Journal Entry history mismatch: {key}")
        if str(record["critic_block_sha256"]) != _sha256_text(""):
            raise ValueError(f"Prompt unexpectedly contains VIF Critic input: {key}")
        boundaries[key] = PromptBoundary(
            persona_id=persona_id,
            week_start=week_start,
            week_end=week_end,
            review_at_date=str(record["review_at_date"]),
            cutoff_t_index=int(record["cutoff_t_index"]),
            visible_t_indices=visible_t_indices,
            current_t_indices=current_t_indices,
            declared_values=declared_values,
            prompt_sha256=str(record["prompt_sha256"]),
        )
        expected_coordinates = {
            (
                str(record["persona_id"]),
                _normalize_value(str(item["dimension"])),
                int(item["t_index"]),
            )
            for item in record["expected_coordinates"]
        }
        if len(expected_coordinates) != len(record["expected_coordinates"]):
            raise ValueError(f"Duplicate expected prompt coordinate: {key}")
        if not expected_coordinates <= case_coordinates:
            raise ValueError(f"Unknown expected prompt coordinate: {key}")
        for coordinate in expected_coordinates:
            if coordinate in coordinate_to_prompt:
                raise ValueError(
                    f"Coordinate appears in multiple prompts: {coordinate}"
                )
            coordinate_to_prompt[coordinate] = key
            text = record["entry_text_by_t_index"].get(str(coordinate[2]))
            if text != entry_text[(coordinate[0], coordinate[2])]:
                raise ValueError(f"Prompt Journal Entry mismatch: {coordinate}")
    if set(coordinate_to_prompt) != case_coordinates:
        raise ValueError(
            "Prompt join is incomplete: "
            f"{len(coordinate_to_prompt)}/{len(case_coordinates)} coordinates"
        )
    return prompt_map, coordinate_to_prompt, boundaries


def _validate_assessments(
    response: dict[str, Any], record: dict[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    parsed = response.get("parsed")
    if not isinstance(parsed, dict) or not isinstance(parsed.get("assessments"), list):
        raise ValueError("Valid response is missing assessments")
    assessments = {}
    entry_text = record["entry_text_by_t_index"]
    valid_verdicts = {"conflict", "not_conflict", "abstain"}
    valid_confidences = {"low", "medium", "high"}
    for assessment in parsed["assessments"]:
        key = (
            _normalize_value(str(assessment["dimension"])),
            int(assessment["t_index"]),
        )
        if key in assessments:
            raise ValueError(f"Duplicate assessment coordinate: {key}")
        verdict = str(assessment["verdict"])
        confidence = str(assessment["confidence"])
        if verdict not in valid_verdicts or confidence not in valid_confidences:
            raise ValueError(f"Invalid assessment values: {key}")
        quote = str(assessment.get("evidence_quote") or "")
        if verdict == "conflict" and (
            not quote or quote not in entry_text[str(key[1])]
        ):
            raise ValueError(f"Conflict quote is not exact: {key}")
        assessments[key] = assessment
    expected = {
        (_normalize_value(str(item["dimension"])), int(item["t_index"]))
        for item in record["expected_coordinates"]
    }
    if set(assessments) != expected:
        raise ValueError(
            f"Assessment coordinate mismatch: {len(assessments)}/{len(expected)}"
        )
    return assessments


def _load_decisions(
    paths: dict[str, Path],
    cases: dict[str, CaseRecord],
    prompt_map: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    dict[tuple[str, int, str, int], Decision],
    dict[tuple[str, int, str, str], str],
    dict[str, dict[str, int]],
]:
    decisions = {}
    receipt_statuses = {}
    summaries = {}
    expected_receipt_keys = {
        (persona_id, week_start, run)
        for persona_id, week_start in prompt_map
        for run in (1, 2, 3)
    }
    for setup_key, spec in SETUP_SPECS.items():
        rows = _read_jsonl(paths[f"responses_{setup_key}"])
        if len(rows) != EXPECTED_COUNTS["persona_weeks"] * 3:
            raise ValueError(f"{spec.label} receipt count mismatch: {len(rows)}")
        observed_keys = set()
        status_counts: Counter[str] = Counter()
        run_counts: Counter[int] = Counter()
        for response in rows:
            key = (
                str(response["persona_id"]),
                str(response["week_start"]),
                int(response["repeat"]),
            )
            if key in observed_keys:
                raise ValueError(f"Duplicate {spec.label} receipt: {key}")
            observed_keys.add(key)
            prompt = prompt_map[(key[0], key[1])]
            if str(response["prompt_sha256"]) != str(prompt["prompt_sha256"]):
                raise ValueError(f"{spec.label} prompt receipt mismatch: {key}")
            if response.get("requested_model") != spec.model:
                raise ValueError(f"{spec.label} requested model mismatch: {key}")
            if response.get("resolved_model") != spec.model:
                raise ValueError(f"{spec.label} resolved model mismatch: {key}")
            status = str(response.get("status"))
            if status not in {"ok", "invalid"}:
                raise ValueError(f"Unexpected {spec.label} receipt status: {status}")
            status_counts[status] += 1
            run_counts[key[2]] += 1
            receipt_statuses[(setup_key, key[2], key[0], key[1])] = status
            assessments = (
                _validate_assessments(response, prompt) if status == "ok" else {}
            )
            for coordinate in prompt["expected_coordinates"]:
                dimension = _normalize_value(str(coordinate["dimension"]))
                t_index = int(coordinate["t_index"])
                case_id = f"{key[0]}:{dimension}"
                assessment = assessments.get((dimension, t_index))
                decision_key = (setup_key, key[2], case_id, t_index)
                if decision_key in decisions:
                    raise ValueError(f"Duplicate decision: {decision_key}")
                decisions[decision_key] = Decision(
                    setup_key=setup_key,
                    run=key[2],
                    case_id=case_id,
                    persona_id=key[0],
                    dimension=dimension,
                    t_index=t_index,
                    week_start=key[1],
                    response_status=status,
                    verdict=str(assessment["verdict"]) if assessment else None,
                    confidence=str(assessment["confidence"]) if assessment else None,
                    reason_code=str(assessment["reason_code"]) if assessment else None,
                    evidence_quote=str(assessment.get("evidence_quote") or "")
                    if assessment
                    else "",
                    validation_error=str(response.get("validation_error"))
                    if status != "ok"
                    else None,
                )
        if observed_keys != expected_receipt_keys:
            raise ValueError(
                f"{spec.label} receipt join mismatch: "
                f"{len(observed_keys)}/{len(expected_receipt_keys)}"
            )
        expected_statuses = {
            "ok": spec.expected_valid,
            "invalid": spec.expected_invalid,
        }
        if dict(status_counts) != expected_statuses:
            raise ValueError(
                f"{spec.label} response validity mismatch: "
                f"{dict(status_counts)} != {expected_statuses}"
            )
        if dict(run_counts) != {1: 951, 2: 951, 3: 951}:
            raise ValueError(
                f"{spec.label} Run receipt counts differ: {dict(run_counts)}"
            )
        summaries[setup_key] = dict(status_counts)

    expected_decisions = sum(len(case.entries) for case in cases.values()) * 9
    if len(decisions) != expected_decisions:
        raise ValueError(
            f"Decision join mismatch: {len(decisions)}/{expected_decisions}"
        )
    return decisions, receipt_statuses, summaries


def _episode_frame(rows: list[DriftSpan]) -> pl.DataFrame:
    schema = {
        "episode_id": pl.String,
        "persona_id": pl.String,
        "dimension": pl.String,
        "onset_t_index": pl.Int64,
        "confirmation_t_index": pl.Int64,
        "end_t_index": pl.Int64,
        "delivery_state": pl.String,
    }
    payload = [
        {
            "episode_id": row.drift_id,
            "persona_id": row.persona_id,
            "dimension": row.dimension,
            "onset_t_index": row.onset_t_index,
            "confirmation_t_index": row.confirmation_t_index,
            "end_t_index": row.end_t_index,
            "delivery_state": "active",
        }
        for row in rows
    ]
    return (
        pl.DataFrame(payload, schema=schema, strict=False)
        if payload
        else pl.DataFrame(schema=schema)
    )


def _derive_drifts_and_metrics(
    *,
    cases: dict[str, CaseRecord],
    decisions: dict[tuple[str, int, str, int], Decision],
    receipt_statuses: dict[tuple[str, int, str, str], str],
    reference_drifts: dict[str, tuple[DriftSpan, ...]],
    max_confirmation_lag: int,
) -> tuple[
    dict[tuple[str, int, str], tuple[DriftSpan, ...]],
    dict[tuple[str, int, str], dict[str, Any]],
]:
    predicted: dict[tuple[str, int, str], tuple[DriftSpan, ...]] = {}
    metrics = {}
    all_references = [row for rows in reference_drifts.values() for row in rows]
    reference_by_id = {row.drift_id: row for row in all_references}
    for setup_key in SETUP_SPECS:
        for run in (1, 2, 3):
            setup_rows = []
            covered = {}
            for case_id, case in cases.items():
                labels = []
                for entry in case.entries:
                    decision = decisions[(setup_key, run, case_id, entry.t_index)]
                    labels.append(
                        None
                        if decision.response_status != "ok"
                        or decision.verdict == "abstain"
                        else decision.verdict == "conflict"
                    )
                t_indices = [entry.t_index for entry in case.entries]
                covered[case_id] = trajectory_covered(labels, t_indices)
                entry_by_t_index = {entry.t_index: entry for entry in case.entries}
                case_rows = []
                for index, (onset, confirmation, end) in enumerate(
                    drift_spans(labels, t_indices), start=1
                ):
                    onset_entry = entry_by_t_index[onset]
                    confirmation_entry = entry_by_t_index[confirmation]
                    end_entry = entry_by_t_index[end]
                    case_rows.append(
                        DriftSpan(
                            drift_id=(
                                f"{setup_key}:run_{run}:{case_id}:drift_{index:02d}"
                            ),
                            case_id=case_id,
                            persona_id=case.persona_id,
                            dimension=case.dimension,
                            onset_t_index=onset,
                            confirmation_t_index=confirmation,
                            end_t_index=end,
                            onset_date=onset_entry.date,
                            confirmation_date=confirmation_entry.date,
                            end_date=end_entry.date,
                            crosses_week=(
                                _week_start(onset_entry.date)
                                != _week_start(end_entry.date)
                            ),
                            run=run,
                            detection_date=_weekly_detection_date(
                                confirmation_entry.date
                            ),
                        )
                    )
                predicted[(setup_key, run, case_id)] = tuple(case_rows)
                setup_rows.extend(case_rows)

            matches = match_episodes(
                _episode_frame(all_references),
                _episode_frame(setup_rows),
                max_confirmation_lag=max_confirmation_lag,
            ).to_dicts()
            match_by_prediction = {
                str(row["predicted_episode_id"]): row for row in matches
            }
            matched_reference_ids = {
                str(row["reference_episode_id"]) for row in matches
            }
            for case_id in cases:
                enriched = []
                for row in predicted[(setup_key, run, case_id)]:
                    match = match_by_prediction.get(row.drift_id)
                    if match is None:
                        enriched.append(replace(row, result="false Drift alert"))
                        continue
                    reference = reference_by_id[str(match["reference_episode_id"])]
                    enriched.append(
                        replace(
                            row,
                            result="hit",
                            matched_reference_id=reference.drift_id,
                            delay_days=(
                                date.fromisoformat(str(row.detection_date))
                                - date.fromisoformat(reference.confirmation_date)
                            ).days,
                            delay_entries=(
                                row.confirmation_t_index
                                - reference.confirmation_t_index
                            ),
                        )
                    )
                predicted[(setup_key, run, case_id)] = tuple(enriched)

                references = reference_drifts[case_id]
                hits = sum(row.drift_id in matched_reference_ids for row in references)
                false_alerts = sum(
                    row.result == "false Drift alert" for row in enriched
                )
                unique_weeks = {
                    _week_start(entry.date).isoformat()
                    for entry in cases[case_id].entries
                }
                statuses = [
                    receipt_statuses[
                        (
                            setup_key,
                            run,
                            cases[case_id].persona_id,
                            week_start,
                        )
                    ]
                    for week_start in unique_weeks
                ]
                delays_days = [
                    row.delay_days for row in enriched if row.delay_days is not None
                ]
                delays_entries = [
                    row.delay_entries
                    for row in enriched
                    if row.delay_entries is not None
                ]
                reference_count = len(references)
                predicted_count = len(enriched)
                metrics[(setup_key, run, case_id)] = {
                    "reference_drifts": reference_count,
                    "predicted_drift_alerts": predicted_count,
                    "drift_hits": hits,
                    "missed_drifts": reference_count - hits,
                    "false_drift_alerts": false_alerts,
                    "drift_recall": hits / reference_count if reference_count else 0.0,
                    "drift_precision": hits / predicted_count
                    if predicted_count
                    else 0.0,
                    "covered": covered[case_id],
                    "coverage": 1.0 if covered[case_id] else 0.0,
                    "abstention": 0.0 if covered[case_id] else 1.0,
                    "median_delay_days": median(delays_days) if delays_days else None,
                    "median_delay_entries": (
                        median(delays_entries) if delays_entries else None
                    ),
                    "valid_responses": statuses.count("ok"),
                    "invalid_responses": len(statuses) - statuses.count("ok"),
                }
    return predicted, metrics


def _aggregate_results(
    base_metrics: dict[str, Any], low_metrics: dict[str, Any]
) -> dict[str, tuple[dict[str, Any], ...]]:
    sources = {
        "mini_none": base_metrics["models"]["gpt_5_4_mini"],
        "luna_none": base_metrics["models"]["gpt_5_6_luna"],
        "luna_low": low_metrics["models"]["luna_low"],
    }
    results = {}
    for setup_key, payload in sources.items():
        spec = SETUP_SPECS[setup_key]
        if (
            payload["model"] != spec.model
            or payload["reasoning_effort"] != spec.reasoning_effort
            or payload["resolved_models"] != [spec.model]
        ):
            raise ValueError(f"Aggregate identity mismatch for {spec.label}")
        rows = tuple(sorted(payload["results"], key=lambda row: int(row["repeat"])))
        if tuple(int(row["repeat"]) for row in rows) != (1, 2, 3):
            raise ValueError(f"Aggregate Runs differ for {spec.label}")
        results[setup_key] = rows
    return results


def _validate_aggregate_parity(
    *,
    cases: dict[str, CaseRecord],
    case_metrics: dict[tuple[str, int, str], dict[str, Any]],
    aggregate_results: dict[str, tuple[dict[str, Any], ...]],
) -> None:
    for setup_key, rows in aggregate_results.items():
        for row in rows:
            run = int(row["repeat"])
            observed = [case_metrics[(setup_key, run, case_id)] for case_id in cases]
            totals = {
                "known_drifts": sum(item["reference_drifts"] for item in observed),
                "predicted_drift_alerts": sum(
                    item["predicted_drift_alerts"] for item in observed
                ),
                "drift_hits": sum(item["drift_hits"] for item in observed),
                "false_drift_alerts": sum(
                    item["false_drift_alerts"] for item in observed
                ),
                "coverage_count": sum(bool(item["covered"]) for item in observed),
            }
            for key, actual in totals.items():
                if actual != int(row[key]):
                    raise ValueError(
                        f"Aggregate parity mismatch for {setup_key} Run {run} {key}: "
                        f"{actual} != {row[key]}"
                    )


def _find_examples(data: ReviewData) -> dict[str, str]:
    examples = {}
    queues = {
        "run_disagreement": "Run disagreement",
        "model_disagreement": "Model disagreement",
        "false_drift_alert": "False Drift alert",
        "unresolved_abstain": "Unresolved because of Abstain",
        "invalid_response": "Invalid response",
        "uncertain_reference": "Uncertain LLM-Judge Conflict Label",
    }
    for key, queue in queues.items():
        matches = data.queue_case_ids(queue, "luna_low")
        if matches:
            examples[key] = sorted(matches)[0]
    for case_id in sorted(data.cases):
        hit_groups = {
            setup: sum(
                data.case_metrics[(setup, run, case_id)]["drift_hits"]
                for run in (1, 2, 3)
            )
            for setup in data.setup_specs
        }
        if len(set(hit_groups.values())) > 1:
            examples["cross_setup_reference_difference"] = case_id
            break
    return examples


def load_review_data(root: Path) -> ReviewData:
    """Load all frozen inputs and fail before rendering if integrity drifts."""
    root = root.resolve()
    paths = _require_paths(root)
    base_config, base_manifest, base_metrics, low_metrics = _validate_protocol_hashes(
        paths
    )
    profiles, wrangled_entries = _load_profiles(root)
    cases, reference_drifts, _targets, _outcomes = _load_cases(
        paths, profiles, wrangled_entries
    )
    prompt_map, _coordinate_to_prompt, prompt_boundaries = _load_prompts(paths, cases)
    decisions, receipt_statuses, response_summaries = _load_decisions(
        paths, cases, prompt_map
    )
    max_confirmation_lag = int(base_config["study"]["max_confirmation_lag_entries"])
    predicted_drifts, case_metrics = _derive_drifts_and_metrics(
        cases=cases,
        decisions=decisions,
        receipt_statuses=receipt_statuses,
        reference_drifts=reference_drifts,
        max_confirmation_lag=max_confirmation_lag,
    )
    aggregate_results = _aggregate_results(base_metrics, low_metrics)
    _validate_aggregate_parity(
        cases=cases,
        case_metrics=case_metrics,
        aggregate_results=aggregate_results,
    )
    data = ReviewData(
        profiles=profiles,
        cases=cases,
        setup_specs=dict(SETUP_SPECS),
        decisions=decisions,
        receipt_statuses=receipt_statuses,
        prompt_boundaries=prompt_boundaries,
        reference_drifts=reference_drifts,
        predicted_drifts=predicted_drifts,
        case_metrics=case_metrics,
        aggregate_results=aggregate_results,
        integrity={
            **EXPECTED_COUNTS,
            "receipts_per_setup": 2853,
            "receipts_per_run": 951,
            "response_summaries": response_summaries,
            "max_confirmation_lag": max_confirmation_lag,
            "reasoning_effort_provenance": (
                "Frozen manifest and registered configuration; individual receipts "
                "do not record reasoning effort."
            ),
            "reviewer_input_provenance": (
                "Frozen prompt records verify declared Core Values, cumulative "
                "Journal Entries through each review week, and an empty VIF Critic "
                "input block."
            ),
            "base_manifest_schema": base_manifest["schema_version"],
        },
    )
    data.integrity["examples"] = _find_examples(data)
    return data
