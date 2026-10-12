"""Verify archived studies and adapt their receipt formats for inspection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.drift_review_app.data import (
    CURRENT_DIR,
    CURRENT_HIGHER_DIR,
    DEFINITIONS_DIR,
    EXPECTED_COUNTS,
    HIGHER_DIR,
    MODEL_DIR,
    SETUP_SPECS,
    _canonical_json,
    _read_json,
    _read_jsonl,
    _read_yaml,
    _sha256_file,
    _sha256_text,
    _validate_assessments,
)
from src.prompt_boundary import render_live_prompt_receipt


@dataclass
class ExpandedInputs:
    """Inputs already verified against each archived study's provenance."""

    prompt_groups: dict[str, list[dict[str, Any]]]
    response_rows: dict[str, list[dict[str, Any]]]
    aggregates: dict[str, tuple[dict[str, Any], ...]]
    definitions_results: dict[str, Any]
    calculated_cost_usd: dict[str, float]


def expanded_paths(root: Path) -> tuple[Path, ...]:
    """Files needed by the hosted inspection app, including provider receipts."""
    paths = [
        root / "config/evals/twinkl_ck3w_luna_higher_reasoning_v1.yaml",
        root / "config/evals/twinkl_q6pt_luna_model_v1.yaml",
        root / "config/evals/twinkl_gv3x_luna_higher_v1.yaml",
        root
        / "logs/experiments/reports"
        / "experiment_review_2026-07-14_twinkl_qtwz_complete_development_review.md",
    ]
    for directory in (CURRENT_DIR, CURRENT_HIGHER_DIR, DEFINITIONS_DIR):
        paths.extend(
            root / directory / name
            for name in (
                "manifest.json",
                "responses.jsonl",
                "attempts.jsonl",
                "results.json" if directory == DEFINITIONS_DIR else "metrics.json",
            )
        )
    paths.extend(
        root / DEFINITIONS_DIR / name
        for name in (
            "requests.jsonl",
            "baseline_snapshot.json",
        )
    )
    paths.extend(root / HIGHER_DIR / name for name in ("manifest.json", "metrics.json"))
    paths.extend(
        root / SETUP_SPECS[f"luna_{effort}"].response_path
        for effort in ("medium", "high", "xhigh")
    )
    return tuple(paths)


def _check_hash(path: Path, expected: str, label: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"Frozen {label} hash mismatch: {path}")


def _check_runs(payload: dict[str, Any], setup_key: str) -> tuple[dict[str, Any], ...]:
    spec = SETUP_SPECS[setup_key]
    if (payload["model"], payload["reasoning_effort"]) != (
        spec.model,
        spec.reasoning_effort,
    ):
        raise ValueError(f"Aggregate identity mismatch for {setup_key}")
    rows = tuple(
        sorted(
            payload.get("runs", payload.get("results", [])),
            key=lambda row: int(row["repeat"]),
        )
    )
    if tuple(int(row["repeat"]) for row in rows) != (1, 2, 3):
        raise ValueError(f"Aggregate Runs differ for {setup_key}")
    return rows


def _load_higher(
    root: Path,
) -> tuple[dict[str, list[dict]], dict[str, tuple[dict, ...]]]:
    config_path = root / "config/evals/twinkl_ck3w_luna_higher_reasoning_v1.yaml"
    config = _read_yaml(config_path)
    manifest_path = root / HIGHER_DIR / "manifest.json"
    manifest = _read_json(manifest_path)
    metrics = _read_json(root / HIGHER_DIR / "metrics.json")
    if _sha256_text(_canonical_json(config)) != manifest["config_sha256"]:
        raise ValueError("Frozen higher reasoning config hash mismatch")
    _check_hash(
        manifest_path, metrics["provenance"]["manifest_sha256"], "higher manifest"
    )
    if manifest["models"] != config["models"] or (
        manifest["study_id"],
        manifest["setup"],
        manifest["repeats"],
    ) != ("twinkl_ck3w_luna_higher_reasoning_v1", "weekly_without_critic", 3):
        raise ValueError("Frozen higher reasoning identity differs")
    if manifest["prompt_contract"] != {
        "development_labels_in_prompt": False,
        "fresh_final_test_inspected": False,
        "same_as_luna_low": True,
        "store": False,
        "vif_critic_input": False,
    }:
        raise ValueError("Frozen higher reasoning prompt contract differs")
    for baseline_key, config_key in (
        ("frozen_luna_low", "baseline"),
        ("frozen_luna_none", "no_reasoning_baseline"),
    ):
        baseline = config[config_key]
        expected = baseline["expected_sha256"]
        for key in ("config", "manifest", "responses", "metrics"):
            _check_hash(root / baseline[f"{key}_path"], expected[key], f"higher {key}")
            if manifest[baseline_key][f"{key}_sha256"] != expected[key]:
                raise ValueError(f"Frozen higher baseline {key} identity differs")
        _check_hash(
            root / MODEL_DIR / "prompts.jsonl",
            manifest[baseline_key]["prompts_sha256"],
            "higher shared prompts",
        )
    responses, aggregates = {}, {}
    for effort in ("medium", "high", "xhigh"):
        setup_key = f"luna_{effort}"
        spec = SETUP_SPECS[setup_key]
        path = root / spec.response_path
        _check_hash(path, metrics["provenance"]["responses_sha256"][effort], setup_key)
        raw = _read_jsonl(path)
        # The archived executor retries error receipts. Only its final terminal
        # receipt contributes to scoring; the earlier error remains in the file.
        terminals = [
            row for row in raw if row["status"] in {"ok", "invalid", "refusal"}
        ]
        for row in raw:
            if (
                row.get("reasoning_effort") != effort
                or row["requested_model"] != spec.model
            ):
                raise ValueError(
                    f"Frozen higher receipt identity differs for {setup_key}"
                )
            if row.get("resolved_model") not in (None, spec.model):
                raise ValueError(f"Unexpected provider model for {setup_key}")
        responses[setup_key] = terminals
        aggregates[setup_key] = _check_runs(metrics["models"][setup_key], setup_key)
    return responses, aggregates


def _normalize_prompts(rows: list[dict], variant: str) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        request = row["request"]
        payload = row["variants"][variant]
        if (
            _sha256_text(
                _canonical_json(
                    {k: v for k, v in payload.items() if k != "request_sha256"}
                )
            )
            != payload["request_sha256"]
        ):
            raise ValueError(
                f"Frozen request hash mismatch: {row['case_id']} {variant}"
            )
        if _sha256_text(request["prompt"]) != request["prompt_sha256"]:
            raise ValueError(f"Frozen source prompt hash mismatch: {row['case_id']}")
        if (
            _sha256_text(_canonical_json(request["history"]))
            != request["runtime_text_sha256"]
        ):
            raise ValueError(f"Frozen request history hash mismatch: {row['case_id']}")
        expected_input = {
            "declared_core_values": request["core_values"],
            "current_week_entry_t_indices": request["current_t_indices"],
            "journal_entries": [
                {"t_index": entry["t_index"], "text": entry["text"]}
                for entry in request["history"]
            ],
        }
        if payload["input_data"] != _canonical_json(expected_input):
            raise ValueError(
                f"Frozen request input differs from history: {row['case_id']}"
            )
        prompt = render_live_prompt_receipt(
            instructions=payload["instructions"], input_data=payload["input_data"]
        )
        records.append(
            {
                "arm": "weekly_without_critic",
                "persona_id": request["persona_id"],
                "week_start": request["week_start"],
                "week_end": request["week_end"],
                "review_at_date": request["week_end"],
                "review_event_id": row["case_id"],
                "current_t_indices": request["current_t_indices"],
                "cutoff_t_index": max(request["current_t_indices"]),
                "declared_values": request["core_values"],
                "entry_text_by_t_index": {
                    str(entry["t_index"]): entry["text"] for entry in request["history"]
                },
                "expected_coordinates": [
                    {"t_index": index, "dimension": value}
                    for index in request["current_t_indices"]
                    for value in request["core_values"]
                ],
                "critic_block_sha256": _sha256_text(""),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "runtime_text_sha256": request["runtime_text_sha256"],
            }
        )
    return records


def _normalize_responses(
    *,
    directory: Path,
    rows: list[dict],
    prompts: dict[str, list[dict]],
    arm_keys: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    _check_hash(
        directory / "responses.jsonl", metrics["responses_sha256"], "study responses"
    )
    _check_hash(
        directory / "attempts.jsonl", metrics["attempts_sha256"], "study attempts"
    )
    response_rows = _read_jsonl(directory / "responses.jsonl")
    expected = {
        f"{arm}:{run}:{row['case_id']}"
        for arm in arm_keys
        for run in (1, 2, 3)
        for row in rows
    }
    if (
        len(response_rows) != len(expected)
        or {row["request_key"] for row in response_rows} != expected
    ):
        raise ValueError("Frozen study receipt join is incomplete or duplicated")
    row_map = {row["case_id"]: row for row in rows}
    prompt_maps = {
        arm: {record["review_event_id"]: record for record in records}
        for arm, records in prompts.items()
    }
    attempts: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    finished_keys = set()
    for event in _read_jsonl(directory / "attempts.jsonl"):
        key, number = str(event["request_key"]), int(event["attempt_number"])
        if key not in expected or event["event"] not in {"started", "finished"}:
            raise ValueError("Unknown frozen provider attempt")
        arm, _run, case_id = key.split(":", 2)
        if (
            event["request_sha256"]
            != row_map[case_id]["variants"][arm]["request_sha256"]
        ):
            raise ValueError("Provider attempt request hash differs")
        if event["event"] == "started":
            if number in attempts[key] or number != len(attempts[key]) + 1:
                raise ValueError("Duplicate or nonconsecutive frozen provider attempt")
        else:
            if number not in attempts[key] or (key, number) in finished_keys:
                raise ValueError("Frozen provider finish lacks a unique start")
            finished_keys.add((key, number))
            spec = SETUP_SPECS[arm_keys[arm]]
            if event.get("resolved_model") not in (None, spec.model):
                raise ValueError(f"Unexpected provider model for {key}")
        attempts[key][number] = event
    normalized: dict[str, list[dict[str, Any]]] = {key: [] for key in arm_keys.values()}
    for receipt in response_rows:
        arm, run, case_id = (
            receipt["variant"],
            int(receipt["repeat"]),
            receipt["case_id"],
        )
        key = receipt["request_key"]
        if key != f"{arm}:{run}:{case_id}" or key not in attempts:
            raise ValueError("Frozen receipt request identity differs")
        row = row_map[case_id]
        if receipt["request_sha256"] != row["variants"][arm]["request_sha256"]:
            raise ValueError("Frozen receipt request hash differs")
        history = attempts[key]
        if receipt["attempts"] != len(history):
            raise ValueError("Frozen receipt attempt count differs")
        final = history[max(history)]
        interrupted = final["event"] == "started"
        status = "error" if interrupted else final["status"]
        if (
            receipt["status"] != status
            or receipt["unknown_interrupted_attempt"] != interrupted
        ):
            raise ValueError("Frozen receipt final status differs")
        prompt = prompt_maps[arm][case_id]
        source = row["request"]
        assessments = (
            _validate_assessments(
                {"parsed": final.get("parsed")}, prompt, strict_quotes=True
            )
            if status == "ok"
            else {}
        )
        entry_dates = {entry["t_index"]: entry["date"] for entry in source["history"]}
        coordinates = set()
        for decision in receipt["decisions"]:
            coordinate = (decision["core_value"], decision["t_index"])
            if coordinate in coordinates:
                raise ValueError("Duplicate effective receipt coordinate")
            coordinates.add(coordinate)
            if (
                decision["persona_id"],
                decision["week_start"],
                decision["week_end"],
                decision["date"],
                decision["review_status"],
            ) != (
                source["persona_id"],
                source["week_start"],
                source["week_end"],
                entry_dates[decision["t_index"]],
                status,
            ):
                raise ValueError("Effective decision boundary differs from its request")
            assessment = assessments.get(coordinate)
            expected_fields = (
                {
                    key: assessment[key]
                    for key in (
                        "verdict",
                        "confidence",
                        "reason_code",
                        "evidence_quote",
                    )
                }
                if assessment
                else {
                    "verdict": "abstain",
                    "confidence": None,
                    "reason_code": None,
                    "evidence_quote": "",
                }
            )
            if any(decision[key] != value for key, value in expected_fields.items()):
                raise ValueError(
                    "Effective decision differs from validated provider output"
                )
        if coordinates != {
            (item["dimension"], item["t_index"])
            for item in prompt["expected_coordinates"]
        }:
            raise ValueError("Frozen effective receipt coordinates are incomplete")
        spec = SETUP_SPECS[arm_keys[arm]]
        normalized[spec.key].append(
            {
                "persona_id": source["persona_id"],
                "week_start": source["week_start"],
                "week_end": source["week_end"],
                "repeat": run,
                "status": status,
                "arm": "weekly_without_critic",
                "requested_model": spec.model,
                "resolved_model": final.get("resolved_model"),
                "reasoning_effort": spec.reasoning_effort,
                "prompt_sha256": prompt["prompt_sha256"],
                "runtime_text_sha256": prompt["runtime_text_sha256"],
                "parsed": final.get("parsed") if status == "ok" else None,
                "validation_error": final.get(
                    "validation_error", final.get("error_type", status)
                ),
            }
        )
    return normalized


def load_expanded_inputs(root: Path, *, reference_cases_sha256: str) -> ExpandedInputs:
    """Verify all report studies without rendering prompts or calling a provider."""
    missing = [str(path) for path in expanded_paths(root) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing review input(s): " + ", ".join(missing))
    responses, aggregates = _load_higher(root)
    calculated_cost_usd = {}
    directory = root / DEFINITIONS_DIR
    manifest = _read_json(directory / "manifest.json")
    definitions_results = _read_json(directory / "results.json")
    _check_hash(
        directory / "manifest.json",
        definitions_results["manifest_sha256"],
        "definitions manifest",
    )
    _check_hash(
        directory / "baseline_snapshot.json",
        manifest["baseline_snapshot_sha256"],
        "baseline snapshot",
    )
    _check_hash(
        directory / "requests.jsonl",
        manifest["requests_sha256"],
        "definitions requests",
    )
    settings = manifest["settings"]
    if (
        settings["model"],
        settings["reasoning_effort"],
        settings["repeats"],
        settings["store"],
        manifest["baseline_prompt_version"],
        manifest["treatment_prompt_version"],
        manifest["personas"],
        manifest["weeks"],
        manifest["expected_terminal_requests"],
    ) != (
        "gpt-5.6-luna",
        "low",
        3,
        False,
        "3.0",
        "4.0",
        204,
        951,
        5706,
    ):
        raise ValueError("Frozen definitions study identity differs")
    rows = _read_jsonl(directory / "requests.jsonl")
    if len(rows) != 951 or len({row["case_id"] for row in rows}) != 951:
        raise ValueError("Frozen definitions request count differs")
    baseline_prompts = _normalize_prompts(rows, "baseline")
    definitions_prompts = _normalize_prompts(rows, "definitions")
    for row in rows:
        if (
            row["variants"]["baseline"]["prompt_version"] != "3.0"
            or row["variants"]["definitions"]["prompt_version"] != "4.0"
        ):
            raise ValueError("Frozen prompt version differs")
        if (
            row["variants"]["baseline"]["input_data"]
            != row["variants"]["definitions"]["input_data"]
        ):
            raise ValueError("Definitions study changed its Journal Entry input")
    prompt_groups = {
        "definitions_off_low": baseline_prompts,
        "definitions_on_low": definitions_prompts,
    }
    responses.update(
        _normalize_responses(
            directory=directory,
            rows=rows,
            prompts={"baseline": baseline_prompts, "definitions": definitions_prompts},
            arm_keys={
                "baseline": "definitions_off_low",
                "definitions": "definitions_on_low",
            },
            metrics=definitions_results,
        )
    )
    for directory, config_name in (
        (CURRENT_DIR, "twinkl_q6pt_luna_model_v1"),
        (CURRENT_HIGHER_DIR, "twinkl_gv3x_luna_higher_v1"),
    ):
        path = root / directory
        config_path = root / f"config/evals/{config_name}.yaml"
        config = _read_yaml(config_path)
        manifest = _read_json(path / "manifest.json")
        metrics = _read_json(path / "metrics.json")
        _check_hash(config_path, manifest["config_sha256"], "current config")
        _check_hash(
            path / "manifest.json", metrics["manifest_sha256"], "current manifest"
        )
        _check_hash(
            root / config["source_requests"],
            manifest["source_requests_sha256"],
            "current source requests",
        )
        _check_hash(
            root / config["reference_config"],
            manifest["reference_config_sha256"],
            "current reference config",
        )
        reference_config = _read_yaml(root / config["reference_config"])
        if reference_cases_sha256 != manifest["reference_cases_sha256"]:
            raise ValueError("Frozen current reference case content hash differs")
        for key, expected in manifest["reference_source_sha256"].items():
            _check_hash(
                root / reference_config["sources"][key], expected, f"current {key}"
            )
        if (
            manifest["models"] != config["models"]
            or manifest["settings"] != config["api"]
            or (
                manifest["study_id"],
                metrics["study_id"],
                manifest["prompt_version"],
                manifest["repeats"],
                manifest["personas"],
                manifest["weeks"],
            )
            != (config_name, config_name, "4.0", 3, 204, 951)
        ):
            raise ValueError("Frozen current study identity differs")
        if (
            config["study"]["max_confirmation_lag_entries"] != 2
            or config["api"]["store"]
        ):
            raise ValueError("Frozen current evaluation contract differs")
        if manifest["prompt_sha256s"] != [
            row["request"]["prompt_sha256"] for row in rows
        ] or manifest["request_sha256s"] != [
            row["variants"]["definitions"]["request_sha256"] for row in rows
        ]:
            raise ValueError("Frozen current request identities differ")
        arm_keys = {
            arm: (
                f"gpt{'56' if model['model'] == 'gpt-5.6-luna' else '6'}_"
                f"{model['reasoning_effort']}"
            )
            for arm, model in config["models"].items()
        }
        adapted_rows = [
            {
                **row,
                "variants": {arm: row["variants"]["definitions"] for arm in arm_keys},
            }
            for row in rows
        ]
        responses.update(
            _normalize_responses(
                directory=path,
                rows=adapted_rows,
                prompts={arm: definitions_prompts for arm in arm_keys},
                arm_keys=arm_keys,
                metrics=metrics,
            )
        )
        for arm, setup_key in arm_keys.items():
            prompt_groups[setup_key] = definitions_prompts
            aggregates[setup_key] = _check_runs(metrics["models"][arm], setup_key)
            calculated_cost_usd[setup_key] = float(
                metrics["models"][arm]["calculated_cost_usd"]
            )
    return ExpandedInputs(
        prompt_groups, responses, aggregates, definitions_results, calculated_cost_usd
    )


def definitions_summary(results: dict[str, Any]) -> dict[str, Any]:
    """Original detected Drift counts and changed cells, without accuracy claims."""
    return {
        "baseline_counts": results["counts"]["all"]["baseline"]["runs"],
        "definitions_counts": results["counts"]["all"]["definitions"]["runs"],
        "changed_entry_decisions": [
            len(results["runs"][str(run)]["changed_entry_decisions"])
            for run in (1, 2, 3)
        ],
        "response_statuses": dict(
            Counter(
                {
                    status: sum(
                        results["runs"][str(run)]["request_statuses"][variant].get(
                            status, 0
                        )
                        for run in (1, 2, 3)
                        for variant in ("baseline", "definitions")
                    )
                    for status in ("ok", "invalid")
                }
            )
        ),
        "cell_count": EXPECTED_COUNTS["entry_value_cells"],
    }
