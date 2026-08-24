"""Generate approved-path Coach Digest responses and build an evaluation sample.

Runs the approved Weekly Drift Reviewer + Drift Detector cycle
(``run_weekly_drift_coach_cycle``) for an explicit roster of personas, so the
Coach Digest Evals score responses as the product's approved path
actually produces them — not leftover ``vif_runtime`` demo-tool outputs.

For the deployed-Persona replacement this:
  1. reads the stored Weekly Drift Detection output from the exact key week's
     ``weekly_digest_built`` event in each public scenario bundle,
  2. makes no Weekly Drift Reviewer calls,
  3. generates one Coach Digest response per Persona with validation-guided retry,
  4. writes the accepted responses to the checked-in scenario response fixture,
  5. rebuilds the public scenario bundles, and
  6. builds the judge manifest from those rebuilt public bundles.

Then it rebuilds the judge sample manifest from the freshly written narratives.

Paid calls are gated behind ``--execute``. Without it, this prints the plan and
makes no calls.

Run:
    .venv/bin/python scripts/coach/generate_approved_judge_sample.py \
        --personas 11de77e8 23d101f8 8f83c818 988d1a65 02fb94f3 \
        --reuse-scenario-key-weeks --execute
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Allow running as a bare script (`python scripts/coach/...`) by putting the
# repo root on sys.path before importing the `src` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from prompts import get_prompt_metadata  # noqa: E402
from src.coach.llm_client import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    OPENAI_LUNA_PRICING_SOURCE,
    build_llm_complete,
    summarize_llm_call_metrics,
)
from src.coach.schemas import (  # noqa: E402
    CoachNarrative,
    LLMCallMetrics,
    WeeklyDigest,
)
from src.coach.weekly_digest import (  # noqa: E402
    LLMCompleteFn,
    attach_coach_artifacts,
    generate_weekly_digest_coach_diagnostic,
    persist_weekly_digest_record,
    render_digest_markdown,
)
from src.coach.weekly_drift_runtime import run_weekly_drift_coach_cycle  # noqa: E402
from src.demo.contracts import ContractFixtureSet  # noqa: E402
from src.demo.scenarios import (  # noqa: E402
    CATALOG_PATH,
    COACH_RESPONSES_PATH,
    SCENARIO_DIRECTORY,
    SELECTIONS,
    export_scenarios,
)

DEFAULT_PARQUET = Path("logs/exports/weekly_digests/weekly_digests.parquet")
DEFAULT_WEEKLY_DRIFT_OUTPUT_DIR = Path("logs/exports/weekly_drift_coach")
DEFAULT_MANIFEST = Path(
    "logs/experiments/reports/coach_digest_sample_20260824/"
    "judge_sample_manifest.json"
)
DEFAULT_RESPONSE_FIXTURE = COACH_RESPONSES_PATH


def _digest_to_manifest_entry(
    digest: WeeklyDigest,
    *,
    provenance: dict[str, object] | None = None,
) -> dict | None:
    """Convert a generated digest into a judge-manifest {digest, narrative} entry."""
    if digest.coach_narrative is None:
        return None
    entry = {
        "digest": digest.model_dump(
            mode="json",
            exclude={"coach_narrative", "validation"},
        ),
        "narrative": digest.coach_narrative.model_dump(),
    }
    if provenance is not None:
        entry["provenance"] = provenance
    return entry


def _find_stored_digest_path(persona_id: str, output_dir: Path) -> Path:
    """Find one stored approved Weekly Drift Detection output."""
    candidates = [
        path
        for path in output_dir.glob(f"{persona_id}_*.json")
        if not path.name.endswith(
            (
                ".coach_diagnostic.json",
                ".drift.json",
                ".weekly_drift_review.json",
            )
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one stored output for {persona_id}; found {len(candidates)}."
        )
    return candidates[0]


def _write_diagnostic(output_dir: Path, stem: str, payload: dict) -> Path:
    """Write one timestamped Coach Digest diagnostic file."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = output_dir / f"{stem}.{stamp}.coach_diagnostic.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _model_contract() -> dict[str, str]:
    provider = os.environ.get("TWINKL_COACH_PROVIDER", "openai").strip().lower()
    default_model = (
        DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_OPENAI_MODEL
    )
    contract = {
        "coach_provider": provider,
        "coach_model": os.environ.get("TWINKL_COACH_MODEL", default_model),
    }
    if provider == "openai":
        contract["coach_reasoning_effort"] = DEFAULT_OPENAI_REASONING_EFFORT
        contract["coach_service_tier"] = DEFAULT_OPENAI_SERVICE_TIER
        contract["coach_pricing_source"] = OPENAI_LUNA_PRICING_SOURCE
    return contract


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_for_persona(persona_id: str):
    try:
        return next(
            selection for selection in SELECTIONS if selection.persona_id == persona_id
        )
    except StopIteration as error:
        raise ValueError(
            f"Persona is not in the deployed roster: {persona_id}"
        ) from error


def _extract_scenario_key_week_digests(
    personas: list[str],
    output_dir: Path,
    *,
    root: Path = _REPO_ROOT,
) -> dict[str, dict[str, object]]:
    """Write the exact key-week Weekly Drift Detection inputs from public bundles."""
    if len(set(personas)) != len(personas):
        raise ValueError("Persona IDs must be unique.")
    catalog = json.loads((root / CATALOG_PATH).read_text(encoding="utf-8"))
    catalog_by_scenario = {
        str(item["scenario_id"]): item for item in catalog["scenarios"]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, object]] = {}

    for persona_id in personas:
        selection = _selection_for_persona(persona_id)
        item = catalog_by_scenario.get(selection.scenario_id)
        if item is None or item.get("persona_id") != persona_id:
            raise ValueError(f"Scenario catalog identity mismatch: {persona_id}")
        bundle_path = root / SCENARIO_DIRECTORY / str(item["file"])
        bundle_bytes = bundle_path.read_bytes()
        bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
        if bundle_hash != item.get("content_sha256"):
            raise ValueError(f"Scenario catalog hash mismatch: {selection.scenario_id}")
        fixture = ContractFixtureSet.model_validate_json(bundle_bytes)
        key_week = next(
            (
                week
                for week in fixture.scenario.weeks
                if week.week_start == selection.coach_week_start
            ),
            None,
        )
        if key_week is None:
            raise ValueError(f"Scenario key week is missing: {selection.scenario_id}")
        digest_event = next(
            (
                event
                for event in fixture.trace_events
                if event.event_id in key_week.event_ids
                and event.event_type == "weekly_digest_built"
            ),
            None,
        )
        if digest_event is None:
            raise ValueError(
                f"Scenario key week has no weekly_digest_built event: "
                f"{selection.scenario_id}"
            )
        digest = digest_event.details.digest.model_copy(
            update={"coach_narrative": None, "validation": None}
        )
        if (
            digest.persona_id != persona_id
            or digest.week_start != selection.coach_week_start
            or digest.week_end != key_week.week_end
        ):
            raise ValueError(f"Scenario key-week digest mismatch: {persona_id}")
        digest_path = output_dir / f"{persona_id}_{digest.week_end}.json"
        digest_path.write_text(digest.model_dump_json(indent=2) + "\n")
        provenance[persona_id] = {
            "scenario_id": selection.scenario_id,
            "source_bundle_path": str(bundle_path.relative_to(root)),
            "source_bundle_content_sha256": bundle_hash,
            "weekly_digest_event_id": digest_event.event_id,
            "week_start": digest.week_start,
            "week_end": digest.week_end,
        }
    return provenance


def _write_scenario_response_fixture(
    generated_manifest: list[dict],
    output_path: Path,
) -> None:
    """Write accepted responses in the fixture consumed by scenario export."""
    responses: dict[str, object] = {}
    for item in generated_manifest:
        digest = WeeklyDigest.model_validate(item["digest"])
        narrative = CoachNarrative.model_validate(item["narrative"])
        provenance = dict(item["provenance"])
        selection = _selection_for_persona(digest.persona_id)
        source = dict(provenance["scenario_source"])
        if digest.week_start != selection.coach_week_start:
            raise ValueError(
                f"Generated response is not for the key week: {digest.persona_id}"
            )
        key = f"{selection.scenario_id}::{digest.week_start}"
        responses[key] = {
            "scenario_id": selection.scenario_id,
            "persona_id": digest.persona_id,
            "week_start": digest.week_start,
            "week_end": digest.week_end,
            "narrative": narrative.model_dump(mode="json"),
            "generation": {
                "model_contract": {
                    "provider": provenance["coach_provider"],
                    "model": provenance["coach_model"],
                    "reasoning_effort": provenance["coach_reasoning_effort"],
                },
                "service_tier": provenance["coach_service_tier"],
                "prompt_name": provenance["coach_prompt_name"],
                "prompt_version": provenance["coach_prompt_version"],
                "prompt_sha256": provenance["coach_prompt_sha256"],
                "prompt": provenance["coach_prompt"],
                "raw_output": provenance["coach_raw_output"],
                "response_sha256": provenance["coach_response_sha256"],
                "attempt_count": provenance["coach_attempt_count"],
                "diagnostic_paths": provenance["coach_diagnostic_paths"],
                "call_metrics": provenance["coach_call_metrics"],
                "weekly_drift_input_sha256": provenance[
                    "weekly_drift_input_sha256"
                ],
                "generated_response_path": provenance[
                    "weekly_drift_output_path"
                ],
                "source_bundle_path": source["source_bundle_path"],
                "source_bundle_content_sha256": source[
                    "source_bundle_content_sha256"
                ],
                "weekly_digest_event_id": source["weekly_digest_event_id"],
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "coach-digest-scenario-fixture-v1",
                "responses": responses,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _manifest_from_public_scenarios(
    personas: list[str],
    *,
    root: Path = _REPO_ROOT,
    response_fixture_path: Path = DEFAULT_RESPONSE_FIXTURE,
) -> list[dict]:
    """Build the evaluation manifest from exact rebuilt public responses."""
    catalog = json.loads((root / CATALOG_PATH).read_text(encoding="utf-8"))
    catalog_by_scenario = {
        str(item["scenario_id"]): item for item in catalog["scenarios"]
    }
    saved = json.loads((root / response_fixture_path).read_text(encoding="utf-8"))
    manifest: list[dict] = []

    for persona_id in personas:
        selection = _selection_for_persona(persona_id)
        catalog_item = catalog_by_scenario[selection.scenario_id]
        bundle_path = root / SCENARIO_DIRECTORY / str(catalog_item["file"])
        bundle_bytes = bundle_path.read_bytes()
        bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
        if bundle_hash != catalog_item["content_sha256"]:
            raise ValueError(f"Rebuilt scenario hash mismatch: {selection.scenario_id}")
        fixture = ContractFixtureSet.model_validate_json(bundle_bytes)
        key_week = next(
            week
            for week in fixture.scenario.weeks
            if week.week_start == selection.coach_week_start
        )
        digest_event = next(
            event
            for event in fixture.trace_events
            if event.event_id in key_week.event_ids
            and event.event_type == "weekly_digest_built"
        )
        digest = digest_event.details.digest
        if digest.coach_narrative is None or digest.validation is None:
            raise ValueError(f"Rebuilt scenario response is missing: {persona_id}")
        if not digest.validation.all_passed:
            raise ValueError(f"Rebuilt scenario response is rejected: {persona_id}")
        key = f"{selection.scenario_id}::{digest.week_start}"
        saved_response = saved["responses"][key]
        narrative = digest.coach_narrative
        if narrative.model_dump(mode="json") != saved_response["narrative"]:
            raise ValueError(f"Displayed response differs from fixture: {persona_id}")
        generation = saved_response["generation"]
        response_hash = _canonical_sha256(narrative.model_dump(mode="json"))
        source_digest_hash = _canonical_sha256(
            digest.model_dump(
                mode="json",
                exclude={"coach_narrative", "validation"},
            )
        )
        if response_hash != generation["response_sha256"]:
            raise ValueError(f"Displayed response hash mismatch: {persona_id}")
        if source_digest_hash != generation["weekly_drift_input_sha256"]:
            raise ValueError(f"Displayed source hash mismatch: {persona_id}")
        entry = _digest_to_manifest_entry(
            digest,
            provenance={
                "scenario_id": selection.scenario_id,
                "scenario_bundle_path": str(bundle_path.relative_to(root)),
                "scenario_bundle_content_sha256": bundle_hash,
                "weekly_digest_event_id": digest_event.event_id,
                "weekly_drift_input_sha256": source_digest_hash,
                "coach_response_sha256": response_hash,
                "display_source": "rebuilt_public_scenario_bundle",
                "generation": generation,
            },
        )
        if entry is None:
            raise ValueError(f"Rebuilt scenario response is missing: {persona_id}")
        manifest.append(entry)
    return manifest


def _write_generation_report(
    manifest: list[dict],
    output_dir: Path,
    *,
    command: str,
) -> None:
    """Write generation call, failure, identity, and location evidence."""
    call_metrics: list[LLMCallMetrics] = []
    failed_attempts: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    for item in manifest:
        digest = item["digest"]
        provenance = item["provenance"]
        generation = provenance["generation"]
        metrics = [
            LLMCallMetrics.model_validate(metric)
            for metric in generation["call_metrics"]
        ]
        call_metrics.extend(metrics)
        for diagnostic_path in generation["diagnostic_paths"]:
            diagnostic = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))
            if not diagnostic["accepted"]:
                failed_attempts.append(
                    {
                        "persona_id": digest["persona_id"],
                        "diagnostic_path": diagnostic_path,
                        "failure_stage": diagnostic.get("failure_stage"),
                        "failure_details": diagnostic.get("failure_details", []),
                        "raw_output_preserved": (
                            diagnostic.get("raw_output") is not None
                        ),
                    }
                )
        responses.append(
            {
                "scenario_id": provenance["scenario_id"],
                "persona_id": digest["persona_id"],
                "week_start": digest["week_start"],
                "week_end": digest["week_end"],
                "generated_response_path": generation["generated_response_path"],
                "public_scenario_path": provenance["scenario_bundle_path"],
                "weekly_digest_event_id": provenance["weekly_digest_event_id"],
                "scenario_bundle_content_sha256": provenance[
                    "scenario_bundle_content_sha256"
                ],
                "weekly_drift_input_sha256": provenance[
                    "weekly_drift_input_sha256"
                ],
                "coach_response_sha256": provenance["coach_response_sha256"],
                "attempt_count": generation["attempt_count"],
                "diagnostic_paths": generation["diagnostic_paths"],
            }
        )
    usage = summarize_llm_call_metrics(call_metrics)
    report = {
        "run": "coach_digest_sample_20260824",
        "command": command,
        "coach_prompt_version": "4.1",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "weekly_drift_reviewer_calls": 0,
        "accepted_responses": len(responses),
        "generation_api_usage": usage,
        "retry_count": max(len(call_metrics) - len(responses), 0),
        "failed_attempts": failed_attempts,
        "responses": responses,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    cost = usage["calculated_cost_usd"]
    lines = [
        "# Deployed Persona Coach Digest Sample",
        "",
        "- Source: each public scenario bundle's stored `weekly_digest_built` output",
        "- Weekly Drift Reviewer calls: 0",
        "- Coach Digest prompt: `weekly_digest_coach` v4.1",
        "- Model: `gpt-5.6-luna`",
        "- Reasoning effort: `none`",
        f"- Accepted responses: {len(responses)}",
        f"- Paid generation calls: {usage['n_calls']}",
        f"- Validation-guided retries: {report['retry_count']}",
        f"- Input tokens: {usage['input_tokens']}",
        f"- Cached input tokens: {usage['cached_input_tokens']}",
        f"- Output tokens: {usage['output_tokens']}",
        (
            f"- Calculated published-rate cost: `${float(cost):.8f}`"
            if cost is not None
            else "- Calculated published-rate cost: unavailable"
        ),
        f"- Total request latency: {float(usage['total_latency_seconds'] or 0):.3f}s",
        "- Cost basis: response token usage and published standard-tier Luna "
        "rates; not a billing export",
        "- Prompt tuning after final scores: none",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Responses",
        "",
        "| Scenario | Persona | Week | Attempts | Response hash | "
        "Generated response | Public bundle |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for response in responses:
        lines.append(
            f"| {response['scenario_id']} | {response['persona_id']} | "
            f"{response['week_start']} to {response['week_end']} | "
            f"{response['attempt_count']} | `{response['coach_response_sha256']}` | "
            f"`{response['generated_response_path']}` | "
            f"`{response['public_scenario_path']}` |"
        )
    lines += ["", "## Failed Attempts and Review"]
    if failed_attempts:
        for failure in failed_attempts:
            lines.append(
                f"- `{failure['persona_id']}`: {failure['failure_stage']}; "
                f"raw output preserved at `{failure['diagnostic_path']}`."
            )
    else:
        lines.append("- No failed generation attempts.")
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


async def _generate_reusing_weekly_drift(
    personas: list[str],
    parquet_path: Path,
    output_dir: Path,
    *,
    llm_complete: LLMCompleteFn | None = None,
    call_metrics: list[LLMCallMetrics] | None = None,
    diagnostic_output_dir: Path | None = None,
    source_provenance_by_persona: dict[str, dict[str, object]] | None = None,
) -> list[dict]:
    """Generate Coach Digest responses from stored Weekly Drift Detection output."""
    collected_metrics = call_metrics if call_metrics is not None else []
    coach_llm_complete = llm_complete or build_llm_complete(
        call_metrics=collected_metrics
    )
    if coach_llm_complete is None:
        raise SystemExit(
            "No Coach Digest provider is available because the API key is missing. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY."
        )

    prompt_metadata = get_prompt_metadata("weekly_digest_coach")
    diagnostics_dir = diagnostic_output_dir or output_dir
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[WeeklyDigest, Path, str, dict[str, object]]] = []
    failures: list[str] = []
    for persona_id in personas:
        digest_path = _find_stored_digest_path(persona_id, output_dir)
        source_bytes = digest_path.read_bytes()
        digest = WeeklyDigest.model_validate_json(source_bytes)
        digest = digest.model_copy(
            update={"coach_narrative": None, "validation": None}
        )
        input_bytes = json.dumps(
            digest.model_dump(
                mode="json",
                exclude={"coach_narrative", "validation"},
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        print(f"[coach-only] {persona_id} ...", flush=True)
        repair_requirements: list[str] | None = None
        diagnostic_paths: list[Path] = []
        persona_call_metrics: list[LLMCallMetrics] = []
        accepted_prompt = ""
        for attempt in range(1, 3):
            metrics_before_call = len(collected_metrics)
            diagnostic, attempt_prompt = (
                await generate_weekly_digest_coach_diagnostic(
                    digest,
                    coach_llm_complete,
                    repair_requirements=repair_requirements,
                )
            )
            if len(collected_metrics) > metrics_before_call:
                call_metric = collected_metrics[-1]
                call_metric.call_label = (
                    f"coach_generation:{persona_id}:{digest.week_end}:"
                    f"attempt_{attempt}"
                )
                persona_call_metrics.append(call_metric)
                diagnostic = diagnostic.model_copy(
                    update={"llm_call": call_metric}
                )
            diagnostic_path = _write_diagnostic(
                diagnostics_dir,
                digest_path.stem,
                diagnostic.model_dump(mode="json"),
            )
            diagnostic_paths.append(diagnostic_path)
            if diagnostic.accepted:
                accepted_prompt = attempt_prompt
                break
            if diagnostic.failure_stage != "coach_validation" or attempt == 2:
                break
            repair_requirements = diagnostic.failure_details
            print(
                f"[retry] {persona_id} after {diagnostic.failure_stage}; "
                f"diagnostic={diagnostic_path}",
                flush=True,
            )
        if (
            not diagnostic.accepted
            or diagnostic.narrative is None
            or diagnostic.validation is None
        ):
            detail = "; ".join(diagnostic.failure_details) or "No details."
            failures.append(
                f"{persona_id}: {diagnostic.failure_stage or 'unknown'}: {detail}"
            )
            print(
                f"[error] {failures[-1]} diagnostic={diagnostic_path}",
                flush=True,
            )
            continue

        enriched = attach_coach_artifacts(
            digest,
            diagnostic.narrative,
            diagnostic.validation,
        )
        response_hash = _canonical_sha256(
            diagnostic.narrative.model_dump(mode="json")
        )
        provenance: dict[str, object] = {
            "weekly_drift_output_path": str(digest_path),
            "weekly_drift_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "coach_prompt_name": str(prompt_metadata["name"]),
            "coach_prompt_version": str(prompt_metadata["version"]),
            "coach_prompt_sha256": hashlib.sha256(
                accepted_prompt.encode()
            ).hexdigest(),
            "coach_response_sha256": response_hash,
            "coach_prompt": accepted_prompt,
            "coach_raw_output": diagnostic.raw_output,
            "coach_attempt_count": len(diagnostic_paths),
            "coach_diagnostic_paths": [str(path) for path in diagnostic_paths],
            "coach_call_metrics": [
                metric.model_dump(mode="json") for metric in persona_call_metrics
            ],
            **_model_contract(),
        }
        if source_provenance_by_persona is not None:
            provenance["scenario_source"] = source_provenance_by_persona[
                persona_id
            ]
        pending.append((enriched, digest_path, accepted_prompt, provenance))
        print(f"[accepted] {persona_id} diagnostic={diagnostic_path}", flush=True)

    if failures:
        raise RuntimeError(
            "Coach Digest-only generation did not produce all responses:\n"
            + "\n".join(failures)
        )

    manifest: list[dict] = []
    for digest, digest_path, accepted_prompt, provenance in pending:
        stored_output = json.dumps(digest.model_dump(mode="json"), indent=2) + "\n"
        digest_path.write_text(stored_output)
        provenance["weekly_drift_output_sha256"] = hashlib.sha256(
            stored_output.encode()
        ).hexdigest()
        digest_path.with_suffix(".md").write_text(render_digest_markdown(digest))
        digest_path.with_suffix(".prompt.txt").write_text(accepted_prompt)
        persist_weekly_digest_record(digest, parquet_path)
        entry = _digest_to_manifest_entry(digest, provenance=provenance)
        if entry is None:
            raise RuntimeError(
                f"Accepted response was missing for {digest.persona_id}."
            )
        manifest.append(entry)
    return manifest


async def _generate(
    personas: list[str],
    parquet_path: Path,
) -> list[dict]:
    coach_llm_complete = build_llm_complete()
    if coach_llm_complete is None:
        raise SystemExit(
            "No Coach Digest provider is available because the API key is missing. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY."
        )

    manifest: list[dict] = []
    for persona_id in personas:
        print(f"[generate] {persona_id} ...", flush=True)
        digest, artifacts = await run_weekly_drift_coach_cycle(
            persona_id=persona_id,
            parquet_path=parquet_path,
            coach_llm_complete=coach_llm_complete,
        )
        entry = _digest_to_manifest_entry(digest)
        if entry is None:
            diagnostic_path = artifacts.get("coach_diagnostic_path")
            diagnostic = (
                json.loads(Path(diagnostic_path).read_text())
                if diagnostic_path is not None
                else {}
            )
            failure_stage = diagnostic.get("failure_stage", "unknown")
            failure_details = "; ".join(diagnostic.get("failure_details") or [])
            print(
                f"[warn] {persona_id}: no narrative produced "
                f"(failure_stage={failure_stage}; {failure_details}); "
                f"diagnostic={diagnostic_path}; skipping from manifest.",
                flush=True,
            )
            continue
        manifest.append(entry)
        print(
            f"[done] {persona_id} {digest.week_start}->{digest.week_end} "
            f"mode={digest.response_mode} drift_states={digest.drift_states}",
            flush=True,
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--personas",
        nargs="+",
        required=True,
        help="Explicit Persona IDs to regenerate. The runner has no default roster.",
    )
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--weekly-drift-output-dir",
        type=Path,
        default=DEFAULT_WEEKLY_DRIFT_OUTPUT_DIR,
    )
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-weekly-drift-output",
        action="store_true",
        help=(
            "Generate only Coach Digest responses from stored Weekly Drift "
            "Detection output. Do not call the Weekly Drift Reviewer."
        ),
    )
    reuse_group.add_argument(
        "--reuse-scenario-key-weeks",
        action="store_true",
        help=(
            "Generate from the exact key-week weekly_digest_built output in each "
            "public scenario bundle, rebuild the bundles, and build the manifest "
            "from the displayed responses. Make no Weekly Drift Reviewer calls."
        ),
    )
    parser.add_argument(
        "--response-fixture-out",
        type=Path,
        default=DEFAULT_RESPONSE_FIXTURE,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize paid Weekly Drift Reviewer and Coach Digest calls.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = _build_parser().parse_args()

    if not args.execute:
        if args.reuse_scenario_key_weeks:
            print(
                "[dry run] Would generate one Coach Digest response for each of "
                f"{len(args.personas)} deployed Persona key weeks. This reads "
                "stored public scenario Weekly Drift Detection outputs and makes "
                "no Weekly Drift Reviewer calls. Re-run with --execute to make "
                "the paid Coach Digest calls."
            )
            return 0
        if args.reuse_weekly_drift_output:
            print(
                "[dry run] Would generate one Coach Digest response for each of "
                f"{len(args.personas)} stored Weekly Drift Detection outputs. "
                "This makes no Weekly Drift Reviewer calls. Re-run with "
                "--execute to make the paid Coach Digest calls."
            )
            return 0
        print(
            "[dry run] Would run the approved Coach Digest cycle for "
            f"{len(args.personas)} persona(s): {', '.join(args.personas)}.\n"
            "This makes paid Weekly Drift Reviewer calls (one per week of "
            "history per persona) and one Coach Digest response call per "
            "persona, and overwrites their rows in\n"
            f"  {args.parquet_path}\n"
            f"then rebuilds the judge manifest at\n  {args.manifest_out}\n"
            "Re-run with --execute to make the calls."
        )
        return 0

    if args.reuse_scenario_key_weeks:
        required_personas = {selection.persona_id for selection in SELECTIONS}
        if set(args.personas) != required_personas or len(args.personas) != 5:
            raise SystemExit(
                "--reuse-scenario-key-weeks requires the five deployed Persona IDs."
            )
        sample_directory = args.manifest_out.parent
        response_directory = sample_directory / "generated_responses"
        diagnostic_directory = sample_directory / "generation_diagnostics"
        source_provenance = _extract_scenario_key_week_digests(
            args.personas,
            response_directory,
        )
        generated_manifest = asyncio.run(
            _generate_reusing_weekly_drift(
                args.personas,
                args.parquet_path,
                response_directory,
                diagnostic_output_dir=diagnostic_directory,
                source_provenance_by_persona=source_provenance,
            )
        )
        _write_scenario_response_fixture(
            generated_manifest,
            _REPO_ROOT / args.response_fixture_out,
        )
        export_scenarios(_REPO_ROOT)
        manifest = _manifest_from_public_scenarios(
            args.personas,
            response_fixture_path=args.response_fixture_out,
        )
    elif args.reuse_weekly_drift_output:
        manifest = asyncio.run(
            _generate_reusing_weekly_drift(
                args.personas,
                args.parquet_path,
                args.weekly_drift_output_dir,
            )
        )
    else:
        manifest = asyncio.run(_generate(args.personas, args.parquet_path))
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.reuse_scenario_key_weeks:
        command = (
            ".venv/bin/python scripts/coach/generate_approved_judge_sample.py "
            f"--personas {' '.join(args.personas)} "
            "--reuse-scenario-key-weeks --execute"
        )
        _write_generation_report(
            manifest,
            args.manifest_out.parent,
            command=command,
        )
    print(
        f"\nWrote {len(manifest)} approved-path narrative(s) to "
        f"{args.manifest_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
