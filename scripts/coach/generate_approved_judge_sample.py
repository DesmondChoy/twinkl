"""Generate approved-path Coach Digest responses and build an evaluation sample.

Runs the approved Weekly Drift Reviewer + Drift Detector cycle
(``run_weekly_drift_coach_cycle``) for an explicit roster of personas, so the
Coach Digest Evals score responses as the product's approved path
actually produces them — not leftover ``vif_runtime`` demo-tool outputs.

For each persona this:
  1. runs the Weekly Drift Reviewer over every week of wrangled history
     (paid OpenAI calls, one per week),
  2. detects Drift and builds the approved Weekly Drift Detection output,
  3. generates the Coach Digest response via the configured provider
     (paid call; provider from ``TWINKL_COACH_PROVIDER``, defaults to openai),
  4. upserts the output and response into the consolidated parquet.

Then it rebuilds the judge sample manifest from the freshly written narratives.

Paid calls are gated behind ``--execute``. Without it, this prints the plan and
makes no calls.

Run:
    .venv/bin/python scripts/coach/generate_approved_judge_sample.py \
        --personas 11de77e8 --execute
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
)
from src.coach.schemas import LLMCallMetrics, WeeklyDigest  # noqa: E402
from src.coach.weekly_digest import (  # noqa: E402
    LLMCompleteFn,
    attach_coach_artifacts,
    generate_weekly_digest_coach_diagnostic,
    persist_weekly_digest_record,
    render_digest_markdown,
)
from src.coach.weekly_drift_runtime import run_weekly_drift_coach_cycle  # noqa: E402

DEFAULT_PARQUET = Path("logs/exports/weekly_digests/weekly_digests.parquet")
DEFAULT_WEEKLY_DRIFT_OUTPUT_DIR = Path("logs/exports/weekly_drift_coach")
DEFAULT_MANIFEST = Path(
    "logs/experiments/reports/coach_digest_sample_20260824/"
    "judge_sample_manifest.json"
)


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


async def _generate_reusing_weekly_drift(
    personas: list[str],
    parquet_path: Path,
    output_dir: Path,
    *,
    llm_complete: LLMCompleteFn | None = None,
    call_metrics: list[LLMCallMetrics] | None = None,
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
                output_dir,
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
        response_bytes = json.dumps(
            diagnostic.narrative.model_dump(mode="json"),
            sort_keys=True,
        ).encode()
        provenance = {
            "weekly_drift_output_path": str(digest_path),
            "weekly_drift_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "coach_prompt_name": str(prompt_metadata["name"]),
            "coach_prompt_version": str(prompt_metadata["version"]),
            "coach_prompt_sha256": hashlib.sha256(
                accepted_prompt.encode()
            ).hexdigest(),
            "coach_response_sha256": hashlib.sha256(response_bytes).hexdigest(),
            "coach_attempt_count": len(diagnostic_paths),
            "coach_diagnostic_paths": [str(path) for path in diagnostic_paths],
            "coach_call_metrics": [
                metric.model_dump(mode="json") for metric in persona_call_metrics
            ],
            **_model_contract(),
        }
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
    parser.add_argument(
        "--reuse-weekly-drift-output",
        action="store_true",
        help=(
            "Generate only Coach Digest responses from stored Weekly Drift "
            "Detection output. Do not call the Weekly Drift Reviewer."
        ),
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

    if args.reuse_weekly_drift_output:
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
    print(
        f"\nWrote {len(manifest)} approved-path narrative(s) to "
        f"{args.manifest_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
