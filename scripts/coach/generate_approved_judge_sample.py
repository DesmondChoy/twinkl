"""Generate approved-path Weekly Coach narratives and build a judge sample.

Runs the approved Weekly Drift Reviewer + Drift Detector cycle
(``run_weekly_drift_coach_cycle``) for a fixed roster of personas, so the
Coach Digest Evals score responses as the product's approved path
actually produces them — not leftover ``vif_runtime`` demo-tool outputs.

For each persona this:
  1. runs the Weekly Drift Reviewer over every week of wrangled history
     (paid OpenAI calls, one per week),
  2. detects Drift and builds the approved Weekly Digest,
  3. generates the Weekly Coach narrative via the configured provider
     (paid call; provider from ``TWINKL_COACH_PROVIDER``, defaults to openai),
  4. upserts the digest+narrative into the Weekly Digest parquet.

Then it rebuilds the judge sample manifest from the freshly written narratives.

Paid calls are gated behind ``--execute``. Without it, this prints the plan and
makes no calls.

Run:
    .venv/bin/python scripts/coach/generate_approved_judge_sample.py --execute
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as a bare script (`python scripts/coach/...`) by putting the
# repo root on sys.path before importing the `src` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.coach.llm_client import build_llm_complete  # noqa: E402
from src.coach.schemas import WeeklyDigest  # noqa: E402
from src.coach.weekly_drift_runtime import run_weekly_drift_coach_cycle  # noqa: E402

# The fixed roster: the personas that previously had demo-tool narratives.
DEFAULT_PERSONAS = [
    "7cc5cf92",
    "0ad04582",
    "20730018",
    "b7b942ab",
    "61d7d490",
]

DEFAULT_PARQUET = Path("logs/exports/weekly_digests/weekly_digests.parquet")
DEFAULT_MANIFEST = Path(
    "logs/experiments/reports/coach_digest_validations_20260727/"
    "judge_sample_manifest.json"
)


def _digest_to_manifest_entry(digest: WeeklyDigest) -> dict | None:
    """Convert a generated digest into a judge-manifest {digest, narrative} entry."""
    if digest.coach_narrative is None:
        return None
    return {
        "digest": {
            "persona_id": digest.persona_id,
            "persona_name": digest.persona_name,
            "week_start": digest.week_start,
            "week_end": digest.week_end,
            "response_mode": digest.response_mode,
            "mode_source": digest.mode_source,
            "mode_rationale": digest.mode_rationale,
            "signal_source": digest.signal_source,
            "n_entries": digest.n_entries,
            "overall_mean": digest.overall_mean,
            "overall_uncertainty": digest.overall_uncertainty,
            "core_values": digest.core_values,
            "drift_states": digest.drift_states,
            "drift_reasons": digest.drift_reasons,
            "top_tensions": digest.top_tensions,
            "top_strengths": digest.top_strengths,
            "dimensions": [row.model_dump() for row in digest.dimensions],
            "evidence": [row.model_dump() for row in digest.evidence],
        },
        "narrative": digest.coach_narrative.model_dump(),
    }


async def _generate(
    personas: list[str],
    parquet_path: Path,
) -> list[dict]:
    coach_llm_complete = build_llm_complete()
    if coach_llm_complete is None:
        raise SystemExit(
            "No Weekly Coach provider available (missing API key). "
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
            print(
                f"[warn] {persona_id}: no narrative produced "
                "(coach LLM returned nothing); skipping from manifest.",
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
        default=DEFAULT_PERSONAS,
        help="Persona IDs to regenerate (default: the demo-tool roster).",
    )
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize paid Weekly Drift Reviewer and Weekly Coach calls.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = _build_parser().parse_args()

    if not args.execute:
        print(
            "[dry run] Would run the approved Weekly Coach cycle for "
            f"{len(args.personas)} persona(s): {', '.join(args.personas)}.\n"
            "This makes paid Weekly Drift Reviewer calls (one per week of "
            "history per persona) and one Weekly Coach narrative call per "
            "persona, and overwrites their rows in\n"
            f"  {args.parquet_path}\n"
            f"then rebuilds the judge manifest at\n  {args.manifest_out}\n"
            "Re-run with --execute to make the calls."
        )
        return 0

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
