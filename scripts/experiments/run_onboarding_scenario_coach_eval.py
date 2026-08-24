"""Generate Coach Digest responses for the onboarding app scenario Personas.

`run_coach_drift_control_eval.py` selects targets from the Drift episodes
Parquet. The onboarding app ships a fixed set of scenarios instead, one of which
(`stable-meera`) has no Drift episode at all, so episode selection cannot
produce a week for it.

This runner takes the scenario roster directly. For each scenario it:

  1. reads the confirmed Profile that the scenario ships, so Core Values and
     goal context match what the onboarding app presents, rather than the
     synthetic persona defaults,
  2. reports on the last week of the Persona history, which is the
     current-state snapshot the scenarios describe,
  3. records the Coach Narrative and its validation verdict even when
     validation fails, so the failure rate stays measurable.

Run:
    .venv/bin/python scripts/experiments/run_onboarding_scenario_coach_eval.py \
        --execute
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Allow running as a bare script by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.coach.llm_client import build_llm_complete, resolve_coach_model  # noqa: E402
from src.coach.schemas import WeeklyDigest  # noqa: E402
from src.coach.weekly_drift_runtime import run_weekly_drift_coach_cycle  # noqa: E402

DEFAULT_SCENARIO_DIR = Path("frontend/onboarding/public/scenarios")
DEFAULT_REPORT_DIR = Path(
    "logs/experiments/reports/coach_narrative_onboarding_20260824"
)
DEFAULT_PARQUET = Path(
    "logs/exports/weekly_digests/coach_onboarding_eval_20260824.parquet"
)
DEFAULT_OUTPUT_DIR = Path("logs/exports/weekly_drift_coach/onboarding_20260824")


@dataclass(frozen=True)
class ScenarioTarget:
    """One onboarding scenario to generate a Coach Digest response for."""

    scenario_id: str
    persona_id: str
    title: str
    profile_path: str
    top_values: list[str]


def load_scenario_targets(
    scenario_dir: Path, profile_dir: Path
) -> list[ScenarioTarget]:
    """Read the onboarding scenario roster and its confirmed Profiles."""
    targets: list[ScenarioTarget] = []
    for path in sorted(scenario_dir.glob("*.json")):
        if path.stem == "index":
            continue
        scenario = json.loads(path.read_text()).get("scenario") or {}
        scenario_id = scenario.get("scenario_id")
        profile = scenario.get("profile") or {}
        if not scenario_id or not scenario.get("persona_id"):
            continue
        profile_path = profile_dir / f"{scenario_id}.profile.json"
        if not profile_path.exists():
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps(profile, indent=2) + "\n")
        targets.append(
            ScenarioTarget(
                scenario_id=scenario_id,
                persona_id=scenario["persona_id"],
                title=scenario.get("title", ""),
                profile_path=str(profile_path),
                top_values=list(profile.get("top_values") or []),
            )
        )
    return targets


def _digest_to_manifest_entry(
    digest: WeeklyDigest,
    target: ScenarioTarget,
    generator_model: str,
) -> dict[str, Any] | None:
    """Build a judge manifest entry that also names its onboarding scenario."""
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
            "goal_context": digest.goal_context,
            "drift_states": digest.drift_states,
            "drift_reasons": digest.drift_reasons,
            "top_tensions": digest.top_tensions,
            "top_strengths": digest.top_strengths,
            # state_comparisons feeds the state_claims validation check.
            "state_comparisons": [row.model_dump() for row in digest.state_comparisons],
            "dimensions": [row.model_dump() for row in digest.dimensions],
            "evidence": [row.model_dump() for row in digest.evidence],
        },
        "narrative": digest.coach_narrative.model_dump(),
        "validation": (
            digest.validation.model_dump() if digest.validation is not None else None
        ),
        "generator_model": generator_model,
        "target": {**asdict(target), "arm": "onboarding_scenario"},
    }


async def _generate(
    targets: list[ScenarioTarget],
    *,
    parquet_path: Path,
    output_dir: Path,
    wrangled_dir: Path,
) -> list[dict[str, Any]]:
    coach_llm_complete = build_llm_complete()
    if coach_llm_complete is None:
        raise SystemExit(
            "No Weekly Coach provider available (missing API key). "
            "Set OPENAI_API_KEY or GEMINI_API_KEY."
        )
    generator_model = resolve_coach_model()

    manifest: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        print(
            f"[{index}/{len(targets)}] {target.scenario_id} "
            f"({target.persona_id}) ...",
            flush=True,
        )
        digest, _artifacts = await run_weekly_drift_coach_cycle(
            persona_id=target.persona_id,
            wrangled_dir=wrangled_dir,
            output_dir=output_dir,
            parquet_path=parquet_path,
            profile_path=target.profile_path,
            coach_llm_complete=coach_llm_complete,
            attach_failed_validation=True,
        )
        entry = _digest_to_manifest_entry(digest, target, generator_model)
        if entry is None:
            print(
                f"[warn] {target.scenario_id}: no Coach Narrative produced; "
                "excluded from the manifest.",
                flush=True,
            )
            continue
        manifest.append(entry)
        passed = digest.validation.all_passed if digest.validation else None
        print(
            f"[done] {target.scenario_id} {digest.week_start}->{digest.week_end} "
            f"mode={digest.response_mode} core_values={digest.core_values} "
            f"validation_passed={passed}",
            flush=True,
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Coach Digest responses for the onboarding app scenario "
            "Personas."
        )
    )
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--profile-dir", type=Path, default=DEFAULT_REPORT_DIR / "profiles"
    )
    parser.add_argument("--wrangled-dir", type=Path, default=Path("logs/wrangled"))
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_REPORT_DIR / "judge_sample_manifest.json",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize the paid Weekly Drift Reviewer and Coach calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    targets = load_scenario_targets(args.scenario_dir, args.profile_dir)

    print(f"Onboarding scenarios: {len(targets)}")
    for target in targets:
        print(
            f"  {target.scenario_id:<18} {target.persona_id} "
            f"top_values={target.top_values}"
        )
    print("Reports on the last week of each Persona history.")

    if not args.execute:
        print("\n[dry run] No calls made. Re-run with --execute.")
        return 0

    manifest = asyncio.run(
        _generate(
            targets,
            parquet_path=args.parquet_path,
            output_dir=args.output_dir,
            wrangled_dir=args.wrangled_dir,
        )
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(manifest)} manifest entries: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
