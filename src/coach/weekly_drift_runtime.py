"""Approved Weekly Drift Reviewer to Weekly Coach runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.coach.schemas import WeeklyDigest
from src.coach.weekly_digest import (
    _default_output_stem,
    attach_coach_artifacts,
    build_weekly_drift_reviewer_digest,
    generate_weekly_digest_coach,
    persist_weekly_digest_record,
    render_digest_markdown,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)
from src.drift_detector import detect_drift
from src.vif.state_encoder import concatenate_entry_text
from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    WeeklyDriftReviewerEntry,
    WeeklyDriftReviewerFn,
    build_weekly_drift_reviewer_request,
    persist_weekly_drift_reviewer_receipt,
)
from src.wrangling.parse_wrangled_data import parse_wrangled_file


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _week_bounds(raw: str) -> tuple[date, date]:
    entry_date = _parse_date(raw)
    start = entry_date - timedelta(days=entry_date.weekday())
    return start, start + timedelta(days=6)


def _normalize_core_value(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _load_runtime_input(
    *,
    persona_id: str,
    wrangled_dir: Path,
    end_date: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wrangled_path = wrangled_dir / f"persona_{persona_id}.md"
    if not wrangled_path.exists():
        raise FileNotFoundError(f"Wrangled file not found: {wrangled_path}")
    profile, entries, _warnings = parse_wrangled_file(wrangled_path)
    if end_date is not None:
        cutoff = _parse_date(end_date)
        entries = [row for row in entries if _parse_date(str(row["date"])) <= cutoff]
    entries.sort(key=lambda row: int(row["t_index"]))
    if not entries:
        raise ValueError(f"No Journal Entries available for persona_id={persona_id}")
    return profile, entries


async def run_weekly_drift_coach_cycle(
    *,
    persona_id: str,
    wrangled_dir: str | Path = "logs/wrangled",
    output_dir: str | Path = "logs/exports/weekly_drift_coach",
    parquet_path: str | Path = "logs/exports/weekly_digests/weekly_digests.parquet",
    end_date: str | None = None,
    reviewer: WeeklyDriftReviewerFn | None = None,
    coach_llm_complete=None,
) -> tuple[WeeklyDigest, dict[str, str]]:
    """Run the approved weekly-only path through the Weekly Digest."""
    wrangled_path = Path(wrangled_dir)
    output_path = Path(output_dir)
    profile, entries = _load_runtime_input(
        persona_id=persona_id,
        wrangled_dir=wrangled_path,
        end_date=end_date,
    )
    core_values = list(
        dict.fromkeys(
            _normalize_core_value(value)
            for value in profile.get("core_values") or []
            if isinstance(value, str) and value.strip()
        )
    )
    if not core_values:
        raise ValueError(f"No Core Values available for persona_id={persona_id}")

    reviewer_call = reviewer or OpenAIWeeklyDriftReviewer()
    entries_by_week: dict[tuple[date, date], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        entries_by_week[_week_bounds(str(entry["date"]))].append(entry)

    receipts = []
    receipt_paths: list[Path] = []
    for week_start, week_end in sorted(entries_by_week):
        history_rows = [
            entry for entry in entries if _parse_date(str(entry["date"])) <= week_end
        ]
        history = [
            WeeklyDriftReviewerEntry(
                t_index=int(entry["t_index"]),
                date=str(entry["date"]),
                text=concatenate_entry_text(
                    entry.get("initial_entry"),
                    entry.get("nudge_text"),
                    entry.get("response_text"),
                ),
            )
            for entry in history_rows
        ]
        request = build_weekly_drift_reviewer_request(
            persona_id=persona_id,
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            core_values=core_values,
            history=history,
            current_t_indices=[
                int(entry["t_index"])
                for entry in entries_by_week[(week_start, week_end)]
            ],
        )
        receipt = await reviewer_call(request)
        receipts.append(receipt)
        receipt_path = output_path / (
            f"{persona_id}_{week_end.isoformat()}.weekly_drift_review.json"
        )
        persist_weekly_drift_reviewer_receipt(receipt, receipt_path)
        receipt_paths.append(receipt_path)

    decisions = [decision for receipt in receipts for decision in receipt.decisions]
    drift_result = detect_drift(decisions, persona_id=persona_id)
    target_receipt = receipts[-1]
    digest = build_weekly_drift_reviewer_digest(
        persona_id=persona_id,
        wrangled_dir=wrangled_path,
        week_start=target_receipt.week_start,
        week_end=drift_result.cutoff_date,
        core_values=core_values,
        decisions=decisions,
        drift_result=drift_result,
    )

    if coach_llm_complete is not None:
        narrative, _prompt = await generate_weekly_digest_coach(
            digest,
            coach_llm_complete,
        )
        validation = (
            validate_weekly_digest_narrative(digest, narrative)
            if narrative is not None
            else None
        )
        digest = attach_coach_artifacts(digest, narrative, validation)

    output_path.mkdir(parents=True, exist_ok=True)
    stem = _default_output_stem(digest)
    drift_path = output_path / f"{stem}.drift.json"
    digest_path = output_path / f"{stem}.json"
    markdown_path = output_path / f"{stem}.md"
    prompt_path = output_path / f"{stem}.prompt.txt"
    drift_path.write_text(
        json.dumps(drift_result.model_dump(mode="json"), indent=2) + "\n"
    )
    digest_path.write_text(json.dumps(digest.model_dump(mode="json"), indent=2) + "\n")
    markdown_path.write_text(render_digest_markdown(digest))
    prompt_path.write_text(render_digest_prompt(digest))
    persist_weekly_digest_record(digest, Path(parquet_path))

    artifacts = {
        "drift_json_path": str(drift_path),
        "digest_json_path": str(digest_path),
        "digest_md_path": str(markdown_path),
        "prompt_path": str(prompt_path),
        "parquet_path": str(parquet_path),
    }
    for index, receipt_path in enumerate(receipt_paths, start=1):
        artifacts[f"review_receipt_{index}_path"] = str(receipt_path)
    return digest, artifacts


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the approved Weekly Drift Reviewer and Drift Detector path."
    )
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--wrangled-dir", default="logs/wrangled")
    parser.add_argument("--output-dir", default="logs/exports/weekly_drift_coach")
    parser.add_argument(
        "--parquet-path",
        default="logs/exports/weekly_digests/weekly_digests.parquet",
    )
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize the paid Weekly Drift Reviewer calls.",
    )
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _build_cli_parser().parse_args()
    if not args.execute:
        raise SystemExit("Refusing paid model calls without --execute")
    digest, artifact_paths = asyncio.run(
        run_weekly_drift_coach_cycle(
            persona_id=args.persona_id,
            wrangled_dir=args.wrangled_dir,
            output_dir=args.output_dir,
            parquet_path=args.parquet_path,
            end_date=args.end_date,
        )
    )
    print(
        f"Built approved Weekly Digest for {digest.persona_id} "
        f"({digest.week_start} -> {digest.week_end})"
    )
    for name, path in artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
