"""Verify or re-score the frozen GPT-5.6/GPT-6 Luna comparisons offline.

The live Reviewer model changed after these experiments. Check its exact
historical source from the manifest revision, then allow only that one model
constant change while the frozen experiment runners verify everything else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.experiments import compare_twinkl_gv3x_luna_higher as higher
from scripts.experiments import compare_twinkl_q6pt_luna as initial
from scripts.experiments import weekly_drift_definitions as frozen

ROOT = Path(__file__).resolve().parents[2]
REVIEWER_SOURCE = "src/weekly_drift_reviewer.py"
OLD_MODEL = b'WEEKLY_DRIFT_REVIEWER_MODEL = "gpt-5.6-luna"'
NEW_MODEL = b'WEEKLY_DRIFT_REVIEWER_MODEL = "gpt-6-luna"'
RUNNERS = {"q6pt": initial, "gv3x": higher}


def replay(study: str, command: str, config_path: Path | None = None) -> dict:
    runner = RUNNERS[study]
    config_path = config_path or runner.DEFAULT_CONFIG
    config = runner._load_config(config_path)
    paths = runner._paths(config)
    manifest = json.loads(paths["manifest"].read_text())
    revision = manifest["repo_head"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Manifest has no valid source revision")
    original = subprocess.run(
        ["git", "show", f"{revision}:{REVIEWER_SOURCE}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    original_hash = hashlib.sha256(original).hexdigest()
    if original_hash != manifest["code_sha256"][REVIEWER_SOURCE]:
        raise ValueError("Historical Reviewer source does not match the manifest")
    live_source = (ROOT / REVIEWER_SOURCE).read_bytes()
    if original.count(OLD_MODEL) != 1 or live_source not in (
        original,
        original.replace(OLD_MODEL, NEW_MODEL),
    ):
        raise ValueError("Live Reviewer differs beyond the adopted model change")

    current_file_hash = frozen.file_hash

    def recorded_file_hash(path: Path) -> str:
        if path.resolve() == (ROOT / REVIEWER_SOURCE).resolve():
            return original_hash
        return current_file_hash(path)

    with patch.object(frozen, "file_hash", recorded_file_hash):
        if command == "verify":
            checked, _rows, _config, _paths = runner.verify(config_path)
            return {
                "status": "verified",
                "study_id": checked["study_id"],
                "planned_calls": checked["expected_terminal_requests"],
                "source_revision": revision,
            }
        metrics = runner.score(config_path)
        return {"status": "scored", "study_id": metrics["study_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=RUNNERS)
    parser.add_argument("command", choices=("verify", "score"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.study, args.command, args.config), indent=2))


if __name__ == "__main__":
    main()
