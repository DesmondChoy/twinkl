"""Run saved research assertions with their recorded source versions.

Temporary archives do not change the working tree. Saved artifacts remain the
current files so their original receipt checks still detect tampering.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path


def source_snapshot(root: Path, revision: str, destination: Path) -> Path:
    """Materialize source/configuration at an immutable Git revision."""
    archive = subprocess.check_output(
        ["git", "archive", revision, "config", "prompts", "scripts", "src"],
        cwd=root,
    )
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(destination, filter="data")
    for name in ("logs", "docs", "tests"):
        (destination / name).symlink_to(root / name, target_is_directory=True)
    return destination


def assert_in_snapshot(snapshot: Path, test_file: Path, assertion: str) -> None:
    """Import frozen application modules in a separate interpreter."""
    code = """
import runpy
import sys
from pathlib import Path

namespace = runpy.run_path(sys.argv[1])
check = namespace[sys.argv[2]]
check.__globals__.update(ROOT=Path.cwd(), REPO_ROOT=Path.cwd())
check()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(test_file), assertion],
        cwd=snapshot,
        env={
            **os.environ,
            "PYTHONPATH": str(snapshot),
            # Historical validators use git show to verify earlier source blobs.
            "GIT_DIR": str(Path(__file__).resolve().parents[1] / ".git"),
            "GIT_WORK_TREE": str(snapshot),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
