"""Thin wrapper around the canonical frontier uncertainty review module."""

import sys
from pathlib import Path

_dir = Path.cwd().resolve()
while _dir != _dir.parent:
    if (_dir / "src").is_dir() and (_dir / "pyproject.toml").is_file():
        sys.path.insert(0, str(_dir))
        break
    _dir = _dir.parent

from src.vif.frontier_uncertainty import main


if __name__ == "__main__":
    main()
