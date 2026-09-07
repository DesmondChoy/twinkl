"""Export completed v4 Run 1 Weekly Drift and NSM results without provider calls."""

from pathlib import Path

from src.demo.north_star_replay import export_records
from src.demo.scenarios import export_scenarios


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    export_records(root)
    export_scenarios(root)


if __name__ == "__main__":
    main()
