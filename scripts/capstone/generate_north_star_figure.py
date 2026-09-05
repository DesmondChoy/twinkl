"""Render the NSM feasibility figure from preserved development results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capstone.generate_report_figures import (  # noqa: E402
    BLUE,
    CORAL,
    GRID,
    MUTED,
    TEAL,
    configure_matplotlib,
    save_figure,
)


def main() -> None:
    directory = ROOT / "logs/experiments/reports"
    paths = [
        directory / "north_star_phase0_20260905/retrieval.json",
        directory / "north_star_phase0b_20260905/report.json",
    ]
    retrieval = json.loads(paths[0].read_text())["retrieval"]
    review = json.loads(paths[1].read_text())["summary"]
    proxy = retrieval["metrics"]["persisted_positive"]
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    panels = [
        (
            axes[0],
            ["k = 1", "k = 3", "k = 5"],
            [proxy[str(k)] for k in (1, 3, 5)],
            [BLUE, TEAL, BLUE],
            0.9,
            "A  Persisted-label proxy recall",
            "90% retrieval gate",
        ),
        (
            axes[1],
            ["Quotation\nprecision", "Correct\nomission"],
            [review["precision"], review["correct_no_card"]],
            [CORAL, CORAL],
            1.0,
            "B  Independent AI reference review",
            "Zero-error criteria",
        ),
    ]
    for axis, labels, metrics, colors, threshold, title, gate_label in panels:
        bars = axis.bar(labels, [m["rate"] for m in metrics], color=colors, width=0.55)
        axis.set_ylim(0, 1.2)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(title, loc="left", fontsize=12, pad=15, weight="bold")
        axis.axhline(threshold, color=MUTED, linestyle="--", linewidth=1)
        axis.text(0.98, 1.075, gate_label,
                  transform=axis.get_yaxis_transform(), ha="right", fontsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, linewidth=0.6)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for bar, metric in zip(bars, metrics, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                metric["rate"] - 0.14,
                f"{metric['numerator']}/{metric['denominator']}",
                color="white", ha="center", weight="bold", fontsize=12,
            )
    fig.text(0.5, 0.01,
             "33 synthetic development episodes; denominators differ by task. "
             "No final-test or human-validation claim.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.5)
    save_figure(fig, "north-star-feasibility.png")
    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "inputs": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "output": "docs/capstone_report/images/north-star-feasibility.png",
        "scope": "AI-reviewed synthetic development evidence; Phase 0B failed",
    }
    (ROOT / "docs/capstone_report/images/north-star-feasibility.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
