"""Generate the capstone paper figures from committed evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "capstone_report" / "images"

INK = "#14233B"
MUTED = "#637083"
TEAL = "#248D83"
BLUE = "#4F6FE8"
GOLD = "#D8A62A"
CORAL = "#C05A40"
PAPER = "#FBFAF6"
GRID = "#D8DDE5"


def configure_matplotlib() -> None:
    """Set one print-safe style for all charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure as a high-resolution PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def parse_float(value: str) -> float:
    """Parse a decimal value from one Markdown table cell."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace("`", ""))
    if match is None:
        raise ValueError(f"No number in {value!r}")
    return float(match.group())


def markdown_rows(path: Path, header_start: str) -> list[list[str]]:
    """Read the first Markdown table below a selected header row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip().startswith(header_start)
    )
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one committed YAML evidence file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def draw_synthetic_data_lifts() -> None:
    """Show corpus growth and the bounded effects of two targeted data lifts."""
    first_snapshot = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_681_5_baseline_snapshot.yaml"
    )
    second_snapshot = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_691_2_baseline_snapshot.yaml"
    )
    second_config = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_691_2.yaml"
    )
    reference_run = load_yaml(
        ROOT / "logs" / "experiments" / "runs" / "run_019_BalancedSoftmax.yaml"
    )
    first_report = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-03-08_twinkl_681_5.md"
    )
    second_report = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-03-09_twinkl_691_3.md"
    )

    baseline_entries = sum(reference_run["data"].values())
    first_text = first_report.read_text(encoding="utf-8")
    first_added_match = re.search(r"(\d+) new entries", first_text)
    if first_added_match is None:
        raise ValueError("Could not read the first targeted-batch entry count")
    first_added_entries = int(first_added_match.group(1))
    final_entries = pl.read_parquet(
        ROOT / "logs" / "judge_labels" / "judge_labels.parquet"
    ).height

    personas = [
        int(first_snapshot["registry_persona_count"]),
        int(second_snapshot["registry_persona_count"]),
        int(second_snapshot["registry_persona_count"])
        + int(second_config["generation"]["num_personas"]),
    ]
    entries = [baseline_entries, baseline_entries + first_added_entries, final_entries]
    stages = [
        "Corrected-split\nbaseline",
        "Power/Security\nlift",
        "Hedonism/Security\nlift",
    ]

    first_rows = markdown_rows(first_report, "| Dimension | Baseline QWK")
    first_by_dimension = {row[0].replace("`", ""): row for row in first_rows}
    second_rows = markdown_rows(second_report, "| Family | `hedonism qwk`")
    second_by_family = {row[0]: row for row in second_rows}

    effect_labels = [
        "Power/Security batch\nPower Conflict recall",
        "Power/Security batch\nSecurity Conflict recall",
        "Hedonism/Security batch\nHedonism QWK",
        "Hedonism/Security batch\nSecurity QWK",
    ]
    before = [
        parse_float(first_by_dimension["power"][4]),
        parse_float(first_by_dimension["security"][4]),
        parse_float(second_by_family["Current default BalancedSoftmax"][1]),
        parse_float(second_by_family["Current default BalancedSoftmax"][4]),
    ]
    after = [
        parse_float(first_by_dimension["power"][5]),
        parse_float(first_by_dimension["security"][5]),
        parse_float(second_by_family["New BalancedSoftmax"][1]),
        parse_float(second_by_family["New BalancedSoftmax"][4]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), gridspec_kw={"wspace": 0.34})
    x = list(range(3))
    axes[0].plot(x, entries, color=TEAL, linewidth=2.5, zorder=2)
    axes[0].scatter(x, entries, s=100, color=TEAL, zorder=3)
    annotation_layout = [(18, "left"), (0, "center"), (-10, "right")]
    for index, (entry_count, persona_count) in enumerate(
        zip(entries, personas, strict=True)
    ):
        horizontal_offset, alignment = annotation_layout[index]
        axes[0].annotate(
            f"{entry_count:,} entries\n{persona_count} personas",
            (index, entry_count),
            xytext=(horizontal_offset, 14),
            textcoords="offset points",
            ha=alignment,
            weight="bold",
        )
    axes[0].set_xticks(x, stages)
    axes[0].set_ylim(1390, 1725)
    axes[0].set_ylabel("Persisted Journal Entries")
    axes[0].set_title("(a) Train-only corpus growth", fontsize=13, weight="bold")
    axes[0].yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)

    y = list(range(len(effect_labels)))
    for index, (start, end) in enumerate(zip(before, after, strict=True)):
        end_color = TEAL if end >= start else CORAL
        axes[1].plot([start, end], [index, index], color=GRID, linewidth=2.2, zorder=1)
        axes[1].scatter(start, index, s=66, color=MUTED, marker="o", zorder=2)
        axes[1].scatter(end, index, s=78, color=end_color, marker="D", zorder=3)
        if start == end:
            axes[1].annotate(
                f"{end:.3f} unchanged",
                (end, index),
                xytext=(8, 8),
                textcoords="offset points",
                ha="left",
                fontsize=8.5,
                color=end_color,
                weight="bold",
            )
            continue
        axes[1].annotate(
            f"{start:.3f}",
            (start, index),
            xytext=(-7, -14),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=MUTED,
        )
        axes[1].annotate(
            f"{end:.3f}",
            (end, index),
            xytext=(7, 8),
            textcoords="offset points",
            ha="left",
            fontsize=8.5,
            color=end_color,
            weight="bold",
        )
    axes[1].set_yticks(y, effect_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 0.66)
    axes[1].set_xlabel("Held-out metric value")
    axes[1].set_title(
        "(b) Selected target-dimension effects", fontsize=13, weight="bold"
    )
    axes[1].xaxis.grid(True, color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)

    for ax in axes:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.text(
        0.56,
        0.012,
        (
            "Circles show the pre-lift reference; diamonds show the post-lift "
            "result. Compare each row only within its named metric."
        ),
        color=MUTED,
        fontsize=9.7,
    )
    fig.subplots_adjust(bottom=0.20, top=0.90)
    save_figure(fig, "synthetic-data-lifts.png")


def vif_conflict_recall() -> tuple[list[str], list[float], list[int]]:
    """Compute per-value median Conflict recall across the three VIF seeds."""
    run_paths = [
        ROOT / "logs" / "experiments" / "runs" / f"run_{run}_BalancedSoftmax.yaml"
        for run in ("019", "020", "021")
    ]
    per_run: list[dict[str, float]] = []
    support_by_dimension: dict[str, int] = {}
    for run_path in run_paths:
        run = load_yaml(run_path)
        outputs = pl.read_parquet(ROOT / run["artifacts"]["test_outputs"])
        run_recalls: dict[str, float] = {}
        for dimension in outputs.get_column("dimension").unique().sort().to_list():
            rows = outputs.filter(pl.col("dimension") == dimension)
            support = rows.filter(pl.col("target") == -1).height
            true_positives = rows.filter(
                (pl.col("target") == -1) & (pl.col("predicted_class") == -1)
            ).height
            run_recalls[dimension] = true_positives / support
            support_by_dimension[dimension] = support
        per_run.append(run_recalls)

    dimensions = [
        "self_direction",
        "stimulation",
        "hedonism",
        "achievement",
        "power",
        "security",
        "conformity",
        "tradition",
        "benevolence",
        "universalism",
    ]
    median_recalls = [
        median(run[dimension] for run in per_run) for dimension in dimensions
    ]
    supports = [support_by_dimension[dimension] for dimension in dimensions]
    return dimensions, median_recalls, supports


def weekly_reviewer_conflict_recall() -> tuple[list[str], list[float], list[int]]:
    """Read per-value Luna-low entry Conflict recall from saved metrics."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "artifacts"
        / "twinkl_52zz_luna_low_20260714"
        / "metrics.json"
    )
    metrics = json.loads(source.read_text(encoding="utf-8"))
    results = metrics["models"]["luna_low"]["results"]
    dimensions = [
        "self_direction",
        "stimulation",
        "hedonism",
        "achievement",
        "power",
        "security",
        "conformity",
        "tradition",
        "benevolence",
        "universalism",
    ]
    recalls = [
        median(
            result["entry"]["per_dimension"][dimension]["recall"]
            for result in results
        )
        for dimension in dimensions
    ]
    supports = [
        int(results[0]["entry"]["per_dimension"][dimension]["negative_support"])
        for dimension in dimensions
    ]
    return dimensions, recalls, supports


def draw_per_value_conflict_recall() -> None:
    """Contrast per-value difficulty without implying a direct model comparison."""
    vif_dimensions, vif_recall, vif_support = vif_conflict_recall()
    weekly_dimensions, weekly_recall, weekly_support = weekly_reviewer_conflict_recall()
    if vif_dimensions != weekly_dimensions:
        raise ValueError("Per-value evidence uses inconsistent value order")

    labels = [dimension.replace("_", " ").title() for dimension in vif_dimensions]
    labels[0] = "Self-Direction"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 7.0),
        sharey=True,
        gridspec_kw={"wspace": 0.26},
    )
    y = list(range(len(labels)))
    panels = (
        (
            axes[0],
            vif_recall,
            vif_support,
            "(a) VIF Critic (Offline)",
            "LLM-Judge VIF Labels · frozen 221-entry test set",
            TEAL,
        ),
        (
            axes[1],
            weekly_recall,
            weekly_support,
            "(b) Weekly Drift Reviewer · Luna-low",
            "LLM-Judge Conflict Labels · complete development data",
            BLUE,
        ),
    )
    for ax, values, supports, title, subtitle, color in panels:
        ax.hlines(y, 0, values, color=GRID, linewidth=2.0, zorder=1)
        ax.scatter(values, y, s=82, color=color, zorder=3)
        for index, (value, support) in enumerate(zip(values, supports, strict=True)):
            ax.annotate(
                f"{value:.3f} · n={support}",
                (value, index),
                xytext=(7, -1),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=color,
                weight="bold",
            )
        ax.set_xlim(0, 0.92)
        ax.set_xlabel("Median entry-level Conflict recall")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, weight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.018,
        (
            "The panels use different data, label rubrics, and model inputs. "
            "Compare value patterns within each panel, not scores across panels."
        ),
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.13, top=0.88)
    save_figure(fig, "per-value-conflict-recall.png")


def draw_label_agreement() -> None:
    """Compare human-human and LLM-Judge-human agreement by dimension."""
    source = ROOT / "docs" / "evals" / "judge_validation_summary.md"
    rows = markdown_rows(source, "| Value Dimension")[:10]
    dimensions = [row[0] for row in rows]
    human = [parse_float(row[1]) for row in rows]
    judge = [parse_float(row[2]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    y = list(range(len(dimensions)))
    human_y = [index - 0.14 for index in y]
    judge_y = [index + 0.14 for index in y]
    ax.scatter(
        human,
        human_y,
        s=70,
        color=MUTED,
        label="Human-human Fleiss' κ",
        zorder=2,
    )
    ax.scatter(
        judge,
        judge_y,
        s=76,
        color=TEAL,
        label="Mean LLM-Judge-human Cohen's κ",
        zorder=3,
    )
    for value, position in zip(human, human_y, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, position),
            xytext=(-7, -1),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.3,
            color=MUTED,
        )
    for value, position in zip(judge, judge_y, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, position),
            xytext=(7, -1),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.3,
            color=TEAL,
        )
    ax.set_yticks(y, dimensions)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 0.9)
    ax.set_xlabel("Chance-corrected agreement (κ)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        ncol=2,
        fontsize=9.5,
        borderaxespad=0,
    )
    fig.subplots_adjust(top=0.91)
    save_figure(fig, "label-agreement.png")


def draw_vif_handoff() -> None:
    """Show the VIF Critic input ablation without mixing unrelated metrics."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-07-14_twinkl_752_5_reassessment.md"
    )
    rows = markdown_rows(source, "| Setup")[:3]
    labels = [
        "Without VIF Critic\ninput",
        "With VIF Critic\nPredictions",
        "Early + weekly\nVIF triggers",
    ]
    recall = [parse_float(row[2]) for row in rows]
    false_alerts = [parse_float(row[5]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"wspace": 0.28})
    x = list(range(3))
    colors = [TEAL, CORAL, GOLD]
    axes[0].bar(x, recall, color=colors, width=0.62)
    axes[0].set_ylim(0, 0.34)
    axes[0].set_ylabel("Median Drift recall")
    axes[0].set_title("(a) Drift recall", fontsize=13, weight="bold")
    axes[1].bar(x, false_alerts, color=colors, width=0.62)
    axes[1].set_ylim(0, 4)
    axes[1].set_ylabel("Median false Drift alerts")
    axes[1].set_title("(b) False Drift alerts", fontsize=13, weight="bold")

    for ax, values, decimals in ((axes[0], recall, 3), (axes[1], false_alerts, 0)):
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=9.2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for index, value in enumerate(values):
            label = f"{value:.{decimals}f}" if decimals else f"{value:.0f}"
            ax.text(
                index,
                value + (0.012 if decimals else 0.12),
                label,
                ha="center",
                weight="bold",
            )

    fig.text(
        0.06,
        0.01,
        (
            "gpt-5.4-mini-2026-03-17 at reasoning none; 33 known Drifts across "
            "106 cases; medians across three repeats."
        ),
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.91, bottom=0.22)
    save_figure(fig, "vif-handoff-ablation.png")


def draw_weekly_drift_tradeoff() -> None:
    """Show the selected Weekly Drift Reviewer operating point."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md"
    )
    rows = markdown_rows(source, "| Reasoning effort")[:5]
    effort = [row[0].replace("`", "") for row in rows]
    recall = [parse_float(row[2]) for row in rows]
    false_alerts = [parse_float(row[3]) for row in rows]
    coverage = [parse_float(row[5]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = [MUTED, TEAL, BLUE, GOLD, CORAL]
    for index, label in enumerate(effort):
        ax.scatter(
            false_alerts[index],
            recall[index],
            s=230,
            color=colors[index],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        offset = {
            "none": (-76, -31),
            "low": (13, -38),
            "medium": (13, 10),
            "high": (13, -18),
            "xhigh": (13, 8),
        }[label]
        suffix = " · selected" if label == "low" else ""
        ax.annotate(
            f"{label}{suffix}\ncoverage {coverage[index]:.3f}",
            (false_alerts[index], recall[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            weight="bold" if label == "low" else "normal",
            arrowprops=(
                {"arrowstyle": "-", "color": TEAL, "linewidth": 0.8}
                if label == "low"
                else None
            ),
        )

    ax.set_xlim(2, 14.5)
    ax.set_ylim(0.43, 0.70)
    ax.set_xlabel("Median false Drift alerts (lower is better)")
    ax.set_ylabel("Median Drift recall (higher is better)")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    fig.subplots_adjust(top=0.95)
    save_figure(fig, "weekly-drift-tradeoff.png")


def main() -> None:
    """Generate all static figures used by the capstone paper."""
    configure_matplotlib()
    draw_synthetic_data_lifts()
    draw_label_agreement()
    draw_per_value_conflict_recall()
    draw_vif_handoff()
    draw_weekly_drift_tradeoff()


if __name__ == "__main__":
    main()
