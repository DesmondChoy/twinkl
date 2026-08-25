"""Generate the capstone paper figures from committed evidence."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "capstone_report" / "images"

INK = "#14233B"
MUTED = "#637083"
TEAL = "#248D83"
BLUE = "#4F6FE8"
GOLD = "#D8A62A"
CORAL = "#C05A40"
PAPER = "#FBFAF6"
PALE_TEAL = "#E7F3F0"
PALE_BLUE = "#EAF0FF"
PALE_GOLD = "#FBF2D8"
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


def draw_profile_construction() -> None:
    """Explain the deterministic Profile construction in three steps."""
    fig, ax = plt.subplots(figsize=(11.4, 4.25))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.25)
    ax.axis("off")

    cards = [
        (
            0.15,
            PALE_BLUE,
            "1  Complete 11 sets",
            (
                "Choose the Most and Least\nimportant item in each set of six.\n"
                "Each item appears six times."
            ),
            "11 sets × 6 items",
        ),
        (
            3.98,
            PALE_TEAL,
            "2  Calculate scores",
            (
                "Raw score = (Most − Least) ÷ 6.\nMerge the two Universalism\n"
                "facets. Shift and normalize\nall ten scores."
            ),
            "Ten Profile weights",
        ),
        (
            7.81,
            PALE_GOLD,
            "3  Confirm Core Values",
            (
                "Show the highest scores.\nIf more than two values tie,\n"
                "ask the user to select\nexactly two."
            ),
            "At most two Core Values",
        ),
    ]
    for x, fill, title, body, result in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.48),
                3.42,
                3.25,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=fill,
                edgecolor=INK,
                linewidth=1.0,
            )
        )
        ax.text(x + 0.2, 3.42, title, fontsize=15, weight="bold", va="top")
        ax.text(x + 0.2, 2.77, body, fontsize=10.7, linespacing=1.38, va="top")
        ax.text(
            x + 1.71,
            0.81,
            result,
            fontsize=11.2,
            weight="bold",
            ha="center",
            va="center",
            color=INK,
        )

    for x in (3.67, 7.50):
        ax.add_patch(
            FancyArrowPatch(
                (x - 0.03, 2.1),
                (x + 0.26, 2.1),
                arrowstyle="-|>",
                mutation_scale=15,
                color=MUTED,
                linewidth=1.3,
            )
        )

    ax.set_title(
        "Profile construction is deterministic after the user's choices",
        loc="left",
        fontsize=18,
        weight="bold",
        pad=8,
    )
    save_figure(fig, "profile-construction.png")


def draw_label_agreement() -> None:
    """Compare human-human and LLM-Judge-human agreement by dimension."""
    source = ROOT / "docs" / "evals" / "judge_validation_summary.md"
    rows = markdown_rows(source, "| Value Dimension")[:10]
    dimensions = [row[0] for row in rows]
    human = [parse_float(row[1]) for row in rows]
    judge = [parse_float(row[2]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    y = list(range(len(dimensions)))
    for index in y:
        ax.plot(
            [human[index], judge[index]],
            [index, index],
            color=GRID,
            linewidth=2.2,
            zorder=1,
        )
    ax.scatter(human, y, s=70, color=MUTED, label="Human-human Fleiss' κ", zorder=2)
    ax.scatter(
        judge,
        y,
        s=76,
        color=TEAL,
        label="Mean LLM-Judge-human Cohen's κ",
        zorder=3,
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
    fig.suptitle(
        "Agreement varies by value dimension",
        x=0.14,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.14,
        0.902,
        (
            "Shared benchmark: 115 Journal Entries from 19 personas and three "
            "human annotators"
        ),
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.81)
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
    axes[0].set_title("No Drift-recall gain", fontsize=14, weight="bold")
    axes[1].bar(x, false_alerts, color=colors, width=0.62)
    axes[1].set_ylim(0, 4)
    axes[1].set_ylabel("Median false Drift alerts")
    axes[1].set_title(
        "VIF Critic Predictions added false Drift alerts",
        fontsize=14,
        weight="bold",
    )

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

    fig.suptitle(
        "VIF Critic Predictions did not improve Weekly Drift Detection",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.06,
        0.01,
        (
            "Development union: 33 known Drifts across 106 cases; medians across "
            "three repeats."
        ),
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.80, bottom=0.22)
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
    sizes = [330 * value for value in coverage]
    colors = [MUTED, TEAL, BLUE, GOLD, CORAL]
    for index, label in enumerate(effort):
        ax.scatter(
            false_alerts[index],
            recall[index],
            s=sizes[index],
            color=colors[index],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        offset = {
            "none": (7, -18),
            "low": (10, -31),
            "medium": (10, 10),
            "high": (8, -18),
            "xhigh": (8, 8),
        }[label]
        suffix = " · selected" if label == "low" else ""
        ax.annotate(
            f"{label}{suffix}",
            (false_alerts[index], recall[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            weight="bold" if label == "low" else "normal",
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
    fig.suptitle(
        "Reasoning effort changes the operating point",
        x=0.095,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.095,
        0.90,
        "Bubble area represents coverage. Results are medians across three repeats.",
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.81)
    save_figure(fig, "weekly-drift-tradeoff.png")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font that is available on macOS and common Linux images."""
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_inside(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize one image to fit inside a box without distortion."""
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def resize_to_fit(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize one image up or down to fit inside a box."""
    scale = min(width / image.width, height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.LANCZOS)


def draw_case_study() -> None:
    """Combine the Experience, Coach Digest, and saved AI review evidence."""
    source_dir = OUTPUT_DIR
    experience = Image.open(source_dir / "lukas-key-week-experience.png").convert("RGB")
    digest = Image.open(source_dir / "lukas-key-week-coach-digest.png").convert("RGB")
    review = Image.open(source_dir / "lukas-key-week-ai-review.png").convert("RGB")

    canvas = Image.new("RGB", (2200, 1830), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(46, bold=True)
    panel_font = load_font(30, bold=True)
    note_font = load_font(25)
    draw.text(
        (80, 48),
        "One saved replay connects the result, response, and evidence",
        fill=INK,
        font=title_font,
    )

    top = fit_inside(experience, 2040, 1050)
    canvas.paste(top, ((2200 - top.width) // 2, 130))
    draw.rounded_rectangle((60, 112, 2140, 1220), radius=18, outline=GRID, width=3)
    draw.text((86, 139), "A  Experience", fill=INK, font=panel_font)

    digest_scaled = resize_to_fit(digest, 600, 455)
    review_scaled = resize_to_fit(review, 660, 455)
    left_box = (260, 1250, 1000, 1770)
    right_box = (1160, 1250, 1980, 1770)
    for box in (left_box, right_box):
        draw.rounded_rectangle(box, radius=18, fill="white", outline=GRID, width=3)

    canvas.paste(
        digest_scaled,
        (
            left_box[0] + (left_box[2] - left_box[0] - digest_scaled.width) // 2,
            left_box[1] + 55,
        ),
    )
    canvas.paste(
        review_scaled,
        (
            right_box[0] + (right_box[2] - right_box[0] - review_scaled.width) // 2,
            right_box[1] + 55,
        ),
    )
    draw.text(
        (left_box[0] + 22, left_box[1] + 16),
        "B  Coach Digest",
        fill=INK,
        font=panel_font,
    )
    draw.text(
        (right_box[0] + 22, right_box[1] + 16),
        "C  Saved AI evidence",
        fill=INK,
        font=panel_font,
    )
    draw.text(
        (120, 1772),
        (
            "The replay is synthetic evidence for the proof of concept. "
            "It is not a user study."
        ),
        fill=MUTED,
        font=note_font,
    )
    canvas.save(OUTPUT_DIR / "lukas-case-study.png", dpi=(300, 300))


def main() -> None:
    """Generate all static figures used by the capstone paper."""
    configure_matplotlib()
    draw_profile_construction()
    draw_label_agreement()
    draw_vif_handoff()
    draw_weekly_drift_tradeoff()
    draw_case_study()


if __name__ == "__main__":
    main()
