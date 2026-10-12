"""Display figures preserved from the submitted capstone report."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from htmltools import Tag, tags

_METADATA_PATH = Path(__file__).parent / "static" / "report" / "metadata.json"


@lru_cache(maxsize=1)
def _figure_metadata() -> dict[str, dict[str, Any]]:
    metadata = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    return dict(metadata["figures"])


def report_figure(key: str) -> Tag:
    """Return an accessible figure with its report caption and full-size link.

    The retained ``matching`` figure explains the report's scoring window.
    Only packaged static assets are needed at runtime.
    """
    figure = _figure_metadata()[key]
    number = figure["figure_number"]
    image_url = f"report/{figure['file']}?v={figure['sha256'][:12]}"
    return tags.figure(
        tags.a(
            tags.img(
                src=image_url,
                alt=figure["alt"],
                width=figure["width_px"],
                height=figure["height_px"],
                loading="lazy",
                decoding="async",
                class_="report-figure-image",
                style="display: block; width: 100%; height: auto;",
            ),
            href=image_url,
            target="_blank",
            rel="noopener noreferrer",
            aria_label=f"Open report Figure {number} at full size in a new tab",
            class_="report-figure-zoom",
        ),
        tags.figcaption(
            tags.strong(f"Figure {number} · Report p. {figure['printed_page']}. "),
            figure["caption"],
            tags.a(
                " Open full-size figure ↗",
                href=image_url,
                target="_blank",
                rel="noopener noreferrer",
                aria_label=f"Open report Figure {number} at full size in a new tab",
            ),
        ),
        tags.details(
            tags.summary("Read figure description"),
            tags.p(figure["description"]),
            class_="report-figure-description",
        ),
        class_=f"report-figure report-figure--{key}",
    )
