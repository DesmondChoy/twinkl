from __future__ import annotations

import json
import struct
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from src.drift_review_app.report_figures import _figure_metadata, report_figure

REPORT_DIR = Path(__file__).resolve().parents[2] / "src/drift_review_app/static/report"
SUBMITTED_PDF_SHA256 = (
    "4335740e6a1b3d05cc822bedf4b8d5dc814262de5894f5a3d49443835d99e9f8"
)


def test_packaged_figures_retain_submitted_report_provenance() -> None:
    metadata = json.loads((REPORT_DIR / "metadata.json").read_text())
    assert metadata["source_pdf"]["sha256"] == SUBMITTED_PDF_SHA256
    assert metadata["source_pdf"]["page_count"] == 65
    assert {
        key: (figure["figure_number"], figure["pdf_page"], figure["printed_page"])
        for key, figure in metadata["figures"].items()
    } == {
        "matching": (10, 23, 19),
    }

    for figure in metadata["figures"].values():
        asset = (REPORT_DIR / figure["file"]).read_bytes()
        assert sha256(asset).hexdigest() == figure["sha256"]
        assert asset[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", asset[16:24]) == (
            figure["width_px"],
            figure["height_px"],
        )
        extraction = figure["extraction"]
        assert extraction["bounds_px"]["width"] == figure["width_px"]
        assert extraction["bounds_px"]["height"] == figure["height_px"]
        assert extraction["dpi"] == 300
        assert f'-f {figure["pdf_page"]} ' in extraction["command"]
        assert '"$REPORT_PDF"' in extraction["command"]


class _FigureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str | None]] = []
        self.links: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "img":
            self.images.append(dict(attrs))
        elif tag == "a":
            self.links.append(dict(attrs))


def test_report_figures_have_accessible_full_size_static_links() -> None:
    for key, metadata in _figure_metadata().items():
        html = str(report_figure(key))
        parsed = _FigureParser()
        parsed.feed(html)
        assert len(parsed.images) == 1
        image = parsed.images[0]
        assert image["alt"]
        assert image["src"] == (
            f"report/{metadata['file']}?v={metadata['sha256'][:12]}"
        )
        assert all(link["href"] == image["src"] for link in parsed.links)
        assert all(link["aria-label"] for link in parsed.links)
        assert all(link["target"] == "_blank" for link in parsed.links)
        assert metadata["caption"] in unescape(html)
        assert metadata["description"] in unescape(html)
        assert f"Report p. {metadata['printed_page']}" in html
        assert "/Users/" not in html
        assert ".pdf" not in html


def test_figure_rendering_needs_only_the_packaged_metadata(monkeypatch) -> None:
    original_open = Path.open
    opened_paths: list[Path] = []

    def confined_open(path: Path, *args, **kwargs):
        resolved = path.resolve()
        assert resolved.is_relative_to(REPORT_DIR.resolve())
        opened_paths.append(resolved)
        return original_open(path, *args, **kwargs)

    _figure_metadata.cache_clear()
    monkeypatch.setattr(Path, "open", confined_open)
    try:
        assert "<figure" in str(report_figure("matching"))
    finally:
        _figure_metadata.cache_clear()
    assert opened_paths == [(REPORT_DIR / "metadata.json").resolve()]
