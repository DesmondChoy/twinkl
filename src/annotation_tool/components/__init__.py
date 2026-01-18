"""UI components for the Schwartz Value Annotation Tool.

This package provides modular Shiny components for the annotation interface:

- header: App header with annotator input, progress bar, navigation
- sidebar: Left sidebar with persona info and entry navigation
- entry_display: Center column with entry content and status
- scoring_grid: Right column with value scoring and save button
- modals: Modal dialogs for warnings and comparisons

Usage:
    from src.annotation_tool.components import (
        header,
        sidebar,
        entry_display,
        scoring_grid,
        modals,
    )
"""

from src.annotation_tool.components import entry_display, header, modals, scoring_grid, sidebar

__all__ = [
    "header",
    "sidebar",
    "entry_display",
    "scoring_grid",
    "modals",
]
