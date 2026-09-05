# Capstone Report Materials

This directory contains the maintained Phase 2 Technical Paper and its
supporting capstone report materials. Submitted milestone files remain under
`docs/archive/capstone/`.

## Current Report

- [`capstone_project_report.md`](capstone_project_report.md) — Quarto source for
  the Phase 2 Technical Paper
- [`capstone_project_report.pdf`](capstone_project_report.pdf) — rendered PDF
- [`capstone_requirements.pdf`](capstone_requirements.pdf) — NUS-ISS capstone
  briefing and requirements
- [`images/`](images/) — report figures, interface captures, and evaluation
  charts
- [`../../scripts/capstone/generate_report_figures.py`](../../scripts/capstone/generate_report_figures.py)
  — deterministic figure generation from committed evidence
- [`../../scripts/capstone/generate_north_star_figure.py`](../../scripts/capstone/generate_north_star_figure.py)
  — NSM feasibility figure and source-hash manifest from preserved run outputs

## Report Controls

- **Document status:** Maintained Phase 2 Technical Paper source and rendered PDF
- **NUS deliverable:** Phase 2 Technical Paper formatted as a publishable paper
- **Product source:** [`../prd.md`](../prd.md)
- **Required terms:** [`../canonical_nouns.md`](../canonical_nouns.md)
- **Prior submission:** [April 2026 Project
  Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)
- **Evidence date:** Core paper 2026-08-31; NSM supplement 2026-09-05
- **Status key:** complete, partial, development-only, experimental, in progress,
  or outside the time-boxed capstone
- Identify AI-reviewed synthetic evidence as AI-reviewed synthetic evidence.
- Identify human annotation as human annotation.
- Do not claim a fresh final test or deployment approval.
- Keep the user-facing Drift path separate from VIF Critic (Offline) research.

## Reproduce the Report

Run the following commands from the repository root:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
MPLCONFIGDIR=/tmp/twinkl-matplotlib \
  uv run python scripts/capstone/generate_report_figures.py
MPLCONFIGDIR=/tmp/twinkl-matplotlib \
  uv run python scripts/capstone/generate_north_star_figure.py
quarto render docs/capstone_report/capstone_project_report.md --to pdf
```

The figure script reads committed configuration, Parquet data, JSON metrics,
and Markdown reports. It writes two architecture diagrams and five evaluation
charts under
[`images/`](images/). The Quarto front matter selects XeLaTeX and records the
fonts, page geometry, table of contents, and PDF presentation settings.

The rendered report cites the committed
[Coach Digest sample](../../logs/experiments/reports/coach_digest_sample_20260824/report.md),
[Coach Digest Validations](../../logs/experiments/reports/coach_digest_validations_20260824/report.md),
and [Coach Digest Evals](../../logs/experiments/reports/coach_digest_evals_20260824/report.md)
for the five saved Persona key weeks. This result is same-model AI review. The
independent-provider Coach Digest and Drift/control tooling has no committed paid result and
does not change the report's evidence claim.

The September NSM supplement is separate from Coach Digest Evals. Its
[Phase 0A retrieval gate](../../logs/experiments/reports/north_star_phase0_20260905/README.md)
passed, but [Phase 0B](../../logs/experiments/reports/north_star_phase0b_20260905/README.md)
failed: independent AI reference review accepted 12/19 selected quotations,
correct omission was 5/9, and two of 29 OpenAI attempts were contract-invalid.
The 61 attempts cost US$0.2062 at frozen rates. The paper adds the experimental
architecture, measured results, limitations, and an offline evidence walkthrough;
it does not claim NSM application integration or browser QC. The supplement is
backed by frozen code and raw evidence in
[`f3030c8d`](https://github.com/DesmondChoy/twinkl/tree/f3030c8deb9400685185e7c620bda53a6f58e8aa),
with input and code hashes linked from Appendix A. The older evidence links still support
only the earlier studies.

The initial 37-page NSM supplement PDF was visually inspected on every page. The
[verification record](../../logs/experiments/reports/north_star_phase0b_20260905/validation.json)
preserves that render's report hashes, validation scope, and PDF QC captures;
it does not verify later report revisions, and those captures are distinct from
the blocked NSM browser QC.

The subsequent readability revision was regenerated as a 37-page PDF and
visually inspected on every page. That pass corrected overflowing source-path
labels and table pagination; its validation and final report hashes are recorded
in Beads issue `twinkl-ity7`.

## Submission Checks

- Confirm the required paper format and page limit with the advisor.
- Confirm whether the advisor requires a short team-contribution statement.
- Render the PDF after any source or figure change and inspect every page for
  clipped content, stale figures, broken links, and inconsistent references.
- Keep the repository commit and durable evidence links pinned to the evidence
  snapshot used by the paper.

## Submitted Material

The April 2026 proposal has already been submitted and is preserved unchanged
under [`../archive/capstone/2026-04-proposal-submission/`](../archive/capstone/2026-04-proposal-submission/).
An older draft is kept separately under
[`../archive/capstone/2026-04-proposal-drafts/`](../archive/capstone/2026-04-proposal-drafts/).

Do not copy the submitted files back into this directory for routine updates or
regenerate their PDF and figures in place. Create a newly named, dated report
or proposal version for any later revision.

For current project truth, start with [`../prd.md`](../prd.md), the active VIF
documentation under [`../vif/`](../vif/), and the evaluation specifications
under [`../evals/`](../evals/).
