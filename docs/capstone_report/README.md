# Capstone Report Materials

This directory contains the maintained Phase 2 Technical Paper and its
supporting capstone report materials. Submitted milestone files remain under
`docs/archive/capstone/`.

## Current Report

- [`capstone_project_report.md`](capstone_project_report.md) — Quarto source for
  the Phase 2 Technical Paper
- [`capstone_project_report.pdf`](capstone_project_report.pdf) — stale rendered PDF;
  regenerate and visually verify it after the NSM evaluation reset
- [`capstone_requirements.pdf`](capstone_requirements.pdf) — NUS-ISS capstone
  briefing and requirements
- [`images/`](images/) — report figures, interface captures, and evaluation
  charts
- [`../../scripts/capstone/generate_report_figures.py`](../../scripts/capstone/generate_report_figures.py)
  — deterministic figure generation from committed evidence

## Report Controls

- **Document status:** Maintained Phase 2 Technical Paper source; rendered PDF
  pending regeneration after the NSM evaluation reset
- **NUS deliverable:** Phase 2 Technical Paper formatted as a publishable paper
- **Product source:** [`../prd.md`](../prd.md)
- **Required terms:** [`../canonical_nouns.md`](../canonical_nouns.md)
- **Prior submission:** [April 2026 Project
  Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)
- **Evidence date:** Core paper 2026-08-31; NSM integration status corrected
  2026-09-06; numerical results and walkthrough incorporation paused
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

The previous NSM results, figure, and source-level walkthrough have been
removed from the maintained paper at the user's request. The [fresh evaluation
protocol](../north_star/luna_evaluation_20260905.md) uses `gpt-5.6-luna` with
`low` reasoning for selection and `xhigh` reasoning for evaluation, followed
by independent AI review of the evaluator. The run and review are complete;
incorporation of results into this paper remains paused at the user's request.
The first POC integration and browser checks are also complete. The
[experiment log](../../logs/experiments/north_star_moment.md) links the separate
evidence. Sections 3.9 and 4.5 and Appendix C retain the pause on numerical
results and walkthrough incorporation. When report updates resume, regenerate and inspect every PDF page.
The existing PDF contains superseded NSM results and does not represent the
revised source.

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
