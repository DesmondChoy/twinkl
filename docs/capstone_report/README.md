# Capstone Report Materials

This directory contains the maintained Phase 2 Technical Paper and its
supporting capstone report materials. Submitted milestone files remain under
`docs/archive/capstone/`.

## Current Report

- [`capstone_project_report.md`](capstone_project_report.md) — Quarto source for
  the Phase 2 Technical Paper
- [`capstone_project_report.pdf`](capstone_project_report.pdf) — rendered report,
  with all 45 pages visually verified on 14 September 2026
- [`capstone_requirements.pdf`](capstone_requirements.pdf) — NUS-ISS capstone
  briefing and requirements
- [`images/`](images/) — report figures, interface captures, and evaluation
  charts
- [`../../scripts/capstone/generate_report_figures.py`](../../scripts/capstone/generate_report_figures.py)
  — deterministic figure generation from committed evidence

## Report Controls

- **Document status:** Maintained Phase 2 Technical Paper source and verified PDF;
  regenerate and visually verify the PDF after source or figure changes
- **NUS deliverable:** Phase 2 Technical Paper formatted as a publishable paper
- **Product source:** [`../prd.md`](../prd.md)
- **Required terms:** [`../canonical_nouns.md`](../canonical_nouns.md)
- **Prior submission:** [April 2026 Project
  Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)
- **Evidence date:** Core paper 2026-08-31; North Star Moment targeted v4 Run 1
  results and saved Persona evidence 2026-09-07; application verification
  2026-09-12; local Coach source-context diagnostic 2026-09-13
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
cd docs/capstone_report
quarto render capstone_project_report.md --to pdf
```

The figure script reads committed configuration, Parquet data, JSON metrics,
and Markdown reports. It writes two architecture diagrams and five evaluation
charts under
[`images/`](images/). The Quarto front matter selects XeLaTeX and records the
fonts, page geometry, table of contents, and PDF presentation settings.

The paper cites the historical August
[Coach Digest sample](../../logs/experiments/reports/coach_digest_sample_20260824/report.md),
[Coach Digest Validations](../../logs/experiments/reports/coach_digest_validations_20260824/report.md),
and [Coach Digest Evals](../../logs/experiments/reports/coach_digest_evals_20260824/report.md)
for the previous five-Persona roster and inputs. This result is same-model AI review. The
independent-provider Coach Digest and Drift/control tooling has no committed paid result and
does not change the report's evidence claim. This workstream is Done under the
[accepted capstone scope](../prd.md#capstone-closeout-decisions). Human
calibration and the external user pilot are closed without execution. The
report preserves its recorded evidence; the PRD records the final scope
decisions, including completion of longitudinal Core Value history.

The [13 September source-context diagnostic](../../logs/experiments/reports/coach_context_eval_20260913/report.md)
contains six responses and nine same-model evaluator assessments for one
synthetic Persona-week. Complete selected-entry context reduced clear event
attribution errors in this sample, but causal overgeneralisation remained and
the evaluator failed to flag the known incorrect response. These results are
separate from the August scores and saved September replays. The diagnostic and
associated Coach changes are preserved in commit
[`af22ab4`](https://github.com/DesmondChoy/twinkl/tree/af22ab457994fd2863fe5b5b7735f768f211519b).

The [7 September Coach Digest sample](../../logs/experiments/reports/demo_v4_run1_20260907/report.md)
contains five accepted key-week responses for its recorded Persona roster and
Weekly Drift inputs. The current 27-week replay uses the
[voice refresh and repair receipts](../../logs/experiments/reports/coach_voice_refresh_20260909/report.md)
and [22 saved comparisons](../north_star/demo_coach_comparison.md). Each saved
response retains its input hashes and validation policy. These responses have
no corresponding Coach Digest Evals or human review; the August scores do not
describe them.

The [16 September application verification](../../logs/experiments/reports/onboarding_qa_fixes_20260916/report.md)
records prompt-`4.7` validation, explicit Coach Digest retry, historical Moment
recovery, and desktop browser evidence. The [current runtime guide](../weekly/weekly_drift_detection.md)
describes ordinary generation; the paper's dated prompt-`4.6` diagnostic remains
the source for that experiment's findings.

The [NSM experiment methodology](../north_star/nsm_experiment_methodology.md)
and [targeted v4 Run 1 results](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
support Sections 3.9 and 4.5. The comparison covers 501 weeks from 105 synthetic
Personas, with 38 reassessed cases and 463 retained observations. Full eligible
history has higher Card precision and Opportunity recall than Nomic top-three
retrieval in both partitions. These are AI assessments with qualified final
histories that have prior upstream research exposure, not human validation or
an untouched final test. Appendix C describes saved full-history outcomes for
five Personas and 27 weeks. Live North Star Moment review uses its separately
pinned US$1 allowance across sessions and restarts. The final professor
walkthrough evidence remains open.

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
