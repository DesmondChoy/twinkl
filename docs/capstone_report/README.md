# Capstone Report Materials

This directory contains supporting capstone report materials. The
Phase 2 Technical Paper is also available in the [Google Doc](https://docs.google.com/document/d/1L_tMcrcF9yuJz3tMs0yCMw5NAAszKsA63nY64dsgbnM/edit), tab
“Twinkl Report (Sept ’26)”. The reference Markdown and its PDF are corrected editions of the
user-selected 50-page `Twinkl IEEE Single Column.pdf`
with the checklist audit applied and validated on 11 October 2026. Submitted milestone files remain under
`docs/archive/capstone/`.

## Report Files

- [Phase 2 Technical Paper](https://docs.google.com/document/d/1L_tMcrcF9yuJz3tMs0yCMw5NAAszKsA63nY64dsgbnM/edit) — current Technical Paper (Google Doc)
- [`capstone_report.md`](capstone_report.md) —
  editable corrected edition of the selected IEEE PDF; applies the
  checklist audit, including factual corrections and verified citation updates
- [`capstone_report.pdf`](capstone_report.pdf) —
  PDF rendered from the corrected Markdown on 11 October 2026
- [`capstone_requirements.pdf`](capstone_requirements.pdf) — NUS-ISS capstone
  briefing and requirements
- [`images/ieee-source/`](images/ieee-source/) — all 33 original raster images
  used by the selected IEEE PDF, verified against its embedded images
- [`images/corrected/`](images/corrected/) — Figures 22 and 29 with corrected
  embedded figure and table references
- [`ieee_mirror_validation.json`](ieee_mirror_validation.json) — source and output
  hashes, audit dispositions, independent reviews, and rendering validation
- [`render-support/`](render-support/) and [`IEEEtran.cls`](IEEEtran.cls) —
  local PDF rendering support
- [`images/`](images/) — also retains figures from earlier report versions
- [`../../scripts/capstone/generate_report_figures.py`](../../scripts/capstone/generate_report_figures.py)
  — deterministic figure generation from committed evidence

## Report Controls

- **Document status:** The Markdown and its PDF are corrected reference editions of the
  selected IEEE PDF with `twinkl-report-checklist-audit.pdf`
  applied. The audit refers to an earlier 68-page LaTeX snapshot, so fixes use
  the selected edition’s figure and table numbering. Fonts and page breaks may
  differ from the source; the validation record documents the final output.
- **NUS deliverable:** Phase 2 Technical Paper formatted as a publishable paper
- **Product source:** [`../prd.md`](../prd.md)
- **Required terms:** [`../canonical_nouns.md`](../canonical_nouns.md)
- **Prior submission:** [April 2026 Project
  Proposal](../archive/capstone/2026-04-proposal-submission/April_Project_Proposal.md)
- **Evidence date:** Core paper 2026-08-31; North Star Moment targeted v4 Run 1
  results and saved Persona evidence 2026-09-07; application verification
  2026-09-12; local Coach source-context diagnostic 2026-09-13; matched Luna
  model comparisons and live migration 2026-09-23. Current model descriptions
  reconciled on 2026-10-03.
- **Status key:** complete, partial, development-only, experimental, in progress,
  or outside the time-boxed capstone
- Identify AI-reviewed synthetic evidence as AI-reviewed synthetic evidence.
- Identify human annotation as human annotation.
- Do not claim a fresh final test or deployment approval.
- Keep the user-facing Drift path separate from VIF Critic (Offline) research.

## Reproduce the Report

Run from the repository root with Quarto and XeLaTeX installed. The report
uses Times New Roman, Arial, and Courier New; these fonts must be available.
The figures are already stored locally and do not need to be regenerated.

```sh
cd docs/capstone_report
quarto render capstone_report.md --to pdf
```

Rendering checks consecutive section and appendix numbers, figure and table
labels, and the numbers shown in figure and table links. Academic citation
numbers link to their exact bibliography entries; rendering also checks that
each citation number matches its reference anchor.

On macOS with TinyTeX, add its binary directory to `PATH` if XeLaTeX is not
already available:

```sh
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
```

The front matter selects the local IEEE class and rendering support. The Lua
filter preserves the explicit figure and table numbers, renders pipe tables,
and retains working internal destinations. The LaTeX header fits images
without changing their aspect ratio and permits long visible URLs to wrap.
The selected source PDF and its adjacent LaTeX files are left intact.

## Historical Evidence Notes

The following notes were recorded for the earlier 3 October reference copy;
the selected IEEE PDF supplies the baseline wording and section numbers, with
the checklist corrections recorded in the validation file.

The current live Weekly Drift Reviewer and North Star Moment selection use
GPT-6 Luna at low reasoning effort. Coach Digest and nudge generation still
default to GPT-5.6 Luna at no reasoning effort. Saved Persona replays and
historical experiments retain their recorded model identities. Section 4.3
distinguishes the original GPT-5.6 prompt-v2 operating-point study from the
[matched prompt-v4 comparison](../../logs/experiments/reports/experiment_review_2026-09-23_twinkl_q6pt_luna_model.md)
that supported the September migration through higher coverage and lower
calculated token cost, without an established Drift-recall gain. The NSM
comparison retains GPT-5.6 selection and evaluation provenance; it does not
measure live GPT-6 NSM quality.

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
support Sections 3.5.2 and 4.4.2. The comparison covers 501 weeks from 105 synthetic
Personas, with 38 reassessed cases and 463 retained observations. Full eligible
history has higher Card precision and Opportunity recall than Nomic top-three
retrieval in both partitions. These are AI assessments with qualified final
histories that have prior upstream research exposure, not human validation or
an untouched final test. Appendix B describes saved full-history outcomes for
five Personas and 27 weeks. Live North Star Moment review uses its separately
pinned US$1 allowance across sessions and restarts. The final professor
walkthrough of all five saved Personas is complete (`twinkl-rklc.10`,
user-confirmed 9 October 2026); Coach Digest feedback remains future work.

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
