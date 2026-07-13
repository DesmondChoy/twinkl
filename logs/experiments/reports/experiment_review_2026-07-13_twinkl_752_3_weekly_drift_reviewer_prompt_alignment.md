# `twinkl-752.3`: Weekly Drift Reviewer Prompt Alignment

**Date:** 2026-07-13

**Decision:** the published `0.40` median Drift recall was not materially limited
by the prompt differences tested here. The aligned prompt reduced median Drift
recall to `0.20`, raised median false Drift alerts from `1` to `5`, and recovered
neither cross-week reference Drift in any repeat.

**Disposition for `twinkl-752.2`:** do not treat a rubric-rich, pair-explicit
Weekly Drift Reviewer at reasoning effort `none` as a stronger Weekly Drift
Reviewer setup. Keep the earlier Weekly Drift Reviewer without VIF Critic input
as a conditional development option, or make no deployment architecture claim. This study does
not test higher reasoning effort and does not authorize deployment.

## Why this study exists

`twinkl-752.1` found median Drift recall of `0.40` for the Weekly Drift Reviewer
without VIF Critic input. That setup differed from the AI-reviewed reference-label
contract: current Journal Entries were repeated only as IDs, adjacent pairs that
crossed a week boundary were split across calls, the model returned no explicit
pair decision, and no shared Core Value rubric was supplied.

This study asks whether those prompt differences explain the weak result. It
reuses the exact three-repeat `twinkl-752.1` receipts for the current prompt and
runs only the aligned prompt, keeping the model, reasoning effort, student-visible
text, structured-output policy, and development target fixed.

## Contract comparison

| Contract element | Current prompt | Aligned prompt | AI-reviewed reference labels | Historical LLM-Judge |
|---|---|---|---|---|
| Time view | Cumulative history; current week assessed | Same history; every adjacent pair ending in the current week shown together | Complete trajectory, including later Journal Entries | Current Journal Entry plus limited prior context |
| Current evidence | Full text in history; IDs in assessment section | Full text repeated under each Journal Entry and adjacent pair | Full text for the complete trajectory | Full Journal Entry text |
| Core Value context | Names only | Versioned definition and core motivation | Name only | Detailed ten-value rubric plus persona context |
| Entry decision | `conflict`, `not_conflict`, or `abstain` | Same | `yes`, `no`, or `uncertain` | `-1`, `0`, or `+1` |
| Drift decision | Derived later in Python | Explicit pair decision, checked against entry decisions | Explicit case-level decision | None |
| Evidence | Exact quote required for Conflict | Same | Rationale, but no matching quote constraint | Rationale for non-zero labels |
| Attitudes | Excluded without observable behavior or choice | Explicitly excluded | Excluded without observable behavior or choice | May count as `-1` or `+1` |

The versioned shared rubric is
[`drift_v1_conflict_rubric_v1.yaml`](../../../config/evals/drift_v1_conflict_rubric_v1.yaml).
It is deliberately narrower than the historical LLM-Judge contract. Complete
text without qualifying behavior is not Conflict rather than abstain; abstention
is reserved for missing, broken, or irreducibly ambiguous evidence.

## Registered design

- Development target only: 28 personas, 217 Journal Entries, 126 persona-weeks,
  41 resolved trajectories, 316 resolved Journal Entry/Core Value cells, and 5
  reference Drifts.
- Current prompt: the frozen `twinkl-752.1` receipts without VIF Critic input,
  three repeats.
- Aligned prompt: 126 prompts and three repeats, for 378 new calls.
- Adjacent-pair decisions: 293 adjacent-pair/Core Value decisions; 152 cross a week
  boundary.
- Model: `gpt-5.4-mini-2026-03-17`, reasoning effort `none`, no tools, stored
  responses disabled.
- Decision rule: prompt-limited only if median Drift recall improved by at least
  `0.20`, a cross-week reference Drift was recovered in at least two repeats,
  false Drift alerts did not rise, and coverage did not fall by more than `0.05`.
- The final test set, retired Drift benchmark, VIF Critic inputs, and reference
  label edits were excluded.

## API execution and validation

| Calls | Valid | Invalid and failed closed | Input tokens | Output tokens | Cost |
|---:|---:|---:|---:|---:|---:|
| 378 | 351 | 27 | 1,180,203 | 114,406 | $1.4000 |

The 27 invalid receipts contained duplicate entry coordinates, inconsistent
entry/pair decisions, or one fabricated quote. They remain in the raw artifact
and contribute no Drift claim. Among valid responses, entry abstentions were
rare: 6 `missing_text`, 2 `ambiguous`, and 1 `feeling_or_intent_only`; no valid
pair output explicitly abstained. Missing or invalid calls still reduce
trajectory coverage because decisions fail closed.

## Primary Drift results

| Prompt | Repeat | Drift recall | Drift precision | False Drift alerts | Coverage | Same-week recovered | Cross-week recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | 1 | 0.60 | 1.000 | 0 | 0.756 | 3/3 | 0/2 |
| Current | 2 | 0.40 | 0.667 | 1 | 0.659 | 2/3 | 0/2 |
| Current | 3 | 0.20 | 0.500 | 1 | 0.829 | 1/3 | 0/2 |
| Aligned | 1 | 0.20 | 0.167 | 5 | 0.829 | 1/3 | 0/2 |
| Aligned | 2 | 0.20 | 0.167 | 5 | 0.829 | 1/3 | 0/2 |
| Aligned | 3 | 0.20 | 0.167 | 5 | 0.854 | 1/3 | 0/2 |
| **Current median** | — | **0.40** | **0.667** | **1** | **0.756** | — | **0/2** |
| **Aligned median** | — | **0.20** | **0.167** | **5** | **0.829** | — | **0/2** |

The aligned prompt made more complete decisions, but not better Drift decisions.
It repeatedly found only one of the three same-week reference Drifts and none of
the two cross-week reference Drifts. Its five false Drift alerts per repeat came
from five non-reference trajectories, although the identities varied across
repeats.

## Journal Entry diagnostics

| Prompt | Median macro `recall_-1` | Median Conflict precision | Median coverage | Median predicted-Conflict rate |
|---|---:|---:|---:|---:|
| Current | 0.306 | 0.583 | 0.804 | 0.038 |
| Aligned | 0.353 | 0.423 | 0.918 | 0.085 |

On the cells covered by both prompts, aligned-prompt Conflict recall was higher
in every repeat: `0.556` vs `0.444`, `0.400` vs `0.300`, and `0.571` vs `0.238`.
That gain did not survive the product rule. The aligned prompt more than doubled
the predicted-Conflict rate, lowered precision, and joined extra Conflict
decisions into false Drift alerts.

Repeat stability also worsened. The current prompt returned the same trajectory
decision across all repeats for 29 of 41 trajectories (`0.707`); the aligned
prompt did so for 25 of 41 (`0.610`).

## Interpretation for Twinkl

The prompt mismatch was real, but fixing it did not reveal a stronger Weekly
Drift Reviewer. Better Journal Entry coverage and recall are not enough when the
extra Conflict decisions form the wrong adjacent pairs. Twinkl needs reliable
Drift evidence for the Weekly Coach, not merely more negative labels.

The result supports a narrow conclusion: at reasoning effort `none`, this
rubric-rich, pair-explicit prompt is not a credible improvement over the current
prompt. It does not establish an intrinsic LLM ceiling because the reference
reviewers saw later Journal Entries, the reviewer and model differ, exact-quote
validation is stricter, and higher reasoning effort was not tested.

## Reproduction and artifacts

```bash
source .venv/bin/activate
python -m scripts.experiments.weekly_drift_reviewer_prompt_alignment prepare
python -m scripts.experiments.weekly_drift_reviewer_prompt_alignment score
pytest -q tests/experiments/test_weekly_drift_reviewer_prompt_alignment.py
```

- [Manifest](../artifacts/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_20260713/manifest.json)
- [Prompts](../artifacts/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_20260713/prompts.jsonl)
- [Raw responses](../artifacts/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_20260713/responses.jsonl)
- [Metrics](../artifacts/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_20260713/metrics.json)
- [Preregistered config](../../../config/evals/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_v1.yaml)
- [Aligned prompt](../../../prompts/weekly_drift_reviewer_aligned.yaml)
- [Runner and scorer](../../../scripts/experiments/weekly_drift_reviewer_prompt_alignment.py)
