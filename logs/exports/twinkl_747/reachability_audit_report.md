# twinkl-747 Judge Reachability Audit

## Audit Scope

- Sample size: `50` cases
- Hard dimensions: `security, hedonism, stimulation`
- Control dimensions: `self_direction`, `universalism`
- Reviewed manual rows: `9`
- Overall recommendation: `change_distillation_target`

## 1. Persisted Label vs Full Context

| Dimension | Source Sign | Flip Count | Cases | Band |
| --- | --- | --- | --- | --- |
| hedonism | -1 | 2 | 5 | ambiguous |
| hedonism | 0 | 0 | 1 | low |
| hedonism | 1 | 4 | 8 | substantive |
| security | -1 | 1 | 2 | low |
| security | 1 | 9 | 12 | substantive |
| self_direction | 1 | 1 | 4 | low |
| stimulation | -1 | 2 | 7 | ambiguous |
| stimulation | 0 | 0 | 3 | low |
| stimulation | 1 | 3 | 4 | ambiguous |
| universalism | -1 | 0 | 1 | low |
| universalism | 1 | 1 | 3 | low |

This table tests whether a fresh rich-context rerun reproduces the persisted training labels or surfaces pre-existing label drift.

## 2. Full Context vs Profile Only

| Dimension | Source Sign | Flip Count | Cases | Band |
| --- | --- | --- | --- | --- |
| hedonism | -1 | 0 | 3 | low |
| hedonism | 0 | 2 | 6 | ambiguous |
| hedonism | 1 | 1 | 5 | low |
| security | -1 | 0 | 2 | low |
| security | 0 | 0 | 9 | low |
| security | 1 | 1 | 3 | low |
| self_direction | 0 | 1 | 1 | low |
| self_direction | 1 | 1 | 3 | low |
| stimulation | -1 | 0 | 5 | low |
| stimulation | 0 | 1 | 7 | low |
| stimulation | 1 | 1 | 2 | low |
| universalism | -1 | 1 | 2 | low |
| universalism | 1 | 0 | 2 | low |

These flips isolate how much the richer teacher signal depends on biography and prior-entry trajectory.

## 3. Profile Only vs Student Visible

| Dimension | Source Sign | Flip Count | Cases | Band |
| --- | --- | --- | --- | --- |
| hedonism | -1 | 0 | 4 | low |
| hedonism | 0 | 0 | 4 | low |
| hedonism | 1 | 0 | 6 | low |
| security | -1 | 0 | 2 | low |
| security | 0 | 2 | 10 | ambiguous |
| security | 1 | 0 | 2 | low |
| self_direction | 0 | 1 | 1 | low |
| self_direction | 1 | 0 | 3 | low |
| stimulation | -1 | 2 | 7 | ambiguous |
| stimulation | 0 | 0 | 6 | low |
| stimulation | 1 | 1 | 1 | low |
| universalism | -1 | 1 | 1 | low |
| universalism | 1 | 1 | 3 | low |

These flips estimate how much of the remaining signal depends on declared profile hints rather than the current session text alone.

## Hard-Dimension Recommendation Grid

| Dimension | Max Path Flip | Path Band | Persisted↔Full Flip | Blind Unrecoverable | Manual Reduced/Text Pref | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| security | 2 | ambiguous | 9 | 2 | 2 | change_distillation_target |
| hedonism | 2 | ambiguous | 4 | 1 | 1 | targeted_relabeling |
| stimulation | 2 | ambiguous | 3 | 0 | 0 | targeted_relabeling |

Band interpretation for hard dimensions:
- `0-1` flips: low concern
- `2-3` flips: ambiguous, use manual review to break the tie
- `4+` flips: substantive mismatch

## Manual Review Notes

Manual review rows were detected in `manual_review_workbook.csv` and were incorporated into the recommendation grid above.

## Output Files

- `joined_results.csv`: manifest plus focal labels for each rerun condition
- `comparison_rows.csv`: case-level flip rows for each comparison pair
- `flip_summary.csv`: aggregated flip counts by comparison, dimension, and sign bucket
