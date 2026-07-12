# `twinkl-1r3d`: Conformity and Self-Direction Shortcut Audit

**Date:** 2026-07-12

**Checkpoint:** `run_020` BalancedSoftmax

**Split:** corrected validation split only

**Decision:** the audit found no evidence that confident correct active
predictions depend on one brittle lexical cue or a tested repeated/multiword
cue. This clears the bounded
`twinkl-1r3d` prerequisite, but it does not prove that the MLP learned either
Schwartz construct.

## Why this audit exists

`run_020` scores above the limited human-agreement benchmark on Conformity and
Self-Direction. That could reflect useful signal, or it could mean the MLP
learned easy words that correlate with the Judge. Before giving the MLP credit
in the final with-Critic versus without-Critic verifier study, this audit tests
the simplest version of the shortcut hypothesis: remove a small wording cue and
watch the prediction collapse.

## Method

- Selected the ten most confident correct `-1` and `+1` validation predictions
  per dimension. Only five correct Conformity `-1` cases exist, so all five were
  included. Final sample: 15 Conformity and 20 Self-Direction dimension-cases.
- Reconstructed the exact `window_size: 1` runtime text and profile state. The
  deterministic replay matches every saved baseline probability within
  `1e-5`.
- Removed one whole content-word occurrence at a time, re-embedded the text,
  and rescored the unchanged checkpoint with dropout disabled.
- Also removed all repetitions of candidate cue words and nine
  theory-motivated phrases per dimension when present.
- Ran 3,406 perturbations. Recorded target-probability change, class flip,
  membership in a narrow theory-motivated cue list, and exact word overlap with
  the Judge rationale.
- Hashed every reconstructed runtime text and rationale into a selected-case
  manifest so later source drift is detectable.

This is a validation diagnostic, not a new performance estimate or promotion
surface.

## Results

| Dimension | Target | Cases | Word removals | Class flips | Median per-case maximum probability drop | Largest drop |
|---|---:|---:|---:|---:|---:|---:|
| Conformity | `-1` | 5 | 610 | 0 | 0.060 | 0.088 |
| Conformity | `+1` | 10 | 769 | 0 | 0.012 | 0.042 |
| Self-Direction | `-1` | 10 | 822 | 0 | 0.015 | 0.061 |
| Self-Direction | `+1` | 10 | 1,205 | 0 | 0.030 | 0.092 |

No single-word removal changed any of the 35 predicted classes.

The suspected cue families were also quiet:

- Conformity cues appeared nine times across six cases. Maximum target-class
  probability drop was 0.052; no class flipped.
- Self-Direction cues appeared 53 times across 18 cases. Maximum drop was
  0.029; no class flipped.
- A listed candidate cue was never the most influential removal in a case.
- The 20 repeated-word or phrase removals also caused no flips. Their largest
  drop was 0.044 for Self-Direction; Conformity's largest was 0.001.

The largest changes came from scene-specific words such as `apartment`,
`alone`, `husband`, `lying`, `software`, and `feedback`. None caused a flip.
This pattern is more consistent with distributed sentence-level evidence than
with a single trigger-word rule. Exact rationale overlap was uncommon among the
most influential words—3/15 Conformity cases and 2/20 Self-Direction cases—but
Judge rationales are paraphrases, so that overlap rate is not a fidelity score.

### Per-dimension verdicts

- **Conformity — no bounded shortcut evidence.** For `d9bf06c1`, entry 5,
  removing the rationale-backed phrase `gave in` changed the correct `+1`
  probability from 0.9520 to 0.9515 and did not change the class. Across all 15
  cases, neither individual cues nor the tested phrases caused a flip.
- **Self-Direction — no bounded shortcut evidence.** For `7adc5866`, entry 5,
  the Judge rationale says the writer suppresses her own view. Removing the
  matching surface phrase `kept quiet` produced the largest grouped-cue change,
  from 0.8551 to 0.8114 for the correct `-1`, but the class remained `-1`.
  Across all 20 cases, no individual, repeated, or phrase cue caused a flip.

## Verdict and consequence

The narrow shortcut hypothesis is **not supported**: on the selected confident
correct active cases, the MLP does not collapse when one rule/duty/family or
choice/independence-style word is removed.

That is enough to stop treating a brittle single-word cue as an established
explanation for the MLP's strong scores. It is not enough to call the dimensions
solved. The test selected correct high-confidence cases, deletion can alter
grammar and meaning, and the cue lists cannot cover arbitrary multiword feature
interactions. The human benchmark is also limited and not directly comparable
to holdout QWK.

For `twinkl-752.1`, keep the MLP family as a baseline and allow its unique wins
to enter the bounded study, but describe them as model evidence—not proof of
construct understanding. The consensus replay and human-anchor checks required
by that issue remain necessary.

## Reproduction and artifacts

```bash
source .venv/bin/activate
python -m scripts.experiments.audit_1r3d_shortcuts
```

- Summary and provenance:
  [`audit_summary.json`](../artifacts/twinkl_1r3d_shortcut_audit_20260712/audit_summary.json)
- Per-occurrence results:
  [`token_perturbations.parquet`](../artifacts/twinkl_1r3d_shortcut_audit_20260712/token_perturbations.parquet)
- Repeated-word and phrase results:
  [`grouped_cue_perturbations.parquet`](../artifacts/twinkl_1r3d_shortcut_audit_20260712/grouped_cue_perturbations.parquet)
- Reproduction script:
  [`scripts/experiments/audit_1r3d_shortcuts.py`](../../../scripts/experiments/audit_1r3d_shortcuts.py)
