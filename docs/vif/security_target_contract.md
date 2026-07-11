# Security Distillation Target Contract

## Decision

The current `security` target must be justified by the state available to the
student Critic at inference time. For the active `window_size: 1` frontier,
that state contains:

- the current journal entry, nudge, and response as one text input; and
- equal-weight declared core-value indicators.

It does not contain persona biography, synthetic-generation metadata, or prior
journal entries. Those sources must not silently determine the training target.
Compact legal history remains a separate student-input experiment under
`twinkl-749`.

The primary policy for `twinkl-a30f` is therefore
`student_visible_current_session_v1`: a non-neutral Security label requires
evidence in the current session that the active student can observe. Original
persisted and consensus labels remain immutable.

## Initial audit result

The deterministic `twinkl-747` sample contains 14 Security cases selected from
known frontier errors. The `twinkl-a30f` artifact compares three existing
reruns: current-session-only, profile-only, and full-context judging.

This artifact is diagnostic, not a replacement training table. The cases were
selected because the frontier missed them, so using the same subset as a broad
retraining or evaluation target would introduce selection bias. A full-corpus
or independently sampled target pass is required before controlled retraining.

The initial result is:

- 14 sampled Security cases;
- 10 proposed label changes under the current-session policy;
- 11 cases whose three fresh reruns agree, so additional context does not
  explain the original frontier miss;
- 8 cases where that unanimous fresh judgment also disagrees with the persisted
  label, indicating likely stored-label drift or overreach;
- 2 cases whose judgment changes when profile context is added; and
- 1 case whose judgment changes when prior trajectory is added.

The selected sample therefore points more strongly to stale or over-broad
stored Security supervision than to hidden biography alone. Because selection
started from known errors, these counts describe the failure sample and must not
be generalized to the full corpus.

Run:

```bash
python scripts/experiments/build_a30f_security_target.py
```

Outputs:

- `security_target_variant.parquet`: provenance-rich case-level target proposal
- `audit_summary.json`: bucket counts, changed-label count, and the explicit
  training-readiness gate

## Bucket meanings

- `visible_from_student_input`: all three reruns agree; extra context does not
  change the judgment.
- `requires_profile_or_bio_context`: profile and full-context reruns agree, but
  current-session-only judging differs.
- `requires_prior_trajectory`: current-session and profile-only judging agree,
  but prior entries change the full-context judgment.
- `ambiguous_security_vs_conformity_or_tradition`: the context path does not
  yield a stable two-of-three interpretation and requires case-level review.

`likely_label_error_or_overreach` is recorded separately when the persisted
label disagrees with unanimous fresh reruns. This distinguishes stale or broad
stored supervision from genuinely hidden context.

## Next gate

Do not retrain from this selected subset. First produce an unbiased Security
target population under the policy above, with immutable case coordinates and
review provenance. Then retrain the incumbent corrected-split configuration
with architecture, split, and optimization held fixed, and report both the old
label lens and repaired-target lens.
