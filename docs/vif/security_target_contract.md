# Security Distillation Target Contract

## Active student state

The Security target must be justified by the exact state available to the
student Critic at inference time. For the active `window_size: 1` frontier, that
state is:

```text
embedding(concatenate_entry_text(initial_entry, nudge_text, response_text))
+ normalized 10-dimensional declared-value profile
```

The synthetic-persona profile assigns equal mass to each valid declared core
value and falls back to a uniform vector if none match. With `window_size: 1`,
there is no time-gap feature and no earlier journal entry in the state.

The active state excludes:

- persona name, age, profession, culture, and biography;
- current-entry date and prior or future entries;
- persisted labels, rationales, model predictions, and uncertainty values; and
- synthetic-generation instructions or metadata.

The primary policy is `security_active_critic_state_v1`. A non-neutral Security
label may use the declared-value profile to interpret a trade-off, but it must
be grounded in behavior, choices, or attitudes visible in the current session.
Original persisted and consensus labels remain immutable.

## Historical audit boundary

The completed `twinkl-747` audit remains useful diagnostic evidence, but none
of its three original prompt arms exactly represented the active Critic state.
In particular, its legacy `student_visible` arm omitted the profile while
retaining persona demographics and date. Its `profile_only` arm retained raw
core-value names plus the same unavailable metadata. Both used the historical
Judge-session text format rather than the runtime serialization.

The 14-row target artifact introduced in commit
`cc114f49f6af992e312a7f61790fad269306820e` therefore did not establish an
active-state Security target. It has been removed from the active artifact tree
and recorded in the
[retirement note](../archive/vif/retired_security_target_a30f_20260711.md).
Its former counts are historical only; they are not proposed-label results
under the replacement policy.

## Completed full-corpus review

The training-ready review used the complete 1,651-entry corpus. Each entry
received three isolated `gpt-5.4-mini` reviews under the exact contract; 25
three-way ties received a fourth review. The receipt-bound materializer
accepted all entries and wrote a non-destructive target variant.

- 1,162 unanimous decisions, 464 two-of-three majorities, 25 tie breaks;
- 678 changed Security labels;
- historical `-1 / 0 / +1`: `151 / 1,212 / 288`;
- repaired `-1 / 0 / +1`: `265 / 697 / 689`; and
- artifacts under `logs/exports/twinkl_a30f_active_critic_state_full_v1/` and
  `logs/exports/twinkl_a30f_security_target_full_v1/`.

The selected policy remains `security_active_critic_state_v1`. The reviews are
repeated model judgments with recorded disagreement, not independent human
ground truth. Historical labels remain immutable.

## Exact-state review workflow

First prepare a separate receipt-bound review bundle for the same selected
Security cases:

```bash
source .venv/bin/activate
python scripts/experiments/prepare_a30f_security_target_audit.py
```

The generated prompt manifest contains:

- the runtime-formatted current session;
- the normalized profile vector in canonical Schwartz order;
- explicit included/excluded-context flags;
- a canonical state-input SHA-256 digest; and
- a rendered-prompt SHA-256 digest.

The review runner must present only the manifest record's `prompt` value to the
Judge or human reviewer. Case coordinates and hashes remain outside the review
input and are used only to reconcile the returned result.

Each external review result must copy the contract version and both hashes,
name the reviewer or runtime, record a timezone-aware review timestamp and
confidence, and provide all ten ordinal scores. Bundle preparation rejects
duplicate entry coordinates. Materialization independently re-renders the
canonical prompt from the hashed state input, so a self-consistently rehashed
prompt with added context is also rejected. Missing, duplicate, extra, stale,
or non-canonical results are rejected.

Only after every selected Security case has a receipt-matching result can the
diagnostic target be materialized:

```bash
python scripts/experiments/build_a30f_security_target.py \
  --active-state-manifest \
    logs/exports/twinkl_a30f_active_critic_state_v1/active_critic_state_manifest.jsonl \
  --active-state-results \
    logs/exports/twinkl_a30f_active_critic_state_v1/active_critic_state_results.jsonl
```

The builder derives `new_label` only from the exact `active_critic_state` arm.
It has no fallback to the legacy `student_visible`, `profile_only`, or
`full_context` labels. It validates all evidence before creating the output
directory, refuses to overwrite an existing artifact, and binds the resulting
summary to the source, manifest, results, and parquet hashes.

## Conservative context buckets

Legacy labels may appear in a target artifact only as diagnostic provenance.
Their relationship to the exact-state review uses observed-delta names:

- `matches_full_context`: the exact-state and legacy rich-context judgments
  agree.
- `changes_with_bio_or_history`: the exact-state result matches the legacy
  profile-only result but differs from the full-context result. The cause is not
  isolated because biography and earlier entries were added together.
- `changes_between_active_state_and_legacy_profile_prompt`: the legacy
  profile-only and full-context results agree while the exact-state result
  differs. This records a prompt/input-contract delta, not a biography claim.
- `unresolved_context_sensitivity`: the three judgments differ and require
  case-level review.

No bucket may claim `requires_prior_trajectory`, `requires_biography`, or a
Security-versus-Conformity/Tradition cause unless a later audit isolates and
reviews those factors separately.

## Training and evaluation gate

The selected 14 cases were chosen from stable frontier misses on the historical
frozen test population. Even a completed exact-state review therefore remains:

- diagnostic-only;
- unsuitable for training or target-repair fitting;
- unsuitable as an old-label or repaired-label evaluation population; and
- unable to support a promotion or production-readiness claim.

A controlled retrain therefore used the separate full-corpus target rather
than this selected subset. `run_057`-`run_062` held architecture, split, and
optimization fixed across three paired seeds and reported both historical and
repaired lenses. Median test Security QWK improved from `0.156` to `0.328`
under the repaired lens and from `0.205` to `0.372` under the historical lens.
See the [decision report](../../logs/experiments/reports/experiment_review_2026-07-11_twinkl_a30f_security_target.md).
