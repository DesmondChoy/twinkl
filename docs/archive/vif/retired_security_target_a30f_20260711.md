# Retired `twinkl-a30f` Security Target Artifact (2026-07-11)

## Status

The Security target artifact introduced in commit
`cc114f49f6af992e312a7f61790fad269306820e` is retired and removed from the
active experiment-artifact path.

It must not be used as:

- a training or relabeling source;
- an evaluation target or comparison lens;
- evidence that the selected Security cases were reviewed under the active
  Critic input contract; or
- provenance for a later full-corpus target pass.

## Why it was retired

The artifact copied `student_visible_label` from the historical `twinkl-747`
reachability audit into `new_label`. That legacy audit arm did not match the
active `window_size: 1` Critic state:

- it omitted the normalized declared-value profile that the Critic receives;
- it retained persona name, age, profession, culture, and entry date that the
  Critic does not receive; and
- it used the Judge-session text serialization rather than the runtime
  `concatenate_entry_text` serialization.

The old reachability buckets also attributed a full-context delta to prior
trajectory even though the full-context arm added biography and previous
entries together. The audit could observe a combined context change, but it
could not identify which added source caused it.

`training_ready: false` prevented immediate retraining, but leaving a parquet
named `security_target_variant.parquet` in the active artifact tree created an
avoidable reuse risk.

## Historical receipt

Git history preserves the original files at the commit above. Their SHA-256
digests were:

- `audit_summary.json`:
  `8ddc7ce98748fca63392e21c0d2beee0098196b93b582295c68e3c2469ba9fc3`
- `security_target_variant.parquet`:
  `24a7b20e5b2a2b9e6622912baf6dee5543a986fb20d94d670586b2740d5115cc`

The original `twinkl-747` prompts, results, report, joined table, and manual
review workbook remain unchanged as historical diagnostic evidence.

## Replacement gate

The replacement policy is `security_active_critic_state_v1`. A target variant
can be materialized only after a separate review uses:

- the runtime-formatted current journal session;
- the exact normalized ten-dimensional profile vector; and
- no date, demographics, biography, history, labels, rationales, or generation
  metadata.

The review manifest binds the canonical state input and rendered prompt with
SHA-256 digests. The target builder requires complete, receipt-matching review
results and has no fallback to any legacy `twinkl-747` label. Even after that
review, the selected 14-case frozen-test subset remains diagnostic-only and is
not a training or evaluation target.
