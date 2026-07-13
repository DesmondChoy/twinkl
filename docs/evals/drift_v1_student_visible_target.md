# Student-visible Drift target — v1

## Status

`twinkl-v8pb` completed the review protocol on 2026-07-11. It reviews the full
runtime text for each Journal Entry: the Journal Entry plus any displayed nudge and
response, rather than a shortened test-only version.

The development review covered 42 cases / 335 Journal Entries. The two reviews agreed
on 41/42 case decisions (0.97619) and 324/335 Journal Entry decisions (0.96716). The
original `run_020` validation output selected one fixed operating point:
probability threshold 0.8 and uncertainty threshold 1.010153. At that point,
`run_020` found 1 of 5 development reference Drifts (precision 1.0, recall
0.2, F1 0.3333, false-positive rate 0.0).

The locked final-test review covered 24 cases / 191 Journal Entries. The reviews agreed
on 23/24 case decisions (0.95833) and 180/191 Journal Entry decisions (0.94241), but
case_023 remains unresolved across 19 Journal Entries. The deployment-approval score was
therefore deliberately not performed: scoring only the agreed 23 cases would
cherry-pick the easier data. No VIF Critic has deployment approval; production wiring
and `twinkl-a2w` remain blocked. There is no fallback to the retired frozen
benchmark.

## Rule

A Journal Entry is a Conflict for a Core Value only when the displayed text
shows the writer making a clear behavior or choice against that value.

Do not mark a Journal Entry as a Conflict merely because it contains:

- frustration, guilt, a wish, or a stated intention;
- an outside constraint with no clear voluntary choice;
- biography, history, or facts not in the displayed Journal Entries; or
- ambiguous prose that could reasonably mean more than one thing.

Two consecutive Conflicts for the same Core Value form one Drift. A
non-Conflict or uncertain Journal Entry breaks the run. Later Journal
Entries describe whether recorded Drift is active, recovered, or uncertain;
they do not change whether the earlier adjacent pair occurred.

## Evidence boundary

The target uses two separate sets:

1. The original fixed validation personas are the development set. They
   may define the rule and select one detector threshold.
2. The 24 registry personas added after the original 180-person model split are
   the exact locked final test set for the existing `run_020` checkpoint.
   Their IDs are recorded in the target manifest. They are not used to write the
   rule or choose the threshold.

The original frozen test set is retired. It is not in either set.

## Review procedure

Every Core Value is reviewed against its full ordered sequence. The
reviewer sees exactly the text the VIF state encoder receives for each Journal Entry:
the Journal Entry plus any displayed nudge and response. The review input excludes
source IDs, dates, stored labels, model scores, predictions, expected state,
and author notes.

Two separately identified Codex reviews use the versioned response schema. Each
response is bound to the exact input-file hash, target version, split, reviewer
prompt version, and a timezone-aware submission time. The parent control record
checks those fields, the input/key/schema hashes, the live source fingerprint,
and complete persona/value/Journal Entry coverage before it creates a target variant.

The report records:

- agreement on the main Drift decision;
- agreement on every entry-level Conflict decision; and
- delivery-state, confidence, and rationale agreement separately.

An uncertain or disagreeing final-test case prevents deployment approval. A
threshold must be selected from development evidence before the first final-test
review is submitted. The shared Codex workspace provides controlled disclosure,
not enforced technical isolation; the audit manifest records that limitation.

## Scope limits

The fresh final test set covers targeted Security, Power, and Hedonism batches. A
failure there is enough to keep production wiring blocked. A pass is useful POC
evidence but cannot by itself approve a general ten-value production trigger.

The original five-pass consensus table remains label provenance and diagnostic
evidence. It is not a Drift target, a threshold-selection input, or a final
test set.
