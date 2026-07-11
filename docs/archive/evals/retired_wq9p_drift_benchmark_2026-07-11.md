# Retired consensus-derived drift benchmark — 2026-07-11

**Status: historical evidence only — do not rerun, score, tune, or promote from it.**

The consensus-derived episode benchmark was retired after the `twinkl-16ar`
audit found that its frozen cases did not reliably meet the student-visible
drift rule. Its scripts, artifacts, report, and dedicated tests are deliberately
absent from the active repository.

The original five-pass consensus table remains unchanged as label provenance and
diagnostic evidence. It is not an active drift reference, a threshold-selection
input, or a promotion surface.

Historical record: commits `e33c2f3`, `e3f1a1f`, and `81c1954`; closed Beads
issues `twinkl-wq9p` and `twinkl-16ar`; GitHub issue #48.

The active successor is `twinkl-v8pb`, which must publish a separate repaired
target and untouched promotion surface before scorer promotion or production
trigger wiring.

## Later outcome

`twinkl-v8pb` did publish the separate full-runtime-text target and ran its
locked review on 2026-07-11. One 19-entry promotion case remained unresolved,
so it deliberately did not score or promote `run_020`. Production wiring is
still blocked. See the [student-visible target record](../../evals/drift_v1_student_visible_target.md).
