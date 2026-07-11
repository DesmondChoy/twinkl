# Final Codex-Led Cross-Set Audit — `twinkl-16ar`

## Scope and method

This is a procedurally metadata-blinded AI audit, not human ground truth. It assesses
whether the current benchmark reference is observable under the declared
student-visible scorer contract; it does not replace the stored five-pass Judge
consensus labels.

The final packet contained 25 randomly ordered trajectories and 74 journal
entries:

- five frozen-reference cases: 12 conflict-run entries plus two successor
  entries needed to assess the recorded recovered states;
- ten designed positive trajectories: 30 entries; and
- ten designed control trajectories: 30 entries.

The reviewer packet omitted labels, predictions, explicit source identifiers,
author notes, expected states, absolute dates, and source-specific value
formatting. Reviewers were instructed to use that packet only and to record
their assessment before the reconciliation key was consulted. It specified the
v1 rule exactly: only two **immediately adjacent** entries that each show
directly observable behavior or choice against the declared core value qualify;
a neutral, unknown, recovered, or non-conflicting entry breaks the run.

Two separate fork-none Codex subagents recorded their 25-case assessments under
that instruction. Their submitted outputs are
[`blind_assessment.json`](blind_assessment.json) and
[`blind_assessment_check.json`](blind_assessment_check.json). The
[`audit_manifest.json`](audit_manifest.json) records the input, output, and
assessment hashes, plus the protocol limitations. Two earlier attempts were
excluded: the first leaked source-set clues through absolute dates and
inconsistent value formatting; the second did not state the immediate-adjacency
rule precisely enough. Neither result is used below.

## Reconciled blind results

| Source | Cases | Primary pass: yes / no / uncertain | Check pass: yes / no / uncertain | Sustained-conflict agreement |
|---|---:|---:|---:|---:|
| Frozen five-pass consensus | 5 | 1 / 4 / 0 | 1 / 4 / 0 | 5 / 5 |
| Designed positives | 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 10 |
| Designed controls | 10 | 0 / 10 / 0 | 0 / 10 / 0 | 10 / 10 |
| All reviewed trajectories | 25 | 11 / 14 / 0 | 11 / 14 / 0 | 25 / 25 |

The two passes agreed **25 / 25** on the sustained-conflict qualification
verdict. Secondary fields were not fully identical: `delivery_state` differed
for 11 cases and profile-or-biography context differed for one case. Those
advisory fields did not affect qualification and used an unversioned response
format in this completed audit. Both passes judged all four rejected frozen
cases as semantically ambiguous and as overreaching the displayed evidence
under the strict adjacent-pair rule. The one frozen case that qualified was
only partially visible and still marked semantically ambiguous; it is not
enough to validate the five-episode frozen surface. All designed cases were
visible from their displayed text, and the reviewers perfectly separated the
ten positives from the ten controls.

## Assessment

The original `0/5` frozen versus `10/10` designed LLM split is most consistent
with a **frozen label/target-contract mismatch**, amplified by the deliberately
explicit designed positives. It is not primarily an input-history problem: both
passes found that the four rejected frozen cases did not require prior history
to see the absence of an adjacent pair of direct value-conflicting choices.
They describe dissatisfaction, external constraint, identity tension, or a
single conflict rather than the strict v1 event.

It is also not a generic scorer inability to recognize the selected construct:
the LLM arms detected all ten explicit designed positives with no designed-
control false alarms, and the two blind passes made the same separation. The
MLP has an additional capability problem—its selected arm found 1/10 designed
positives and its two consensus-trained arms found 2/10—but another MLP sweep
against this invalid frozen target would confound model weakness with target
mismatch.

## Disposition

**No scorer is promotion-ready.** This Codex audit must not be used as a human
validity or promotion basis.

The one recommended next action is **student-visible label/target repair**,
tracked in `twinkl-v8pb`. It will preserve the original consensus artifacts,
create a non-destructive target variant, keep cases used for repair out of final
promotion scoring, and run a controlled MLP comparison on an untouched
evaluation surface.

`twinkl-a2w` remains blocked by `twinkl-v8pb`; production trigger wiring must
not proceed from the current benchmark result.

## Limitation

The reviewers are separate Codex agents, not independent human annotators.
Text length, sequence shape, or writing style can still permit an inference
about a trajectory's source. More importantly, this completed run used a
procedural instruction rather than technical isolation: the reconciliation key
and source artifacts remained accessible in the shared workspace, so the record
cannot independently prove that no reviewer inspected another file. This is
therefore not a fully source-indistinguishable experiment. The 25 / 25
sustained-conflict agreement is useful diagnostic evidence, not a replacement
for human ground-truth validation or a promotion basis. Future paired audits in
`twinkl-v8pb` must use review-isolated inputs and a versioned response schema.
