# North Star Moment: saved-Persona coverage and case checks

> Historical scope: this report preserves the earlier checks limited to Active
> Drift, before frontend integration. The subsequent [integration report](../north_star_integration_20260906/README.md)
> supersedes its statements about omitted cards and outstanding integration by adding
> encouragement and reminders for No Active Drift and completing both frontend
> paths and browser checks. Original results below remain unchanged.

This supplement adds Meera and checks the actual saved frontend replay states.
It also diagnoses the rejected selection and unresolved history from the
[completed experiment](../north_star_luna_20260905/report.json). Original
judgments, precision, source assignments and raw responses are unchanged.
The user accepts the original result as useful POC evidence with acknowledged
limitations; its stricter original gate remains recorded as failed.

## Saved frontend coverage

All five catalog entries and their bundle hashes/provenance validate. All 36
saved weeks were projected through the existing replay function. Four weeks
have Active Drift; the other 32 are no-trigger controls under the NSM rule.
These controls are not added to quotation precision or semantic omission rates.

| Saved Persona | Saved weeks | NSM coverage outcome |
|---|---:|---|
| Meera Krishnamurthy | 7 | Both Achievement and Security remain No Active Drift throughout. All seven weeks correctly require no NSM request and no card. |
| Wei Jun Chen | 6 | The Active Drift week matches the original Universalism case exactly; its quotation was accepted. |
| Marc Vandenberghe | 8 | The Active Drift week matches the original Power case exactly; its quotation was accepted. Later No Active Drift weeks require no new card. |
| Noor Haddad | 6 | The saved Active Drift week matches the first Tradition case, which correctly omitted a quotation. Her accepted later experiment case is not the saved frontend trigger. |
| Lukas Vermeer | 9 | The saved Active Drift week concerns Conformity at onset index 4. Its separate supplementary case completed, and the evaluator accepted the selected quotation. |

Reuse requires identical Persona, Core Value definition, onset date/index and
complete original-source inputs. The additional Lukas case contains entries
0–3 only. All its writing was already verified as synthetic in the original
experiment. Meera contributes a no-trigger control, not a manufactured Drift
or a forced positive selection.

**Execution status:** Lukas's new case completed on September 6, 2026, after
the user-authorized repair of the local API environment variable name. All three
generation requests succeeded without retries: Luna low for runtime, Luna xhigh
for source evaluation and the separate exact-quotation assessment. New spending
was US$0.01085415, bringing cumulative spending to US$0.62978415.

The runtime selected entry 3: “I told Pieter I'd think about it over the
weekend.” In the complete source, Lukas delays joining an unsanctioned project
despite wanting to participate immediately, considering his manager's expected
disapproval. The evaluator accepted this as actual restraint supporting
Conformity. Two independent AI reviews also accepted it before reading the
evaluator's labels. The primary review assigned medium confidence because the
choice also reflects career concerns; the separate reviewer assigned high
confidence. This supports the observed deferral, not a final refusal or lasting
compliance. These judgments are AI review, not human validation.

All four runtime source classifications agreed with the reference. This
supplement adds one accepted selection separately; the original experiment's
24/25 result and failed strict gate remain unchanged.

This is offline coverage of the saved inputs and replay states. NSM is not yet
integrated into the frontend, so actual UI rendering, NSM dispatch suppression,
and response-availability integration still require implementation checks.

## Rejected selection: Ricardo Mendoza / Conformity

The runtime selected “I have been working ten-hour days trying to keep us on
track for the holiday release.” That establishes effort, not the restraint
specified by Conformity. Its factual explanation also changed a responsibility
in the source, “has to explain,” into a performed explanation, “tried to
explain.” The evaluator correctly withheld acceptance; the independent reviews
also withheld acceptance, with a more cautious ambiguous classification.

All eight eligible earlier entries were supplied. The runtime and evaluator
both accepted an older, stronger November 17 entry: the writer withheld a
defensive reply to the VP and instead gave a cooperative answer. The
newest-first rule allowed the newer false positive to displace that option.
This is a semantic classification error, not a missing-source or quotation-
fidelity error. It belongs in the acknowledged limitations; no broad redesign
or retrospective score change is needed.

Source: [Ricardo's original writing](../../../../logs/wrangled/persona_7c712a0a.md).

## Unresolved history: Marcus Chen / Security

Drift starts on December 8, 2025, at index 1. The sole earlier entry, on
December 1, says: “App keeps glitching today, lost two rides because of it.
My head's all over the place.” It describes external problems and distress,
without an actual protective or opposing action by the writer.

Runtime and evaluator both abstained for insufficient text, and the independent
review also abstained. No quotation was displayed; the original failed flag is
false. Unresolved means the reference cannot establish whether a valid earlier
example exists. The practical outcome is a safe omission, not a crash or an
incorrect selection. Keep that evidence category while explaining the behavior.

There is no response attached to this earlier entry. The later December 15
response is after Drift onset and hypothetical, so it cannot supply the missing
evidence. No eligible original writing was left out.

Source: [Marcus's original writing](../../../../logs/wrangled/persona_8fcd7947.md).

The separate courtesy-as-Universalism case remains officially accepted with a
medium-confidence independent disagreement. It is not a second official
rejection. Its uncertainty concerns value specificity, not fabricated action.

## Verification and reproduction

- 271 relevant NSM, supplement and scenario tests passed, including six new
  coverage/recovery checks. Scoped Ruff and MyPy passed.
- Hashes, chronology and exact eligible-source completeness passed for all
  three diagnosed cases.
- Offline execution generated no calls and preserved every original run file.
- After the paid run, offline replay with generation and counting forbidden
  reproduced the report and ledger byte for byte. All 15 original experiment
  files remained unchanged, and the frozen source/frontend hashes validated.
- All three input counts matched actual usage (1,974, 1,974 and 1,800 tokens),
  below the 16,000-token limit. Exact quotation identity and text passed.
- Original experiment results and frontend bundles remain unchanged. Capstone
  report updates remain paused.

Artifacts: [coverage and prepared case](manifest.json), [completed result](report.json),
[validation](validation.json), [primary independent review](independent_quote_judgment.json),
[separate independent review](independent_agent_quote_judgment.json),
[case diagnostics](case_diagnostics.json), and the
[reproduction script](../../../../scripts/experiments/north_star_saved_checks.py).
Reproduce the completed report offline:

```sh
source .venv/bin/activate
python scripts/experiments/north_star_saved_checks.py report
```
