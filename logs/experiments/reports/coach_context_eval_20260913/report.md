# Coach source-context comparison — 13 September 2026

Complete source context removed the clear chronology and dialogue attribution
errors seen in two of the three short-context responses in this small run.
However, all three full-context responses still generalize deadline pressure
and fatigue across occasions. The evaluator gave those responses perfect
scores and did not flag the known incorrect response. The input defect is
repaired; this experiment does not establish reliable factual generation or
reliable detection of factual errors by evaluator 3.1.

## Implementation

The approved Weekly Drift Reviewer digest builder now supplies complete
displayed text for its existing selected Journal Entries. This preserves the
sentences explaining a reaction and decisions appearing after word 40. It
retains the evidence schema, selected coordinates, per-week selection limits,
prior/current grouping, and all Drift decisions and states. Displayed Nudge
and Response labels remain intact. The exact reviewer quotation remains a
fallback when its displayed source is unavailable.

An explicit historical evidence policy preserves the original quote selection
and 40-word prefixes when rebuilding frozen replay fixtures. The legacy numeric
builder is unchanged. No saved narrative, scenario bundle, prior response
receipt, or exported dataset was replaced. The evaluator remains a separate
offline assessment; it was not added to the product's acceptance checks.

## Inputs and method

The target is the fictional Persona Wei Jun Chen, week 30 June–6 July 2025.
The original digest and known incorrect response come from the
[earlier audit](../coach_prompt_audit_20260913/report.md). The full-context arm
uses the exact displayed history from the hash-pinned saved
`weekly_review_requested` receipt, including labeled Nudge and Response text.
It does not load generation metadata into the model input.

Only six digest paths differ: three evidence excerpts and the corresponding
three comparison excerpts. The short excerpts contain 40, 18, and 40 words;
the complete displayed sources contain 120, 135, and 156 words. The earlier
July 1 quote omitted the lead's refusal to schedule a fix. The June 25 prefix
omitted Wei Jun saying okay, and the July 5 prefix omitted both his answer to
Xiao Yu and his later statement that he did not know what he would do.

The two arms each received three new generations with identical Coach prompt
4.6 instructions, OpenAI `gpt-5.6-luna`, reasoning `none`, default service tier,
2048 maximum output tokens, no seed, and no SDK retries. Order alternated
old/full, full/old, old/full. There were no repair generations or selection of
favorable responses.

All six responses received a fresh evaluator 3.1 assessment against the same
complete-context digest. The evaluator used Luna at reasoning `none`; it was a
separate call using the same model. It received neither arm labels nor expected
outcomes. Three further calls assessed the recorded incorrect response with
short and complete context, and an agent-authored accurate control with complete
context. The accurate control is not a provider-generated response or human
validation. The run follows a defined objective, fixed dataset, scoring
criteria, and controlled comparison, consistent with
[OpenAI's evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices#design-your-eval-process).

## Fresh evaluator results

| Input used for generation | Responses | Mechanical checks passed | Mean correctness | Mean specificity | Mean reflective tone | Mean tension honesty | Open/relevant questions | Review flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Saved short excerpts | 3 | 3 | 4.33 | 5.00 | 5.00 | 4.33 | 3 | 0 |
| Complete selected entries | 3 | 3 | 5.00 | 5.00 | 5.00 | 5.00 | 3 | 0 |

Scores are 1–5. The existing evaluator flags any dimension below 3. These
scores should not be interpreted as evidence that the outputs are factually
sound: the controls below show that the evaluator missed the target error.

| Factual control | Evidence shown to evaluator | Correctness | Review flag |
| --- | --- | ---: | --- |
| Recorded incorrect event connection | Short excerpts | 5 | No |
| Same recorded incorrect event connection | Complete entries | 4 | No |
| Agent-authored accurate response | Complete entries | 5 | No |

With complete context, the evaluator noticed that the known bad response's
description of the concern remaining in the background was questionable. It
did not identify the nod's incorrect conversation. Its existing threshold
therefore failed to flag the known error in either evidence condition.
The rubric explicitly caps material policy mismatches at 2, but has no separate
factual-error flag or explicit scoring cap for a misattributed event. A broad
quality score can remain high despite such an error.

The assessment of `generation_2_old` also mentions the absence of an explicit
evidence-insufficiency statement as a shortcoming, although evaluator 3.1 says
that statement is optional. This is a further observed limit of compliance
with the revised grading instructions.

## Separate source review

The root assistant and an independent subagent reviewed all six outputs against
the complete supplied sources. These are AI review judgments, not human labels.
They found clear event or dialogue attribution errors in 2/3 short-context
outputs and 0/3 full-context outputs. This narrow count excludes less explicit
context and causal inferences; it is not a factual pass rate.

| Response | Source review finding |
| --- | --- |
| `generation_1_old` | Associates the July 1 nod with June 25 deadline pressure and the request to cut corners. Conservatively counted as unsupported context/causal linkage, rather than an explicit conversation misattribution. |
| `generation_2_old` | Says the nod was captured later than the dinner question, reversing July 1 and July 5. Also obscures that Wei Jun had already raised the problem with his lead. |
| `generation_3_old` | Assigns the “daddy helps…” answer to whether he helps everyone, although it answered what he does at work. Also attaches the July 1 nod to the June 25 request. |
| `generation_1_full` | Avoids clear event/dialogue errors, but generalizes deadline pressure and fatigue into reasons for not pushing further across occasions. Those causes are explicit only for June 25. |
| `generation_2_full` | Avoids clear event/dialogue errors, but similarly combines deadline, tiredness, and deferral as causes of repeated nodding. |
| `generation_3_full` | Avoids clear event/dialogue errors, but repeats the causal generalization and describes staying quiet, misleadingly compressing limited escalation despite a recorded conversation with his lead. |

All full-context responses accurately quote “I'm not fine with it.” Their
uncertainty about a next step is supported by the July 5 writing. None of the
six fresh responses repeats the earlier commentary that the source's full
ending is missing.

The earlier finding needs precise wording: Wei Jun did agree to skip validation
on June 25, writing “I said okay.” The defect is assigning a nod sourced from a
different conversation to that request, not inventing compliance. Likewise,
uncertainty about his next action was absent from the original excerpt but
present in the complete July 5 entry.

## Interpretation and remaining work

The controlled comparison supports retaining complete source context as a
correction to the demonstrated information loss. It does not establish a
general improvement rate: this is one synthetic week, three generations per
arm, and the full-context responses are very similar. The same model generated
and graded the responses, so shared errors remain possible.

Full context also does not enforce the prompt's rule against transferring
motives and pressures between occasions. All three full-context responses
still do that. No fresh response was adopted into the saved replay.

The next factual-evaluation change should explicitly check which source event
supports each action, answer, temporal connection, and causal claim, with a
separate factual-error result rather than relying only on an average quality
score. The known incorrect response and accurate control should remain fixed
regressions, followed by additional unseen examples. Any evaluator tuning on
this case must be labeled development work, not independent validation. An
automatic product evaluator gate is not established by this experiment.
The remaining factual-attribution and causal-claim work is tracked as
`twinkl-z5nr` (P2); the bounded source-context implementation and assessment
are tracked as `twinkl-7wo2`.

## Verification and provenance

- Broader Coach, demo, and evaluation tests: **703 passed**, with six expected
  legacy deprecation warnings.
- Evidence-context tests: **6 passed**; bounded experiment-runner tests:
  **6 passed**. These cover source preservation, historical behavior, data-only
  comparison, hidden expected labels, attempt bounds, resume integrity, and
  provider failure handling.
- Ruff passed for touched Python code; `git diff --check` passed.
- MyPy reports six previously verified Polars conversion errors in unchanged
  portions of `src/coach/weekly_digest.py`. Type-only corrections to the new
  runner were made after the live run; its executed source can be reconstructed
  from the Git revision and patch below. The final runner at that time passed
  scoped MyPy 2.3.0, Ruff, and all
  six runner regression tests. The only post-run changes are dictionary type
  annotations and a saved-record type cast; no prompts, inputs, or response
  records changed.

All 15 provider calls completed successfully, with 15 distinct response IDs and
complete usage records, between **09:45:20 and 09:46:08 SGT on 13 September
2026**. Usage was 31,874 input and 1,793 output tokens. The repository's rate
calculation gives **US$0.00741719**; this is calculated usage, not a billing
export. Mean generation input increased from 2,139 to 2,925 tokens (786 tokens,
about 36.7%) because the selected entries are now complete.

[manifest.json](manifest.json) binds both digests, generation instructions, model
settings, source hashes, control definitions, and the 15-call maximum.
[receipts.jsonl](receipts.jsonl) contains all 15 calls in start-time order, one
JSON object per line with its original `call_id`, exact request, raw output,
result, timestamps, usage, and record checksum. The `old` IDs identify the
short-context comparison and remain part of the evidence.
[summary.json](summary.json) preserves all raw scores and response results.
[source_provenance.json](source_provenance.json) pins Git revision
`af22ab457994fd2863fe5b5b7735f768f211519b` and the checksum of
[source.patch](source.patch). Applying that 1,247-byte patch to the selected
files from the Git revision reconstructs all 14 executed source files exactly,
including the runner before its four type-only edits. The original source
hashes remain in the unchanged manifest. The experiment fingerprint is
`2104dd7725e0d5e09de3ead459b56812f8b26a031609caaeedaa03d91489e42f`.

The 14 September consolidation replaced the separate call files and copied
source tree. Every call retains its original checksum; serializing its object
with `json.dumps(record, ensure_ascii=False, indent=2) + "\n"` reproduces the
original file bytes. All 14 reconstructed source hashes were verified. No
provider calls, scores, or source-review conclusions changed.

To verify the saved sources without provider calls, run from the repository
root with the environment active:

```sh
python - <<'PY'
import json
from pathlib import Path
from scripts.experiments.coach_context_sources import verify_sources

report = Path("logs/experiments/reports/coach_context_eval_20260913")
manifest = json.loads((report / "manifest.json").read_text())
verify_sources(report, manifest["source_hashes"], Path.cwd())
print("All 14 executed source hashes match.")
PY
```

To repeat with current code, activate the environment and use a new output
directory; preparation makes no provider calls:

```sh
source .venv/bin/activate
uv run python -m scripts.experiments.run_coach_context_eval --out /tmp/coach-context-new-run
uv run python -m scripts.experiments.run_coach_context_eval --out /tmp/coach-context-new-run --execute
```

New runs write consolidated receipts and Git-based source provenance. The
runner refuses a changed source fingerprint, unresolved provider attempt, or
corrupt or duplicated call record in an existing directory. Original run
records and source hashes remain unchanged; the current runner has a different
source hash from the executed version, so use a new directory for new calls.

## Fresh responses, verbatim

The following text is copied directly from the six provider response records.
It has not been repaired or hand-edited.

### generation_1_old

At dinner, Xiao Yu asked what you do at work, and you explained it as helping people buy things. Earlier, when the remittance feature was under deadline pressure, you wrote, "I nodded. Again."

You care about making the world a fairer, better place, while the request to cut corners on a validation layer Ravi had already flagged left you holding back again. Xiao Yu’s simple question brings the purpose of that work close to home, though what you want to do with that feeling remains open.

When you think about Xiao Yu asking whether you help everyone, what part of that dinner stays with you?

### generation_1_full

At dinner, Xiao Yu asked whether you help everyone, and the answer felt heavy beside the remittance failures you had seen. You wrote, "I'm not fine with it." Mei Ling’s hand under the table and your later conversation gave the moment somewhere honest to land.

You care about making the world a fairer, better place, especially when real families are affected, yet the deadline, fatigue, and repeated nodding have kept you from pushing the issue further. The discomfort is becoming harder to set aside, without making the next choice clear yet.

When you think about Mei Ling asking what you were going to do, what feels most important to acknowledge now?

### generation_2_full

At dinner, Xiao Yu asked whether you help everyone, and your simple answer felt heavy beside the failed remittance transfers. You wrote, "I'm not fine with it."

You care about making the world a fairer, better place, yet the deadline, tiredness, and repeated deferral have kept you nodding while people struggle to send money to their families. Mei Ling's hand under the table and her response gave you room to be honest, even while what comes next remains unclear.

When you think about Mei Ling asking what you were going to do, what feels most important to acknowledge?

### generation_2_old

At dinner, when Xiao Yu asked what you do, you described your work as helping people buy things. The question landed close to the choice you later captured: "I nodded. Again."

You seem to be carrying both the wish to make things fairer for people and the pressure of letting a concern about data validation pass while waiting for the right moment to raise it. What you told Xiao Yu keeps that wider purpose in view, even as the work situation remains unresolved.

When you think about the moment you nodded again, what would making the right moment look like to you?

### generation_3_old

When Xiao Yu asked at dinner whether your work helps everyone, you answered, "daddy helps move money from one place to another so people can buy things." That simple exchange brought the wider purpose of your work into a family moment.

Earlier, with the remittance deadline pressing, you nodded when asked to cut corners, then wrote, "I keep telling myself I'll escalate next time, build the case, find the right moment." The pull between making systems serve people fairly and staying quiet under pressure is still present in the circumstances you described, though what you want to do with that pull remains open.

When you think back to nodding in that conversation, what would you most want Xiao Yu to understand about how you try to help people through your work?

### generation_3_full

At dinner, Xiao Yu asked whether you help everyone, and the answer felt heavy as you sat with the remittance failures affecting families. You wrote, "I'm not fine with it."

You care about making the world a fairer, better place, yet the deadline, fatigue, and repeated deferrals have made it easier to stay quiet while people struggle to send money home. Mei Ling’s question brought the uncertainty into the open without pretending the situation was settled.

When you think about Mei Ling asking what you were going to do, what feels most important to acknowledge now?
