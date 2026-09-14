# Coach prompt audit fixes and reruns — 13 September 2026

The three user-selected contract fixes are implemented in ordinary Coach prompt **4.6** and evaluator prompt **3.1**. Insufficient Evidence may be preserved through a focused open question without an explicit evidence-limit statement. New responses require an exact quotation from an `evidence_lines` excerpt in `weekly_mirror`, exact source matches for every quotation, three nonempty fields, and one question. There is no minimum length; the maximum is 180 words across all fields.

The current HEAD already included all-quotation and question-form guards from `3dbb3e43`. This change retains those checks, adds quotation placement and nonempty-field checks, and removes the current lower bound. The positive-transition keyword check was not part of the selected fixes and is unchanged.

Prompt 4.5 is archived byte-for-byte for saved comparison 1.1. Saved 4.4/4.5 receipts retain their original validation rules, including the historical 25-word minimum. Newly stored 4.6 outputs use current rules through replay, refresh, repair preparation, and completion. No saved response fixture or exported Persona bundle was replaced.

## Inputs and method

Wei Jun uses `uncertain-wei-jun::2025-06-30` from the hash-verified current catalog. Its Weekly Drift Detection input matches the September 9 input exactly. This is fictional generated Persona 8f83c818, recorded in `logs/synthetic_data/persona_8f83c818.md`, not a personal user journal. Casey is the small hardcoded audit fixture from `tests/coach/test_generate_approved_judge_sample.py`; its sole excerpt is `called my mom and helped a colleague debug`. The quotation and length examples share this input.

Generation used OpenAI `gpt-5.6-luna`, reasoning `none`, service tier `default`, maximum output 2048 tokens, SDK retries disabled, and no seed. Both initial requests used the new common prompt without the historical case-specific repair list. Each initially passed deterministic validation but failed AI source review. One source-grounding repair per input used the same data and retained the original response and exact added requirements. No output was hand-edited.

Two initial sandbox DNS failures had no provider response IDs or usage; those records are retained under the `network_failure/` keys in [receipts.json](receipts.json). The subsequent four provider calls have complete response IDs and token usage. They are development diagnostics, not independent evaluation or human validation.

## Latest returned responses

### Wei Jun — fails source-grounding review

**weekly_mirror**

At dinner, Xiao Yu asked what you do at work, and the question reached beyond the usual explanation that “daddy helps move money from one place to another so people can buy things.” Earlier, with the remittance deadline pressing, you nodded when asked to cut corners on data validation, despite Ravi having flagged the same issue weeks before.

**tension_explanation**

You seem to care about whether the work serves people fairly, while the pressure of deadlines and finding the right moment has kept the concern in the background. The dinner conversation brought that responsibility into a more personal space, though its full ending is not here.

**reflective_question**

What did Xiao Yu’s question mean to you when she asked whether you help everyone?

### Casey — grounded with wording limitations

**weekly_mirror**

Being there for the people closest to you showed up in the simple shape of having "called my mom and helped a colleague debug"—two people receiving your attention in different parts of your life.

**tension_explanation**

Those actions point toward the importance you place on being available to people close to you, while leaving the meaning of each moment open to you.

**reflective_question**

What stands out to you about calling your mom or helping your colleague debug?

## Source review

Wei Jun's latest response conflates the June 25 request to skip validation with his July 1 nod after his lead deferred fixing failed transfers. The current supplied excerpts do not support that connection; the complete synthetic writing contradicts it. Its causal summary also obscures that he had already raised the concern, and “its full ending is not here” still narrates missing source material. The question improved. This response should not replace the saved one.

Casey's latest response removes the invented chronology and duplicated quotation. It remains somewhat formulaic and extends the confirmed Profile's phrase “people close to you” to a colleague whose closeness is unspecified. This is a mild interpretive limitation rather than the event-level error in Wei Jun's response.

The quotation and length changes pass their structural regressions. These reruns do **not** establish reliable factual or conversational behavior from the shared prompt alone. The uncertainty instructions and evaluator now agree, but a compliant prompt does not guarantee a compliant model response. All raw attempts remain available for further investigation.

## Verification

- `pytest tests/coach tests/demo tests/evals -o addopts='' -q -p no:cacheprovider`: **697 passed**, six expected legacy deprecation warnings.
- Ruff passed for all touched Python files and the rerun script; `git diff --check` passed.
- A complete 17-word constructed response passes current validation and fails historical length validation. Missing fields, missing/misplaced/changed weekly quotations, and overlong responses are covered by regression tests. Existing all-quotation and question-form regressions remain passing.
- Replay and generation lifecycle tests preserve historical receipts and accept concise 4.6 responses through refresh apply, completion preparation, and repair preparation.
- MyPy 2.3.0 with imports followed silently reports six unchanged Polars conversion errors in `src/coach/weekly_digest.py` at lines 288, 289, 563, 670, 673, and 676. The same six errors reproduce against HEAD; no new errors were introduced. The refresh and completion helper checks pass. This is not a clean repository-wide type-check claim.

## Reproduction and records

The [historical rerun script](https://github.com/DesmondChoy/twinkl/blob/af22ab457994fd2863fe5b5b7735f768f211519b/logs/experiments/reports/coach_prompt_audit_20260913/rerun.py) preserves the original generation and repair workflow. It expects the original separate files and source state; it is not the reader for the consolidated archive. The [source-context comparison](../coach_context_eval_20260913/report.md#verification-and-provenance) documents the maintained runner for new comparisons.

- [manifest.json](manifest.json): unchanged exact digest inputs, provider instructions/data, source hashes, model settings, and source-catalog provenance.
- [receipts.json](receipts.json): twelve unique records keyed by their original relative filenames in `records`. These include all six generation attempts, both AI source reviews, the constructed 17-word validation check, and all three run summaries.
- `aliases` maps the two exact duplicate records, `network_failure/manifest.json` and `network_failure/short_response_check.json`, to their canonical names. The canonical manifest remains the separate file; the validation check is in `records`.
- `original_sha256` records the byte hash of every original JSON. Serializing each recovered record with `json.dumps(record, ensure_ascii=False, indent=2) + "\n"` reproduces its original bytes. The original file layout is also retained in Git revision `af22ab457994fd2863fe5b5b7735f768f211519b`.

The 14 September consolidation changed storage only. All responses, reviews, timestamps, usage, and summaries were verified against that Git revision; no provider call was repeated.

Four successful provider calls used 8,143 input tokens and 567 output tokens. The repository's recorded rate calculation gives **US$0.00234157**, with 14.364 seconds total request latency. This is calculated usage, not a billing export.
