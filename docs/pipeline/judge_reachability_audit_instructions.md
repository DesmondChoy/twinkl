# Judge Reachability Audit Instructions

Instructions for running the `twinkl-747` judge reachability audit with **Claude Code, Codex, or any other agent** that can read files, write files, and follow structured instructions.

> **Status:** Completed on 2026-03-20; retained as the reproducible workflow for the one-off teacher-student reachability audit.
> **Goal:** Measure how much the persisted Judge labels depend on context the student Critic cannot see.
> **No API keys required:** This workflow assumes the judging is done inside an agent session, not through a local API client.

This document is intentionally **LLM-agnostic**. It does not assume Claude-specific commands or Codex-specific tools. If your environment supports subagents or background workers, you can use them. If it does not, run the same steps sequentially.

The finalized bundle for the completed run lives in `logs/exports/twinkl_747/`. Its final report recommends `change_distillation_target`, driven primarily by severe `security` mismatch between the stored labels and the rerun conditions.

---

## What This Audit Produces

The audit answers three questions on the same 50-case sample:

1. **Persisted label vs full context**
   - Are the stored labels in `judge_labels.parquet` reproducible under the rich prompt path?
2. **Full context vs profile only**
   - How much does biography and trajectory context change the label?
3. **Profile only vs student visible**
   - How much does declared profile information change the label once bio/history are removed?

The final output is a short report recommending one of:
- keep current labels
- targeted relabeling
- change distillation target

---

## Prerequisites

Before running the audit:

1. Activate the environment:

```bash
source .venv/bin/activate
```

2. Confirm the audit scripts exist:

```bash
ls scripts/journalling/twinkl_747_prepare_audit.py
ls scripts/journalling/twinkl_747_summarize_audit.py
```

3. Confirm the source artifacts exist:

```bash
ls logs/judge_labels/judge_labels.parquet
ls logs/wrangled
ls logs/annotations
ls logs/experiments/runs/run_020_BalancedSoftmax.yaml
```

---

## Workflow Overview

The audit has four phases:

1. **Prepare bundle**
   - Build the deterministic 50-case sample.
   - Render prompt bundles for the three rerun conditions.
   - Create the manual review workbook and packets.

2. **Run the three judging conditions**
   - `full_context`
   - `profile_only`
   - `student_visible`

3. **Complete manual review**
   - Especially the blind text-only review for the top 3 hard cases in each hard dimension.

4. **Summarize**
   - Generate flip tables and the final recommendation report.

---

## Phase 1: Prepare the Audit Bundle

Run:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_747_prepare_audit.py
```

By default this writes to:

```text
logs/exports/twinkl_747/
```

You may override the output directory:

```bash
python scripts/journalling/twinkl_747_prepare_audit.py --output-dir /tmp/twinkl_747_bundle
```

### Expected Files

After preparation, the bundle directory should contain:

```text
logs/exports/twinkl_747/
├── README.md
├── sample_manifest.csv
├── manual_review_blind_packet.md
├── manual_review_reference.md
├── manual_review_workbook.csv
├── prompts/
│   ├── full_context.jsonl
│   ├── profile_only.jsonl
│   └── student_visible.jsonl
└── results/
    ├── full_context_results.jsonl
    ├── profile_only_results.jsonl
    └── student_visible_results.jsonl
```

### What the Prep Script Already Decides

The prep script already fixes:
- the 50-case sample
- the focal dimension for each case
- the exact prompt text for each condition
- the output `case_id` keys

Do **not** resample the cases and do **not** rewrite the prompt wording by hand. The point of the audit is to keep the conditions reproducible.

If you are inspecting the already-completed `logs/exports/twinkl_747/` bundle, do not rerun the prep step in place. Create a fresh output directory instead.

---

## Phase 2: Run the Three Judging Conditions

You do **not** need to invent prompts for this phase. The prompts are already rendered in:

- `prompts/full_context.jsonl`
- `prompts/profile_only.jsonl`
- `prompts/student_visible.jsonl`

Each line is one case with:
- `case_id`
- `condition`
- `dimension`
- prompt metadata
- `prompt` — the exact text the agent should judge from

### Output Format

For each condition, write one JSON object per line to the matching file in `results/`:

```json
{
  "case_id": "security__013d8101__1",
  "scores": {
    "self_direction": 0,
    "stimulation": 0,
    "hedonism": 0,
    "achievement": 0,
    "power": 0,
    "security": 1,
    "conformity": 0,
    "tradition": 0,
    "benevolence": 1,
    "universalism": 0
  },
  "rationales": {
    "security": "Feels steady during budget cuts because her job is permanent.",
    "benevolence": "Offers to help a colleague update her CV after work."
  }
}
```

Rules:
- `case_id` must match the prompt bundle exactly.
- `scores` must contain all 10 Schwartz dimensions.
- Every score must be exactly `-1`, `0`, or `1`.
- `rationales` is optional for the summarizer, but strongly recommended for audit review.
- Write exactly one result row per `case_id` per condition.

### Condition Semantics

Treat the three condition files as **already-authoritative**:

- `full_context`
  - rich teacher comparison baseline
  - includes session text, core values, bio, and previous entries
- `profile_only`
  - removes bio and previous-entry history
  - keeps the declared core values
- `student_visible`
  - removes bio, declared core values, and previous-entry history
  - leaves only the current session text visible

Do **not** try to “fix” or reinterpret the condition yourself. If the prompt bundle says a field is omitted, keep it omitted.

### Execution Strategy

Use whichever mode fits your agent environment:

#### Option A: Sequential

Work through one condition file at a time:
1. read one line from `prompts/<condition>.jsonl`
2. judge that prompt
3. append one JSON line to `results/<condition>_results.jsonl`
4. repeat until all 50 cases are done

#### Option B: Parallel Workers

If your environment supports parallel workers/subagents:
1. split one condition file into chunks
2. assign each chunk to a worker
3. have each worker return valid JSONL rows
4. merge them into the final `results/<condition>_results.jsonl`

When merging:
- keep exactly one row per `case_id`
- do not reorder fields in a way that drops information
- verify the final file has 50 rows

### Recommended Worker Prompt

If you are using subagents, give each worker an instruction like:

```text
Read the assigned prompt rows from logs/exports/twinkl_747/prompts/<condition>.jsonl.
For each row:
- use the provided prompt exactly as written
- produce one JSON object with case_id, scores, and optional rationales
- write valid JSONL rows only
Do not change case_id values.
Do not skip any assigned rows.
```

---

## Phase 3: Complete Manual Review

The bundle contains two review aids:

- `manual_review_blind_packet.md`
  - current session text only
  - intended for blind text-only judgments on the highest-priority hard cases
- `manual_review_reference.md`
  - richer context reference for all sampled cases

Record review judgments in:

```text
manual_review_workbook.csv
```

### Minimum Manual Review Requirement

At minimum, complete the blind-review rows for:
- the top 3 `security` cases
- the top 3 `hedonism` cases
- the top 3 `stimulation` cases

For each reviewed row:
1. judge the focal dimension from text alone
2. then read the richer context
3. record whether the label changes
4. record the preferred target if the richer label seems unreachable from text alone

### Workbook Fields That Matter Most

The summarizer pays attention to:
- `blind_text_label`
- `rich_context_label`
- `preferred_target`
- `recommendation_notes`

Suggested `preferred_target` values:
- `keep_current`
- `reduced_context`
- `text_only_ambiguous`

Use consistent spelling so the final report stays easy to interpret.

---

## Phase 4: Summarize the Audit

After the three result files are complete, run:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_747_summarize_audit.py
```

Or point it at a specific bundle:

```bash
python scripts/journalling/twinkl_747_summarize_audit.py --bundle-dir logs/exports/twinkl_747
```

This writes:

```text
logs/exports/twinkl_747/reachability_audit_report.md
logs/exports/twinkl_747/joined_results.csv
logs/exports/twinkl_747/comparison_rows.csv
logs/exports/twinkl_747/flip_summary.csv
```

### What the Summary Script Checks

It compares:
- persisted label vs full-context rerun
- full-context rerun vs profile-only rerun
- profile-only rerun vs student-visible rerun

It then maps hard-dimension flip counts into:
- `low` = 0–1 flips
- `ambiguous` = 2–3 flips
- `substantive` = 4+ flips

And combines that with the manual review workbook to recommend:
- keep current labels
- targeted relabeling
- change distillation target

---

## Quality Checks Before You Finish

Before declaring the audit complete, verify:

1. `sample_manifest.csv` has 50 rows.
2. Each result file has 50 JSONL rows.
3. Every `case_id` in each result file appears exactly once.
4. `manual_review_workbook.csv` has been filled for the blind-review priority rows.
5. `reachability_audit_report.md` was regenerated after the latest result/manual-review edits.

Quick row-count check:

```bash
wc -l logs/exports/twinkl_747/results/*.jsonl
```

---

## Notes for Fresh Agent Sessions

- Read `docs/prd.md` first.
- Read `docs/pipeline/claude_judge_instructions.md` for the original judge workflow context.
- The reachability audit is **not** a full relabeling pass.
- Do not modify the main `judge_labels.parquet` dataset in this workflow.
- Treat this as measurement and reporting, not training-time intervention.
- Keep all outputs inside the audit bundle so the run is reproducible.

---

## Related Files

| File | Purpose |
|------|---------|
| `scripts/journalling/twinkl_747_prepare_audit.py` | Build the sample, prompt bundles, and review materials |
| `scripts/journalling/twinkl_747_summarize_audit.py` | Aggregate condition outputs into the final report |
| `prompts/judge_alignment.yaml` | Base rich-context judge prompt template |
| `src/judge/labeling.py` | Prompt rendering helpers used by the prep script |
| `docs/pipeline/claude_judge_instructions.md` | Historical judge labeling workflow |
| `logs/exports/twinkl_747/README.md` | Bundle-local execution reminder after prep is run |
