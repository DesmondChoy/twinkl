# Consensus Re-judging Instructions

Instructions for running the `twinkl-754` consensus re-judging workflow with
an orchestrator agent plus short-lived worker sub-agents.

## Goal

Run the same Judge prompt path `k=5` times over all `1,651` labeled entries
using the `profile_only` prompt variant. The goal is repeated-call
self-consistency of one judge workflow, not simulated independence among five
human-like raters:

- keep `name`, `age`, `profession`, `culture`, and `core values`
- remove `bio`
- remove previous-entry history

The bundle is prepared by:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_prepare_consensus.py
```

Run the audit-recovery pilot first:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_prepare_consensus.py \
  --output-dir logs/exports/twinkl_754_pilot \
  --pilot-size 50 \
  --pilot-hard-dimensions security,hedonism,stimulation
```

The pilot bundle remains format-compatible with the full bundle. The only
difference is that `manifest.csv`, `prompts/`, `shards/`, and downstream
outputs are restricted to the selected 50 entries. The generated
`bundle_status.json` records whether a bundle is a `pilot` or `full` rerun.

Useful prepare options:

- `--labels-path` to point at a non-default persisted label parquet
- `--wrangled-dir` to point at a different wrangled corpus
- `--schwartz-path` to point at a different Schwartz value config
- `--pilot-size` and `--pilot-hard-dimensions` to build a deterministic pilot bundle

## Roles

### Main Agent

The main agent is the orchestrator only. It must:

1. prepare the bundle
2. spawn worker sub-agents
3. validate each returned shard result
4. write shard result files in the main workspace
5. merge validated shard files into per-pass result files while recording provenance
6. run the summarizer only after the merge step succeeds

The main agent does **not** judge entries itself.

### Worker Sub-agents

Use one fresh worker per shard with:

- `agent_type=default`
- `model=gpt-5.4`
- `reasoning_effort=medium`
- `fork_context=false`

Each worker gets only:

1. this instructions file path
2. one shard JSONL path
3. one target result path

Each worker must:

1. read only the assigned shard
2. judge every row in that shard
3. copy `entry_id` exactly from input
4. return valid JSONL rows only, with no prose

Close each worker immediately after its shard is accepted or rejected.

## Bundle Layout

```text
logs/exports/twinkl_754/
├── README.md
├── bundle_status.json
├── manifest.csv
├── shard_manifest.csv
├── prompts/
│   ├── pass_1.jsonl
│   ├── pass_2.jsonl
│   ├── pass_3.jsonl
│   ├── pass_4.jsonl
│   └── pass_5.jsonl
├── shards/
│   └── pass_N/
│       └── pass_N_shard_XXX.jsonl
└── results/
    ├── pass_1_results.jsonl
    ├── ...
    └── pass_N/
        └── shards/
            └── pass_N_shard_XXX_results.jsonl
```

## Shard Policy

Shards are deterministic and persona-preserving:

- sort by `persona_id`, then `t_index`
- keep a persona's entries together
- cap at `5 personas` or `24 entries`, whichever comes first

Run passes sequentially for safety. Within a pass, use waves of up to `10`
workers at a time.

## Worker Input Format

Each shard row includes:

```json
{
  "entry_id": "013d8101__3",
  "persona_id": "013d8101",
  "t_index": 3,
  "date": "2025-03-04",
  "persisted_scores": {
    "self_direction": 0,
    "stimulation": 0,
    "hedonism": 1,
    "achievement": 0,
    "power": 0,
    "security": -1,
    "conformity": 0,
    "tradition": 0,
    "benevolence": 0,
    "universalism": 0
  },
  "prompt": "<rendered profile_only judge prompt>"
}
```

## Worker Output Format

Return one JSON object per line:

```json
{
  "entry_id": "013d8101__3",
  "scores": {
    "self_direction": 0,
    "stimulation": 0,
    "hedonism": 1,
    "achievement": 0,
    "power": 0,
    "security": -1,
    "conformity": 0,
    "tradition": 0,
    "benevolence": 0,
    "universalism": 0
  },
  "rationales": {
    "hedonism": "Accepted a relaxing weekend plan and described enjoyment.",
    "security": "Described anxiety about losing predictable routine."
  }
}
```

Rules:

- `entry_id` must match the shard exactly
- `scores` must contain all 10 dimensions
- every score must be exactly `-1`, `0`, or `1`
- every non-zero score must have a matching rationale
- rationales for zero-valued scores are discouraged
- do not skip or add rows

## Validation

Validate each shard before merging it:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_validate_results.py \
  --manifest logs/exports/twinkl_754/manifest.csv \
  --expected-jsonl logs/exports/twinkl_754/shards/pass_1/pass_1_shard_001.jsonl \
  --results logs/exports/twinkl_754/results/pass_1/shards/pass_1_shard_001_results.jsonl
```

If validation fails:

1. discard the shard result
2. spawn a brand-new worker for the same shard
3. retry up to two additional times
4. pause the pass if the third attempt still fails

Merge only validated shard result files into `pass_N_results.jsonl`.

## Merge And Provenance

After all accepted shard result files are ready, run:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_merge_pass_results.py \
  --bundle-dir logs/exports/twinkl_754 \
  --worker-model gpt-5.4
```

This step:

1. re-validates every shard result
2. merges shard files into `results/pass_N_results.jsonl`
3. writes `provenance/shard_provenance.csv`
4. writes `provenance/pass_provenance.csv`
5. writes `provenance/pass_similarity.csv`
6. fails closed if any pass pair is byte-identical or score-identical
7. fails closed if any non-zero label is missing a rationale

If you retried any shard, provide an attempt manifest with `pass_index`,
`shard_id`, `attempt`, and optional `worker_model` columns via
`--attempt-manifest`.

## Pilot Acceptance Gate

Accept the subagent execution path only if the pilot bundle satisfies all of
the following after merge and summarization:

1. all 5 passes complete successfully
2. `provenance/pass_similarity.csv` has no `raw_hash_match=true`
3. `provenance/pass_similarity.csv` has no `score_hash_match=true`
4. every pass has `non_zero_rationale_coverage == 1.0`
5. the summarizer completes without provenance mismatch errors

If the pilot fails any of those checks, stop the subagent-based rerun. Replace
the execution layer with a scripted runner built around
`src/judge/labeling.judge_session()` that preserves the exact same bundle
layout, shard result files, provenance CSVs, and summarizer inputs.

## After All 5 Passes

Run:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_summarize_consensus.py \
  --bundle-dir logs/exports/twinkl_754 \
  --consensus-output logs/judge_labels/consensus_labels.parquet
```

Useful summarize options:

- `--output` to write the markdown report to a non-default location
- `--annotations-dir` to point at a different set of human annotation parquet files

This writes:

- `joined_results.csv`
- `comparison_rows.csv`
- `flip_summary.csv`
- `irr_summary.csv`
- `confidence_summary.csv`
- `stability_summary.csv`
- `consensus_rejudging_report.md`
- `logs/judge_labels/consensus_labels.parquet`

The markdown report includes the full-corpus stability gate, per-dimension
stability diagnostics, and the diagnostic retrain comparison for the consensus
label branch.
