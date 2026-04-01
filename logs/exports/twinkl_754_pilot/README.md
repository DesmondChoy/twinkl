# twinkl-754 Consensus Re-judging Bundle

This bundle was prepared by `scripts/journalling/twinkl_754_prepare_consensus.py`.

## Scope

- Bundle mode: `pilot`
- Selected entries: `50`
- Prompt condition: `profile_only`

## Files

- `manifest.csv`: the selected entries with deterministic `entry_id` keys and persisted labels.
- `shard_manifest.csv`: shard-level execution plan for all 5 passes.
- `bundle_status.json`: bundle lifecycle state, selection mode, and operator warnings.
- `prompts/pass_<n>.jsonl`: full pass prompt files.
- `shards/pass_<n>/shard_*.jsonl`: persona-preserving worker shards.
- `results/pass_<n>_results.jsonl`: merged per-pass result placeholders.
- `results/pass_<n>/shards/shard_*_results.jsonl`: shard result placeholders.
- `provenance/shard_provenance.csv`: accepted shard validation and hash records.
- `provenance/pass_provenance.csv`: merged pass fingerprints and rationale coverage.
- `provenance/pass_similarity.csv`: pairwise pass similarity diagnostics.

## Worker Model

- Main agent: orchestrator only
- Worker sub-agents: `gpt-5.4`, `fork_context=false`, one worker per shard
- Suggested concurrency: up to 10 workers per wave

## Shard Policy

- Max personas per shard: `5`
- Max entries per shard: `24`
- Total shards across all passes: `35`

## Validation

Validate each shard before merging it into a pass file:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_validate_results.py \
  --manifest logs/exports/twinkl_754/manifest.csv \
  --expected-jsonl logs/exports/twinkl_754/shards/pass_1/pass_1_shard_001.jsonl \
  --results logs/exports/twinkl_754/results/pass_1/shards/pass_1_shard_001_results.jsonl
```

After all validated shard files are ready, merge them and persist provenance:

```bash
source .venv/bin/activate
python scripts/journalling/twinkl_754_merge_pass_results.py \
  --bundle-dir logs/exports/twinkl_754 \
  --worker-model gpt-5.4
```

Only after the merge step succeeds should you run `twinkl_754_summarize_consensus.py`.

## Pilot Selection

- Requested entries: `50`
- Selected entries: `50`
- Selected non-zero `security` entries: `30`
- Selected non-zero `hedonism` entries: `18`
- Selected non-zero `stimulation` entries: `18`
