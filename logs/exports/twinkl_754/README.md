# twinkl-754 Consensus Re-judging Bundle

This bundle was prepared by `scripts/journalling/twinkl_754_prepare_consensus.py`.

> Warning: the staged twinkl-754 results in this bundle were later found to contain duplicate passes and should not be used for retraining or merge decisions. Regenerate the bundle and rerun the workflow after the recovery hardening lands.

The invalid generated artifacts from that run were intentionally removed on
March 20, 2026. This directory now keeps only the lightweight invalidation
markers until a fresh rerun recreates the bundle.

## Files

- `README.md`: explains why the previous bundle was invalidated and cleared.
- `bundle_status.json`: lifecycle status for this bundle, including the list of removed invalid artifacts.

## Worker Model

- Main agent: orchestrator only
- Worker sub-agents: `gpt-5.4`, `fork_context=false`, one worker per shard
- Suggested concurrency: up to 10 workers per wave

## Shard Policy

- Max personas per shard: `5`
- Max entries per shard: `24`
- Total shards across all passes: `415`

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
