# twinkl-a30f Active-Critic-State Review Bundle

This full-corpus bundle may supply a training and evaluation target only after all three review passes, any required tie-break reviews, and strict materialization succeed.

## Contract

Every prompt contains only the active `window_size: 1` Critic state:

- runtime-formatted current session text; and
- the normalized 10-dimensional profile vector.

It excludes date, demographics, biography, prior entries, raw core-value names,
labels, rationales, and generation metadata.

## Files

- `active_critic_state_manifest.jsonl`: immutable reviewer prompts with state and prompt hashes.
- `active_critic_state_results.jsonl`: fill one JSON object per manifest case.

The review runner must send only each record's `prompt` value to the judging
model or human reviewer. The surrounding case coordinates and hashes are
reconciliation metadata; they are not part of the review input.

## Result format

Each result must bind to the supplied contract and hashes:

```json
{
  "case_id": "security__example__1",
  "state_contract_version": "active_critic_state_v1",
  "state_input_sha256": "<copied from manifest>",
  "prompt_sha256": "<copied from manifest>",
  "reviewer": "reviewer-or-runtime-id",
  "reviewed_at": "2026-07-11T00:00:00+00:00",
  "confidence": "high",
  "rationale_status": "provided",
  "scores": {
    "self_direction": 0,
    "stimulation": 0,
    "hedonism": 0,
    "achievement": 0,
    "power": 0,
    "security": 1,
    "conformity": 0,
    "tradition": 0,
    "benevolence": 0,
    "universalism": 0
  },
  "rationales": {"security": "Cites behavior from the current journal session."}
}
```

For a neutral Security score, use `"rationale_status": "not_applicable_neutral"`
and an empty `rationales` object. The target builder rejects missing cases,
duplicate cases, hash mismatches, unavailable-context flags, and any attempt to
use the legacy `twinkl-747` reduced-context labels as a fallback.
