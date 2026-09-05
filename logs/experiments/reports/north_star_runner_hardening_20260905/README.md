# NSM development runner hardening

**Issue:** `twinkl-fz34.9`. **Date:** 5 September 2026.

The [v2 runner](../../../../scripts/experiments/north_star_phase0b_v2.py) fixes
preparation and recovery defects in the frozen Phase 0B runner. It preserves
the original code, provider/review modules, policy, and
[failed experiment](../north_star_phase0b_20260905/README.md).
No prompt, embedding, retrieval setting, or application behavior changes.
The [NSM experiment results log](../../north_star_moment.md) brings the Phase 0A,
Phase 0B, and runner-hardening results together.

## Changes

Preparation verifies the frozen cohort, query/encoder configuration, source
files, and episode records before parsing development writing. Each ranked
Journal Entry must match its owner, text hash, date, and stored order; the
ranking must contain every eligible earlier source, with no duplicates.
Reserved Persona files receive byte-hash checks only. Their writing is not
parsed or evaluated.

The manifest binds those inputs. A separate execution freeze binds the manifest,
the inherited budget seed, and the runner, validation, parsing, review, and
provider code. Every run verifies these records and rebuilds the manifest for
comparison before constructing a provider. A missing or incomplete freeze
cannot resume paid work.

Recovery uses the request's durable attempt numbers. Completed selections and
omissions are reconstructed from raw receipts. Exhausted and nonretryable
failures return their saved terminal outcome. A pending reservation has an
unknown transport outcome, so it remains reserved and cannot be retried
automatically. Requests already in flight in the current process are awaited
and shared. Other cases can still be reconstructed. An interrupted response
that was saved before validation is validated during recovery; an allowed
remaining retry requires `--run`.

A new run directory inherits the complete prior ledger, including earlier costs
and reservations, as an offline snapshot with an immutable seed. Every paid v2
run uses one locked cumulative ledger at
`logs/experiments/north_star_development_budget_v2.json`. Separate run directories
share its budget and retry allowance. An adjacent origin record prevents a
deleted cumulative ledger from silently resetting that allowance. Offline replay
can recover receipts from this ledger after an interrupted report write.
Use the latest cumulative ledger when preparing a later run. The historical
ledger is only read. Missing exact-candidate reference checks count as failed
cases; selected text remains offline diagnostic evidence, not an approved card.

## Commands

From the repository root, prepare a new directory and reconstruct existing
receipts without provider calls:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
uv run python scripts/experiments/north_star_phase0b_v2.py --prepare \
  --directory /tmp/twinkl-nsm-development-replay \
  --prior-ledger logs/experiments/reports/north_star_phase0b_20260905/budget.json
uv run python scripts/experiments/north_star_phase0b_v2.py --replay \
  --directory /tmp/twinkl-nsm-development-replay
```

Preparation refuses an existing directory or a location inside the historical
run. Replay returns exit code 2 when the gate fails, including missing receipts.
It never requests provider transport. `--run` permits unfinished provider work
within the inherited budget; a revised paid protocol remains separate work in
`twinkl-fz34.1`.

The preserved provider adapter uses runtime reasoning `none` and reference
thinking `low`. The v2 runner rejects policy settings that the adapter cannot
honor. A later model/reasoning experiment must version the adapter and freeze its
actual execution settings. This fix does not establish that any such change
improves review quality.

## Verification

Synthetic regressions reproduce both original defects and cover stale inputs,
ownership, chronology, missing/duplicate rankings, changed execution freezes,
interrupted preparation, pending reservations, lost case-result files, bounded
retries, concurrent request reuse, shared cumulative budgets, missing review
roles, and exact-candidate quotation validation.
The combined NSM suite has 193 passing tests. Ruff and scoped MyPy pass for the
new code. Full-import MyPy retains five existing wrangling/registry errors;
application and browser tests were not run because application code is unchanged.

A separate temporary replay disabled `BudgetedProvider.complete` and reproduced
all 33 original case statuses and selections, all 61 attempts, the failed gate,
and US$0.20617785 in calculated cost. It made zero provider calls. All 39 files
in the original Phase 0B report directory and all five original execution-freeze
hashes remained unchanged. [Validation record](validation.json).

The v2 report names retrieval-only **source** precision explicitly and omits the
old unmatched verification-lift subtraction. Quotation precision remains a
separate measure. This recovery evidence does not improve the failed semantic
results or evaluate the reserved histories; Phase 0B and dependent integration
remain blocked.
