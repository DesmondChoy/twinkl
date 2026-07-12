# Experiment Review — `twinkl-748` Hedonism matched hard-set

## Decision

Keep the new Hedonism hard-set as a **Codex-reviewed diagnostic**, stop the
epic at the evaluation arm, and do not retrain from these cases yet.

The hard-set answers the narrow question it was built for: the active Critic
still struggles to distinguish choosing pleasure from rejecting it when the
surface wording is closely matched. The result does not change the active
frontier and is not human validation or a production-promotion surface.

Any later training augmentation remains gated by `twinkl-kof2`, which must
first establish that the current model is still data-limited. Security and
Stimulation expansion is not justified before the final VIF scope decision.

## Review protocol

The parent authored 24 matched pairs / 48 short journal entries across six
Hedonism boundary families:

1. quiet sensory pleasure;
2. guilt-coded rest;
3. protected leisure boundaries;
4. celebration and savoring;
5. lifestyle-quality choices; and
6. pleasure chosen despite another commitment.

Hedonism was scored independently. Choosing present comfort can be Hedonism
`+1` even when it conflicts with Achievement, Benevolence, or another value.
Hedonism `-1` required clear self-denial, cancellation, or rejection of
available pleasure, rest, or comfort.

Two separate packet-only Codex reviewers saw opaque pair and entry IDs, the
journal text, and the rubric. They did not receive author labels, family names,
generation notes, model predictions, or the reconciliation key. A pair entered
the frozen set only when both reviewers agreed on both entry labels, accepted
the realism and comparability of the pair, and raised no quality flags.

The requested reviewer runtime was "5.6 Sol / Light," but the available
subagent launcher exposed no model or reasoning-level selector. Both responses
therefore record the actual runtime as platform-default and the audit manifest
records that limitation. Packet-only instructions provided controlled
disclosure, not enforced filesystem isolation.

## Review outcome

| Measure | Result |
|---|---:|
| Candidate pairs / entries | 24 / 48 |
| Entry-label agreement | 47 / 48 (97.9%) |
| Accepted pairs / entries | 20 / 40 |
| Excluded pairs | 4 |
| Accepted labels matching author intent | 40 / 40 |

Three lifestyle-choice pairs were excluded because one reviewer found that
their pay, prestige, role, or commute changes moved other values as well as
Hedonism. One meal pair was excluded because rushed eating while working was
read as `0` by one reviewer and `-1` by the other. The exclusions are the
protocol working as intended: disagreement was preserved rather than
adjudicated into a convenient label.

The 97.9% agreement shows that the deliberately explicit boundary is highly
reviewable by two instances of the same model family. It does **not** establish
human validity or external agreement.

## Model evaluation

The frozen 20-pair set was evaluated directly on the Hedonism head using the
checkpoint-specific runtime contract and `core_values: [Hedonism]`. The active
incumbent family (`run_019`-`run_021`) and tail-sensitive reference
(`run_034`-`run_036`) were scored separately across all three seeds. Pair
metrics are primary; QWK is secondary because the set is small and balanced by
construction.

| Family | Exact accuracy | Recall `-1` | Recall `+1` | Both entries correct | Directional accuracy | High-confidence error rate |
|---|---:|---:|---:|---:|---:|---:|
| Incumbent | 0.525 | 0.050 | 1.000 | 0.050 | 0.650 | 0.450 |
| Tail-sensitive reference | 0.575 | 0.200 | 0.950 | 0.150 | 0.750 | 0.175 |

Values are family medians across three seeds. The tail-sensitive branch moves
the paired cases in the right direction more often and reduces confident
errors, but it still correctly classifies both members of only 15% of pairs.
The incumbent recognizes almost every pleasure-protecting `+1` entry while
recovering only 5% of matched self-denial `-1` entries. That is a strong
positive-class bias, not a subtle ranking miss.

## Interpretation

The result strengthens the existing semantic diagnosis: the local Critic can
often recognize explicit enjoyment, but it does not reliably treat the
rejection of available pleasure as Hedonism misalignment. The matched design
removes much of the topic and tone variation that complicated earlier replay
examples, so another loss or context tweak should not be assumed to solve the
problem.

This does not yet say that more training data will help. The hard-set is an
evaluation instrument, and training on it would contaminate that instrument.
`twinkl-kof2` must answer whether the learning curve is still rising before a
separate training augmentation population is considered.

For the capstone, the honest scope is now clearer: Hedonism remains a
documented hard dimension, the default model must not claim reliable
Hedonism-conflict detection, and downstream feedback should remain selective
or fall back to a stronger verifier until independent evidence supports more.

## Artifacts

- Candidate spec: `config/evals/twinkl_748_hedonism_hard_set_v1.yaml`
- Review workflow: `src/vif/hedonism_hard_set.py`
- Review bundle: `logs/experiments/artifacts/hedonism_hard_set_twinkl_748_20260712/`
- Frozen set SHA-256: `c2c825b27867c6f15e61ca8ed945fc92a42db43ffbac6953994d81cb8bca0637`
- Evaluation report: `logs/experiments/artifacts/hedonism_hard_set_twinkl_748_20260712/parent_control/evaluation/evaluation_report.md`

## Limitations

- Reviewers were Codex subagents, not humans.
- The requested model/reasoning setting could not be enforced by the launcher.
- The shared workspace was procedurally packet-only, not technically isolated.
- Pairwise agreement between two instances of one model family can share the
  same blind spots.
- The balanced 40-entry set is diagnostic and too small for broad population
  claims.
- No training or promotion decision should use these cases as an untouched
  holdout after they become visible to implementation work.
