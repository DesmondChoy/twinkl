# `twinkl-752.1`: Weekly Verifier Critic-Input Ablation

**Date:** 2026-07-12

**Decision:** negative. Supplying `run_020` Critic probabilities and uncertainty
made the weekly verifier more conservative, but it halved median episode recall.

**Conditional recommendation:** reject the tested raw Critic-input path. If the
choice is limited to these two development arms, carry the no-Critic verifier
into `twinkl-752.2` for approval. This is not a production or promotion decision:
the no-Critic arm still produced a median one false episode, and no acceptable
false-alert tolerance has been adopted.

## Why this study exists

`twinkl-752` narrowed the Value Identity Function (VIF) to conflict screening,
but left the runtime architecture open. Historical entry-level results suggested
that the local MLP and an LLM catch different negative cases. This study asks the
smaller product question: when the same weekly LLM verifier sees the same
student-visible history, do Critic scores improve sustained-conflict detection?

The prerequisite `twinkl-1r3d` audit found no single-word or tested phrase
shortcut that flipped the selected Conformity or Self-Direction predictions.
That kept the MLP family eligible as a baseline without claiming that it learned
the constructs.

## Registered design

- Development surface only: 28 personas, 217 entries, 126 ISO weeks, 42 review
  cases, and 41 resolved trajectories containing 5 reference episodes.
- Primary paired arms: the same `gpt-5.4-mini-2026-03-17` verifier without and
  with fixed `run_020` `P(-1)` and MC-uncertainty inputs.
- Three repeats per arm and persona-week: 252 unique prompts and 756 calls.
- Exact cumulative runtime text; only current-week entries were assessed.
- Reviewer-agreed entry cells only: 316 cells. Case 037 was excluded because it
  remained unresolved.
- Abstentions and genuinely invalid outputs suppress claims and reduce coverage.
- Locked promotion data and the retired benchmark were not used.

The decision rule was registered before execution: Critic inputs are positive
only if median episode recall rises, false alerts do not rise, and coverage does
not fall.

## API execution and validation

The run used OpenAI Responses structured parsing with reasoning effort `none`,
no tools, no stored responses, concurrency 4, and a $5 hard cap.

| Calls | Raw valid | Raw invalid | Contract-recovered | Genuinely invalid | Input tokens | Output tokens | Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 756 | 677 | 79 | 69 | 10 | 1,086,615 | 111,208 | $1.3154 |

The first validator incorrectly required optional `not_conflict` and `abstain`
explanations to be exact quotes. The registered prompt required exact evidence
only for `conflict`. Raw receipts were preserved; 69 complete outputs were
revalidated under that prompt contract, while 10 genuinely invalid outputs
remained fail-closed. Tests now pin this distinction.

## Primary results

Episode matching uses the repository's standard onset-to-lag matcher. Coverage
means the trajectory contains a detected episode or every adjacent pair can be
ruled out by at least one explicit non-conflict decision.

| Arm | Repeat | Episode recall | Episode precision | False episodes | Coverage |
|---|---:|---:|---:|---:|---:|
| Without Critic | 1 | 0.60 | 1.00 | 0 | 0.756 |
| Without Critic | 2 | 0.40 | 0.667 | 1 | 0.659 |
| Without Critic | 3 | 0.20 | 0.50 | 1 | 0.829 |
| With Critic | 1 | 0.00 | 0.00 | 0 | 0.707 |
| With Critic | 2 | 0.20 | 1.00 | 0 | 0.732 |
| With Critic | 3 | 0.20 | 1.00 | 0 | 0.756 |
| **Without Critic median** | — | **0.40** | **0.667** | **1** | **0.756** |
| **With Critic median** | — | **0.20** | **1.00** | **0** | **0.732** |

Critic inputs removed the median false episode but reduced median recall by
0.20 and coverage by 0.024. Under the recall-first registered rule, that is a
negative result: the arm became safer by becoming blinder.

## Entry-level diagnostics

`recall_-1` is the macro average across dimensions with negative support. The
pooled micro recall is reported separately.

| Arm | Repeat | Macro `recall_-1` | Micro recall | `-1` precision | Predicted-conflict rate | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Without Critic | 1 | 0.383 | 0.323 | 0.833 | 0.038 | 0.804 |
| Without Critic | 2 | 0.306 | 0.226 | 0.583 | 0.038 | 0.801 |
| Without Critic | 3 | 0.187 | 0.161 | 0.556 | 0.028 | 0.845 |
| With Critic | 1 | 0.405 | 0.290 | 0.900 | 0.032 | 0.804 |
| With Critic | 2 | 0.351 | 0.258 | 0.727 | 0.035 | 0.829 |
| With Critic | 3 | 0.315 | 0.194 | 0.545 | 0.035 | 0.807 |

The Critic arm improves median macro entry recall (`0.351` versus `0.306`) and
precision (`0.727` versus `0.583`), but those isolated detections do not join
into the correct adjacent episodes. Entry-level gains therefore do not answer
Twinkl's downstream product question by themselves.

## Baselines and target sensitivity

### MLP family on the same reviewer-resolved cells

| Run | Macro `recall_-1` | Micro recall | `-1` precision | Predicted-conflict rate |
|---|---:|---:|---:|---:|
| `run_019` | 0.607 | 0.548 | 0.327 | 0.165 |
| `run_020` | 0.601 | 0.516 | 0.281 | 0.180 |
| `run_021` | 0.530 | 0.516 | 0.262 | 0.193 |

The MLP family has much higher entry recall, but precision is only `0.262` to
`0.327` and it flags 16% to 19% of cells. It is useful evidence, not a safe
standalone product rule. Across all 335 available core-value cells, the median
seed range in `P(-1)` is 0.129; 29% of cells have a range of at least 0.20.

### `twinkl-754` consensus replay

The same 316 coordinates are fully available under the five-pass consensus
target. Consensus agrees with the two-reviewer target on 95.3% of cells and has
24 negative cells rather than 31. Median macro entry recall is `0.452` without
Critic and `0.571` with Critic; MLP-family median macro recall is `0.679`, with
median precision `0.246`. This confirms entry-level complementarity, but it does
not overturn the primary episode result because the consensus labels do not
form a separately reviewed weekly episode target.

### Human anchor and historical entry LLM

The existing three-annotator human anchor is unavailable on the matched
development inputs: among these 28 personas, annotation coverage is 12 entries
from one annotator and zero from the other two. No new annotation was authorized.

The historical student-visible `gpt-5.4-mini` entry baseline remains context
only because it uses a separate frozen 221-entry test split: QWK `0.434`,
`recall_-1` `0.188`, minority recall `0.428`, and hedging `0.789`. The matched
no-Critic weekly arm is the causal LLM baseline for this study.

## Interpretation for Twinkl

Twinkl needs a trustworthy weekly mirror, not merely a model that emits more
negative labels. The raw Critic numbers made the verifier cautious, but caused
it to miss more sustained conflict. The tested integration should therefore
not be adopted. `twinkl-752.2` should decide whether to approve the no-Critic
verifier as the capstone architecture, request a narrower Critic handoff, or
defer architecture adoption. A fresh resolved promotion surface and an adopted
false-alert tolerance remain necessary before any deployment claim.

## Reproduction and artifacts

```bash
source .venv/bin/activate
python -m scripts.experiments.weekly_verifier_ablation \
  --config config/evals/twinkl_752_1_weekly_verifier_ablation_v1.yaml prepare
python -m scripts.experiments.weekly_verifier_ablation \
  --config config/evals/twinkl_752_1_weekly_verifier_ablation_v1.yaml score
pytest -q tests/experiments/test_weekly_verifier_ablation.py
```

- [Manifest](../artifacts/twinkl_752_1_weekly_verifier_ablation_20260712/manifest.json)
- [Prompts](../artifacts/twinkl_752_1_weekly_verifier_ablation_20260712/prompts.jsonl)
- [Raw responses](../artifacts/twinkl_752_1_weekly_verifier_ablation_20260712/responses.jsonl)
- [Metrics](../artifacts/twinkl_752_1_weekly_verifier_ablation_20260712/metrics.json)
- [Preregistered config](../../../config/evals/twinkl_752_1_weekly_verifier_ablation_v1.yaml)
- [Prompt template](../../../prompts/weekly_vif_verifier.yaml)
- [Runner and scorer](../../../scripts/experiments/weekly_verifier_ablation.py)
