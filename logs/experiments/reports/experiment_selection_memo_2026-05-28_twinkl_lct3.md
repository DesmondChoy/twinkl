# Experiment Selection Memo - 2026-05-28 - twinkl-lct3

## Goal

Find a promotable next VIF Critic frontier candidate under the no-new-data
constraint, or prove the current persisted-label frontier is not reachable by a
scientifically honest no-new-data intervention.

This memo is written before any new candidate test evaluation or retraining on
this branch. No synthetic personas, journal entries, nudges, or labels will be
generated. No LLM call will be used to create training or evaluation data.

## Baseline and Promotion Floor

Active persisted-label baseline: `run_019`-`run_021` `BalancedSoftmax`.

| Metric | Baseline median | Promotion floor |
|---|---:|---:|
| QWK | 0.362 | >= 0.400 |
| recall_-1 | 0.313 | >= 0.400 |
| minority recall | 0.448 | >= 0.480 |
| hedging | 0.621 | <= 0.580 |
| calibration | 0.713 | >= 0.720 |

Hard-dimension guardrail:

- `hedonism`, `security`, and `stimulation` may each drop by at most 0.02 QWK
  versus `run_019`-`run_021`.
- At least one of those dimensions must improve by >= 0.05 QWK or >= 0.05
  active-class recall.

## Repo Evidence

- `docs/prd.md` and `docs/vif/03_model_training.md` define the active Critic
  as a frozen-encoder, vector-valued student trained to predict immediate
  Judge labels on the corrected persona-stratified split.
- `logs/experiments/index.md` keeps `run_019`-`run_021` as the active
  persisted-label default. The strongest single-family challengers each solve
  only part of the floor:
  - `run_034`-`run_036` `BalancedSoftmax + dimweight` improves
    `recall_-1`, minority recall, hedging, and calibration, but loses QWK and
    hard-dimension stability.
  - `run_042`-`run_044` `Qwen3-0.6B + BalancedSoftmax` improves QWK and
    hedging, but gives back calibration and weakens `hedonism` / `security`.
  - `run_045`-`run_047` `TwoStageBalancedSoftmax` improves calibration and
    `stimulation`, but hedges too much and loses `recall_-1`.
  - `run_048`-`run_050` consensus-label retrains are diagnostic only because
    the evaluation labels changed.
- `logs/experiments/reports/experiment_review_twinkl_729.md` says previous
  post-hoc logit adjustment over individual checkpoints is probably exhausted.
- `logs/experiments/reports/experiment_review_2026-03-14_twinkl_730.md`
  says promotion should require family-level uncertainty checks, not lucky
  point medians.
- `logs/experiments/reports/experiment_review_2026-03-19_twinkl_746.md`
  shows the two-stage branch mainly loses active recall, not active-sign
  polarity once active.
- `logs/experiments/reports/experiment_review_2026-04-01_twinkl_754_6.md`
  warns that hard majority relabeling can erase disagreement-heavy active
  signals.

## External Inspiration

- Balanced Softmax: Ren et al. show standard softmax gives biased gradients
  under long-tailed label distributions and introduce Balanced Softmax for
  train/test prior mismatch:
  https://papers.nips.cc/paper_files/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- Logit adjustment: Menon et al. formalize prior-based logit adjustment as
  both a post-hoc and training-time long-tail method:
  https://research.google/pubs/long-tail-learning-via-logit-adjustment/
- Decoupled classifier work: Kang et al. argue long-tail gains can come from
  classifier/decision-boundary adjustment on top of existing representations:
  https://openreview.net/pdf?id=r1gRTCVFvB
- Effective-number weighting: Cui et al. motivate reweighting by effective
  samples rather than raw counts:
  https://arxiv.org/abs/1901.05555
- Calibration: Guo et al. show simple post-processing such as temperature
  scaling can materially affect calibration:
  https://arxiv.org/abs/1706.04599
- Uncertainty: Gal and Ghahramani justify using MC Dropout predictions as a
  practical uncertainty signal:
  https://arxiv.org/abs/1506.02142
- Multi-annotator disagreement: Davani et al. show majority aggregation can
  discard meaningful subjective disagreement and that multi-annotator modeling
  can improve uncertainty:
  https://transacl.org/index.php/tacl/article/view/3173
- Ordinal heads: CORAL's rank-consistency result supports preserving ordinal
  structure rather than treating `-1/0/+1` as unrelated nominal classes:
  https://arxiv.org/abs/1901.07884

## Idea 1 - Conservative: Two-stage active-recall reweighting

Hypothesis: `TwoStageBalancedSoftmax` already improved activation precision,
calibration, and `stimulation`, but missed too many active cases. A Stage-A
active-positive reweighting or recall floor could recover active recall while
keeping the two-stage hard-dimension gains.

Repo evidence:

- `run_045`-`run_047` reached comparable QWK and better calibration, but
  `recall_-1` fell to 0.266 and hedging rose to 0.708.
- The two-stage review says active-sign accuracy was comparable to the
  incumbent once active, so the failure is mostly activation recall.

External inspiration:

- Long-tail classifier-side adjustment from Balanced Softmax and the decoupled
  classifier literature.
- Ordinal decomposition from CORAL-style rank-consistent ordinal thinking.

Expected metric movement:

- `recall_-1`: +0.05 to +0.10 versus two-stage, maybe +0.00 to +0.04 versus
  incumbent.
- Hedging: should fall if active recall rises, but may still miss <= 0.580.
- QWK: likely 0.36-0.39 unless the active-recall lift is unusually clean.
- Calibration: likely remains >= 0.720 because two-stage already does well.

Implementation cost:

- Medium. Requires training-code changes and a 3-seed rerun.

Leakage or benchmark risk:

- Low if the fixed holdout and persisted labels are preserved.
- Risk is overfitting validation via an overly aggressive active-recall floor.

Why it could plausibly hit the promotable floor:

- It attacks the exact two-stage failure mode and could improve `stimulation`
  without new data. The problem is that it needs a large simultaneous lift in
  `recall_-1`, QWK, and hedging. That is a lot to ask from one reweighting knob.

## Idea 2 - Creative/high-upside: Seed-aligned ensemble and decision policy

Hypothesis: The existing persisted-label model families make different mistakes.
A validation-only, seed-aligned ensemble or dimension policy over their saved
probability outputs can combine:

- incumbent `BalancedSoftmax`: best all-around hard-dimension stability
- dimweighted `BalancedSoftmax`: best tail package
- Qwen `BalancedSoftmax`: best QWK / hedging challenger
- `TwoStageBalancedSoftmax`: best calibration and strongest `stimulation`

The candidate would use only validation outputs to select a simple policy, then
run one untouched evaluation on the fixed test outputs. No checkpoint is
retrained, and no test output is used for selection.

Repo evidence:

- The frontier board shows complementary strengths across these families.
- Prior individual post-hoc logit adjustment is exhausted, but that does not
  answer cross-family probability ensembling or dimension-level routing.
- The uncertainty review says family-level promotion should use medians and
  intervals, which this design can preserve by evaluating one policy per seed.

External inspiration:

- Decoupled classifier/decision-boundary work supports testing whether the
  bottleneck is in the representation or final decision layer.
- Calibration and MC Dropout work support weighting or gating by confidence,
  but only if validation confirms it helps.
- Long-tail logit-adjustment literature supports prior-aware boundary movement,
  but previous single-checkpoint adjustment failed here, so the intervention
  must be genuinely cross-family.

Expected metric movement:

- QWK: +0.02 to +0.04 if Qwen/two-stage reduce off-by-two mistakes while
  incumbent protects hard dimensions.
- `recall_-1`: +0.05 to +0.09 if dimweighted probabilities recover rare
  misalignment cases.
- Minority recall: +0.03 to +0.05.
- Hedging: -0.03 to -0.06 if decisive families dominate active dimensions.
- Calibration: flat to mildly positive if uncertainty-aware averaging reduces
  overconfident misses.

Implementation cost:

- Low to medium. Needs an artifact script that aligns validation/test rows,
  searches a small policy space on validation only, and writes a report.

Leakage or benchmark risk:

- Main risk is validation overfitting. Mitigation: use a tiny, predeclared
  search space; select one shared policy family; do not tune on test; keep
  consensus-label outputs out of persisted-label promotion.

Why it could plausibly hit the promotable floor:

- This is the only no-new-data idea with a credible path to move all five floor
  metrics at once, because no single family currently does. Holy shit would it
  be a nice result if the errors are complementary. If they are not, the
  diagnostic kills the path cheaply.

## Idea 3 - Artifact-only diagnostic: Complementarity ceiling and validation
upper bound

Hypothesis: Before retraining anything, measure whether existing persisted-label
checkpoints contain enough complementary signal. If even a validation-only
upper-bound policy cannot approach the promotion floor without degenerating, a
new no-data retrain is not credible.

Repo evidence:

- Several single-family approaches are near-misses on different axes but none
  clears the package.
- `twinkl-729` already falsified simple per-checkpoint retargeting.
- `twinkl-730` says point-metric noise is large enough that weak retrains are
  not worth creating just for run IDs.

External inspiration:

- Decoupled classifier literature: test classifier/decision limits before
  changing representation.
- Calibration literature: test post-hoc probability behavior before a training
  run.

Expected metric movement:

- This is diagnostic, not a promoted model. It should estimate whether the
  chosen artifact policy can clear QWK >= 0.400 and `recall_-1 >= 0.400` on
  validation with a realistic policy, not an oracle cheat.

Implementation cost:

- Low.

Leakage or benchmark risk:

- Low if it is explicitly validation-only and not reported as a test win.
- An oracle per-row policy would be useful only as an upper bound, not as a
  deployable candidate.

Why it could plausibly hit the promotable floor:

- It does not hit the floor itself. It decides whether Idea 2 deserves a final
  test evaluation or whether the no-new-data frontier path is dead.

## Idea 4 - Soft-label persisted-board sibling

Hypothesis: Existing consensus votes contain useful disagreement signal, but
hard majority replacement muted active rare cases. A confidence-tiered or
soft-label BalancedSoftmax variant could train on existing vote distributions
while still being evaluated on the persisted-label board.

Repo evidence:

- `run_048`-`run_050` improved consensus-regime QWK/calibration but weakened
  persisted-frontier tail behavior.
- The consensus review identified disagreement-heavy hard dimensions as the
  live problem, especially `security` and `hedonism`.

External inspiration:

- Multi-annotator disagreement work argues against collapsing subjective labels
  to a single majority target too early.
- Calibration and uncertainty literature support preserving uncertainty rather
  than pretending ambiguous labels are clean.

Expected metric movement:

- Calibration could improve.
- Hard dimensions could improve if disagreement is preserved as softness rather
  than erased.
- `recall_-1` is uncertain and could easily regress if soft labels shrink active
  negatives toward neutral.

Implementation cost:

- Medium to high. Requires loss/target changes and careful board separation.

Leakage or benchmark risk:

- Medium. The biggest risk is accidentally treating consensus-label results as
  a direct persisted-label frontier replacement. That is explicitly forbidden.

Why it could plausibly hit the promotable floor:

- It attacks label noise without new data, but the prior consensus branch
  already showed tail risk. This is promising research, less promising as the
  fastest frontier promotion shot.

## Selection

Choose Idea 2, with Idea 3 as the mandatory falsification gate.

Reason: the promotable floor requires simultaneous movement in QWK,
`recall_-1`, minority recall, hedging, calibration, and hard-dimension
stability. The repo evidence says single-family training knobs keep trading one
axis for another. The highest-upside honest path is therefore to test whether
the existing checkpoint families contain complementary signal that a simple
validation-selected ensemble or decision policy can unlock.

Stop rule before retraining:

- If validation-only complementarity is weak, stop. Do not retrain.
- If the selected validation policy cannot plausibly clear the test floors
  without a large validation/test generalization miracle, stop.
- If the policy only works by per-row oracle choices or test-aware tuning, stop.

Candidate constraints:

- Use only persisted-label checkpoint outputs for promotion evaluation.
- Keep consensus-label outputs diagnostic only.
- Align rows by `persona_id`, `t_index`, `date`, and `dimension`.
- Select policy on validation only.
- Evaluate once on test after policy selection.
- Report family medians across seeds `11`, `22`, and `33`.
