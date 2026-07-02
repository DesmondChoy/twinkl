# VIF Strategic Review — 2026-07-02 (resume-from-break)

Scope: full cross-run review of all 56 run IDs / 120 persisted configs, plus a
strategy assessment of the QWK + `recall_-1` metric regime, unexplored levers,
and web corroboration. Provenance/observation backfill: **complete** (0 empty
fields across 120 YAMLs; the final six backfills for `run_051`–`run_056` were
sitting uncommitted from the `twinkl-upb5` session and are committed with this
report).

## 1. Experiment Overview

Two evaluation regimes exist. The active corrected-split regime (`run_016+`,
persona-stratified split from `d937094`, split seed 2025, model seeds
11/22/33) is the only basis for leaderboard claims here; `run_001`–`run_015`
are archival. Constants across the active frontier: nomic-embed-text-v1.5 at
256d, `ws=1`, `hd=64`, dropout 0.3, 23,454 params, `n_train` 1,022 → 1,213
after the two targeted data lifts. Axes varied: loss family (CORN, SoftOrdinal,
CDWCE, LDAM-DRW, SLACE, BalancedSoftmax ± dimweight/circreg/two-stage),
encoder (nomic 256/768d, v2-moe, Qwen3-0.6B), capacity (hd 28–256), post-hoc
retargeting, checkpoint policy, and label regime (persisted vs consensus —
consensus is diagnostic-only because it changes the eval labels).

## 2. Head-to-Head (corrected split, family medians)

| Family | Runs | QWK | recall_-1 | MinR | Hedging | Cal |
|---|---|---:|---:|---:|---:|---:|
| **BalancedSoftmax (default)** | 019–021 | 0.362 | 0.313 | **0.448** | 0.621 | 0.713 |
| Qwen3-0.6B + BSM | 042–044 | **0.370** | 0.318 | 0.436 | **0.591** | 0.691 |
| BSM + dimweight (tail ref.) | 034–036 | 0.342 | **0.378** | 0.449 | 0.599 | **0.726** |
| TwoStage BSM | 045–047 | 0.360 | 0.266 | 0.382 | 0.708 | 0.743 |
| CDWCE_a3 (conservative) | 016–018 | 0.353 | 0.104 | 0.276 | 0.804 | 0.762 |
| CORN (calibration anchor) | 016–018 | 0.315 | 0.089 | 0.273 | 0.801 | **0.818** |

No frontier change since 2026-03. All deltas among the top three families are
within noise per the twinkl-730 BCa gates, except the dimweight branch's
`recall_-1` lift (+0.065, CI [+0.021, +0.128]) which still pays an unresolved
QWK cost. Artifact audits (selection traces, dimension-weight traces) were
re-verified as documented in twinkl-721/upb5; no new runs exist since.

## 3. Per-Dimension Analysis — with the human-agreement ceiling

New cross-reference: per-dimension model QWK (family medians) vs human
inter-annotator Fleiss' κ on the 115-entry shared subset
(`logs/exports/agreement_report_20260318_130642.md`). Unweighted κ and QWK are
not identical scales, but directionally this is the honest headroom map:

| Dimension | Model QWK | Human Fleiss κ | Headroom verdict |
|---|---:|---:|---|
| hedonism | 0.11 | 0.64 | **Largest real gap — model failure, not task ambiguity** |
| stimulation | 0.25 | 0.58 | Real gap |
| power | 0.15–0.37 | 0.61 | Real gap, volatile |
| security | 0.25 | 0.48 | Gap, but target partly unreachable (twinkl-747) |
| universalism | 0.43 | 0.72 | Gap, extreme imbalance (87% neutral) |
| benevolence | 0.34 | 0.61 | Moderate gap |
| achievement | 0.35 | 0.47 | Near ceiling |
| tradition | 0.49 | 0.50 | **At ceiling** |
| self_direction | 0.52 | 0.44 | **At/above ceiling** |
| conformity | 0.54 | 0.43 | **Above human agreement — check for shortcut/judge-shared bias** |

**Error analysis (fresh, from `run_020` saved validation outputs):** all top
hedonism/security misses are affect-vs-intent polarity flips, not noise. A
celebrated promotion ("late nights, stress, sacrifices… it paid off") →
predicted −1 hedonism (true +1). Declining a promotion to protect a working
life → −1 at 0.86 confidence (true +1). A happy first week at a chaotic
startup → +1 security (true −1: stability was abandoned). One entry
(`5fcf93f5` t=2, a shift-change fight over a threatened holiday) requires
hedonism −1 *and* security +1 from the same text; the model got both wrong by
following sentiment. A frozen sentence embedding encodes valence strongly and
behavioral intent weakly — this is the representation-level explanation for
the twinkl-fncm dead end.

## 4. Calibration

BalancedSoftmax families hold global calibration 0.65–0.73 with all/most
dimensions positive; CORN remains the anchor at 0.818. No dimension below
−0.4 in the active families. Literature note: calibration failures concentrate
in minority classes, so the −1 class is exactly where confidence gating is
least trustworthy — selective prediction can silently drop the cases the
product exists to catch. Uncertainty quality on −1 specifically should be
tracked, not just global correlation.

## 5. Hedging vs Minority Recall

| Family | Hedging | MinR | Verdict |
|---|---:|---:|---|
| BSM default | 62.1% | 0.448 | Balanced, slightly hedgy |
| BSM + dimweight | 59.9% | 0.449 | **Decisive + balanced** |
| Qwen + BSM | 59.1% | 0.436 | Decisive + balanced |
| TwoStage | 70.8% | 0.382 | Hedgy |
| CDWCE/CORN/SoftOrdinal | ~80% | 0.27–0.28 | Majority-locked |

The conservative-loss hedging wall (~80%) never broke in 56 runs; only prior
correction (BalancedSoftmax) broke it. This is consistent with the documented
QWK pathology on imbalanced data: QWK rewards asymmetric marginals and admits
all-zero-column optima, i.e., QWK-first selection structurally favors hedging.

## 6. Capacity & Overfitting

All active families sit at param/sample ≈ 19–23 (high) with small training
gaps. Every capacity increase tested (768d run_037: 46.3 ratio; hd≥128
historical) overfit or regressed; the 768d probe reached train QWK 1.0
(twinkl-fncm). Capacity is not the constraint and should not be revisited
before the target is repaired.

## 7. Systemic Insights & Hypotheses

**The story:** three independent ceiling confirmations (lct3 policy hunt: 0/5
promotion floors met across 143 policies while a validation *oracle* reached
QWK 0.769; fncm representation probe; 747 reachability audit) relocate the
bottleneck from model to **target contract**. The student is graded against a
teacher that (a) uses context the student never sees, (b) collapses 5-vote
uncertainty into hard labels, and (c) has only moderate agreement with humans
(Cohen's κ vs annotators 0.50–0.80; human-human Fleiss κ 0.56).

**Hypothesis 1 (supported by fresh error analysis):** the residual hard-dim
error is affect/intent conflation in the frozen embedding + label
unreachability, in roughly equal parts; neither is fixable by loss, capacity,
or post-hoc levers — all confirmed dead ends.

**Hypothesis 2:** the promotion floors (QWK ≥ 0.40 AND recall_-1 ≥ 0.40) sit
above the achievable ceiling *under the current entry-level metric regime*,
so continuing to gate on them guarantees "no promotion" verdicts forever. The
aggregate QWK floor is dragged by dimensions whose human ceiling is ~0.45.

**Confound check:** BalancedSoftmax's rise and the recall gains involved both
loss change (same-data run_019–021 vs run_016–018) and later data lifts
(n_train 1,022→1,213); both contributions are separately evidenced and stated
in prior reviews — no single-cause claims made here.

## 8. Actionable Recommendations

Web research conducted (see sources in the session log / index entry): QWK
imbalance pathologies; ValueEval'23/24 benchmarks (best fine-tuned macro-F1
≈ 0.28 sentence-level; reasoning LLMs 0.62–0.64 F1, human-level value
selection); soft-label/disagreement training (LeWiDi-2025, Crowd-Calibrator);
zero-shot-vs-fine-tuned evidence; selective-prediction minority-class
miscalibration.

1. **Reframe the primary metric at the decision level (feeds twinkl-752,
   twinkl-a2w).** The product acts on confidence-gated weekly crash/rut
   triggers, not per-entry labels (PRD eval #3: ≥8/10 crisis-week hit rate).
   Build that benchmark from existing checkpoints + drift bridge and measure
   trigger hit-rate / false-alarm / evidence quality. Entry-level primary
   becomes **recall_-1 at a precision floor** (screening framing, PR-style);
   QWK demoted to diagnostic. Per-entry QWK 0.36 may already be sufficient
   after weekly aggregation — nobody has measured this, and it decides the
   capstone claim.
2. **Run the zero-shot/few-shot LLM critic baseline on the frozen test split**
   (student-visible context contract only). ValueEval'24 shows reasoning LLMs
   near human-level value selection; if an LLM beats the 23k-param student
   materially, the honest capstone story is "what distillation loses and why
   the student is still justified (cost/latency/privacy)"; if it doesn't, the
   labels themselves are the ceiling. Cheapest highest-information experiment
   available. Watch: QWK + recall_-1 vs run_020, per-dimension.
3. **Proceed with twinkl-a30f then twinkl-j0ck (unchanged P0s).** Soft
   vote-distribution training is strongly corroborated: disagreement-aware
   soft-label training reliably improves macro-F1, calibration, and reduces
   false certainty on subjective tasks. Evaluate under recommendation 1's
   metrics, with BCa family gates.
4. **Buy the data-scaling curve before more generation.** Subsample train at
   25/50/75/100% (3 seeds, incumbent config, ~12 cheap runs). If QWK/recall
   curves are flat, more synthetic personas are not the lever and twinkl-748
   should stay eval-only; if rising, targeted generation is justified. Also
   probe transfer from Touché-ValueEval (~60k Schwartz-labeled sentences with
   attained/constrained polarity) as head pretraining — the only external-data
   lever never pulled.
5. **Target hedonism first in twinkl-748** (human κ 0.64 vs model QWK 0.11 is
   the largest defensible gap); audit conformity/self_direction for
   shortcut features (model above human agreement). Optionally distill the
   unused `rationales_json` as an auxiliary target.

Hygiene: recover the stranded lct3 report from `git stash@{0}` before it is
lost; wire BWS graded weights (twinkl-1m8) so the state matches the PRD
contract before decision-level evaluation.

## 9. Summary Verdict

- **Best config:** unchanged — `run_019`–`run_021` BalancedSoftmax default;
  `run_034`–`run_036` tail-sensitive reference. No leaderboard change.
- **Key weakness:** the evaluation regime itself — entry-level QWK against
  hard, partly-unreachable, moderately-human-agreed labels, with promotion
  floors above the demonstrated ceiling.
- **Highest-leverage next step:** the zero-shot LLM baseline plus the
  decision-level (weekly trigger) benchmark — together they bound the ceiling
  and test whether the current student already clears the product bar, before
  any further training investment.

## Addendum — full report-corpus sweep (same day)

A follow-up pass extracted the recommendation/verdict sections of all 24
corrected-split-era reports in `reports/` (the five pre-`d937094` reports are
archival and their conclusions are reproduced in the index's historical
section). Nothing above is contradicted; every report's frontier verdict is
consistent with the board. Three residual items surfaced that the index
digests under-weight:

1. **Decoupled head-only retraining with class-balanced sampling was
   recommended four times (v7, v8, twinkl-719.5, twinkl-746) and never
   executed** — verified: no run config or `src/vif` code implements it. It is
   partially subsumed by the fncm (frozen features lack generalizable
   hard-dim signal) and lct3 (no label-free extractable policy) negatives,
   but it remains the only *training-time* lever with a specific untested
   mechanism. Cheapest honest disposition: fold it into `twinkl-j0ck` — when
   the head is retrained on soft vote-distribution targets, run one
   class-balanced-sampling arm rather than giving it a separate cycle.
2. **`twinkl-upb5` carries a conditional recommendation that must not be
   lost:** if soft/consensus labels become the training target, retain and
   evaluate the `0.02` recall-window candidate checkpoint as standard — the
   consensus branch showed real gains under it (QWK 0.374→0.393, recall_-1
   0.257→0.323). Noted on `twinkl-j0ck` in the tracker.
3. **Focal temperature scaling, Kendall log-sigma weighting, PCGrad, and PCA
   ablations** were recommended in v7/v8/731 and never run. The ceiling
   evidence deprioritizes all of them; they should stay dead unless the
   decision-level benchmark shows calibration (not recall) is the binding
   constraint at the trigger layer.
