---
name: experiment-review
description: Review and compare VIF Critic (Offline) experiment runs using saved evidence, compatible evaluation setups, and current project decisions. Use for questions about runs in logs/experiments; a review does not authorize changes to experiment records.
---

# VIF Critic (Offline) Experiment Review

Answer the requested research question with evidence from relevant runs. Explain
trade-offs, investigate plausible causes, and distinguish findings from
hypotheses. Choose the depth and report structure to fit the question.

## Scope and Authority

Read the relevant [PRD](../../../docs/prd.md) section and any specification it
explicitly delegates to before recommending a model, metric policy, or further
research. Use the [experiment index](../../../logs/experiments/index.md) to find
runs and reports. Verify implementation claims against current code and saved
results; historical rankings do not establish the current default.

A review request is read-only. A request to save a report authorizes that report;
updates to run records, the index, or adopted decisions require those changes to
be within the agreed task scope. Apply existing authorization without asking
again. Follow the repository's Beads workflow for authorized documentation edits.

Leave missing provenance explicitly missing. Keep inferred rationale and
retrospective observations in the review, identify them as such, and cite the
supporting evidence. Do not populate original run records with inferred intent.
If correcting provenance is explicitly requested, use verifiable sources and
identify the correction as retrospective so it cannot be mistaken for the
original experiment record.

## Select Comparable Evidence

Start with the requested runs and their relevant baselines. Read all runs only
when the question requires a complete comparison. Record the run IDs, data and
label versions, input contract, split membership or manifest, split and model
seeds, training size, configuration changes, and checkpoint-selection policy
needed to understand the comparison.

- Compare performance within compatible evaluation setups: the same target,
  label source, evaluation data, input visibility, and metric definitions.
  Explain differences before interpreting comparisons across setups.
- Preserve the historical split boundary at commit `d937094`: `run_001` through
  `run_015` precede the persona-level stratified split. Later run numbers alone
  do not prove comparability; labels, data, and selection policies also changed.
- Group configurations that differ only by model seed and summarize their
  median and interquartile range, the spread of the middle half of results.
  Show the seeds and repeat count; identify single-run evidence as provisional.
- Distinguish training, development, and final test evidence. Check prior use
  of the evaluation data before calling it a fresh final test.
- Name the label source, such as LLM-Judge VIF Labels or project-team annotation.
  Preserve the distinction between AI review and human validation.

If metadata is incomplete, check configuration, manifests, Git history, and
source reports. State unresolved gaps and limit the comparison accordingly.

## Investigate What Could Change the Conclusion

Use saved predictions, labels, and traces to investigate relevant errors before
recomputing results. Inspect only the diagnostics needed for the question:

- For checkpoint selection, inspect `artifacts.selection_trace`,
  `selection_policy`, `training_dynamics.selection_source`,
  `promotion_eligible`, and `debug_fallback_used`. Distinguish a selected
  checkpoint from an epoch whose model state was never saved.
- For dimension weighting, inspect `artifacts.dimension_weight_trace`, the
  selected-epoch weights, clamp hits, and loss history. Group weighting and
  circumplex-regularization experiments by their actual settings, even when
  they share a `model_name`.
- For circumplex diagnostics, which summarize relationships between predictions
  for neighboring or opposing Schwartz values, use
  `load_artifact_bundle()` in [posthoc.py](../../../src/vif/posthoc.py) and
  `compute_circumplex_diagnostics()` in [eval.py](../../../src/vif/eval.py) when
  recomputation is needed. Use a consistent basis across compared runs and
  identify recomputed results separately from recorded metrics.
- For qualitative errors, inspect relevant Journal Entries alongside their
  reference labels, label source, and VIF Critic Predictions. Use local
  inference only when saved results cannot answer the question and the
  checkpoint and inputs are available. Report missing evidence without
  inventing substitute results.

Use bounded read-only calculations to resolve material questions about label
distributions or configuration changes. Use the project environment and
existing helpers. Keep temporary analysis outside the repository unless saved
outputs are requested, and record the inputs and commands needed to reproduce
any new numerical findings. Training or paid provider calls must be part of the
authorized task.

Before attributing a change to one intervention, compare data size and
composition, labels, split membership, encoder, loss, capacity, seeds, and
selection policy. Where several factors changed, describe the observed
association and what evidence would distinguish the explanations. Treat prior
conclusions as evidence to reassess, rather than assumptions to preserve.

## Interpret Metrics in Context

Use the metric priorities in the current PRD and delegated specifications.
Report actual values, denominators, and relevant uncertainty. Use paired
comparisons and confidence intervals when available; a fixed percentage
difference does not establish equivalence or a meaningful improvement.

- `calibration_global` is the Spearman correlation between uncertainty and
  absolute prediction error in `compute_calibration_summary()` in
  [eval.py](../../../src/vif/eval.py). It measures whether uncertainty ranks
  errors usefully; it does not establish calibrated probabilities or prove
  systematic overconfidence or underconfidence.
- `gap_at_best` is **validation loss minus training loss** at the selected epoch
  in [experiment_logger.py](../../../src/vif/experiment_logger.py). Compare
  gaps only with compatible loss definitions and training settings. Interpret
  them alongside learning curves and evaluation behavior.
- Interpret neutral prediction rates and minority recall alongside class
  prevalence and per-value errors. A parameter-to-sample ratio alone does not
  establish overfitting. Avoid universal quality thresholds unless an adopted
  evaluation contract specifies them.

## Deliver the Review

Lead with the answer and its strongest evidence. Include only the comparison
tables, error examples, and diagnostic detail needed to assess it. Follow
[canonical product terminology](../../../docs/canonical_nouns.md) and use
connected academic prose for research reports.

Identify material uncertainty beside the relevant claim. State what was
inspected, what was recomputed, and which evidence was unavailable. Cite exact
runs, files, and source reports so another reviewer can check the conclusion.

Recommend next steps only when evidence and current scope justify them. Keeping
the current decision or concluding that no further experiment is justified are
valid outcomes. When recommendations depend on current methods, consult primary
research or official documentation and explain its relevance. An external result
does not establish that the same intervention will help Twinkl.

For authorized index updates, preserve historical sections and identify each
comparison's data, labels, seeds, selection policy, and supporting report. A
review recommendation becomes an adopted product or research decision only when
that decision is authorized.
