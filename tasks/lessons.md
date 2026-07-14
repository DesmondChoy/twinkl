# Lessons

- **2026-07-14 — Historical split provenance is not an exclusion rule:** When a task explicitly makes every historical split development-only, include every valid label in the primary development dataset. Report former split membership as a subgroup or sensitivity check; do not silently turn names such as `retired_audit_only` into eligibility criteria.

- 2026-07-14: Before claiming a candidate-mined benchmark replaces an earlier
  reference, explicitly compare and union the prior confirmed cases. Candidate
  mining can miss known Drifts. Report cohort-only, overlap, net-new, and union
  counts separately.

## 2026-03-11

- When promoting a script out of a notebook path, avoid carrying forward
  "notebook-derived" wording into the canonical name or docs unless the user
  explicitly wants that provenance preserved. Prefer the clearest operational
  label for fresh-session discoverability.
- When a new training lever is intended to drive the active frontier path,
  default it on in that path instead of relying on a per-run opt-in flag that
  can be forgotten. Keep low-level APIs explicit, but avoid operational
  footguns in the primary experiment flow.
- In experiment reviews, separate the intervention recommendation from the
  paper citation used to justify it. Reuse the repo's actual tooling and open
  issue path first, and only cite external work that matches this setup's
  class count, training regime, and model architecture closely enough to avoid
  misleading extrapolation.
- For this repo's Git workflow, stay on the current branch unless the user
  explicitly asks for a new one. If branch isolation seems helpful, ask first
  instead of creating or switching branches unilaterally.
- Before recommending a "next experiment," explicitly verify that it is not
  already completed, de-scoped, or closed in `logs/experiments/index.md`,
  recent report docs, and `bd`. Do not re-suggest a finished experiment as
  if it were untested.
- In this VIF stack, be precise about what counts as "representation" versus
  "head" changes: the sentence encoder is already frozen and the state encoder
  is deterministic, so "head-only retraining" means changing the MLP critic
  layers, not toggling an encoder-freeze path that does not exist.
- Before recommending a "next experiment," verify from the latest report, open
  issues, and close reasons that it has not already been run, explicitly
  de-scoped, or superseded. Treat stale recommendation text in older reports as
  historical context, not current guidance.

## 2026-03-19

- When creating experiment epics, make the dependency posture explicit in the
  issue body: call out which epics can run in parallel, which are hard-blocked,
  and which have only soft sequencing constraints. Do not assume a fresh
  session will infer this from prose alone.
- For research epics, include concrete scoping constraints in the issue itself
  when they are part of the rationale: target dataset size, rough parameter
  budget, and any explicit stopping or narrowing decision points. This prevents
  open-ended execution in later sessions.
- If a repo-local skill or workflow file is named explicitly for a task, read
  it before doing the work and follow its required audit steps, not just its
  output format. For experiment reviews here, that means artifact-aware
  analysis, provenance backfill, and index updates before treating the review
  as done.

## 2026-03-20

- When a task-specific go/no-go rule is an exact threshold on a noisy eval
  metric, do not assume a tiny miss should permanently close the follow-up as
  "not needed." If the user treats the gate as advisory, record that override
  explicitly on the issue and reopen the downstream task instead of leaving the
  hard-stop interpretation in place.
- When an audit invalidates execution data, do not talk about the recovery as
  "done" just because the workflow code is fixed. Separate "rerun-ready" from
  "fresh artifacts regenerated," and avoid suggesting that a partial rerun is
  equivalent to a statistically clean rerun unless the independence claim has
  actually been re-established.

## 2026-03-24

- Be careful when exporting new runtime entrypoints from package `__init__`
  files. If a low-level module imports a schema from that package, eager
  `__init__` imports can create circular dependencies. Prefer lazy wrappers or
  importing the concrete submodule directly when wiring new runtime bridges.

## 2026-03-26

- Do not describe human annotation subsets as "ground truth" unless the user or
  project docs explicitly define them that way. In this repo, human annotations
  can serve as a limited external benchmark, but they may be sparse, non-expert,
  and narrower than the full judged corpus.
- When reporting judge-vs-human agreement, state the evaluation coverage
  explicitly: how many entries and personas are in the overlap, how many are
  excluded, and that the metric is computed on the intersection only rather than
  the full dataset.

## 2026-04-01

- When a repo-local skill is the required workflow for a task, run it before
  closing the issue or epic and before reporting conclusions. Do not rely on
  "roughly equivalent" manual analysis and wait for the user to notice the
  missing workflow step.

## 2026-04-04

- When syncing docs, distinguish carefully between prototype code that exists in
  the repo and project scope that is actually decided. Do not document an idea
  as implemented, or as an active planned dependency, unless the user or the
  authoritative project docs clearly treat it that way. In this repo, evolution
  gating should be described as an undecided idea unless explicitly confirmed
  otherwise.

## 2026-04-10

- In sponsor-facing or evaluative project writing, do not flatten every
  assertive sentence into neutral wording after the user asks to remove
  self-judging language. Keep engineering characterizations that the report can
  support with concrete architecture, benchmarks, or workflow evidence, and cut
  only the lines that make the value judgment for the reader.
- When a user corrects a product or platform name during report review, re-check
  whether that correction changes the strategic assessment before keeping the
  earlier recommendation. In this repo, confusing OpenClaw with Claude would
  wrongly turn a researched messaging-native delivery path into speculative
  vendor integration.

## 2026-04-11

- When iterating on presentation drafts, keep one canonical output file unless
  the user explicitly asks for variants. Do not leave parallel "short" and
  "20min" slide markdown files behind; consolidate the stronger draft into the
  requested filename.
- When turning a report into slides, use the report's table of contents to set
  the section structure, then fold thin subsections into stronger composite
  slides. Do not give lightweight topics like target users their own slide when
  they can be integrated into a fuller introduction or differentiation slide.
- For sponsor-academic presentation decks in this repo, do not stop at the
  proposal prose. Cross-check the implemented system docs and evaluation docs so
  the slides reflect the real engineering depth rather than a thinner summary.

## 2026-06-06

- When updating `logs/experiments/index.md` after an experiment review, update
  both commentary layers: the top "latest" note near the frontier table and the
  newest-first `## Findings` entry below the run log. Run table rows alone are
  not enough.

## 2026-07-07

- Prototype code existing (module + tests + an import site) is not the same as
  a feature being adopted. Before recommending an issue be closed as
  "implemented", check the product-facing docs for adoption/decision status —
  e.g. `src/vif/evolution.py` exists with tests, but
  `docs/weekly/weekly_digest_generation.md` explicitly marks evolution gating
  as undecided and outside the committed Coach flow. Issues about product
  capabilities track the decision, not just the code.

## 2026-07-10

- Treat submitted academic and sponsor-facing deliverables as immutable
  records. Before editing a proposal, report, slide deck, generated figure, or
  PDF, verify whether it has already been submitted. If it has, preserve the
  submitted bundle unchanged and create a separate current-state artifact
  instead of regenerating the submitted files in place.
- When archiving a submitted document, keep its referenced images and source
  assets together, preserve relative paths, and verify byte identity against
  the pre-move Git content. Add an explicit archive note so future sessions do
  not mistake the snapshot for a maintained current-state document.

## 2026-07-11

- When a user points out that an invalid benchmark could be confused with, or
  accidentally used as, an active one, do not solve only the wording problem.
  Audit and retire every runnable, configured, artifact, report, and test
  surface; preserve provenance in a clearly marked archive; and add a
  regression check that prevents the old default or paths from returning.
- When a commit review uncovers actionable defects and the user asks to make
  the changes, carry the review through implementation and verification while
  preserving the existing branch and unrelated work. Do not stop at a revised
  assessment or leave the invalid artifact active.
- When work on a Beads issue is described as complete, check every acceptance
  criterion, record the evidence, and close the issue in the same session. If
  only a safe checkpoint is complete, say that plainly and continue the
  remaining acceptance work when the user has authorized the full task.

## 2026-07-12

- Before proposing a VIF architecture study, separate model-development metrics
  from product deployment gates. Pre-register the with-versus-without-Critic
  ablation, audit suspected shortcut dimensions, and do not invent an episode
  benchmark or false-alert tolerance that the project has not adopted.
- When metric policy changes, inspect the executable metric and checkpoint
  selection paths before calling the change implemented. Document a policy/code
  mismatch explicitly and require a tested implementation before treating new
  runs as decision evidence.

## 2026-07-13

- Do not replace one vague experiment word with another. Instead of `arm` or
  `condition`, name the exact experiment setup being compared, and reserve
  `run` or `repeat` for one execution of that setup. Likewise, do not use a bare `set`;
  say `development set`, `final test set`, or name the exact reviewed cases.
- Do not call a separately reviewed downstream target "training-contaminated"
  merely because the same Journal Entries trained an upstream component.
  Separate label validity from evaluation independence: newly reviewed Drift
  labels can be valid development references, while VIF Critic trigger scores
  on Journal Entries seen during training remain in-sample and cannot establish
  generalization.
