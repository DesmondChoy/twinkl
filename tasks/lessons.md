# Lessons

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

## 2026-03-26

- Do not describe human annotation subsets as "ground truth" unless the user or
  project docs explicitly define them that way. In this repo, human annotations
  can serve as a limited external benchmark, but they may be sparse, non-expert,
  and narrower than the full judged corpus.
- When reporting judge-vs-human agreement, state the evaluation coverage
  explicitly: how many entries and personas are in the overlap, how many are
  excluded, and that the metric is computed on the intersection only rather than
  the full dataset.
