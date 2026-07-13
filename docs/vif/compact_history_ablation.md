# Compact History Ablation (`twinkl-749`)

## Question

Can the VIF Critic use a small, production-legal summary of prior Journal
Entries without repeating the parameter explosion of raw window
concatenation?

The live default remains `window_size: 1`. This is a representation ablation,
not a change to the repaired Security target or the product's Drift contract.

## Input contract

For Journal Entry `t`, the tested state contains:

1. the complete 256-dimensional Nomic embedding for Journal Entry `t`,
   including its already-completed same-session nudge and response;
2. the mean of up to three strictly prior Journal Entry embeddings, truncated
   to the leading 64 Matryoshka dimensions and L2-normalized;
3. one history-availability feature: `real_prior_count / 3`; and
4. the existing ten-dimensional value profile.

The first Journal Entry receives 65 history zeros. Padding is never averaged.
Prior Journal Entries are selected by chronological position inside the same
persona, so future Journal Entries, other personas, labels, LLM-Judge
rationales, biography, demographics, and synthetic metadata cannot enter the
state.

This grows the state from 266 to 331 dimensions. At hidden width 64, the first
layer gains `65 * 64 = 4,160` trainable weights, within the issue's roughly
5,000-parameter increment.

## Design choice

| History representation | Budget | Decision |
|---|---:|---|
| Full pooled 256-d summary | +16,384 first-layer weights | Reject: exceeds budget. |
| Mean-pooled 64-d Matryoshka summary | +4,160 weights | Test first: preserves the current embedding and provides a separate context channel. |
| In-place mean/EMA fusion | 0 weights | Keep as fallback: cheapest, but current evidence and history become inseparable. |
| Tiny GRU | About 4,752 parameters at hidden size 6 before downstream use | Defer: too much estimation complexity before pooling establishes signal. |
| Standard attention | Far over budget | Reject for this POC. |

Mean pooling is the order-invariant baseline motivated by
[Deep Sets](https://arxiv.org/abs/1703.06114). GRU parameterization follows
the [PyTorch GRU contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
and the sequence-encoder design introduced by
[Cho et al.](https://aclanthology.org/D14-1179/). Attention is deferred relative
to the standard projection-heavy formulation in
[Vaswani et al.](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need.pdf).

## Evaluation and stopping rule

Run seed 11 against repaired-Security baseline `run_058`, holding the target,
holdout, encoder, model width, optimizer, and checkpoint policy fixed. Expand
to seeds 22 and 33 only if the first run is directionally promising on the
combined package: QWK, `recall_-1`, minority recall, hedging, calibration, and
the hard dimensions.

Security interpretation is deliberately limited. Its repaired labels were
created from the exact current Journal Entry, including its displayed nudge and
response, so this experiment can show whether legal history helps prediction
under that fixed regime; it cannot prove that a history-visible Security target
is better.
