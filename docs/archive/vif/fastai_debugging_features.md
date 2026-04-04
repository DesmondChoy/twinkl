# fastai Debugging Features: Applicability to VIF Critic Ablations

Research note — 2026-03-04

## Context

The VIF critic training pipeline (`src/vif/train.py`) uses raw PyTorch with
custom training loops. After 12 experiment runs across 5 loss functions
(MSE, CORAL, CORN, EMD, SoftOrdinal), 2 encoders, 3 window sizes, and
4 hidden dimensions, QWK has plateaued at 0.434 (run_010 CORN). Key
unresolved problems include excessive hedging (82%+), poor minority recall,
and systematic Power dimension failure.

fastai provides several diagnostic tools that address blind spots in our
current training infrastructure. Since we don't use fastai's `Learner`, the
recommendations below focus on standalone PyTorch-compatible implementations
or lightweight adaptations that fit our existing loop.

## 1. Learning Rate Finder (`lr_find`)

### What it does

Runs a short mock training where the learning rate increases exponentially
from a tiny value (~1e-7) to a large value (~10) over 100-200 mini-batches.
The loss is recorded at each step. The resulting loss-vs-LR plot reveals
three zones:

- **Shallow zone**: LR too small, loss barely changes
- **Linear zone**: Sharp loss decrease — optimal LR lives here
- **Divergent zone**: LR too large, loss explodes

The recommended LR is the midpoint of the steepest downward slope (not the
minimum). fastai also returns `lr_steep` (steepest slope) and `lr_valley`
(minimum / 10).

### Why this matters for VIF

Our learning rate is fixed at `0.001` for all loss functions. But different
losses have fundamentally different gradient scales:

| Loss | Gradient character |
|------|--------------------|
| MSE | Smooth quadratic |
| CORAL | Binary cross-entropy on cumulative thresholds |
| CORN | Conditional rank probabilities |
| EMD | Squared CDF distance |
| SoftOrdinal | KL divergence with smoothed targets |

A single hardcoded LR is likely suboptimal for at least some of these.
The LR finder would take ~30 seconds per loss variant and could reveal
that some losses need 3-10x different LR to reach their best training
dynamics.

### How to adopt

Use `torch-lr-finder` (standalone, no fastai dependency):

```bash
uv add torch-lr-finder
```

```python
from torch_lr_finder import LearningRateFinder

# After model/optimizer/criterion are set up:
lr_finder = LearningRateFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1, num_iter=200)
lr_finder.plot()  # Shows loss vs LR with steepest-slope annotation
lr_finder.reset()  # Restores original model/optimizer state
```

This works directly with `nn.Module`, `torch.optim`, and `DataLoader` — no
`Learner` required.

### Integration sketch

Add a `--lr-find` flag to `train.py`:

```python
parser.add_argument("--lr-find", action="store_true",
                    help="Run LR finder before training")
```

When active, run the finder, print the suggested LR, save the plot to
`models/vif/lr_find_{loss_fn}.png`, and optionally use the suggestion
automatically.

### Impact estimate

**High**. This is the single highest-ROI diagnostic. If CORN's optimal LR
is 3e-4 while MSE's is 3e-3, we're leaving performance on the table for
every run. The Power dimension instability could also be a symptom of
LR being too high for certain loss surfaces, causing the model to
overshoot narrow optima for rare-class dimensions.

---

## 2. Gradient Norm Monitoring

### What it does

fastai's `GradientClip` callback and custom backward hooks track the L2
norm of gradients at each training step. This detects:

- **Exploding gradients**: Norm spikes → training instability
- **Vanishing gradients**: Norm near zero → model stops learning
- **Layer-wise imbalance**: Early layers getting much smaller gradients
  than later layers (especially relevant for 2-layer MLPs where the
  output layer may dominate)

### Why this matters for VIF

The current training loop has **no gradient clipping and no gradient
monitoring** (`train.py:128-133`):

```python
optimizer.zero_grad()
predictions = model(batch_x)
loss = criterion(predictions, batch_y)
loss.backward()
optimizer.step()  # No clip, no monitoring
```

For the BNN variant (`train_bnn.py`), the loss is `MSE + KL/batch_size`.
The KL divergence term from Bayesian layers can produce gradient magnitudes
that are on a completely different scale from MSE, creating training
instability that would be invisible without monitoring.

The ordinal losses (CORAL, CORN) use binary cross-entropy on cumulative
thresholds, which can produce sharp gradient spikes when logits are far
from targets — a known failure mode that gradient clipping directly
addresses.

### How to adopt

**Option A — Gradient clipping (1 line)**:

```python
# In train_epoch, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Option B — Gradient norm logging (5 lines)**:

```python
# In train_epoch:
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Log grad_norm alongside loss for analysis
history["grad_norms"].append(grad_norm.item())
```

**Option C — Per-layer gradient tracking (for debugging)**:

```python
def log_gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
    return norms
```

### Integration sketch

Add `grad_norms` to the training history dict. Log per-epoch mean gradient
norm. Add a `--grad-clip` flag (default: 1.0) and a `--log-gradients` flag.
Include gradient norm statistics in experiment YAML under
`training_dynamics`.

### Impact estimate

**Medium-High**. Gradient clipping is a near-free safety net that prevents
the worst-case training instabilities. Gradient norm logging would also
help diagnose why Power (a rare-label dimension) fails: if its gradient
contribution is swamped by dominant-class dimensions, that's visible in
per-layer gradient analysis.

---

## 3. Activation Statistics & Dead Neuron Detection

### What it does

fastai's `ActivationStats` callback registers forward hooks on every layer,
recording the mean, standard deviation, and (optionally) full histogram of
activations for each batch. This enables:

- **Dead neuron detection**: Neurons whose output is consistently near zero
  (the `near_zero` metric). These neurons waste parameters.
- **Activation distribution health**: Healthy training shows activations
  with mean near 0 and std near 1. Drift indicates problems.
- **`color_dim` visualization**: A heatmap where the x-axis is
  training batches and the y-axis is activation histogram bins.
  Healthy training shows smooth, bell-shaped distributions.
  Pathological training shows dark bands (dead neurons) or bright
  spikes (exploding activations).

### Why this matters for VIF

The CriticMLP has only 2 hidden layers with GELU activation:

```
Input → Linear → LayerNorm → GELU → Dropout(0.3) → ...
```

With `hidden_dim=64` (current SOTA config) and `dropout=0.3`, we have
only 64 neurons per layer, and 30% are dropped each forward pass.
**If a meaningful fraction of the remaining neurons are dead, the
effective capacity is far below what the parameter count suggests.**

This could explain the QWK plateau: the model may look like it has
22K parameters but only 30-40 effective neurons per layer if GELU is
killing some. LayerNorm should help prevent this, but monitoring
would confirm.

### How to adopt

Simple forward hooks (no fastai needed):

```python
class ActivationMonitor:
    """Track activation statistics during training."""

    def __init__(self, model: nn.Module):
        self.stats = {}
        self._hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.GELU)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            with torch.no_grad():
                self.stats[name] = {
                    "mean": output.mean().item(),
                    "std": output.std().item(),
                    "near_zero_pct": (output.abs() < 0.01).float().mean().item(),
                    "max": output.max().item(),
                    "min": output.min().item(),
                }
        return hook_fn

    def remove(self):
        for h in self._hooks:
            h.remove()
```

### Integration sketch

Enable with `--monitor-activations`. Log stats every N epochs to avoid
overhead. Include summary in experiment YAML:

```yaml
activation_health:
  fc1:
    mean: 0.12
    std: 0.89
    near_zero_pct: 0.04  # 4% dead — healthy
  fc2:
    mean: 0.05
    std: 0.73
    near_zero_pct: 0.15  # 15% dead — concerning
```

### Impact estimate

**Medium**. Not critical for every run, but valuable as a diagnostic when
results are unexpectedly poor. If dead neuron % is high, it directly
motivates either increasing `hidden_dim` or reducing `dropout`.

---

## 4. Training Recorder & Loss Curve Analysis

### What it does

fastai's `Recorder` callback logs the loss and all metrics at every batch
(not just epoch-level averages). It also tracks the learning rate schedule
and provides plotting utilities like `learn.recorder.plot_loss()`.

### Why this matters for VIF

The current training loop logs loss at **epoch granularity** and only
prints every 10th epoch unless there's an improvement. This masks
intra-epoch dynamics:

- **Batch-level loss spikes**: A single adversarial batch can cause a
  gradient explosion that gets averaged away in epoch-level loss but
  permanently damages model weights.
- **Train-val gap trajectory**: The experiment logger stores
  `gap_at_best` (train-val loss difference at best epoch), but not
  the gap trajectory. A widening gap over training indicates overfitting
  onset — useful for deciding whether to reduce capacity or increase
  regularization.
- **LR schedule visualization**: ReduceLROnPlateau reductions are only
  visible indirectly through the `learning_rate` history. A plot of
  LR vs epoch alongside loss would immediately show if the scheduler
  is firing too early, too late, or at the right time.

### How to adopt

Already partially implemented — the `history` dict in `train.py:324-329`
tracks `train_loss`, `val_loss`, and `learning_rate` per epoch. Extensions:

1. **Batch-level loss logging** (opt-in, since it increases log size):

```python
# In train_epoch:
batch_losses = []
for batch_x, batch_y in dataloader:
    ...
    batch_losses.append(loss.item())
return np.mean(batch_losses), batch_losses  # Return both
```

2. **Gap tracking**:

```python
history["train_val_gap"] = [
    t - v for t, v in zip(history["train_loss"], history["val_loss"])
]
```

3. **Plotting utility**:

```python
def plot_training_curves(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_ylabel("Loss"); ax1.legend()
    ax2.plot(history["learning_rate"])
    ax2.set_ylabel("LR"); ax2.set_xlabel("Epoch")
    if save_path:
        fig.savefig(save_path)
```

### Impact estimate

**Low-Medium**. Useful for post-hoc analysis but unlikely to shift the
QWK frontier directly. The train-val gap tracking would be most valuable
for runs with very high param/sample ratios (like the deprecated MiniLM
configurations at 388:1).

---

## 5. Weight Statistics & Model Summary

### What it does

fastai's `learn.summary()` prints a table of every layer with its output
shape, parameter count, and trainable status. For debugging, weight
statistics (mean, std, min, max of each layer's parameters) reveal
initialization health and weight drift during training.

### Why this matters for VIF

The model already logs total parameter count (`train.py:306-307`). But
weight statistics would help diagnose:

- **Tanh saturation in the output layer**: If `fc_out` weights grow large,
  the Tanh output saturates at +/-1, creating "confident wrong" predictions
  with zero gradient — a hard-to-detect failure mode. This could explain
  hedging if the model learns to push most outputs toward 0 (the safe
  center of Tanh's range).
- **LayerNorm parameter drift**: If LayerNorm bias drifts far from 0,
  it shifts the activation distribution and can interact badly with
  dropout.

### How to adopt

```python
def log_weight_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        stats[name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "min": param.data.min().item(),
            "max": param.data.max().item(),
            "norm": param.data.norm(2).item(),
        }
    return stats
```

Log before training (initialization health) and after training (final
state). Include in experiment YAML under `weight_stats`.

### Impact estimate

**Low-Medium**. Mostly useful as a debugging tool when specific runs
produce anomalous results. The Tanh saturation check is worth doing
once to verify the output layer isn't systematically biased.

---

## 6. `TerminateOnNaN` Callback

### What it does

Immediately stops training if the loss becomes NaN or Inf, preserving the
last good checkpoint and reporting which batch/epoch caused the divergence.

### Why this matters for VIF

Currently, if the loss goes NaN, the training loop continues for all
remaining epochs, overwriting the best checkpoint with garbage. The early
stopping logic only checks `val_loss < best_val_loss - delta` — it never
checks for NaN.

### How to adopt

```python
# In train_epoch, after loss.item():
if not np.isfinite(loss.item()):
    raise ValueError(f"Training diverged: loss={loss.item()} at epoch {epoch}")
```

### Impact estimate

**Low** (but near-zero cost). A safety guardrail — NaN probably doesn't
happen often with AdamW + ReduceLROnPlateau, but it's free to add.

---

## Priority-Ordered Recommendations

Based on the current bottleneck (QWK plateau at 0.434, hedging at 82%,
Power dimension failure), here are the features ranked by expected impact
on the ablation experiments:

| Priority | Feature | Effort | Expected Impact |
|----------|---------|--------|-----------------|
| **P0** | LR Finder | 1 hr | Per-loss optimal LR could shift QWK by 5-15% |
| **P1** | Gradient clipping | 5 min | Free safety net, may stabilize Power |
| **P1** | Gradient norm logging | 15 min | Diagnose per-dim gradient imbalance |
| **P2** | Activation monitoring | 30 min | Detect dead neurons in hd=64 config |
| **P2** | NaN termination guard | 5 min | Free safety net |
| **P3** | Train-val gap tracking | 15 min | Better overfitting diagnosis |
| **P3** | Weight statistics | 15 min | Diagnose Tanh saturation / hedging |
| **P4** | Batch-level loss logging | 15 min | Intra-epoch spike detection |

### What NOT to adopt from fastai

- **`Learner` wrapper**: Too invasive; our custom loop is well-structured
  and already has experiment logging, checkpoint management, and evaluation
  integrated.
- **`ClassificationInterpretation` / confusion matrix tools**: Already
  covered by the ordinal eval metrics (QWK, accuracy, per-dimension MAE).
  Our problem is not "we can't see what's wrong" but "we know what's wrong
  (hedging/Power) and need targeted interventions."
- **`ActivationStats.color_dim`**: Useful for deep CNNs with many layers.
  For a 2-layer MLP, simple mean/std/near_zero_pct per layer is sufficient.
- **Mixed precision training**: The critic is ~23K parameters. FP16 gains
  are negligible at this scale.
- **`BnFreeze`**: We use LayerNorm, not BatchNorm, and there's no transfer
  learning involved.

### Relationship to known bottlenecks

| Bottleneck | Relevant fastai feature | Mechanism |
|------------|------------------------|-----------|
| Power QWK instability | LR finder + gradient clipping | Power has very few -1 labels; an aggressive LR causes the gradient signal from these rare samples to get overridden. A lower LR + gradient clipping stabilizes the optimization landscape. |
| Hedging (82%+) | Weight stats (Tanh saturation check) | If the model learns that Tanh(0) = 0 is "safe," weight statistics would show `fc_out.bias` near 0 and `fc_out.weight` with small norms — confirming the hedging is a learned strategy, not a capacity issue. |
| QWK plateau at 0.434 | LR finder | The plateau may simply be that 0.001 is not the optimal LR for CORN. The finder would confirm or refute this in 30 seconds. |
| Minority recall < 30% | Gradient norm monitoring (per-dim) | If minority-class samples generate small gradient norms, they're being outweighed by majority-class samples. This would motivate per-dimension gradient scaling or sample weighting. |
