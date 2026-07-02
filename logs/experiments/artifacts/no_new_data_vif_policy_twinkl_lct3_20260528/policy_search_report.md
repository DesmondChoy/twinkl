# No-New-Data VIF Policy Search

Selected policy: `router[qwk;self_direction=qwen,stimulation=dimweight,hedonism=twostage,achievement=twostage,power=qwen,security=twostage,conformity=dimweight,tradition=twostage,benevolence=twostage,universalism=qwen]`

## Validation Selection

| Policy | QWK | recall_-1 | MinR | Hedge | Cal | floors | hard ok | hard gain |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `router[qwk;self_direction=qwen,stimulation=dimweight,hedonism=twostage,achievement=twostage,power=qwen,security=twostage,conformity=dimweight,tradition=twostage,benevolence=twostage,universalism=qwen]` | 0.464 | 0.425 | 0.494 | 0.671 | 0.762 | 4/5 | True | True |
| `ensemble[dimweight=1.00;temp=1.00]` | 0.405 | 0.462 | 0.513 | 0.606 | 0.721 | 4/5 | False | True |
| `router[balanced;self_direction=qwen,stimulation=dimweight,hedonism=qwen,achievement=qwen,power=qwen,security=dimweight,conformity=twostage,tradition=twostage,benevolence=qwen,universalism=qwen]` | 0.457 | 0.514 | 0.556 | 0.610 | 0.724 | 4/5 | False | True |
| `ensemble[dimweight=0.50,qwen=0.50;temp=0.75]` | 0.448 | 0.439 | 0.492 | 0.621 | 0.697 | 3/5 | True | True |
| `ensemble[dimweight=0.75,twostage=0.25;temp=0.75]` | 0.424 | 0.482 | 0.484 | 0.632 | 0.719 | 3/5 | True | True |
| `ensemble[incumbent=0.25,dimweight=0.25,qwen=0.50;temp=0.75]` | 0.466 | 0.442 | 0.494 | 0.635 | 0.696 | 3/5 | True | True |
| `ensemble[dimweight=0.75,qwen=0.25;temp=0.85]` | 0.437 | 0.449 | 0.481 | 0.635 | 0.703 | 3/5 | True | True |
| `ensemble[qwen=0.75,twostage=0.25;temp=0.75]` | 0.463 | 0.462 | 0.506 | 0.638 | 0.688 | 3/5 | True | True |
| `ensemble[dimweight=0.25,qwen=0.50,twostage=0.25;temp=0.75]` | 0.479 | 0.437 | 0.483 | 0.643 | 0.706 | 3/5 | True | True |
| `ensemble[dimweight=0.50,qwen=0.25,twostage=0.25;temp=0.75]` | 0.458 | 0.443 | 0.474 | 0.645 | 0.722 | 3/5 | True | True |
| `ensemble[incumbent=1.00;temp=1.00]` | 0.404 | 0.429 | 0.498 | 0.590 | 0.704 | 3/5 | True | False |
| `ensemble[dimweight=0.25,qwen=0.75;temp=0.75]` | 0.459 | 0.475 | 0.515 | 0.608 | 0.684 | 3/5 | True | False |
| `ensemble[incumbent=0.25,qwen=0.75;temp=0.75]` | 0.468 | 0.462 | 0.509 | 0.617 | 0.682 | 3/5 | True | False |
| `ensemble[dimweight=0.25,qwen=0.75;temp=0.85]` | 0.454 | 0.457 | 0.501 | 0.621 | 0.671 | 3/5 | True | False |
| `ensemble[incumbent=0.25,dimweight=0.75;temp=0.75]` | 0.421 | 0.471 | 0.497 | 0.629 | 0.686 | 3/5 | True | False |

## Validation Oracle Upper Bound

| QWK | recall_-1 | MinR | Hedge | Cal |
|---:|---:|---:|---:|---:|
| 0.769 | 0.685 | 0.730 | 0.788 | 0.085 |

The oracle is per-cell and label-aware, so it is not deployable. It is only a complementarity ceiling.

## Fixed Test Evaluation

| Policy | QWK | recall_-1 | MinR | Hedge | Cal | floors | hard ok | hard gain |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `ensemble[dimweight=1.00;temp=1.00]` | 0.342 | 0.378 | 0.449 | 0.599 | 0.726 | 1/5 | False | True |
| `ensemble[incumbent=1.00;temp=1.00]` | 0.362 | 0.313 | 0.448 | 0.621 | 0.713 | 0/5 | True | False |
| `ensemble[qwen=1.00;temp=1.00]` | 0.370 | 0.318 | 0.436 | 0.591 | 0.691 | 0/5 | False | True |
| `ensemble[twostage=1.00;temp=1.00]` | 0.360 | 0.266 | 0.382 | 0.708 | 0.743 | 1/5 | False | True |
| `router[qwk;self_direction=qwen,stimulation=dimweight,hedonism=twostage,achievement=twostage,power=qwen,security=twostage,conformity=dimweight,tradition=twostage,benevolence=twostage,universalism=qwen]` | 0.365 | 0.244 | 0.393 | 0.676 | 0.718 | 0/5 | False | True |

## Selected Hard-Dimension Test Deltas vs Incumbent

| Dimension | QWK delta | active-recall delta |
|---|---:|---:|
| hedonism | -0.005 | -0.010 |
| security | -0.021 | -0.075 |
| stimulation | -0.009 | +0.077 |

## Verdict

- Validation hard guardrail: True / True
- Test hard guardrail: False / True
- Promotion floors met on test: 0/5
