# Week 14 — RANS, inverse PINN and neural turbulence closures

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 13](../week13/README.md) · [Week 15 →](../week15/README.md)

Davidson-based RANS, inverse PINN and neural closures.

![Week 14 classical and table-PINN channel profiles against DNS](../../results/week14_validation/profiles.png)

## Lecture and notebooks

**Lecture:** [Lecture 14](../../lectures/week14_rans_pinn_nn.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| RANS, inverse PINN and neural closures (local checkout) | [Open notebook](../../notebooks/week14/W14_pyCALC_RANS_PINN_NN.ipynb) | [Setup and reproduction guide](../../notebooks/week14/README.md) |

**Module guides:** [Week 14 lab](../../notebooks/week14/README.md)

## What you will work on

- RANS closure, inverse PINN and local-feature NN

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results. This module uses a separate local environment; hosted Run All is not verified. Follow its setup guide.

## Experiment and results

**Problem:** Improve turbulent kinetic energy without confusing a coefficient fit
with a validated coupled flow solution.<br>
**CFD / data:** Lars Davidson's pyCALC-RANS source-checkpoint channel restarts;
Lee-Moser DNS reference statistics.<br>
**Learning method:** Inverse PINN diffusion inference and the original small
ReLU coefficient-regression protocol, with a separately labeled interpolation control.



[Executed notebook and CPU setup](../../notebooks/week14/README.md) ·
[Lecture](../../lectures/week14_rans_pinn_nn.pdf) ·
[Run ledger](../../results/week14_validation/README.md) ·
[Paper-claim alignment](../../docs/WEEK14_PAPER_ALIGNMENT.md).

This completed teaching audit retains unsuccessful convergence criteria and
source-target regeneration differences. The plotted table-based PINN comparison
is **not** a verified reproduction of the paper's final PINN-NN curves.
The notebook reruns the small NN fit and checks retained CFD evidence; it does
not silently present saved solver fields as a fresh Run-All CFD calculation.

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 13](../week13/README.md) · [Week 15 →](../week15/README.md)
