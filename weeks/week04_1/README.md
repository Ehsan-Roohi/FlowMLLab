# Week 4.1 — Classical reduced-order models

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4](../week04/README.md) · [Week 4.2 →](../week04_2/README.md)

POD–Galerkin and POD–DEIM.

![Classical cavity POD–Galerkin and POD–DEIM validation and timing](../../results/cavity_rom/cavity_rom_validation.png)

## Lecture and notebooks

**Lecture:** [Week 4 companion](../../lectures/week04_cavity_surrogates_deeponet.pdf); theory in lab

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Classical POD-Galerkin/POD-DEIM | [Open notebook](../../notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) |

## What you will work on

- Classical dynamical ROM and nonlinear cost

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results. The Week 4 lecture is the shared companion; this notebook contains the additional ROM theory.

## Experiment and results

**Problem:** Evolve cavity flow in a reduced state space.<br>
**CFD / data:** The Week-4 cavity equations and finite-difference full-order model.<br>
**Learning method:** POD–Galerkin and POD–DEIM; neither is a neural network.



Compare reduced dynamics, hyper-reduction, blind trajectories and the offline/online
cost tradeoff. [Run the ROM lab](../../notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4](../week04/README.md) · [Week 4.2 →](../week04_2/README.md)
