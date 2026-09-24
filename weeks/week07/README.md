# Week 7 — Unsteady cylinder wakes

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 6](../week06/README.md) · [Week 7.1 →](../week07_1/README.md)

LBM, vortex shedding and autonomous surrogates.

![Week 7: global and worst-frame vorticity errors for the 277-frame autonomous rollout](../../results/cylinder_phase/homepage_week07.png)

## Lecture and notebooks

**Lecture:** [Lecture 7](../../lectures/week07_cylinder_lbm_neural_surrogate.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lattice-Boltzmann cylinder wakes | [Open notebook](../../notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) |
| Modal forecasting | [Open notebook](../../notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb) |

## What you will work on

- Lattice-Boltzmann mechanics
- Cylinder-wake regimes and CFD verification
- Educational unsteady field learning

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict future cylinder-wake vorticity.<br>
**CFD / data:** FlowMLLab D2Q9–TRT lattice Boltzmann solver (LBM).<br>
**Learning method:** The lead result is a phase-stable learned Fourier decoder, not an autoregressive CNN; POD and CNN baselines are linked below.



[Watch the LBM/decoder wake animation](../../results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.webp)

Four initial fields seed **277 autonomous future frames** at unseen **Re = 95**,
with **4.281% global vorticity error** against educational LBM labels.
The grid study passes practical fine-pair limits but fails the formal
asymptotic/GCI gate; these labels are not high-fidelity DNS.
[Video](../../results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4)
· [Model evidence](../../results/cylinder_phase/README.md)
· [Grid study](../../results/cylinder_grid_convergence/README.md)

Continue with the [modal forecasting lab](../../notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb)
and its [reproducible evidence](../../results/modal_labs/README.md).
The [Re100 D40 dataset](../../results/cylinder_d40/README.md) provides force histories,
final fields and a three-grid comparison. A separate
[frozen-model Re115 evaluation](../../results/cylinder_re115_evaluation/README.md)
reports 3.17% global vorticity error, with sampling and pressure limitations.

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 6](../week06/README.md) · [Week 7.1 →](../week07_1/README.md)
