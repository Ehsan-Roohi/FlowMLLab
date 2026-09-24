# Week 7.2 — Sparse-sensor state estimation

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.1](../week07_1/README.md) · [Week 7.3 →](../week07_3/README.md)

Causal filtering of a cylinder wake from noisy sparse sensors.

![Reference and causal state estimates at the final test frame](../../results/week07_2_state_estimation/state_estimation_fields.png)

## Lecture and notebooks

**Lecture:** [Lecture 7.2](../../lectures/week07_2_cylinder_state_estimation.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Sparse-sensor cylinder-wake state estimation | [Open notebook](../../notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb) |

**Module guides:** [Week 7.2 lab](../../notebooks/week07_2/README.md)

## What you will work on

- Linear-Gaussian state estimation and information contracts

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Estimate a wake from noisy velocity sensors.<br>
**CFD / data:** Retained FlowMLLab D2Q9–TRT LBM Re110 trajectory.<br>
**Learning method:** POD–DMD dynamics and a Kalman filter; no neural network in this estimator.



A validation-selected rank-8 Kalman filter assimilates 32 noisy transverse-velocity
sensors on the retained Re110 wake. Mean test relative L2 is **2.18%**, versus
**3.16%** for matched sensor-only POD reconstruction and **4.36%** for open-loop
DMD. Its nominal 95% marginal intervals cover only **55.1%** of sampled values;
the overconfidence is retained as a model failure.
[Protocol, all baselines and limits](../../results/week07_2_state_estimation/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.1](../week07_1/README.md) · [Week 7.3 →](../week07_3/README.md)
