# Week 7.1 — Rarefied hypersonic cylinder

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7](../week07/README.md) · [Week 7.2 →](../week07_2/README.md)

DSMC fields and Mach-to-field operators.

![Original 400 by 400 Mach-8.5 DSMC fields and interpolation errors](../../results/hypersonic_cylinder_week7_1/cylinder_homepage.png)

## Lecture and notebooks

**Lecture:** [Lecture 7.1](../../lectures/week07_1_hypersonic_rarefied_cylinder.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Rarefied hypersonic-cylinder operator learning | [Open notebook](../../notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) |

## What you will work on

- Rarefied hypersonic-cylinder operator learning

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict cylinder fields as Mach number varies.<br>
**CFD / data:** Author-supplied DSMC archives; exact source/checkpoint attribution remains subject to the linked audit.<br>
**Learning method:** 3×96 tanh MLP versus Mach interpolation; separate from the article's Fusion-DeepONet.



The original **400 × 400 Mach-8.5** fields are rendered with continuous contours.
Error panels compare against Mach-8/Mach-9 interpolation; gray marks the masked
solid/sentinel region.
[Figure provenance](../../results/hypersonic_cylinder_week7_1/cylinder_homepage_provenance.json)

Across the five held-out cases in the **compact teaching dataset**, interpolation
errors are **0.398% / 0.609% / 0.779%** for local Mach, temperature and pressure.
The trained **3×96 tanh MLP** gives **1.53% / 2.70% / 2.50%**, with training
errors **1.26% / 2.21% / 1.82%**. Interpolation still wins; the earlier
underfit random-feature ridge is no longer the default classroom comparison.
These are new teaching runs, not the published model's accuracy.
[Data and paper](../../data/hypersonic_cylinder/README.md) · [MLP metrics](../../results/hypersonic_cylinder_week7_1/mlp_metrics.json)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7](../week07/README.md) · [Week 7.2 →](../week07_2/README.md)
