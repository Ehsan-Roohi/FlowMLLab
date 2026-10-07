# Week 22 — Transfer and information budgets

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 21](../week21/README.md)

Target-label accounting, matched updates and representation controls.

![Transfer errors across three Reynolds numbers and three fully counted target-label budgets](../../results/transformer_course_v3/week22.png)

## Lecture and notebooks

**Lecture:** [Lecture 22](../../lectures/week22_cfd_transformer.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Transfer and information budgets | [Open notebook](../../notebooks/week22/W22_CFD_Transformer.ipynb) | [Setup and reproduction guide](../../notebooks/week22/README.md) |

## What you will work on

- Audit transfer before calling it a foundation model

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Audit a transfer benefit before attributing it to pretraining.<br>
**CFD / data:** Re90 source and Re100/Re105/Re110 retained targets; budgets of 44, 84 and 154 target frames include validation and initialization observations.<br>
**Learning method:** Compare scratch, pretrained, matched-update, target-POD-MLP and DMD controls under fixed rollout initialization.

Separate full-field and in-subspace error exposes the source-POD floor. Shading is the range across three seeds, not a confidence interval. Source access, representation fitting and selected/executed updates remain part of the comparison.



[Student notebook](../../notebooks/week22/W22_CFD_Transformer.ipynb) · [Lecture](../../lectures/week22_cfd_transformer.pdf) · [Setup and assignment](../../notebooks/week22/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 21](../week21/README.md)
