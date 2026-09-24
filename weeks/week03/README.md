# Week 3 — Kinetic theory and DSMC

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 2.1](../week02_1/README.md) · [Week 4 →](../week04/README.md)

Maxwellian sampling and particle simulation.

![HS–NTC DSMC wall-pressure validation against Mohammadzadeh reference data](../../results/article_figures/fig10a_mohammadzadeh_validation.png)

## Lecture and notebooks

**Lecture:** [Lecture 3](../../lectures/week03_kinetic_dsmc.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Maxwellian noise and ML | [Open notebook](../../notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) |
| Mini DSMC cavity | [Open notebook](../../notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) |

**Module guides:** [Week 3 labs](../../notebooks/week03/)

## What you will work on

- Maxwellian distributions and macroscopic moments
- DSMC logic
- Noisy field estimation

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict wall pressure from molecular motion.<br>
**CFD / data:** FlowMLLab hard-sphere DSMC with no-time-counter (HS–NTC) collisions.<br>
**Learning method:** No network in the displayed particle-solver validation.



Connect molecular sampling to a macroscopic observable through the executed
HS–NTC wall-pressure validation.
[Validation contract](../../ARTICLE_FIGURE_MAP.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 2.1](../week02_1/README.md) · [Week 4 →](../week04/README.md)
