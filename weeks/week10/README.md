# Week 10 — DSMC cavity and molecular shocks

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 9](../week09/README.md) · [Week 10.1 →](../week10_1/README.md)

Cavity and mono/diatomic shock reproduction.

![Large-format DSMC cavity and diatomic-shock course reproduction](../../results/aescte_dsmc/week10_dsmc_reproduction_summary.png)

## Lecture and notebooks

**Lecture:** [Lecture 10](../../lectures/week10_dsmc_data_driven_surrogates.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| DSMC cavity and mono/diatomic shock reproduction | [Open notebook](../../notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) |

**Module guides:** [Week 10 lab](../../notebooks/week10/README.md)

## What you will work on

- DSMC solver qualification and provenance
- Rarefied-cavity parameter synthesis
- Mono/diatomic shock operators

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Reconstruct rarefied cavity and mono/diatomic shock data.<br>
**CFD / data:** Author-supplied article DSMC tables, not new particle-solver runs.<br>
**Article learning method:** the cavity uses a family of coordinate MLP experts,
one per training Kn: fixed Fourier features of $(x,y)$ feed three 256-unit
Swish dense layers, and neighboring experts are fused by log-Kn interpolation.
It is **not DeepONet**. The article uses DeepONet for its diatomic-shock study.<br>
**Displayed course reproduction:** direct log-Kn interpolation for the cavity and
POD–polynomial profile surrogates for the shocks. These displayed predictions do
not rerun the article's trained neural networks.



The article-data experiment retains **1.281% maximum primary cavity NRMSE**
and **1.018% maximum shock-profile relative L2 error**, with the data contract
and regeneration workflow available for inspection.
[Article (Roohi & Shoja-Sani, 2026)](https://doi.org/10.1016/j.ast.2025.110785)
· [Reproduction evidence](../../results/aescte_dsmc/README.md)
· [Data contract](../../data/aescte_dsmc/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 9](../week09/README.md) · [Week 10.1 →](../week10_1/README.md)
