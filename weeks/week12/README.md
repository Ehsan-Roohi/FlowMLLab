# Week 12 — DSMC moment reconstruction

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 11](../week11/README.md) · [Week 13 →](../week13/README.md)

Additive moments, observation-conditioned reconstruction and support.

![Week 12 real DSMC heat flux reference, observation and reconstruction](../../results/week12_research/cavity_qy_hero.png)

## Lecture and notebooks

**Lecture:** [Lecture 12](../../lectures/week12_dsmc_moment_reconstruction.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Noisy DSMC moment reconstruction | [Open notebook](../../notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb) |

**Module guides:** [Week 12 lab](../../notebooks/week12/README.md)

## What you will work on

- Additive moments, prior-plus-observation reconstruction, support

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Reduce cavity heat-flux sampling noise.<br>
**CFD / data:** Author-supplied multi-seed DSMC with an independent finite-sample reference.<br>
**Learning method:** Archived observation-conditioned estimator in the lead figure; a separate 64×32 tanh patch MLP in the Noise2Noise lab.



Existing author-supplied DSMC cavity results associated with
[Roohi, arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637).
Eight seeds, both heat-flux components, 80 recomputed errors; no new DSMC or neural training.
Mean reference NRMSE for qy: Raw(3) 17.61%, Raw(10) 9.80%, conditioned estimator 4.34%.
[Full comparisons, profiles and all-seed errors](../../results/week12_research/README.md) ·
[Notebook and lecture](../../notebooks/week12/README.md). The independent reference still has sampling noise.

The companion [Noise2Noise-style MLP lab](../../results/week12_noise2noise/README.md)
trains on raw DSMC observations, compares seven estimators and reports both
held-out seeds. Its training results are documented separately from the
archived research reconstruction shown above.

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 11](../week11/README.md) · [Week 13 →](../week13/README.md)
