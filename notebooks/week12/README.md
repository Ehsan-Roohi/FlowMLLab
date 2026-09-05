# Week 12 - Observation-conditioned DSMC moment reconstruction

## Practical extension: actually train on real DSMC

The notebook now includes a fresh CPU Noise2Noise-style patch MLP, not just
archived figures. Four raw-observation seeds train the model, two choose the
Gaussian baseline width, and two evaluate all methods. The high-budget
reference is excluded from fitting and selection. Training takes about 14
seconds on the tested machine; allow another 20-30 classroom minutes.

Use a current complete checkout or the main-branch Colab link below.
No new release or DOI has been created. The lecture now has 12 pages.

[Fresh fit, all scores and failures](../../results/week12_noise2noise/README.md) ·
[Source data and independence limits](../../data/week12_noise2noise/README.md).

The MLP improves qy but loses to simpler baselines on qx; its 160-epoch
convergence warning is retained. This is same-condition denoising of an
already-inspected archive, not a new blind test or the paper's research model.
Every Run All retrains the teaching MLP without rewriting data or results.

## Materials

[Lecture PDF](../../lectures/week12_dsmc_moment_reconstruction.pdf) ·
[Executable notebook](W12_DSMC_Moment_Reconstruction.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb)

Allow 90 minutes after Weeks 3, 4 and 10. CPU only; use Restart and Run All.
Verify additive-first centralisation, fit a development-only spectral prior/gain,
compare noisy sampling budgets and a Gaussian baseline, preserve the observation
mean and test a reference-free support warning. Nothing under results/ is modified.

Research source: Ehsan Roohi, *Geometry-native machine learning reconstruction of
DSMC moment fields with support monitoring*,
[arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637), 2026.
JCP submission is author-confirmed; no JCP publication is claimed.
This **synthetic scalar CPU analog is not the paper's MambaIR or coupled cylinder
estimator** and produces no new DSMC data. See [scope and figures](../../results/week11_12_teaching/README.md).

The final section audits [real DSMC cavity results](../../results/week12_research/README.md)
from the author's existing JCP2 archive: qx/qy, eight observation seeds, 80
recomputed scores and large common-scale contours. The notebook checks retained
hashes and recomputes first-seed errors. This is archive re-evaluation, not new
neural training or a newly blind trial. Full comparisons and profiles retain
the prior-only and Raw(10) baselines alongside the conditioned estimator.
