# Week 12 - Observation-conditioned DSMC moment reconstruction

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
