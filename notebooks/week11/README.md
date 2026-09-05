# Week 11 - Shock and vortex identification

[Lecture PDF](../../lectures/week11_shock_vortex_identification.pdf) ·
[Executable notebook](W11_Shock_Vortex_Identification.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Shock_Vortex_Identification.ipynb)

Allow 75-90 minutes after Weeks 1, 2, 7 and 8. CPU only; use Restart and Run All.
The lab verifies rotation/shear/compression controls, trains a small two-output
MLP, freezes validation thresholds and compares complete held-out analytic cases
against a physical baseline. Outputs remain in memory; results/ is not overwritten.

Research source: Ehsan Roohi, *Physics-audited joint neural segmentation of shocks
and vortex cores: cross-solver transfer and controlled airfoil--cylinder studies*,
author-supplied 2026 manuscript, [ShockVortexML](https://github.com/Ehsan-Roohi/ShockVortexML).
The lab is an original manufactured-field teaching analog, **not** a research
checkpoint, gas-dynamically consistent shock simulation or reproduction of the
paper's segmentation metrics. See [provenance and figures](../../results/week11_12_teaching/README.md).

Next: [Week 12 - noisy DSMC moment reconstruction](../week12/README.md).
