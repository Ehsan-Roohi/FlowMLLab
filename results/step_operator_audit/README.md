# Geometry-operator step-flow teaching audit

This package contains the author's existing `wake_predictions.tgz`, copied without
modification from Downloads, the verified `source/dataset.npz`, and independently
recomputed teaching diagnostics.
Archive SHA256: `190bb252c2739fc2acfa0841233652144b82eb9ca3a01ad6ddb2ae0f0429ade1`.
Dataset SHA256: `28d4d4c440cdc4c1ac1d13749ce00b0690d99f29cf20fd56c65fc00b6a8058fd`.

Run [Week 9 Lab 3](../../notebooks/week09/W9_Lab3_Geometry_Operators_Step_Audit.ipynb).
It uses NumPy, pandas, SciPy and Matplotlib on CPU. No training or GPU is required.
Rebuild the notebook source with `python qa/build_step_operator_notebook.py`;
this clears outputs, so execute all cells again afterwards.

## Evidence boundary

- Nine retained predictions: Geo-DeepONet, FNO, U-FNO; geometry g011; Re 25/50/100;
  seed 17. These are continuum CFD reference fields, not the Week 9 DSMC archive.
- The dataset contains 130 accepted sampled OpenFOAM fields from 51 geometry masks:
  106 training cases (41 geometries), 21 validation cases (9 geometries), and
  3 test cases (one geometry). All Reynolds numbers for a geometry remain together.
- The dataset has no explicit geometry-ID column. IDs are inferred from the source's
  first-occurrence order of unique masks. The g011 mapping is verified bit-for-bit
  against all three retained reference fields; the other ID labels remain an inference
  until the original case-name manifest or training source is recovered.
- All 18 velocity and raw-pressure comparisons match source metrics within
  0.001 percentage points. Four diagnostic tests check derivative convention,
  perfect predictions, pressure-offset removal and exclusion of solid stencils.
- Reverse-flow IoU under the explicit all-fluid/u<0 definition differs from the
  original report. Both scores and deltas are preserved. The original threshold
  and region were not included, so exact reproduction is not claimed.
- Derivative errors use a one-cell-eroded fluid mask; they are not automatically
  comparable to previously reported vorticity errors with different stencils.
- One seed cannot establish training uncertainty. This historically examined
  geometry is development evidence, not a fresh blind test.
- Pressure absolute units cannot be confirmed from this archive alone. Per-case
  relative errors are invariant under a common nonzero pressure scaling.
- Missing: original training source at the recorded hash, model checkpoints,
  complete OpenFOAM case directories and later combined-loss/multi-family prediction bundles.
  This lab does not reproduce their training or their later reported improvements.
- No new license is assigned to the author's upstream research materials.

Generated files: 130-row case manifest, nine-row prediction-metric CSV,
source-member hashes and metadata, execution summary, dataset coverage and split
examples, three four-row field comparisons, metric comparison, and local-error plot.
The archive is read in memory, never blindly extracted.
