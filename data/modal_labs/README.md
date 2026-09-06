# Real LBM wake subset for modal teaching labs

These are **previously generated FlowMLLab cylinder CFD fields**, not synthetic
ML labels, not external package example data, and not a new run of a Roohi paper.
The source is the author-generated [cylinder-cfd-v1 release](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1),
produced by the repository's versioned LBM solver. Do not attribute these low-Mach
LBM fields to the separate hypersonic DSMC/Fusion-DeepONet article.

`manifest.json` records original and derived SHA-256 hashes. Regenerate into a
fresh directory with `python qa/prepare_modal_lab_data.py --output tmp/new_modal_data`.
The first download is approximately 274 MB; notebooks use this bundled 30 MB
subset without downloading or rerunning CFD. Source code is covered by the
repository MIT license; no external package code or data were copied.

Each file has 281 time frames on a 32 x 78 fluid-only rectangular wake ROI.
`u` and `v` are normalized by U; `omega` is normalized by U/D, computed using
centered differences on the native grid BEFORE taking every second spatial point.
`x,y` are in cylinder-diameter units relative to its center; `t` is in D/U units.
No filtering or interpolation changes the data. Subsampling can alias fine scales.
The cylinder surface is outside this ROI; its perimeter is **not a physical wall**.
Filled contour plots interpolate level crossings for display, not for fitting or
scoring, and do not imply additional CFD resolution.

The source has only 12 lattice nodes per diameter. It is educational, not
grid-independent CFD. Existing box/acoustic-mode cautions still apply.
`original_strouhal` comes from the complete source force history and is a
diagnostic reference ONLY, never a training input or model-selection target.

## Two different questions, two frozen splits

- Forecasting: Re110 frames [0,160) training, [160,210) validation, [210,281) test.
  Future prediction within one previously inspected case, not unseen-Re prediction.
  No reset or observation assimilation at the test boundary.
- Sensing: complete Re90/Re110 training, Re100 validation, Re105 test.
  All POD fitting and sensor locations use training cases only. The validation
  case selects budget; synthetic measurement noise is 1% of training v RMS.
  Re105 was inspected in prior work: this is a retained test, not a new blind claim.

See [protocol and results](../../results/modal_labs/README.md). The four NPZ files
are an exact, checksummed derivative of the published author data, not outputs of
PyDMD, PySensors, PySINDy or PDEBench.
