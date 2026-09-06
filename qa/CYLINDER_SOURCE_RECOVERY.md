# Cylinder source recovery — 2026-09-06

The original working directory was located on Unity. Five candidate ensemble
members exist under `modelsold/cylinder_fusion_model_0.h5` through `_4.h5`.
Their SHA-256 hashes and the observed source identity are recorded in
[the recovery record](CYLINDER_SOURCE_RECOVERY.json). These are candidate
weights, not yet authenticated as the published figure-producing ensemble.
No original file was changed, no training or inference was launched.

## Verified observations

- `models/` has member 0 and best-checkpoint variants for members 0/1, but lacks
  the complete set of five identically named ensemble members. `modelsold/`
  contains all five.
- Member 0 HDF5 metadata, inspected without TensorFlow deserialization, records
  Keras 2.15.0, two inputs, eight 256-unit tanh dense layers, four Add layers,
  one Lambda layer and a three-unit linear output. All five members have the same ordered layer classes and dense widths.
  Full configurations have different hashes; topology and Lambda semantics
  remain unchecked. This does not prove a match to a paper figure.
- Both `32fusion.py` and `33fusion.py` name Mach 5/7/9 as training and Mach 10
  as test. Both reference the same checkpoint basename. Filename matching is
  therefore insufficient to identify the producing revision.
- In `33fusion.py`, training uses `N_f=50000` per file (lines 181–194, 340),
  whereas replay rebuilds scalers with `N_f=200000` (lines 241–248). Sampling
  uses `min(N_f, len(df))`, seed 42, before invalid-target filtering (107–125).
  When the sampled rows differ, refitting can alter the input transform and
  inverse output transform for unchanged weights. Numerical impact is unmeasured.
- Recursive filename search found no `*scaler*`, `.pkl` or `.joblib` files.
  This is not proof that scalers cannot be embedded elsewhere or reconstructed.

## Next executable gate

Recover the exact training data list and sampling configuration for these
weights; compare all five stored architecture configurations. Reconstruct the
training scalers only from that identified configuration, persist their numeric
parameters, and compare against historical exported predictions before scoring
Mach 15. Do not substitute the replay's larger-sample scalers or start the old
script directly: its incomplete-ensemble path can trigger fresh training.

The source-directory discovery advances the [paper parity audit](WEEK71_PAPER_PARITY.md).
It does not close quantitative reproduction. The user-reported asymmetric
near-cylinder shock overlay belongs to a separate ShockVortexML diagnostic;
these DSMC surrogate files cannot explain that image by themselves.

## Executable correction

The recovered `33fusion.py` revision is now adapted in
[`examples/cylinder_fusion`](../examples/cylinder_fusion/README.md). Explicit
training writes paired preprocessing and model hashes; replay never refits
scalers or launches training. A fresh-process synthetic smoke run and bundle
regression tests pass (see the checked-in smoke report).

Further source inspection confirms this script trains six 512-wide layers per
branch/trunk with BatchNormalization and Dropout, whereas the historical weights
inspected above have four 256-wide layers. The executable correction is complete;
historical checkpoint-to-paper attribution remains a separate evidence question.
