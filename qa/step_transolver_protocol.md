# Original Transolver: frozen step-flow experiment

Protocol written before training/test inference. Reuse the existing 130-case
OpenFOAM archive; do not generate CFD or use the DSMC micro-step archive.

- Recover the exact historical `wake_focused_geometry_holdout` split from the
  hashed prediction archive: 106 train, 21 validation, three g011 test cases
  at Re 25/50/100. All Reynolds numbers of a geometry stay in one group.
- Seeds 17, 29, 43. No test-based seed choice or hyperparameter sweep.
- Original upstream irregular-mesh Transolver, 3 blocks, width 64, 4 heads,
  16 slices, MLP ratio 2; no pretraining, time input or unified positions.
- Input: x/L, y/H, archived geometry SDF and Re. Train-only standardization.
- Target: u, v, stored p*=Re*p. Train-only target standardization.
- 400 epochs x 16 optimizer updates; one uniform training case and 1,024
  uniformly sampled fluid points per update. No reference-gradient sampling.
- AdamW: LR 1e-3, cosine decay to 1e-5, weight decay 1e-5, gradient clip 1.0.
  Loss: mean squared normalized field error, without target-derived weights.
- Select the checkpoint by the lowest mean per-case validation MSE over all
  fluid points, evaluated at epoch 1 and every 25 epochs. Test only afterwards.
- Validation/test inference uses the complete fluid cloud in one call.
  Splitting a cloud into inference chunks would change contextual attention.
- Save all checkpoints, histories, scaling, predictions, hashes and metrics.
- Recompute all displayed model metrics using identical definitions: joint
  velocity relative L2, mean-centered pressure relative L2, and fluid-cell u<0
  reverse-flow IoU. This IoU is not the filtered -0.01 post-audit metric.
- Plots show fixed seed 17, regardless of which seed performs best.

This is a retrospective reproduction/comparison, not a fresh blind test.
The historical Geo/FNO/U-FNO models have only seed 17 retained. Their original
source and full optimizer configuration are unavailable; loss weights and
target exposures differ. Equal epoch counts do not mean equal FLOPs or a
controlled architecture ranking. This run does not replace the separate
GeoTransolver double-step-family experiment.
