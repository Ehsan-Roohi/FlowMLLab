# Weeks 17–22: CFD contour previews

[Course home](../../../README.md) · [Course contract and setup](../../TRANSFORMER_COURSE.md)

These homepage previews render retained CFD snapshots and frozen predictions.
They do not fit models or replace the original mechanism and performance figures
in `results/transformer_course_v3/`, the notebooks or the lecture PDFs.

| Week | Contour comparison | Fixed selection |
| --- | --- | --- |
| 17 | CFD transverse velocity, 16-sensor SensorSet reconstruction and error | Re105, frame 140, neural seed 17 |
| 18 | CFD vorticity, causal continuous-state decoder and error | Re110, frames 210 and 245, neural seed 17 |
| 19 | CFD, saved-basis POD projection, 16-sensor gappy POD and error | Re105, frame 140 |
| 20 | CFD, 8-sensor SensorSet/gappy POD, 16-sensor control and errors | Re105, frame 140, neural seed 17 |
| 21 | CFD, causal Transformer, History-MLP, DMD and neural-model errors | Re110, frame 245, neural seed 17 |
| 22 | CFD, pretrained, matched-update scratch, target-POD-MLP and errors | Re105, frame 245, 84 total target labels, neural seed 17 |

The displayed region is the native 32 × 78 retained LBM wake grid, in cylinder
diameters. The cylinder lies upstream outside this fluid-only rectangle; its
perimeter is not a wall. Velocity is normalized by U, and vorticity by U/D.
`contourf` interpolates level crossings for display without smoothing or
regridding the CFD arrays. This coarse teaching dataset is not evidence of grid
independence.

Each figure uses one symmetric signed-field scale across reference and
prediction panels and one nonnegative scale across absolute-error panels.
Relative L2 labels are instantaneous full-field norms, `||prediction − CFD||₂ /
||CFD||₂`; they are not the aggregate trajectory scores. Snapshots and seed are
fixed in the renderer, without selecting them by prediction error.

Week 18 shows the continuous-state CFD decoder, while the separate required
character language model is trained and evaluated in its notebook. The POD
projection in Week 19 uses the saved training basis and truth snapshot as a
representation diagnostic, not a predictor. Week 22's matched-update arm matches
optimizer-update ceilings, not FLOPs or source information; target-POD-MLP is a
diagnostic with a different representation. The broader comparisons and their
limitations remain in the [course contract](../../TRANSFORMER_COURSE.md).

To reproduce, use the course environment and run from the repository root:

```bash
python qa/plot_transformer_homepage.py --output build/transformer-contour-preview
```

This writes six PNGs and a manifest in a separate build directory. Omitting
`--output` refreshes this presentation asset directory. The renderer verifies
that numerical inputs, frozen predictions, checkpoints and original
instructional figures retain their SHA256 hashes. It refuses output outside a
subdirectory of `docs/assets` or `build`.

[manifest.json](manifest.json) records the source and renderer hashes,
environment, exact prediction keys and frame indices, panel array hashes,
coordinate bounds, shared color ranges and generated PNG hashes. Rendering/font
differences may change PNG bytes on another platform; input identity and the
underlying numerical panels can be checked independently.
