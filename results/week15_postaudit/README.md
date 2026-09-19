# Week 15 post-audit evidence summary

This directory contains the compact, repository-sized evidence for the expanded
Week 15 teaching edition. No CFD was generated for this update and no historical
run was overwritten.

## Protocol

- 100 training cases, 8 non-double-step validation cases, 19 retrospective
  double-step test cases, and three quarantined `g005` cases.
- The split is reconstructed from the geometry masks. No training or validation
  mask contains the test family's two consecutive descending steps.
- Model and learning-rate selection uses validation only. The double-step family
  is opened after the selection freeze, but is called retrospective because it
  was historically inspected.
- The fresh comparison uses a 19,200-update ceiling and three seeds. Equal update
  count does not imply equal FLOPs or equal architecture-specific optimization.

## Files

- `consolidated_results.csv` and `.json`: nine-model global summary.
- `core_g*_Re*.png`: CFD plus historical DeepONet/Geom/Geo-FNO and tuned
  Geom/SMART/DoMINO for Re=25, 50 and 100.
- `extended_g*_Re*.png`: CFD plus tuned Geom/SMART/DoMINO, matched GeoTransolver,
  and the two transparent company-inspired proxies.
- `week15_training_geometries_homepage.png`: all 39 training masks.
- `figure_provenance.json`: input paths and SHA-256 hashes used by the figures.
- `notebook_execution.json`: clean execution record for the companion notebook.

Every contour figure uses a shared banded speed scale within the figure, a true
5:1 physical aspect ratio, and streamlines computed from the field in its own
row. The orange reverse-flow threshold overlay is intentionally absent. Reverse
IoU values still use the retained `u < -0.01`, four-connected, eight-cell-filter
metric and are therefore threshold-dependent.

## Main results

Historical Geom-DeepONet has the smallest global velocity error (9.86%). Among
the validation-tuned models, Geom-DeepONet has 10.53% velocity error, SMART
11.61%, and DoMINO 15.59%; DoMINO has the largest mean reverse IoU (0.534).
At Re=25 the corresponding IoUs are 0.261, 0.299 and 0.357, versus 0.096 for
historical Geom. The improvement in footprint detection does not remove the
reverse-flow magnitude overshoot.

The complete checkpoints, histories, 342 tuned prediction fields and independent
metric records are in the frozen release/Zenodo evidence archive. Company-inspired
PhysicsX and LIFT rows are proxies disclosed at the layer/feature level; they are
not proprietary implementations.
