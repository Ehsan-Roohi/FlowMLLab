# Week 15 family-holdout ordinary DeepONet (three seeds)

Retained evidence of the ordinary-DeepONet family-holdout protocol
(`qa/train_week15_family_deeponet.py`): one directory per seed (17, 29, 43) with
the trained weights, `manifest.json`, `case_metrics.csv`, `scaling.npz` and the
held-out predictions.

## Provenance note

The seed-17 and seed-43 directories were regenerated on 19 September 2026 by the
`week15-family-deeponet.yml` GitHub Actions workflow, which at that time
committed its output to `main` (commits `0ca9814`, `f08baec`; the v1.8.3
release commit `da086b5` retained the regenerated state). The retraining
follows the same script, data and seeds as the v1.8.0 files, but it is not
bit-stable across machines: the per-case metrics differ from the v1.8.0 values by
at most 1.4e-5 relative (fifth significant figure) and the weight files differ
in their bytes. In the seed-29 directory only the recorded `training_seconds`
changed; its weights, predictions and metrics are the v1.8.0 files.

Since v1.8.3 the workflow has `contents: read` and uploads its output as a run
artifact for review instead of committing it, so the files in this directory
change only through a reviewed commit. Treat the numbers here as the reference
values of this repository version; anyone regenerating them on another machine
should expect differences at the same 1e-5 level.
