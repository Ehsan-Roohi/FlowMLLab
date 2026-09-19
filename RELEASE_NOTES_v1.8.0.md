# FlowMLLab v1.8.0 - Geometry-aware neural operators

This release publishes the complete post-audit Week 15 geometry-generalization
edition. It builds on the public v1.7.0 wake-learning release and brings the
course to 43 notebooks and 26 lecture PDFs. Historical Week 15 runs remain
unchanged.

## Week 15: geometry-aware neural operators

The new executed notebook and 24-page lecture compare ordinary DeepONet,
Geom-DeepONet, Geo-FNO, SMART, GeoTransolver and DoMINO, together with two
explicitly labelled company-inspired proxies. Six seven-row figures cover
Re=25, 50 and 100 at the physical 5:1 aspect ratio, with each prediction's own
streamlines and without the discarded orange threshold overlay.

The frozen 100/8/19 split is verified from geometry masks, `g005` remains
quarantined, and no double-step motif enters training or validation. Because
the double-step family had already been inspected during the research program,
all results are described as retrospective rather than prospectively blind.
No new CFD was generated.

An equal 19,200-update ceiling corrects the earlier undertrained architecture
comparison. A non-double-step validation-only sweep selects peak learning rate
`1e-3` for Geom, SMART and DoMINO. Historical Geom remains best in global
velocity error (9.86%); tuned Geom reaches 10.53%; and DoMINO has the largest
mean reverse-flow IoU in the tuned suite (0.534). The Re=25 low-step regime
remains the key limitation because its Reynolds-number/step-height combination
is absent from training. Negative ablations and failed vortex predictions are
retained.

## Reproduction and evidence

```bash
python -m pip install -e '.[test,reconstruction]'
python -m pytest -q tests qa
flowmllab smoke --root .
flowmllab qa --root .
python qa/validate_course_release.py
```

The compact repository includes Week 15 summary tables, figure provenance and
executed teaching artifacts. Full checkpoints and saved prediction fields are
published as release/Zenodo evidence assets rather than committed to Git history.
Their names and SHA-256 digests are frozen in
[`release_assets_v1.8.0_sha256.txt`](release_assets_v1.8.0_sha256.txt).

The all-versions DOI is `10.5281/zenodo.22074169`. The version-specific v1.8.0
DOI is [`10.5281/zenodo.22840293`](https://doi.org/10.5281/zenodo.22840293).
