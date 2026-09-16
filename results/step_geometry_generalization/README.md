# Step-flow geometry generalization evidence

This directory supports Week 15. It separates two three-seed, 400-epoch
experiments on the 130-case sampled OpenFOAM dataset committed at
`../step_operator_audit/source/dataset.npz`.

- `diverse_geometry_v1`: 107 training, 11 validation, 12 test cases; test
  geometries `g009`, `g023`, `g036`, `g048`.
- `diverse_family_v1`: 103 training, 8 validation, 19 test cases; the held-out
  double-step family comprises `g012`, `g045`–`g051`.
- Original models in these protocols: Geo-DeepONet (`Geo`) and FNO, seeds 17, 29, 43.
- Added fixed-domain baseline: ordinary DeepONet on the same geometry-holdout
  tests, epochs and seeds; Re-only branch and x/y trunk, with geometry withheld.
- `case_metrics.csv`: case-wise three-seed mean and standard deviation.
- `summary.json`: protocol/model summaries and worst cases.
- `geometry_holdout_casebook.pdf`: all 12 geometry-holdout cases.
- `family_holdout_casebook.pdf`: all 19 family-holdout cases.
- `predictions/geometry_holdout`: raw seed-17 fields for two representative
  cases, sufficient to recompute the lecture's CFD/Geo/FNO field figures.

The refreshed teaching figures place CFD, ordinary DeepONet, Geo-DeepONet, and FNO in rows and speed with
streamlines, velocity error, and centered pressure in columns. The notebook
uses only speed/streamlines and centered pressure for the main teaching figure.

The ordinary-DeepONet OpenFOAM bundle is a new reproducible run, not a recovered
historical checkpoint. Its exact code, split, histories and predictions are stored
under `../week15_ordinary_deeponet`. The older V5 metrics remain labelled as the
separate DSMC step-height experiment. U-FNO appears only in the older `g011` audit.
The full sampled OpenFOAM fields are included, while the
original case directories, solver logs, model checkpoints, and exact
train-versus-validation identity lists are not represented as present.
