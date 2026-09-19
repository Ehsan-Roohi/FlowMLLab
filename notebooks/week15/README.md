# Week 15 — Geometry-aware neural operators

## Complete post-audit edition

Start with [`W15_Complete_Geometry_Generalization.ipynb`](W15_Complete_Geometry_Generalization.ipynb).
It is the executed companion to the expanded
[`Week 15 lecture`](../../lectures/week15_geometry_generalization.pdf) and brings
the historical and new evidence into one reproducible narrative:

- ordinary DeepONet, Geom-DeepONet and Geo-FNO historical baselines;
- equal 19,200-update comparisons with SMART, GeoTransolver and DoMINO;
- transparent PhysicsX- and LIFT-inspired proxies, never presented as proprietary code;
- a validation-only sweep of peak learning rates `1e-3`, `3e-4` and `1e-4`;
- the rejected common reverse-scale ablation and other negative results;
- independent checks of 342 saved tuned prediction fields;
- Reynolds-stratified and reverse-flow topology diagnostics; and
- six seven-row CFD/model figures at the true 5:1 aspect ratio, with each
  model's own streamlines and no orange threshold overlay.

No new CFD is generated. No double-step motif enters training or validation,
`g005` remains quarantined, and the historically inspected double-step family
is described as a retrospective test. The original notebook and runs below
remain available as historical records and were not overwritten.

The compact repository retains the executed notebook, lecture, summary tables,
figures and checksums. Full checkpoints and saved fields are distributed with
the release/Zenodo evidence archive rather than duplicated in Git history.

[`W15_Geometry_Operators_Step_Audit.ipynb`](W15_Geometry_Operators_Step_Audit.ipynb)
is the executed companion to the Week 15 lecture. It includes:

- the exact ordinary-DeepONet builder already present in
  `qa/step_architecture_v5.py`, plus its retained three-seed DSMC V5 evidence;
- 130 sampled OpenFOAM fields spanning 51 geometry masks;
- the historical Geo-DeepONet/FNO/U-FNO `g011` audit;
- four unseen-geometry tests and a held-out double-step-family protocol; and
- CFD, ordinary DeepONet, Geo-DeepONet, and FNO velocity/streamline and pressure fields using
  common column scales.

The geometry-only strip separates confirmed ordinary-DeepONet training shapes
from the held-out g009 shape without distorting their physical aspect ratio.
The full-size CFD and model comparison follows with streamlines and pressure.
Geo-DeepONet/FNO train-versus-validation geometry identities were not retained,
so their exact training members are not inferred.

Ordinary DeepONet now also has a separate three-seed OpenFOAM geometry-holdout
run with the same 107/11/12 case counts and 400-epoch budget. Its branch sees
Reynolds number only; geometry masks, SDF and geometry IDs are withheld. The
older retained V5 scores remain clearly labelled as a distinct DSMC study.

Rebuild with `python qa/build_step_operator_notebook.py`, then execute all cells.
The lecture is [`lectures/week15_geometry_generalization.pdf`](../../lectures/week15_geometry_generalization.pdf).
