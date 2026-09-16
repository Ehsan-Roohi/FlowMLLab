# Week 15 — Geometry-aware neural operators

[`W15_Geometry_Operators_Step_Audit.ipynb`](W15_Geometry_Operators_Step_Audit.ipynb)
is the executed companion to the Week 15 lecture. It includes:

- the exact ordinary-DeepONet builder already present in
  `qa/step_architecture_v5.py`, plus its retained three-seed DSMC V5 evidence;
- 130 sampled OpenFOAM fields spanning 51 geometry masks;
- the historical Geo-DeepONet/FNO/U-FNO `g011` audit;
- four unseen-geometry tests and a held-out double-step-family protocol; and
- CFD, Geo-DeepONet, and FNO velocity/streamline and pressure fields using
  common column scales.

Ordinary DeepONet's retained V5 scores come from the rarefied DSMC step-height
study. No ordinary-DeepONet OpenFOAM prediction bundle is claimed. This boundary
keeps the code/results in the lecture without mixing two different datasets.

Rebuild with `python qa/build_step_operator_notebook.py`, then execute all cells.
The lecture is [`lectures/week15_geometry_generalization.pdf`](../../lectures/week15_geometry_generalization.pdf).

