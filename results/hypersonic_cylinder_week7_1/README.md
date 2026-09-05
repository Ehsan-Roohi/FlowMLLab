# Week 7.1 evidence

The DSMC fields come from earlier Roohi et al. research,
[PoF 38, 057108 (2026)](https://doi.org/10.1063/5.0334590), not from an
AI-generated simulation. See the [data card](../../data/hypersonic_cylinder/README.md)
for exact lineage, comparison to the article and pending reuse permissions.

`mlp_metrics.json` records the current trained 3x96 tanh classroom baseline,
including seed, training time, selected epoch and validation loss history.
`metrics.json` preserves the earlier underfit random-feature ridge experiment
for historical traceability; it is **not** the default lab model.

Reproduce into scratch with `python qa/run_hypersonic_cylinder_evidence.py`.
The runner does not overwrite this directory. Promotion of a reviewed result
is an explicit maintainer action, not a notebook side effect.

`cylinder_homepage.png` and its provenance file retain the separate original
400x400 source-grid visualization. Do not mix compact and full-grid errors.
