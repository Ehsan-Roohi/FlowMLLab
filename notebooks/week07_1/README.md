# Week 7.1 - Rarefied hypersonic cylinder

This incremental lecture/lab follows the Week-7 continuum/LBM cylinder and
introduces parameterized DSMC fields, leakage-free Mach-case splits,
Fusion-DeepONet anatomy, a strong field-interpolation baseline, and a trained
3x96 tanh MLP with whole-case validation and training-error diagnostics.

- `W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb`: student notebook;
- `make_week7_1_notebook.py`: deterministic notebook builder;
- `../../data/hypersonic_cylinder/`: compact author-released DSMC derivative;
- `../../results/hypersonic_cylinder_week7_1/`: executed metrics and field audit;
- `../../lectures/week07_1_hypersonic_rarefied_cylinder.pdf`: companion lecture.

The default CPU model is an MLP, not the published operator. The notebook exposes the
reviewed Fusion-DeepONet topology as an optional TensorFlow path and does not
claim to reproduce the full paper model.

The data predate their course integration: see [research provenance and reuse
status](../../DATA_PROVENANCE.md). Historical ridge metrics remain in
`metrics.json`; the revised executed MLP benchmark is `mlp_metrics.json`.
