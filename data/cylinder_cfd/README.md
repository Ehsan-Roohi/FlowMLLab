# Cylinder CFD dataset

The complete nine-case dataset is generated from the versioned FlowMLLab LBM
solver and published as durable assets of the GitHub release
`cylinder-cfd-v1`.  It is not mixed into the source checkout.

| Role | Reynolds numbers |
|---|---|
| Development/training | 60, 80, 90, 110, 120, 140 |
| Validation/model selection | 100 |
| Fresh never-opened test | 95 |
| Retained historical test | 105 |

Regenerate with `python qa/generate_cylinder_cfd_dataset.py --workers 3` and
validate with `python qa/validate_cylinder_cfd_dataset.py data/cylinder_cfd`.
Every NPZ contains normalized `u`, `v`, and pressure snapshots, the solid mask,
coordinates, force histories, Strouhal number, and solver provenance.  The
generated `manifest.json` records byte sizes and SHA-256 checksums.
