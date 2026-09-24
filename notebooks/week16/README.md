> **Week 16 scope (24 September 2026):** the release covers the verified teaching-body CFD, neural model, design checks and Taylor–Maccoll study. NASA SEEB-ALR CFD failed convergence/physical checks and remains a deferred research extension; no successful NASA validation is claimed.

# Week 16: supersonic shape optimization with verified CFD

Start with `W16_Supersonic_Shape_Optimization.ipynb`. The notebook uses independently generated Gmsh/SU2 Euler data, compares POD-based learned models, and inspects fresh CFD evaluations of optimized shapes.

The computed quantity is the **near-field pressure signature**, not perceived ground loudness. The geometry is a fixed-volume body of revolution at zero incidence. All train/test shapes belong to the same two-parameter family. The extrapolation split extends one shape parameter beyond its training range.

- [Assignment and grading](ASSIGNMENT.md)
- [CFD reproduction](../../cases/week16_lowboom/README.md)
- [Lecture](../../lectures/week16_supersonic_shape_optimization.pdf)
- [Data and computational evidence](../../results/week16_lowboom/README.md)

A laptop can execute the data-based notebook. Gmsh and SU2 are required only for regenerating the CFD campaign or checking a new shape. Numerical precision and scope are documented in the evidence report. An extension to atmospheric propagation is a future research task.

## Guided reference study

Read these alongside the executed notebook:

- [NASA geometry, pressure normalization and reference comparison](NASA_REFERENCE_GUIDE.md).
- [Taylor-Maccoll, CFD validation and neural-model testing](MODEL_VALIDATION_GUIDE.md).
- [Comparison with the Beihang journal papers and reproduction limits](REFERENCE_REPRODUCTION.md).

NASA original STEP geometry, wind-tunnel records, coordinate macros and a LAVA solution are distributed in `cases/week16_lowboom/reference/`. These are separate from the two-parameter training bodies.

## Versioned student data and model

The main exercise uses `reference/clean_dataset_v801.npz`, containing all 44 original geometries recomputed with full-field physical and numerical checks, and a separately identified clean-label neural checkpoint. Historical 8.5.0 data/model comparisons remain an explicit failure-analysis lesson. Small residuals did not prevent a local thermodynamic defect in the old fields. Fresh design checks assess the retained geometry directly; they are not predictions from the new checkpoint. See [current publication status](../../PUBLICATION_STATUS.md) before treating the reference extension as a final release.
