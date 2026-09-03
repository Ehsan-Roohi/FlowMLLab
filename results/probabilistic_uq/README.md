# Probabilistic UQ evidence

This directory contains the retained evidence for the incremental Week-2.1
probabilistic-UQ module. Regenerate it with:

```bash
python qa/run_probabilistic_uq_validation.py --root .
```

The experiment fits a rank-4 velocity POD basis and independent fixed-kernel
Gaussian processes using complete Reynolds-number cases
`100, 150, 200, 225, 250, 350, 400`. `Re=300` is withheld for a single
multiplicative interval-width calibration. The existing `Re=175, 275, 375`
test cases remain blind until the protocol is frozen.

`blind_metrics.csv` compares the POD--GP mean with complete-case linear
interpolation and reports proper Gaussian scores, interval coverage, interval
width, an error--spread rank correlation, and physical diagnostics.
`calibration.csv` retains per-case nominal and observed coverage.
`protocol.json` records the data hash, split, fixed hyperparameters, public
sources, originality declaration, and claim boundary.

Pointwise coverage across a CFD grid is descriptive: neighboring residuals are
correlated and do not form independent calibration samples. The retained scale
factor is not a certification guarantee under Reynolds extrapolation, geometry
change, solver discrepancy, or experimental measurement noise.
