# Executed POD-DeepONet cavity evidence

This directory contains the retained, machine-readable result of `common/run_pod_deeponet_validation.py`. The experiment uses complete Reynolds-number cases throughout; no random grid-point split is used.

## Frozen protocol

- development cases: `Re = 100, 150, 200, 225, 250, 300, 350, 400`;
- architecture-selection case: complete `Re = 225` field;
- untouched blind cases: `Re = 175, 275, 375`;
- candidate POD ranks: `3, 4`;
- candidate branch widths: `(16,16)`, `(32,32)`, `(64,64)`;
- predeclared seeds: `690, 691, 692`;
- primary metric: velocity-vector relative L2 error;
- physical diagnostics: exact wall transform, discrete divergence, centerlines, and Ghia error change.

## Files

| File | Meaning |
| --- | --- |
| `deeponet_selection.csv` | Every development-only rank/width candidate and three-seed validation range |
| `deeponet_metrics.csv` | Individual-seed and ensemble metrics for every blind Reynolds case |
| `deeponet_ghia_metrics.csv` | CFD and POD-DeepONet centerline errors against Ghia at `Re = 100, 400` |
| `deeponet_protocol_and_timing.json` | Frozen case lists, selected architecture, POD energy, training times, inference time, and independent CFD timing |
| `deeponet_predictions.csv` | Full `65 x 65` ensemble `u,v` fields for all three blind cases in a compact wide table |
| `pod_deeponet_ghia_validation.svg` | Ghia centerlines, blind CFD/prediction fields, error field, and measured timing |

The local archival course package may additionally contain PNG/PDF rendering variants and a compressed NumPy prediction archive. The text/SVG files above are sufficient to inspect the evidence and pass the public release validator.

## Interpretation

The result does not assert that a neural network improves the Ghia benchmark. It shows that the POD-DeepONet retains the benchmark fidelity of the educational CFD labels while reducing repeated post-training full-field evaluation from seconds to approximately a millisecond on the recorded CPU run. Training cost is reported separately and must be amortized over repeated queries.
