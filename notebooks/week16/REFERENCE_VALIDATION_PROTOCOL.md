# Reference validation protocol

Keep geometry, Mach number, observation line and normalization fixed while evaluating numerical changes. Do not tune alignment or rescale a pressure trace to pass the comparison.

## Retained neural audit

Use the archived predictions verbatim. Verify recovered Git blob hashes, dataset SHA-256, test case identities, shape parameters and sampling coordinates. Compute aggregate waveform relative L2, mean positive-peak relative error and mean pressure-drag relative error. Each aggregate threshold is 10%. Report all per-case waveform errors and the worst error, even when the aggregate passes. A test geometry must not be moved into training after its error is observed.

## NASA CFD

Use NASA SEEB-ALR as-built geometry, M=1.6, zero incidence, and H=21.2 inches. Compare dp/p_infinity within x=25–46 inches using only original NASA macro shifts. Interpolate CFD to measurement points. The finest accepted solution must meet both experiments independently: waveform relative L2 below 20%, positive-peak relative error below 10%. The last two mesh waveforms must differ by less than 5% in relative L2 over the same window. All included accepted mesh solutions must converge; retain failed pilots separately with explicit status. Mesh change is not a formal GCI. The finite nose cap is fixed at two cells and this limitation must be stated.

A solver run completes only when return code is zero and physical fields are finite with positive density and pressure. The density residual must be at most 10^-9 with a drop of at least 5 orders. A small residual does not override a failed experimental comparison.

If a threshold fails, preserve the result and report the failure. Numerical troubleshooting can change the mesh or solver settings, but record the change and rerun the entire comparison. Do not claim that the reconstructed implementation is bitwise identical to an earlier unavailable implementation.

## Release checks

Execute the notebook from distributed files, recompute both audit reports, verify source hashes and inspect rendered lecture pages. Publish geometry, configurations, compact computed signatures, convergence histories, metric definitions and reference citations. Keep full field/mesh files in the associated raw archive. The release must not claim ground-noise reduction, full-aircraft reproduction, original-author weight reproduction or experimental validation of the neural model.
