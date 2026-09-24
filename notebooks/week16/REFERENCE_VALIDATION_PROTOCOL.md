# Reference validation protocol

Keep geometry, Mach number, observation line and normalization fixed while evaluating numerical changes. Do not tune alignment or rescale a pressure trace to pass the comparison.

## Retained neural audit

Use the archived predictions verbatim. Verify recovered Git blob hashes, dataset SHA-256, test case identities, shape parameters and sampling coordinates. Compute aggregate waveform relative L2, mean positive-peak relative error and mean pressure-drag relative error. Each aggregate threshold is 10%. Report all per-case waveform errors and the worst error, even when the aggregate passes. A test geometry must not be moved into training after its error is observed.

## Recomputed checkpoint test cases

Use the already retained portable checkpoint, with no training, weight changes or model selection. Recompute the same eight test geometries at mesh level 2 and Mach 1.8 using the original teaching-body generator. Save each prediction before launching its solver and retain the complete mesh, configuration, restart field, extracted signal and history. Require positive finite fields, density residual at or below -9 in log10 units, residual reduction of at least five decades and final-100-iteration relative drag range below 1e-4. Freeze the aggregate waveform, peak and drag thresholds at 10%; report per-case and worst waveform errors separately.

These are repeated reference geometries whose earlier labels were already available. Fresh raw evidence closes the numerical reproducibility gap; it does not make this a blind test of new geometry generalization. Preserve the historical audit separately rather than replacing its values with the rerun.

## NASA CFD

Use NASA SEEB-ALR as-built geometry, M=1.6, zero incidence, and H=21.2 inches. Compare dp/p_infinity within x=25–46 inches using only original NASA macro shifts. Interpolate CFD to measurement points. The finest accepted solution must meet both experiments independently: waveform relative L2 below 20%, positive-peak relative error below 10%. The last two mesh waveforms must differ by less than 5% in relative L2 over the same window. All included accepted mesh solutions must converge; retain failed pilots separately with explicit status. Mesh change is not a formal GCI. The nose cap must refine with the replacement mesh family; the rejected family used only two cap cells.

The original and mesh-scaled-freeze mesh families are rejected as validation evidence because the nose region is underresolved. See [the nose physics audit](NOSE_PHYSICS_AUDIT.md). A new cap-resolved family uses refinement levels 1, 1.5 and 2, refines the cap with the rest of the mesh, and must also pass the independently prescribed stagnation-pressure, total-enthalpy and density checks. The physical model and experimental error thresholds remain unchanged. Failed pilots remain separate from accepted results.

A solver run completes only when return code is zero and physical fields are finite with positive density and pressure. The density residual must be at most 10^-9 with a drop of at least 5 orders. A small residual does not override a failed experimental comparison.

If a threshold fails, preserve the result and report the failure. Numerical troubleshooting can change the mesh or solver settings, but record the change and rerun the entire comparison. Do not claim that the reconstructed implementation is bitwise identical to an earlier unavailable implementation.

## Release checks

Execute the notebook from distributed files, recompute both audit reports, verify source hashes and inspect rendered lecture pages. Publish geometry, configurations, compact computed signatures, convergence histories, metric definitions and reference citations. Keep full field/mesh files in the associated raw archive. The release must not claim ground-noise reduction, full-aircraft reproduction, original-author weight reproduction or experimental validation of the neural model.
