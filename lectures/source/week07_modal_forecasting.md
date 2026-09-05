# Week 7 companion: modal wake forecasting

## Question and scope
Can a linear reduced dynamical model forecast a real cylinder wake, and what does a small nonlinear network add? This is a continuation of one previously inspected Re110 trajectory, not generalization to a new Reynolds number. Allow 60-75 minutes. Prerequisites: POD, least squares, complex eigenvalues, and Week 7 LBM.

Use the previously generated author FlowMLLab cylinder-cfd-v1 data. Compute vorticity on the native grid, then take a fluid-only wake ROI. The cylinder wall is not included. D/U-normalized vorticity, coordinates in D and time in D/U are used consistently. The 12-node-per-diameter reference is not grid-independent CFD.

## Projected DMD derivation
Fit a centered POD basis on training snapshots only. Let a row a_k contain the modal coefficients at time k. Assemble X from a_0 through a_(m-2) and Y from a_1 through a_(m-1). Solve min_A norm(X A - Y)_F; the least-squares solution is A = pseudoinverse(X) Y.

The convention here is a_(k+1) = a_k A, not A a_k. Reconstruct q_k = mean + a_k Phi-transpose. Keeping the training mean makes this an affine field forecast. POD orders energy content; it does not guarantee optimal predictive coordinates.

## Eigenvalues and physical time
For a discrete eigenvalue lambda, growth rate = log(abs(lambda))/dt and frequency = arg(lambda)/(2 pi dt). The frequency branch is limited by sampling/Nyquist. A conjugate pair represents oscillation. Compare relevant wake frequencies with the source force-history Strouhal diagnostic, but never use that full-history value to tune the model.

---
# Week 7: frozen forecast protocol

## Time windows
Use frame indices [0,160) for training, [160,210) for validation and [210,281) for test. Candidate DMD ranks are 2, 4, 6 and 8. Select the minimum validation field relative L2. Roll out from training frame 159 through BOTH held windows without resetting.

The MLP baseline uses eight training POD coordinates and four-frame history, two tanh hidden layers of 32 units, a fixed seed, and training-only scaling. Its history is longer than DMD's one-state initialization; state this informational difference. Predict recursively without teacher forcing. Do not compare these errors directly with previously archived phase-decoder scores that used another protocol.

## Controls
Persistence repeats the last training field. The mean control repeats the training mean. Full-field POD projections at ranks two and eight are oracle representation floors, not forecasts. Report each control alongside the actual models and keep all candidate validation scores.

## Retained result, not a promised outcome
The initial retained run selects rank eight: DMD test relative L2 is about 6.44%, MLP about 5.54%, and the rank-eight POD floor about 5.45%. Rank-two DMD and the rank-two SINDy bridge have about 36% field error because the representation itself loses substantial structure. Reproduction tolerances, warnings and library versions belong with these figures.

These scores describe agreement with this finite-resolution LBM label dataset, not error against exact continuum flow. Repeated inspection makes this a teaching test rather than a newly blind scientific benchmark.

---
# Week 7: evaluate more than a pretty contour

## Field and regional errors
Use area-weighted global relative L2, RMSE, maximum absolute error and worst-frame relative L2. An all-zero reference makes relative error undefined; return null rather than an arbitrary epsilon-normalized number. The ROI perimeter metric diagnoses that extracted region, not the cylinder wall or inlet/outlet boundary condition.

## Integral error
Integrate the supplied scalar with declared weights and compare its integral. The vorticity integral is not mass. A velocity-area integral is not automatically a mass flux. Conservation needs the correct variable, geometry, normals and flux definitions; this helper does not invent them.

## Spectral error and phase
At the predeclared probe near x/D=4 and y/D=0.5, remove the temporal mean, apply a Hann window, and compare one-sided power spectra. Exclude DC; report frequency-bin spacing and phase at the reference peak. A zero-power reference has no defined peak or phase.

The 71-frame test interval lasts about 7.40 D/U, so its FFT bin spacing is about 0.135 U/D. It cannot substantiate a high-precision Strouhal estimate. A DMD frequency is a model-based estimate, not improved raw Fourier resolution. Short windows, harmonics and known box modes must be considered before interpreting a peak as vortex shedding.

## Display conventions
Use common signed color limits for reference and predictions, a separate nonnegative error scale, and the same final frame for all methods. Filled contours interpolate level crossings for display only; metrics use the original sampled arrays. No sharpening, smoothing or fabricated super-resolution is applied.

---
# Week 7: assessment and references

## Student investigation
1. Verify row/column orientation on a known rotating two-dimensional oscillator.
2. Derive discrete eigenvalue growth and frequency in physical units.
3. Explain why a true-state reset at frame 210 would change the forecast question.
4. Separate the rank-eight representation floor from additional forecast error.
5. Estimate the observation duration needed for a desired Fourier-bin spacing.
6. Propose a long-horizon and a new-Re experiment; label both as new protocols.

## Connection to Week 11
The field contains a low-Mach wake with vortical structures, not a shock. DMD supplies a temporal model; it is not a vortex-core detector. A later comparison with Q-criterion or swirling strength would need its own labels and evaluation protocol. Do not infer segmentation performance from these forecasts.

## Attribution and algorithm boundaries
PyDMD inspired this transparent projected-DMD exercise. This is an original NumPy least-squares implementation, not a call to PyDMD and not its robust/noise-aware variants. The companion Week 5 lab covers QR/D-optimal sensing and integral sparse regression inspired by PySensors and PySINDy. PDEBench inspired explicit multi-metric reporting; scores are not claimed to be numerically identical to its implementation.

PyDMD: https://github.com/PyDMD/PyDMD

PySensors: https://github.com/dynamicslab/pysensors

PySINDy: https://github.com/dynamicslab/pysindy

PDEBench metrics: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/metrics.py

Data: https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1

Executable companion: notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb
