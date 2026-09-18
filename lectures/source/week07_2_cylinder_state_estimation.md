# Week 7.2 - State estimation from sparse wake sensors

Scientific question: can noisy point measurements correct a reduced cylinder-wake forecast without seeing future observations? We separate four information contracts: autonomous open-loop dynamics, causal filtering, instantaneous sensor-only reconstruction, and offline smoothing. Only the first three are evaluated here; smoothing is defined but not used.

Data are prior FlowMLLab Re110 LBM transverse-velocity fields on a 32 x 78 wake ROI. They are finite-resolution educational CFD, not exact continuum truth. The cylinder wall is outside the ROI. Synthetic sensor noise does not create independent CFD realizations.

Prerequisites: POD, least squares, Gaussian conditioning and Week 7 modal forecasting. CPU runtime is about 15 seconds.

Definitions and notation. POD (proper orthogonal decomposition): the singular value decomposition of the matrix of training snapshots; its left singular vectors Phi are the modes and the projection a_k = Phi^T (q_k - mean) gives the modal coefficients (the reduced state) of frame k. ROI: the region of interest, the 32 x 78 wake window on which all fields are stored. DMD (dynamic mode decomposition): a linear model of the coefficient dynamics, a_(k+1) = a_k A, fitted by least squares on training frames; "open-loop DMD" runs that model forward from the last training frame without ever looking at a measurement. H: the rows of Phi at the sensor locations, so that H a_k is the modal prediction of the sensor readings; q_mean,s: the entries of the mean field at those same sensor locations, which must be subtracted from a raw reading before it is compared with H a_k. Kalman filter: the recursion that combines the model forecast with each new measurement, weighting them by their covariances; the Kalman gain K is that weight. Relative L2 error: ||estimate - truth|| / ||truth|| over the ROI at each test frame, averaged over frames.

Worked scalar example of one predict-update step. Suppose one mode, one sensor, F = 1, H = 1. After the previous update the estimate is x+ = 2.0 with variance P+ = 0.5, the process noise is Q = 0.5 and the sensor noise R = 1.0. Predict: x- = 2.0, P- = 0.5 + 0.5 = 1.0. A measurement y = 3.0 arrives: innovation r = 3.0 - 2.0 = 1.0, its variance S = P- + R = 2.0, gain K = P- / S = 0.5, update x+ = 2.0 + 0.5 * 1.0 = 2.5 and P+ = (1 - K) P- = 0.5. The estimate moves halfway towards the measurement because model and sensor are equally uncertain; a noisier sensor (larger R) would move it less. The matrix recursion on page 1 is this arithmetic with vectors and covariance matrices.

Why persistence scores above 100%: the transverse velocity in the wake window oscillates about zero, so a frame frozen at the last training time is out of phase with the truth for most of the test window and its error can exceed the norm of the truth itself (131.8% here). An error above 100% is therefore possible for any estimate that is anti-correlated with the truth; it is not a bookkeeping mistake.

---
# POD-space Gaussian model

Represent the field as q_k = mean + Phi a_k. Fit mean and Phi on training frames only. The row-vector transition is a_(k+1) = a_k A + w_k, with w_k distributed as N(0,Q). Point sensors observe y_k = mean_s + H a_k + v_k, where H is the selected rows of Phi and v_k is N(0,R).

A comes from training-only least squares. Q begins with the covariance of one-step training residuals and is multiplied by a validation-selected scale. R is fixed from the declared synthetic sensor-noise level. D-optimal oversampling chooses sensor positions from training POD modes only.

The model is deliberately imperfect: truncation, nonlinear dynamics and biased covariance estimates remain. Covariance inflation is a model-selection parameter, not a physical noise measurement.

---
# Causal predict-update recursion

Predict: a minus = A-transpose a plus under the stored row convention; P minus = A-transpose P plus A + Q. Innovation: r = y - mean_s - H a minus. Gain: K = P minus H-transpose (H P minus H-transpose + R)^(-1). Update: a plus = a minus + K r.

The implementation solves the linear system rather than explicitly forming an inverse. The Joseph covariance update preserves positive semidefiniteness numerically. Output at time k uses measurements only through k; a prefix-invariance unit test verifies causality.

An RTS smoother would revisit time k using observations after k. It can be valuable for retrospective denoising, but it answers a different question and would be an unfair baseline for online estimation.

---
# Frozen selection and evaluation

Training is frames 0:160; validation 160:210; test 210:281. Candidate ranks are 4, 6 and 8; sensor counts are 8, 16 and 32; process-covariance scales are 1, 10, 100 and 1000. Five validation noise seeds select minimum mean validation relative L2. Ties favor fewer sensors, lower rank and lower inflation.

Final evaluation uses five different seeds. Noise standard deviation is 10% of training-field RMS. Test truth never selects rank, sensors, inflation or noise level. The selected model is rank 8 with 32 sensors and scale 1000.

Baselines: open-loop DMD receives no measurements after frame 159; persistence repeats frame 159; sensor-only reconstruction solves a POD least-squares problem independently at every measured time. Sensor-only receives the same measurements as the filter and is therefore the strongest matched information baseline.

---
# Retained field result

The final test-frame contours use one common signed color scale and the same sampled grid. Filled contours interpolate level crossings only for display; scores use the original arrays. A separate sensor-location panel and enlarged view show all 32 transverse-velocity measurements without covering the contours.

Across five noise seeds, mean test relative L2 is 2.179% for Kalman, 3.163% for sensor-only, 4.359% for open-loop DMD and 131.813% for persistence. The first two have sample SD 0.066% and 0.064%. Open-loop and persistence do not depend on synthetic sensor noise, so their across-seed SD is zero.

The comparison supports a bounded claim: assimilating current noisy observations improves this retained trajectory under this declared observation model. It does not establish new-Re generalization, grid independence or field accuracy against experiment.

---
# Uncertainty audit: a useful failure

Project P_k through the POD basis to obtain marginal point variances. Nominal 95% Gaussian intervals cover only 55.10% of test field values on average. Mean full interval width is 0.00760 v/U. This is severe under-coverage, even though the point estimate is accurate.

Why? Training residual covariance omits structural model error; modes truncate the state; Gaussian white-noise assumptions are incomplete; pointwise field errors are correlated. Validation selected point error, not coverage. We retain the failure rather than calibrating on test data.

Marginal pointwise coverage is descriptive, not simultaneous field coverage. A next study may choose inflation or conformal calibration on validation coverage, then freeze it before a genuinely untouched trajectory.

---
# Reproducibility, exercises and references

Required evidence: exact split, all candidates, all seeds, hashes, environment, matched baselines, common-scale contours and the retained interval failure. Run All writes only to a fresh temporary folder; tracked data/results remain unchanged.

Exercises: derive the row/column covariance convention; explain why sensor-only can beat a poor dynamical model; show why a smoother is non-causal; design a missing-sensor stress test; predeclare a new-Re evaluation and coverage criterion.

Canonical reference: S. Sarkka and L. Svensson, Bayesian Filtering and Smoothing, 2nd ed., Cambridge University Press, 2023, doi:10.1017/9781108917407.

Data: FlowMLLab cylinder-cfd-v1. Code is independently authored for this repository. No restricted handout text, code, figure, exercise or solution is incorporated.
