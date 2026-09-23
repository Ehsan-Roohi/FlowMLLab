# Week 16: what has actually been validated?

Read this guide alongside the notebook. The learning sequence is **geometry → governing equations → numerical checks → data split → surrogate → optimization → fresh CFD**. A model can pass one check and fail the next.

## 1. Three different questions

| Exercise | Geometry and conditions | Output and reference | What the comparison establishes | What it cannot establish |
|---|---|---|---|---|
| Taylor–Maccoll cone | Axisymmetric 7° cone, Mach 1.8, zero incidence | Cone surface pressure versus an independently integrated Taylor–Maccoll solution | A check of the axisymmetric Euler pressure calculation | Accuracy of the full off-body waveform, atmospheric propagation, or a neural network |
| NASA SEEB-ALR | NASA's reference body, with benchmark-specific conditions | Off-body pressure signature versus the original NASA experimental records and a separately identified computational reference | An independent physical comparison for near-field CFD when the geometry, coordinates and conditions match | Experimental validation of a neural network trained on a different two-parameter family |
| Week 16 surrogate | Fixed-volume two-parameter bodies at Mach 1.8 | POD waveform coefficients and pressure drag versus held-out SU2 calculations | Interpolation within the stated family and the tested extrapolation range | Generalization to NASA geometry, a lifting aircraft, a new Mach number, or ground loudness |

NASA experiment and NASA-hosted CFD are different kinds of reference: label both explicitly. An agreement between two numerical solutions is not a wind-tunnel validation. The NASA exercise must retain the actual geometry, original data and their provenance before numerical claims can be checked. This guide supplies no additional measured NASA error or new validation pass.

The analytical cone model is **Taylor–Maccoll**. It is separate from the learned **surrogate model**. Neither is a ground-noise model.

## 2. Follow one sample through the learned model

The input is the pair `(a, b)` that controls the radius distribution. Volume normalization prevents improvement simply by shrinking the body. The CFD output is a pressure waveform sampled at fixed axial locations and an integrated pressure-drag coefficient.

1. Fit the input scaler on the **24 training geometries**.
2. Fit the pressure mean and POD/PCA modes on their training waveforms only.
3. Express each waveform as the training mean plus weighted modes. Predict the mode coefficients and the logarithm of pressure drag.
4. Fit the output scaler using training outputs only. Compare ridge regression, the fixed two-layer MLP and the Gaussian process.
5. Select using the **six validation geometries**, with the stated sum of peak and drag errors.
6. Inspect the **eight test geometries** and **six extrapolation geometries** without changing the model in response. If they inform a revision, they become development data; obtain a new final test set.

POD reduces output dimension; it does not solve the flow equations. A high retained variance also does not guarantee accurate shock peaks. Taking the exponential of predicted log drag enforces positivity, not conservation or constraint satisfaction.

The published implementation computes all comparison rows in one routine, but its selection expression uses validation rows only. For a student experiment, decide the architecture and selection rule before examining the test rows. Never select hyperparameters from the displayed test error.

## 3. Know exactly what each error means

Let `y_ij` be reference pressure for geometry `i` at sample `j`, and let `yhat_ij` be its prediction, on the same sampling grid and in the same pressure convention. The implemented aggregate relative waveform error is

$$E_{wave}=\frac{\sqrt{\sum_i\sum_j(\hat y_{ij}-y_{ij})^2}}{\sqrt{\sum_i\sum_j y_{ij}^2}}.$$

This is not the mean of the per-geometry relative errors. Stronger signals contribute more to the denominator. Report each geometry's error and the worst case as well when assessing reliability.

For positive reference peaks `p_i = max_j y_ij` and positive pressure drag `d_i`, the mean relative errors are

$$E_{peak}=\frac1N\sum_i\frac{|\max_j\hat y_{ij}-p_i|}{p_i},\qquad E_{drag}=\frac1N\sum_i\left|\frac{\hat d_i}{d_i}-1\right|.$$

Multiply these dimensionless ratios by 100 to report percentages. Near-zero denominators require another prespecified normalization; do not silently insert an arbitrary floor. A shock displacement can give a large waveform error even if the peak is accurate. Plot the waveform and signed error together to see which discrepancy dominates.

For NASA comparisons, preserve the source pressure convention. In particular,

$$C_p=\frac{p-p_\infty}{\tfrac12\gamma p_\infty M_\infty^2},\qquad\frac{\Delta p}{p_\infty}=\tfrac12\gamma M_\infty^2 C_p.$$

Do not compare these quantities directly without conversion. State the extraction radius, axial origin, window, interpolation procedure and any source-prescribed coordinate shift. Fitting a shift or amplitude to minimize the reported error changes the validation question and must be disclosed.

## 4. Reproducibility and the existing PCA implementation

The existing model uses `PCA(..., svd_solver='auto')` implicitly. Depending on array dimensions and the installed scikit-learn version, the automatic choice can use randomized SVD. No PCA random seed is supplied. Consequently, the MLP's `random_state=16` does **not** guarantee an identical complete refit. Library versions and numerical backends can also matter.

The current release preserves the original architecture and fitting code so that an implementation change is not confused with validation of the original model. A newly trained model is a **refit**, not evidence that historical weights have been reproduced. For a future reproducibility experiment, explicitly select full deterministic SVD or seed the randomized solver, record the environment and retain the fitted scalers, POD basis and model. Treat changed fitting behavior as a new model version and recompute its results.

A rigorous independent CFD audit freezes predictions **before** the new reference solutions are inspected. Keep those exact predictions, geometry identifiers and hashes. Repeatedly retraining until a favorable comparison appears is not an independent audit. A same-mesh test probes surrogate error relative to that discretization; a finer-mesh test also reveals sensitivity to the CFD labels.

The recovered frozen-prediction audit can be recomputed with `python qa/week16/independent_audit_report.py`. It compares the same eight geometries and the same archived predictions against both original and finer-mesh CFD arrays. Its report, `results/week16_lowboom/reference/neural_audit.json`, verifies the original blob hashes and separates recoverable numerical evidence from unavailable solver-log evidence. Notebook refits do not replace those predictions.

## 5. Optimization needs its own final check

A low average test error is not a guarantee at the optimizer's chosen geometry. The optimizer may deliberately reach regions where the surrogate is optimistic. For every proposed candidate:

- Build its geometry and mesh again, then run CFD to the declared convergence criteria.
- Compare predicted and achieved peak pressure and drag with the baseline using consistent reference areas and observation lines.
- Check the actual `CD_pressure <= 1.02 * CD_baseline` constraint.
- Refine the mesh and inspect whether the claimed improvement survives.

The implementation searches with an additional conservative factor on the surrogate drag limit; explain that margin separately from the final physical constraint. A proposal that violates the final CFD constraint is a failed proposal, even when the surrogate score is excellent.

Residual reduction tests convergence of the discrete equations. Mesh refinement tests sensitivity to spatial discretization. Agreement with experiment tests the modeled physical problem. None substitutes for the others. A lower near-field pressure peak is not a measured reduction in ground PLdB: atmospheric propagation and an appropriate loudness calculation are still required.

## 6. Suggested classroom sequence

First draw the three geometries and predict where their compression waves begin. Then inspect the axisymmetric equations, the mesh and the cone comparison. Next inspect a reference pressure signature and reproduce its normalization. Only then train the surrogate and compare the three regression methods. Finish by checking a proposed optimum against its independent CFD result.

Discuss these questions in the submitted report:

1. Can two models have similar waveform error but different design feasibility? Show a case or explain which errors would cause it.
2. Why must the POD basis be fitted before seeing the test waveforms?
3. Why does a NASA CFD validation not experimentally validate this neural model?
4. If the refined-mesh error rises, what evidence separates surrogate error from discretization error?
5. What additional calculations would justify a ground-noise claim?

For the motivating full-aircraft study, cite Zheng et al., *Aerospace Science and Technology* 178 (2026), 113218, [doi:10.1016/j.ast.2026.113218](https://doi.org/10.1016/j.ast.2026.113218). This teaching geometry and its learned model do not reproduce that aircraft. Consult the [NASA Sonic Boom Prediction Workshop](https://lbpw.larc.nasa.gov/) for the benchmark's original geometry, conditions and data, and the repository's reference documentation for the exact files used.
