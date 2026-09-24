# Week 16: what has actually been validated?

**Current primary teaching model:** `reference/clean_model_v801.npz`, trained once on 24 of the 44 newly regenerated SU2 8.0.1 cases. Every case passed numerical and full-field physical checks. Run `python qa/week16/clean_model_v801.py --check-only` to reload the weights and reproduce all metrics without fitting. Finer CFD waveform/peak/drag errors are 7.2036%/2.2373%/2.2740%; the worst waveform error is 15.3716%. Extrapolation waveform error is 34.2862% and drag error 22.3889%, so extrapolation reliability is not established. The fit used full-SVD POD, training-only statistics, 800 optimizer iterations and no warnings.

The original data and older model identities described below remain historical comparisons. The eight old 8.5.0 refined fields later failed a full-field enthalpy check; their numerical prediction-error tables do not constitute physical acceptance. New data and weights have distinct filenames and hashes.


Read this guide alongside the notebook. The learning sequence is **geometry → governing equations → numerical checks → data split → surrogate → optimization → fresh CFD**. A model can pass one check and fail the next.

## 1. Three different questions

| Exercise | Geometry and conditions | Output and reference | What the comparison establishes | What it cannot establish |
|---|---|---|---|---|
| Taylor–Maccoll cone | Axisymmetric 7° cone, Mach 1.8, zero incidence | Cone surface pressure versus an independently integrated Taylor–Maccoll solution | A check of the axisymmetric Euler pressure calculation | Accuracy of the full off-body waveform, atmospheric propagation, or a neural network |
| NASA SEEB-ALR | NASA's reference body, with benchmark-specific conditions | Off-body pressure signature versus the original NASA experimental records and a separately identified computational reference | An independent physical comparison for near-field CFD when the geometry, coordinates and conditions match | Experimental validation of a neural network trained on a different two-parameter family |
| Week 16 surrogate | Fixed-volume two-parameter bodies at Mach 1.8 | POD waveform coefficients and pressure drag versus held-out SU2 calculations | Measured interpolation and extrapolation errors within the stated study; extrapolation performs poorly | Generalization to NASA geometry, a lifting aircraft, a new Mach number, or ground loudness |

NASA experiment and NASA-hosted CFD are different kinds of reference: label both explicitly. An agreement between two numerical solutions is not a wind-tunnel validation. The NASA exercise must retain the actual geometry, original data and their provenance before numerical claims can be checked. This guide supplies no additional measured NASA error or new validation pass.

The analytical cone model is **Taylor–Maccoll**. It is separate from the learned **surrogate model**. Neither is a ground-noise model.

## 1a. Recompute the cone reference rather than copying a pressure value

For steady inviscid conical flow, the velocity depends on polar angle $\theta$ alone. Normalize velocity by the maximum speed $\sqrt{2h_0}$, where $h_0$ is stagnation enthalpy. The radial and polar components satisfy

$$\frac{dv_r}{d\theta}=v_\theta,\qquad
\frac{dv_\theta}{d\theta}=\frac{v_\theta^2v_r-H(2v_r+v_\theta\cot\theta)}{H-v_\theta^2},
\quad H=\frac{\gamma-1}{2}(1-v_r^2-v_\theta^2).$$

For a guessed shock angle $\beta$, apply the normal-shock density ratio to the polar velocity at the shock. The radial component is unchanged. Integrate toward the cone half-angle $\theta_c$ and adjust $\beta$ until $v_\theta(\theta_c)=0$, the impermeable-wall condition. After the shock, pressure follows the isentropic relation along a streamline. `benchmark.cone_exact()` implements this shooting calculation with adaptive ODE integration and a bracketed scalar root search.

At $M_\infty=1.8$, $\theta_c=7^\circ$ and $\gamma=1.4$, the computed shock angle is 34.035742 degrees, wall pressure ratio is 1.140626 and wall $C_p$ is 0.0620043. The historical SU2 8.5.0 wall value was 0.0613869 (0.996% error). The new official SU2 8.0.1 three-mesh study gives 4.5632%, 2.1563% and 1.1616% pressure error. The finest value is 0.0612840 and the last-two change is 1.0064%; the unchanged refined-family criterion passes while the coarse failure remains recorded. Students should rerun the ODE, inspect its boundary condition, and compare wall pressure away from the apex and outlet. Do not apply this infinite-cone similarity solution to the finite NASA body or to ground propagation.

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

## 4a. Run the retained model without fitting it again

The release also includes a **different, retained full-SVD refit** in `results/week16_lowboom/reference/model_checkpoint.npz`. This portable checkpoint stores the input and output scalers, training pressure mean, 12 POD modes, three weight matrices, three bias vectors, axial sampling coordinates and the 24 training geometry identifiers. It uses ordinary numeric arrays; loading it does not deserialize executable Python objects.

From the repository root, run:

```bash
python qa/week16/freeze_model.py
```

This command **does not retrain or rewrite the checkpoint**. It regenerates `checkpoint_audit.json` using NumPy inference. It checks array shapes and finite values, independently recomputes the training scalers and POD subspace, confirms that the finer CFD samples have the same geometry identifiers and sampling coordinates, and compares batched with one-at-a-time predictions. All 24 training IDs are disjoint from the eight test IDs.

To predict one geometry interactively:

```python
import sys
sys.path.insert(0, "qa/week16")
from freeze_model import FrozenSurrogate
model = FrozenSurrogate()
waveform, drag = model.predict([[0.0, 0.0]])
x_over_L = model.arrays["x"]
# waveform[0] is Cp on the teaching observation line r/L = 0.5.
# drag[0] is the predicted pressure-drag coefficient, not total viscous drag.
```

Each inference follows these equations:

$$h_0=(q-\mu_q)/s_q,\quad h_1=\tanh(h_0W_0+b_0),\quad h_2=\tanh(h_1W_1+b_1),$$
$$z=(h_2W_2+b_2)s_z+\mu_z,\quad\widehat C_p=z_{1:12}\Phi+\overline C_p,\quad\widehat C_D=\exp(z_{13}).$$

Here division and scaling are componentwise, `q=(a,b)`, and each row of `Phi` is a training POD mode. The final MLP layer is linear; applying `tanh` there would produce the wrong model.

| Eight-case finer-CFD comparison | Historical frozen-prediction audit | Retained portable checkpoint |
|---|---:|---:|
| Aggregate waveform relative L2 | 7.1541% | 6.3903% |
| Mean peak relative error | 3.4469% | 1.9271% |
| Mean pressure-drag relative error | 1.6936% | 1.5708% |
| Worst individual waveform relative L2 | 15.7984% | 14.6268% |

These columns describe **two model identities**, not two implementations of identical weights. The historical column uses unchanged frozen predictions whose original weights were not retained. The portable column is a later refit, evaluated retrospectively on the already available finer CFD data. Do not describe the difference as a demonstrated improvement on a fresh blind test. Do not replace historical predictions with portable-model outputs while retaining the historical audit label.

The same portable checkpoint has the following errors against the original CFD mesh:

| Split | Geometries | Waveform relative L2 | Mean peak error | Mean drag error |
|---|---:|---:|---:|---:|
| Training | 24 | 0.2740% | 0.1188% | 0.1031% |
| Validation | 6 | 2.8963% | 1.0569% | 1.6227% |
| Test | 8 | 4.1000% | 1.7557% | 2.5784% |
| Extrapolation | 6 | 34.1628% | 6.0284% | 22.1449% |

The extrapolation errors are substantial. The aggregate interpolation pass does not authorize use beyond the training geometry range. This provides a useful classroom counterexample: positive drag and a reasonable pressure peak do not guarantee an accurate waveform or a reliable design constraint.

The portable checkpoint's Git blob SHA is `11337027584a29e44c78ef52e8dfd9c3cc034109`; its SHA-256 is recorded in `checkpoint_audit.json`. The numerical comparison passes the inherited teaching thresholds of 10% for each **aggregate** metric. The worst individual waveform error exceeds 10%, which is why the report also gives all eight individual errors. Checking training statistics establishes consistency with the training subset; it cannot independently prove every historical model-selection decision. A new prospective generalization claim requires a new locked model and new reference cases.

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
