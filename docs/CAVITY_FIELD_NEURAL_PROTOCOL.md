# Full-field cavity neural experiment: frozen design

This follows the new three-grid CFD qualification. It does not reuse the old
Ghia-centreline neural fit as if that were a field model.

Training Re: **100, 250, 500, 750, 1000, 2500, 4000, 5000, 6000, 9000, 10000**.
Validation Re: **400, 6500**. Test Re: **3200, 7500**.
All locations and fields of each held-out Reynolds number stay out of network
fitting, POD construction, scaling and model selection. CFD numerical audits
of every case are separate from model selection; these are disclosed benchmark
cases, not a claim of prospectively unknown physics.

Use the matched 513-by-513 native CFD velocity grids, after independent equation,
wall, continuity and SHA256 checks, with the three-grid uncertainty reported.
An eleven-case, mean-subtracted, training-only POD supplies at most ten modes.
The branch maps log10(Re)-3 to modal coefficients. This is a scalar-parameter
POD-DeepONet, not a data-free PINN or an arbitrary boundary-function operator.
No pressure prediction is claimed. No Ghia samples enter fitting.

Compare fixed candidates on complete validation fields only:

1. two tanh layers, width 32, Adam;
2. two tanh layers, width 32, Adam then L-BFGS;
3. five tanh layers, width 32, Adam then L-BFGS;
4. the five-layer model with training-only per-mode RMS scaling;
5. five tanh layers of width 17 with per-mode scaling, close in parameter count
   to the two-layer width-32 model (report exact counts).

Each candidate has seeds 7, 23 and 61 and at most 4000 joint function/gradient
evaluations; the hybrid uses 1200 Adam evaluations first. Adam learning rate is
0.003. Keep the best evaluated training-objective state. L-BFGS line-search
trials count in the budget. No minibatches or stochastic target resampling.

Select the configuration by mean full-field relative velocity L2 error across
both validation Re and the three seeds. Report seed spread; the delivered
prediction is the arithmetic mean of the selected three fields. Training-only
POD projection error is a separate lower bound, not neural predictive skill.

Evaluate both linear interpolation in Re and in log(Re), using precisely the
same training fields. Choose the baseline convention on validation alone and
still publish both test scores. Save model/source/trunk hashes, selection and
frozen test coefficients before opening the test target arrays for scoring.

Aim for less than 1% full-field relative velocity discrepancy on validation and
test, but report measured failures and remaining CFD uncertainty honestly.
Compare physical profiles, full fields, signed discrepancies and shared-scale
error maps. Never tune a new model on these disclosed test scores. Boundary and
continuity preservation of a linear CFD basis is not enforcement of the nonlinear
Navier--Stokes equations: report the latter's residual separately.

## Validation-driven amendment before test disclosure

The original nine-training-case experiment is retained in
results/cavity_field_deeponet/selection_round_1, including its source, POD trunk,
models, optimizer moments, validation scores and unscored frozen predictions.
Its best configuration averaged 2.0686% validation error. At Re=400 the
training-only POD projection itself had a 3.2510% lower bound; at Re=6500 it was
0.00382%. Adding depth or optimizing coefficients cannot remove that missing
spatial content.

Two new independent PDE solves at Re=500 and 750 fill the low-Re training gap.
All three grids are checked before rebuilding the POD. The validation and
test Reynolds numbers, candidate architectures, seeds, budgets and selection
criterion remain unchanged. No test field is read for neural scoring before
the amended predictions are frozen. This is an adaptive validation-led study,
not a claim that its final training set was fixed before validation was seen.

The three-grid fields pass the discrete-equation gate but retain material
spatial sensitivity: at Re=10000 the 256-to-512 vector discrepancy is 6.32%
on common native samples. Neural accuracy is measured against a specified
513-by-513 discretization, not a mesh-independent exact physical solution.
