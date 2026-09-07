# Probabilistic theory gap matrix

This matrix compares broad probabilistic-machine-learning topics with the
current FlowMLLab evidence. Topic names are curricular metadata, not source
material. Proposed treatments must follow `THEORY_SOURCE_POLICY.md`.

| Topic | Current FlowMLLab coverage | Important gap | Fluid-mechanics extension | Priority |
| --- | --- | --- | --- | --- |
| Probability and observation models | Week 2.1 connects noisy observations to a Gaussian likelihood | Correlated sampling errors need a separate treatment | Estimate effective sample size of time-correlated DSMC observations | Next |
| Bayesian inference | Week 2.1 computes an exact linear-Gaussian posterior and predictive distribution | Nonlinear inverse inference is not established by that example | Add only when a qualified inverse problem warrants it | Later |
| Information theory and proper scores | Week 2.1 reports Gaussian NLL, CRPS, interval width and coverage | Descriptive pointwise coverage is not simultaneous field coverage | Audit correlated errors and case-level calibration | Next |
| Gaussian processes | Week 2.1 contains a POD--GP cavity baseline against interpolation | Limited-case calibration and extrapolation remain limitations | Extend the existing lab, not a duplicate introductory GP week | Maintain |
| Deep predictive uncertainty | Three-seed ensembles are retained | Spread is correctly limited but not calibrated as a predictive distribution | Compare ensemble spread with GP intervals and observed coverage | High |
| Gaussian filtering and state-space models | Temporal baselines and cylinder rollouts | No data-assimilation formulation | Filter noisy lift/pressure signals and preserve phase and Strouhal evidence | Next |
| Probabilistic graphical models and message passing | No dedicated treatment | Weak connection to the current fixed-grid workflow | Sensor networks or local state estimation | Later |
| Variational inference and MCMC | No dedicated treatment | Computational cost is high relative to the current teaching need | Posterior inference for selected closure parameters | Later |
| Latent-variable and generative models | POD supplies an interpretable linear latent basis | No probabilistic latent model or generative claim | Compare POD with a probabilistic latent representation only when a validated dataset warrants it | Later |
| Decision-making and reinforcement learning | No closed-loop control task | A trustworthy environment and safety envelope are absent | Flow-control policy evaluation after state estimation is validated | Future |

## Existing bounded increment

The first increment is **Week 2.1 — Probabilistic UQ for CFD surrogates**. It
adds an exact Bayesian linear-regression example, a fixed-protocol
POD--Gaussian-process surrogate on the existing cavity archive, and proper
probabilistic metrics. It does not replace the Week-4 POD--DeepONet result or
reinterpret a seed ensemble as a calibrated posterior.

The protocol requires:

1. complete-case train/validation/blind separation;
2. a deterministic interpolation baseline using the same training cases;
3. posterior mean, standard deviation, interval coverage, interval width,
   Gaussian negative log likelihood, and Gaussian CRPS;
4. an explicit warning that pointwise spatial coverage is descriptive because
   grid errors are correlated; and
5. machine-readable protocol and metrics produced without opening blind cases
   during model selection.

See the [existing notebook](notebooks/week02_1/Probabilistic_UQ_CFD.ipynb)
for implementation; these topics are not missing modules.

## Proposed increments, not released lessons

The following are curriculum proposals, not executed research results. Reuse
FlowMLLab's qualified data; author code independently from public references.
Restricted handouts, worked solutions and assessment files are not import sources.

| Placement | Question and bounded experiment | Acceptance before promotion |
| --- | --- | --- |
| 3.1, revisited in 12 | How many independent observations does a DSMC average contain? Compare naive standard errors, autocorrelation-aware effective sample size and block averaging. | Use ordered raw samples, not spatial pixels or unordered seed averages as a time series. Verify on a known correlated process first; defer real-data claims if temporal samples are unavailable. |
| 5.1 | Does a small PINN residual imply an accurate flow solution? Use an independently implemented analytic Kovasznay reference, boundary and divergence errors, then a fixed-budget baseline comparison. | Separate collocation from evaluation points; report field and boundary errors, seed spread and runtime. Time-dependent/causal PINNs are a later extension, not implied by a steady test. |
| 7.2 | Can sparse noisy sensors correct a cylinder-wake forecast? Compare a POD-space Kalman filter with persistence, open-loop dynamics and sensor-only reconstruction. | Fit dynamics and noise models on development sequences only. Distinguish causal filtering from smoothing with future observations; report phase, field error and interval coverage on complete held-out trajectories. |
| Existing 2.1 | Deepen posterior-predictive calibration and GP limitations. | Preserve the same baselines and split; no duplicate introductory Bayesian or GP module. |
| Existing 4.1, later | When is nonlinear latent compression justified against POD? | Equal data and latent dimension; separate reconstruction from rollout; include boundary/divergence checks and total training cost. No automatic VAE superiority claim. |

Prioritize 7.2 and sampling diagnostics over adding generic classifiers or GANs.
The filtering proposal can start on CPU with existing reduced coordinates; it
does not require a new CFD campaign merely to test the implementation.

Public independent-authoring references:

- Särkkä and Svensson, [Bayesian Filtering and Smoothing, second edition](https://doi.org/10.1017/9781108917407), 2023.
- Wang, Sankaran and Perdikaris, [Respecting causality for training physics-informed neural networks](https://doi.org/10.1016/j.cma.2024.116813), 2024, for a subsequent time-dependent extension only.
