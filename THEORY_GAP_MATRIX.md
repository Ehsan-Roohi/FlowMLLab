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
| Gaussian filtering and state-space models | Week 7.2 compares POD-space causal filtering with open-loop and sensor-only baselines | Marginal intervals are severely under-covered; no smoothing or new-Re evidence | Validation-only uncertainty calibration, missing-sensor stress and an untouched trajectory | Next |
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

Week 7.2 is also implemented as a bounded [state-estimation lab](notebooks/week07_2/README.md).
It deliberately retains its nominal-95% coverage failure; a low point error is
not treated as calibrated state uncertainty.

## Proposed increments, not released lessons

The following are curriculum proposals, not executed research results. Reuse
FlowMLLab's qualified data; author code independently from public references.
Restricted handouts, worked solutions and assessment files are not import sources.

| Placement | Question and bounded experiment | Acceptance before promotion |
| --- | --- | --- |
| 3.1, revisited in 12 | How many independent observations does a DSMC average contain? Compare naive standard errors, autocorrelation-aware effective sample size and block averaging. | Use ordered raw samples, not spatial pixels or unordered seed averages as a time series. Verify on a known correlated process first; defer real-data claims if temporal samples are unavailable. |
| Existing 4.2 and 13 | Extend the current PINN reliability audit into a matched representation study. | Compare primitive/FOSLS and streamfunction formulations under identical physics and budgets; add grid-converged smooth-lid references and multiple preregistered seeds before a paper claim. |
| Existing 2.1 | Deepen posterior-predictive calibration and GP limitations. | Preserve the same baselines and split; no duplicate introductory Bayesian or GP module. |
| Existing 4.1, later | When is nonlinear latent compression justified against POD? | Equal data and latent dimension; separate reconstruction from rollout; include boundary/divergence checks and total training cost. No automatic VAE superiority claim. |

Prioritize sampling diagnostics and a genuinely untouched Week 7.2 trajectory
over adding generic classifiers or GANs.

Public independent-authoring references:

- Särkkä and Svensson, [Bayesian Filtering and Smoothing, second edition](https://doi.org/10.1017/9781108917407), 2023.
- Wang, Sankaran and Perdikaris, [Respecting causality for training physics-informed neural networks](https://doi.org/10.1016/j.cma.2024.116813), 2024, for a subsequent time-dependent extension only.
