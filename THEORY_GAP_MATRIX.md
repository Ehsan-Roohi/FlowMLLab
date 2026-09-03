# Probabilistic theory gap matrix

This matrix compares broad probabilistic-machine-learning topics with the
current FlowMLLab evidence. Topic names are curricular metadata, not source
material. Proposed treatments must follow `THEORY_SOURCE_POLICY.md`.

| Topic | Current FlowMLLab coverage | Important gap | Fluid-mechanics extension | Priority |
| --- | --- | --- | --- | --- |
| Probability and observation models | Maxwellian sampling, particle noise, repeated seeds | No explicit chain from measurement model to likelihood | Noisy velocity sensors and CFD surrogate residuals | High |
| Bayesian inference | Uncertainty taxonomy and ensemble spread | Prior, likelihood, posterior, and posterior predictive are not computed | Infer a flow coefficient from noisy measurements; propagate posterior uncertainty | High |
| Information theory and proper scores | Entropy proxy and deterministic losses | Loss assumptions, Gaussian NLL, CRPS, and calibration are not connected | Score probabilistic field predictions rather than only point errors | High |
| Gaussian processes | Not currently a guided method | No probabilistic non-neural surrogate baseline | GP prediction of POD coefficients or scalar flow diagnostics versus Reynolds number | High |
| Deep predictive uncertainty | Three-seed ensembles are retained | Spread is correctly limited but not calibrated as a predictive distribution | Compare ensemble spread with GP intervals and observed coverage | High |
| Gaussian filtering and state-space models | Temporal baselines and cylinder rollouts | No data-assimilation formulation | Filter noisy lift/pressure signals and preserve phase and Strouhal evidence | Next |
| Probabilistic graphical models and message passing | No dedicated treatment | Weak connection to the current fixed-grid workflow | Sensor networks or local state estimation | Later |
| Variational inference and MCMC | No dedicated treatment | Computational cost is high relative to the current teaching need | Posterior inference for selected closure parameters | Later |
| Latent-variable and generative models | POD supplies an interpretable linear latent basis | No probabilistic latent model or generative claim | Compare POD with a probabilistic latent representation only when a validated dataset warrants it | Later |
| Decision-making and reinforcement learning | No closed-loop control task | A trustworthy environment and safety envelope are absent | Flow-control policy evaluation after state estimation is validated | Future |

## First bounded increment

The first increment is **Week 2.1 — Probabilistic UQ for CFD surrogates**. It
adds an exact Bayesian linear-regression example, a fixed-protocol
POD--Gaussian-process surrogate on the existing cavity archive, and proper
probabilistic metrics. It does not replace the Week-4 POD--DeepONet result or
reinterpret a seed ensemble as a calibrated posterior.

Acceptance requires:

1. complete-case train/validation/blind separation;
2. a deterministic interpolation baseline using the same training cases;
3. posterior mean, standard deviation, interval coverage, interval width,
   Gaussian negative log likelihood, and Gaussian CRPS;
4. an explicit warning that pointwise spatial coverage is descriptive because
   grid errors are correlated; and
5. machine-readable protocol and metrics produced without opening blind cases
   during model selection.
