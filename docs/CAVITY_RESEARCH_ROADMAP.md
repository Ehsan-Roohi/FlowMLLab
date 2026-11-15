# Cavity development: literature, implemented evidence and next experiments

Research checked 15 November 2026. The purpose is to deepen the existing FlowMLLab
cavity route, not to replace it with a second framework. Below, an upstream result
is an idea to test locally, not evidence that FlowMLLab has achieved that accuracy.

## What is implemented in this revision

1. One source of Ghia Tables I/II at all seven Re values, including a disclosed
   printed-entry sensitivity policy and unfiltered scores.
2. Separate Week 4 Lab 5/6 notebooks provide explicit missing-field coverage,
   mesh/time-step admission and profile learning; earlier lessons gain navigation
   only, with their original cells and outputs preserved.
3. A three-seed CPU centreline study comparing optimizer, depth and wall weighting.
   The improved neural model still loses to log-Re interpolation on both held-out cases.
4. A separate D/W=5 Week 13 notebook with actual author CFD/PINN fields at four Re,
   regional velocity/pressure/vorticity diagnostics, and retained workflow trade-offs.
5. Matching reading extensions in the four existing cavity lecture PDFs. Full
   high-Re field qualification and a new deep-cavity training ablation remain future work.

## Sources and concrete adaptations

| Primary source / code | Useful idea | Specific FlowMLLab experiment and safeguard |
| --- | --- | --- |
| [Ghia et al. 1982](https://doi.org/10.1016/0021-9991(82)90058-4) | Centreline comparisons across seven Re values | Publish both velocity profiles, signed errors, convergence and mesh/time-step sensitivity. Published samples do not become dense ground truth. |
| [McDevitt, Fowler & Roy](https://arxiv.org/abs/2211.06375), [DeepPlasma LDC](https://github.com/cmcdevitt2/DeepPlasma/tree/fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b/LDC) | Physics-constrained cavity learning; role of sparse data and parameter variation | Compare data-free and sparse-data PINNs at fixed information budgets. Do not call a model Ghia-validated if the same Ghia points supplied its training labels. |
| [JAX-PI cavity configuration](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/examples/ldc/configs/default.py) | Re continuation 100 -> 400 -> 1000, modified MLP, Fourier features, random weight factorization and gradient-norm weighting | Separate warm-start initialization from architecture and weighting. Compare direct versus continuation training with the full prior cost counted; keep independent residual and field audits. |
| [PirateNets](https://arxiv.org/abs/2402.00326) | Adaptive residual connections for deeper PINNs | Test plain versus residual 2/5-layer models with matched parameter count and derivative evaluations. Simply making the current MLP deeper is not the same method. |
| [Rathore et al., ICML 2024](https://proceedings.mlr.press/v235/rathore24a.html), [code](https://github.com/pratikrathore8/opt_for_pinns) | Residual-induced ill-conditioning and staged first-/second-order optimization | Benchmark Adam, L-BFGS, their sequence and the existing self-scaled Broyden route. Count function/gradient evaluations, memory and achieved field error. Keep the quasi-Newton objective fixed. |
| [SOAP gradient-alignment study](https://arxiv.org/abs/2502.00604) | Preconditioning can reduce conflicts between composite loss gradients | Compare SOAP-to-SSB with the retained training route using identical starts and samples. Log per-loss gradient norms/cosines and regional errors, not only the total loss. |
| [DeepXDE residual refinement demo](https://deepxde.readthedocs.io/en/v1.15.0/demos/pinn_forward/burgers.rar.html) | Add collocation where residuals are large | Adapt to the lid corner and weak lower vortices, retain uniform independent audit nodes, and record sampling weights. A residual-selected sample mean is not automatically a uniform-domain mean. |
| [PINNacle benchmark](https://github.com/i207M/PINNacle), [paper](https://arxiv.org/abs/2306.08827) | Common tasks and systematic comparisons of loss weighting / domain treatments | Make a reproducible cavity matrix across Re, depth, method and seed; preserve failure rows and reference identity. Adopt the protocol, not unrelated benchmark bulk. |
| [NVIDIA PhysicsNeMo LDC example](https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/ldc_pinns/README.md) | Explicit geometry, equation and boundary-constraint organization | Improve the student's residual/BC ledger and experiment configuration, while keeping the CPU teaching path small. |
| [Okada's cavity PINN](https://github.com/okada39/pinn_cavity) | Compact streamfunction-pressure and L-BFGS teaching example | Use as a pedagogical comparison to primitive variables. Its Re=100 example and older dependency stack do not qualify high-Re runs. |
| [Roohi & McDevitt D/W=5, v2](https://arxiv.org/abs/2610.38814v2) | Regional assessment reveals different velocity, pressure and vorticity rankings | Reproduce common-grid fields, retain one pressure gauge, compare absolute and relative weak-flow error. Historical workflow gains are not isolated optimizer effects. |

## Order of further work

**First: reference qualification.** Extend the existing CFD production matrix to
3200,5000,7500,10000 with three grids and time-step sensitivity. Inspect stationarity,
corner resolution and pressure recovery separately. Keep failed or nonstationary
cases out of field training; document them rather than tuning the acceptance
threshold to get a pass. Reuse already qualified low-Re results with provenance.

**Second: full-field generalization.** Freeze whole-Re splits, train-only scaling
and POD bases. Compare linear interpolation, POD regression, the existing DeepONet
and a boundary-constrained variant at identical information budgets. The present
centreline result is a warning that a neural gain over Adam alone can still lose
to a simple interpolant. Keep spatial and parameter generalization separate.

**Third: isolated PINN ablations.** Use matched references at Re=100/400/1000 and
D/W=1/2/5, with the correct lid for each family. Change one of network depth,
residual connection, initialization, optimizer, or regional weights at a time.
Compare equal evaluations and achieved accuracy/time; include several seeds.
Do not conflate five network layers with a five-width-deep cavity.

**Fourth: adaptive training.** Test gradient-norm loss balancing and residual
refinement only after the fixed-weight baseline passes derivative/BC checks.
Freeze sampling/weights during line searches, and maintain fresh uniform audit
samples. For weak lower recirculation, include velocity amplitude, pressure
offset/shape and vorticity profiles so local deterioration remains visible.

## Evidence boundaries

The new notebooks can inspect all seven square-cavity reference cases now. They
do not silently fabricate the four missing high-Re full fields. The D/W=5 bundle
contains the paper's retained evidence and can reproduce its sampled regional
numbers; it is not a fresh multi-seed training study. No new large cluster jobs are created by these teaching edits. Software-release metadata
records publication separately from scientific qualification.
