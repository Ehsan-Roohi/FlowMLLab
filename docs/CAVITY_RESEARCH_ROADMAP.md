# Cavity development: literature, implemented evidence and next experiments

Research checked 15 November 2026. The purpose is to deepen the existing FlowMLLab
cavity route, not to replace it with a second framework. Below, an upstream result
is an idea to test locally, not evidence that FlowMLLab has achieved that accuracy.

## Verified evidence already implemented

1. One source of Ghia Tables I/II covers all seven standard Re values, with
   printed-entry sensitivity and unfiltered scores retained. Week 4 Labs 5/6
   examine coverage and three-seed centreline learning; that earlier neural
   profile model still loses to log-Re interpolation on both held-out cases.
2. [New native CFD fields](../results/cavity_highre_ghia/README.md) now cover
   15 Reynolds numbers on 128, 256 and 512 cells per side: 45 fields, including
   all seven Ghia cases. Lab 7 evaluates their equations, boundary conditions,
   mesh sensitivity and short matched-time perturbation checks. The high-Re
   fields are no longer missing. They are not established mesh-independent truth.
3. [Supervised full-field POD-DeepONet](../results/cavity_field_deeponet/README.md)
   is trained and frozen, with whole-Re tests at 3200 and 7500 and matched
   interpolation baselines. Its reported velocity discrepancies against the
   retained 512-cell CFD fields are 0.0951% and 0.0844%. This is not a label-free
   PINN. Large nonlinear residuals and reference-grid sensitivity remain disclosed.
4. The full-field gallery, Lab 8 and Week 4 lecture compare streamlines,
   vorticity and reconstructed pressure at Re1000 (training) and Re7500 (held out).
   Pressure is postprocessed from velocity, not independently learned or validated.
5. [D/W=5 retained author evidence](../results/week13_d5/README.md) reproduces
   regional diagnostics for Re100/250/500/1000 and retains historical workflow
   trade-offs. Those unequal transfer/refinement histories do not isolate one
   optimizer or provide a fresh independent-seed study.
6. [The measured solver comparison](../results/cavity_solver_benchmark/README.md)
   and executable lesson appendices are complete. The tested GMRES/multigrid
   candidate was slower than sparse-direct Newton; both wins and failures are
   retained. Further solver optimization is not part of the current work.

7. [The fresh label-free Ghia baseline](../results/cavity_pinn_campaign/figures/README.md)
   has completed all 42 predeclared endpoints and post-freeze scores: seven Re,
   three seeds and two collars. It **failed the validation targets**: 0/42 pass
   both screened Ghia components, 0/42 pass the independent momentum gate,
   3/42 pass interior vector error, and 0/42 pass all four numerical targets.
   The streamfunction satisfies continuity, but that is not sufficient for the
   momentum solution. All signed profiles and the predeclared Re1000/7500
   physical comparisons are retained. In 14/21 collar pairs, sparse sampling
   produced identical discrete objectives; zero differences in those pairs
   do not establish collar independence.

## Completed controlled follow-up studies

The [D5 optimizer comparison](../results/deep_cavity_ablation/README.md) now
contains six fresh frozen runs and three paired seeds. Architecture, Gaussian
lid, depth envelope, samples and objective/gradient budgets match within each
pair. SOAP followed by compact self-scaled Broyden lowers interior vorticity
error in all three pairs, but improves velocity in only one and pressure in
none. Interior velocity NRMSE is approximately 97-117%; neither route reproduces
the retained deep-cavity reference accurately at this finite budget.

The [12-case sampling comparison](../results/cavity_pinn_stratified/README.md)
tests Re=100/Re=7500 with fixed corner/wall/interior strata and area weights,
matching each original counterpart's work. Interior velocity error improves in
2/12 pairs and worsens in 10/12, including all six Re=7500 pairs. Better coverage
alone did not repair the failed high-Re solution. No further variants were
selected after inspecting these outcomes.

The new [Lab 9](../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb) and
[D5 controlled study](../notebooks/week13/W13_D5_Controlled_Optimizer_Study.ipynb)
teach these measured outcomes with signed profiles, fields and provenance.
The sampling study discloses the earlier four-case partial scoring attempt
and its exact-saved-state inference repair. All training endpoints predate
target access; no retraining or endpoint selection followed that access.

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

## What was completed and what remains a proposal

**Completed admission: preserve the formulation and recovery.** The square Ghia study preserves
constant open-wall lid speed; the D5 study preserves the author's Gaussian lid,
depth envelope, physical derivatives and residual scaling. Their physics must
not be mixed. Independent manufactured-residual, boundary, pressure-gradient,
loss-gradient and interrupted/restarted-update checks passed before the current training work. A new variant needs its own admission.

**Completed baseline: retain the bounded-pilot and full-matrix outcome.** The square-cavity design
uses leading Stokes corner velocity/pressure subtraction and explicitly declared
0.02W/0.01W collocation-collar sensitivity. Raw residuals inside and outside the
collars remain visible. This does not establish full-domain strong-residual
convergence. Use measured pilot cost and the predeclared physics/audit gates to
set a finite case/seed budget; retain failed runs. The 42-case baseline now shows
why sparse collar sampling and low training loss do not establish a solution.
For D5, the authentic SOAP/compact-self-scaled-Broyden source and complete recovery
are admitted. Dense and compact implementations are not treated as interchangeable;
the matched numerical comparison is complete and its finite-budget failures are retained.

**Future ideas, not additional authorized runs or completed improvements.** Compare fixed and
importance-corrected adaptive collocation at matched initialization and work;
freeze samples and weights before quasi-Newton stages. For optimizer attribution,
keep network, points, physics and audit rules fixed within each paired seed.
Count closures, candidate-pool residual evaluations, memory and time. Keep CFD
and Ghia targets out of fitting and model selection; freeze prediction hashes
before reference scoring. Report physical profiles, signed regional errors,
reference amplitude, all seeds and failures alongside objective residuals.

**Teaching and release.** The new lessons append verified fresh results while
preserving older notebook exercises and lecture pages. The homepage keeps its
course structure and distinguishes the supervised successes from the PINN
failures. GitHub and Zenodo publication are verified separately from numerical
completion.

## Evidence boundaries

The prior high-Re CFD qualification and supervised field-model work are complete
within their stated numerical limits. The retained 256-to-512 CFD vector
sensitivities at Re3200/5000/7500/10000 are 2.03%/3.14%/4.72%/6.32%; they can
exceed surrogate error. A steady solution and a short perturbation check do not
establish asymptotic temporal stability.

Fresh case-specific PINN outputs, if admitted, do not demonstrate held-out-Re
operator generalization. The D5 retained paper fields are distinct from new
controlled trials. Literature-inspired depth, weighting, continuation and
residual-architecture ideas remain proposals until tested in a matched study;
simply making a network deeper does not establish a gain. Do not claim the first
neural solution of all seven Ghia cases: the checked literature already includes
case-specific all-seven coverage. The value here is a transparent information
budget, reproducible physical assessment and honest gains and losses.
