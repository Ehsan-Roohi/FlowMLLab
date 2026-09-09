# FlowMLLab

[![FlowMLLab CI](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22651906.svg)](https://doi.org/10.5281/zenodo.22651906)

Learn scientific machine learning through reproducible fluid-mechanics experiments:
generate numerical data, compare transparent baselines with learned models, and
check both prediction error and physical fidelity.

Developed for **MIE 690A: AI in Fluid Mechanics**, University of Massachusetts
Amherst. The working course includes **33 notebooks** and a continuous lecture
sequence from
numerical foundations to continuum and rarefied-flow research examples.

## Start here

**Research provenance:** the article-linked DSMC cases were produced in earlier
research by Ehsan Roohi and collaborators, then brought into FlowMLLab for
teaching. The course adds new code and baselines, not a new origin for those
data. See [per-case papers, data lineage, reuse limits and AI-assistance
disclosure](DATA_PROVENANCE.md).

| Your goal | Open |
| --- | --- |
| Run a first experiment in 20 minutes | [Launch the introductory Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Follow the course | [Course map](COURSE_MAP.md) · [All notebooks](notebooks/README.md) · [Lectures](lectures/README.md) |
| Install and reproduce the results | [Setup and validation](START_HERE.md) |
| Explore the scientific evidence | [Results and technical guide](docs/RESULTS_GUIDE.md) · [Interactive cavity demo](demo/README.md) |

## Continue through the course

Each week has its own row, including the incremental laboratories.
Weeks 5 and 6 share a project pack and lecture guide, but have separate learning goals.

| Week | Topic | Notebook / lab | Lecture |
| --- | --- | --- | --- |
| [1](#week-1--numerical-foundations) | Python, numerical methods and CFD validation | [Week 1 labs](notebooks/week01/) | [Lecture 1](lectures/week01_numerical_foundations.pdf) |
| [1.1](#week-11--ai-assisted-scientific-software) | Specification, verification, physical gates and accountable AI use | [Week 1.1 lab](notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb) | [Lecture 1.1](lectures/week01_1_ai_assisted_scientific_software.pdf) |
| [2](#week-2--supervised-learning-and-rarefaction) | Features, scaling, baselines and model validity | [Week 2 lab](notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) | [Lecture 2](lectures/week02_supervised_learning_rarefaction.pdf) |
| [2.1](#week-21--probabilistic-uncertainty) | Bayesian prediction, calibration and uncertainty | [Week 2.1 lab](notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) | [Lecture 2.1](lectures/week02_1_probabilistic_uq.pdf) |
| [3](#week-3--kinetic-theory-and-dsmc) | Maxwellian sampling and particle simulation | [Week 3 labs](notebooks/week03/) | [Lecture 3](lectures/week03_kinetic_dsmc.pdf) |
| [4](#week-4--cavity-surrogates-and-deeponet) | CFD datasets, field surrogates and operator learning | [Week 4 labs](notebooks/week04/) | [Lecture 4](lectures/week04_cavity_surrogates_deeponet.pdf) |
| [4.1](#week-41--classical-reduced-order-models) | POD–Galerkin and POD–DEIM | [Week 4.1 lab](notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) | [Week 4 companion](lectures/week04_cavity_surrogates_deeponet.pdf); theory in lab |
| [5](#week-5--physics-guided-projects) | POD, physics-guided learning and frozen project protocols | [Week 5 project setup and tracks](notebooks/week05_06/README.md) | [Shared Weeks 5–6 guide](lectures/week05_06_project_guide.pdf) |
| [6](#week-6--physical-validation-and-final-evidence) | Closure testing, physical validation and reproducibility | [Week 6 closure track](notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) · [All tracks](notebooks/week05_06/README.md) | [Shared Weeks 5–6 guide](lectures/week05_06_project_guide.pdf) |
| [7](#week-7--unsteady-cylinder-wakes) | LBM, vortex shedding and autonomous surrogates | [Week 7 lab](notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) | [Lecture 7](lectures/week07_cylinder_lbm_neural_surrogate.pdf) |
| [7.1](#week-71--rarefied-hypersonic-cylinder) | DSMC fields and Mach-to-field operators | [Week 7.1 lab](notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) | [Lecture 7.1](lectures/week07_1_hypersonic_rarefied_cylinder.pdf) |
| [7.2](#week-72--sparse-sensor-state-estimation) | Causal filtering of a cylinder wake from noisy sparse sensors | [Week 7.2 lab](notebooks/week07_2/README.md) | [Lecture 7.2](lectures/week07_2_cylinder_state_estimation.pdf) |
| [8](#week-8--gas-dynamics-and-sciml) | Exact compressible-flow branches and learned inverse maps | [Week 8 labs](notebooks/week08/README.md) | [Lecture 8](lectures/week08_gas_dynamics_sciml.pdf) |
| [9](#week-9--rarefied-micro-step-and-micro-nozzle) | Geometry-dependent and shock-aligned operators | [Week 9 labs](notebooks/week09/README.md) | [Lecture 9](lectures/week09_rarefied_deeponet_case_studies.pdf) |
| [10](#week-10--dsmc-cavity-and-molecular-shocks) | Cavity and mono/diatomic shock reproduction | [Week 10 lab](notebooks/week10/README.md) | [Lecture 10](lectures/week10_dsmc_data_driven_surrogates.pdf) |
| [10.1](#week-101--ab-initio-collision-deeponet) | Molecular scattering and DSMC cylinder contours | [CPU scattering lab](notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb) · [Research fields](results/abinitio_deeponet_cylinder/README.md) | [Lecture companion](lectures/week10_1_abinitio_collision_deeponet.md) |
| [11](#week-11--shock-and-vortex-identification) | Physical diagnostics and overlapping learned labels | [Week 11 lab](notebooks/week11/README.md) | [Lecture 11](lectures/week11_shock_vortex_identification.pdf) |
| [12](#week-12--dsmc-moment-reconstruction) | Additive moments, observation-conditioned reconstruction and support | [Week 12 lab](notebooks/week12/README.md) | [Lecture 12](lectures/week12_dsmc_moment_reconstruction.pdf) |
| [13](#week-13--rectangular-cavity-pinn-research-audit) | Streamfunction PINNs across Reynolds number and cavity depth | [Week 13 audit](notebooks/week13/README.md) | [Lecture 13](lectures/week13_rectangular_cavity_pinn.pdf) |

## Results gallery · in course order

The figures below connect each week to an experiment or teaching example.
Follow the captions for data provenance and validity limits; the
[technical guide](docs/RESULTS_GUIDE.md) retains the detailed protocols and metrics.

### Week 1 — Numerical foundations

**Problem:** Lid-driven cavity benchmark.<br>
**CFD / data:** FlowMLLab finite-difference streamfunction–vorticity Navier–Stokes solver.<br>
**Learning method:** No neural network; CFD is checked against Ghia centreline data.

![Cavity CFD benchmark and Ghia velocity validation](results/article_figures/fig02_cavity_benchmark.png)

Start with a numerical solution and an independent benchmark: cavity fields,
centerlines and Ghia comparisons establish what a useful training label means.
[Figure contract](ARTICLE_FIGURE_MAP.md)

### Week 1.1 — AI-assisted scientific software

**Problem:** Verify cavity diagnostics and physical constraints.<br>
**CFD / data:** Analytic manufactured solution and a retained FlowMLLab cavity field.<br>
**Learning method:** No trained network; independent tests verify AI-proposed code.

![Executable scientific contract for an AI-proposed cavity diagnostic](results/week01_1_scientific_software/week01_1_acceptance_summary.png)

An independently derived manufactured solution establishes second-order
convergence before the accepted `Re=100` cavity case is opened. Seven frozen gates
then bind array semantics, vorticity convention, incompressibility, wall
conditions, thresholds and the complete data hash into one machine-readable
decision. [Run the lab](notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb)
· [Read the scientific specification](notebooks/week01_1/SCIENTIFIC_SPEC.md)
· [Complete the agent-assisted GitHub assignment](notebooks/week01_1/ASSIGNMENT.md)
· [Inspect the retained evidence](results/week01_1_scientific_software/README.md)

### Week 2 — Supervised learning and rarefaction

**Problem:** Predict response across Knudsen number and accommodation.<br>
**CFD / data:** Synthetic teaching equation; no CFD/DSMC run.<br>
**Learning method:** Ridge regression, not a neural CFD model.

![Week 2 synthetic response across Knudsen regimes and accommodation coefficients](assets/week02_synthetic_response.png)

The Week-2 notebook's synthetic response illustrates how Knudsen number and
accommodation affect the learning problem. This is a teaching equation, not a
CFD/DSMC result. [Run the lab](notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb)

### Week 2.1 — Probabilistic uncertainty

**Problem:** Predict cavity quantities and fields with uncertainty.<br>
**CFD / data:** Retained FlowMLLab cavity CFD data.<br>
**Learning method:** Bayesian regression and POD–Gaussian processes; non-neural models.

![Probabilistic CFD prediction, uncertainty and blind calibration checks](results/probabilistic_uq/probabilistic_uq_validation.png)

Bayesian and POD–Gaussian-process predictions are checked with proper scores
and blind coverage; the retained under-coverage is part of the lesson.
[UQ evidence](results/probabilistic_uq/README.md)

### Week 3 — Kinetic theory and DSMC

**Problem:** Predict wall pressure from molecular motion.<br>
**CFD / data:** FlowMLLab hard-sphere DSMC with no-time-counter (HS–NTC) collisions.<br>
**Learning method:** No network in the displayed particle-solver validation.

![HS–NTC DSMC wall-pressure validation against Mohammadzadeh reference data](results/article_figures/fig10a_mohammadzadeh_validation.png)

Connect molecular sampling to a macroscopic observable through the executed
HS–NTC wall-pressure validation.
[Validation contract](ARTICLE_FIGURE_MAP.md)

### Week 4 — Cavity surrogates and DeepONet

**Problem:** Map Reynolds number to cavity fields.<br>
**CFD / data:** FlowMLLab finite-difference streamfunction–vorticity Navier–Stokes solver.<br>
**Learning method:** POD–DeepONet: learned parameter-to-coefficient map with a fixed POD spatial basis.

![POD–DeepONet cavity fields, Ghia checks, blind errors and cost](results/pod_deeponet/pod_deeponet_ghia_validation.png)

Complete-case testing combines field error, wall/divergence checks, reference
centerlines and measured inference cost.
[Model and validation evidence](results/pod_deeponet/README.md)

### Week 4.1 — Classical reduced-order models

**Problem:** Evolve cavity flow in a reduced state space.<br>
**CFD / data:** The Week-4 cavity equations and finite-difference full-order model.<br>
**Learning method:** POD–Galerkin and POD–DEIM; neither is a neural network.

![Classical cavity POD–Galerkin and POD–DEIM validation and timing](results/cavity_rom/cavity_rom_validation.png)

Compare reduced dynamics, hyper-reduction, blind trajectories and the offline/online
cost tradeoff. [Run the ROM lab](notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb)

### Week 5 — Physics-guided projects

**Problem:** Build a cavity surrogate or reconstruct a wake from sparse sensors.<br>
**CFD / data:** FlowMLLab cavity CFD for the animation; D2Q9–TRT LBM for the wake extension.<br>
**Learning method:** POD–DeepONet for the animation; gappy POD, sensor placement and SINDy in the extension.

![Animated cavity comparison for three retained blind POD–DeepONet cases](assets/flowmllab_blind_demo.gif)

Use this retained cavity experiment as a project starting point: freeze a baseline,
change one modeling choice and evaluate complete unseen cases.
[Project pack](notebooks/week05_06/README.md) · [Interactive demo](demo/README.md)

Extend the project with the [sparse sensing lab](notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb):
reconstruct fields from limited measurements and inspect the
[retained modal-method comparisons](results/modal_labs/README.md).

### Week 6 — Physical validation and final evidence

**Problem:** Recover and validate cavity pressure.<br>
**CFD / data:** FlowMLLab cavity CFD with least-squares pressure-gradient reconstruction.<br>
**Learning method:** No neural model in this pressure-recovery figure.

![Independent cavity pressure-recovery validation](results/article_figures/fig08_pressure_recovery.png)

This existing pressure benchmark illustrates the independent physical checks
expected in a final evidence bundle; it is not a learned Fokker–Planck result.
Week 6 completes the selected Week-5 track, including optional closure testing.
[Project completion guide](notebooks/week05_06/README.md)

### Week 7 — Unsteady cylinder wakes

**Problem:** Predict future cylinder-wake vorticity.<br>
**CFD / data:** FlowMLLab D2Q9–TRT lattice Boltzmann solver (LBM).<br>
**Learning method:** The lead result is a phase-stable learned Fourier decoder, not an autoregressive CNN; POD and CNN baselines are linked below.

![Fresh Re=95 cylinder wake: LBM and autonomous learned decoder](results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.webp)

Four initial fields seed **277 autonomous future frames** at unseen **Re = 95**,
with **4.281% global vorticity error** against educational LBM labels.
The grid study passes practical fine-pair limits but fails the formal
asymptotic/GCI gate; these labels are not high-fidelity DNS.
[Video](results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4)
· [Model evidence](results/cylinder_phase/README.md)
· [Grid study](results/cylinder_grid_convergence/README.md)

[![Earlier Re=100 POD failure comparison](results/cylinder_ml/blind_re100_lbm_vs_neural_poster.png)](results/cylinder_ml/blind_re100_lbm_vs_neural.mp4)

The earlier Re=100 POD comparison is retained as a failed baseline, separate
from the autonomous Re=95 result. [Failure analysis](results/cylinder_ml/README.md)

Continue with the [modal forecasting lab](notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb)
and its [reproducible evidence](results/modal_labs/README.md).
The [Re100 D40 dataset](results/cylinder_d40/README.md) provides force histories,
final fields and a three-grid comparison. A separate
[frozen-model Re115 evaluation](results/cylinder_re115_evaluation/README.md)
reports 3.17% global vorticity error, with sampling and pressure limitations.

### Week 7.1 — Rarefied hypersonic cylinder

**Problem:** Predict cylinder fields as Mach number varies.<br>
**CFD / data:** Author-supplied DSMC archives; exact source/checkpoint attribution remains subject to the linked audit.<br>
**Learning method:** 3×96 tanh MLP versus Mach interpolation; separate from the article's Fusion-DeepONet.

![Original 400 by 400 Mach-8.5 DSMC fields and interpolation errors](results/hypersonic_cylinder_week7_1/cylinder_homepage.png)

The original **400 × 400 Mach-8.5** fields are rendered with continuous contours.
Error panels compare against Mach-8/Mach-9 interpolation; gray marks the masked
solid/sentinel region.
[Figure provenance](results/hypersonic_cylinder_week7_1/cylinder_homepage_provenance.json)

Across the five held-out cases in the **compact teaching dataset**, interpolation
errors are **0.398% / 0.609% / 0.779%** for local Mach, temperature and pressure.
The trained **3×96 tanh MLP** gives **1.53% / 2.70% / 2.50%**, with training
errors **1.26% / 2.21% / 1.82%**. Interpolation still wins; the earlier
underfit random-feature ridge is no longer the default classroom comparison.
These are new teaching runs, not the published model's accuracy.
[Data and paper](data/hypersonic_cylinder/README.md) · [MLP metrics](results/hypersonic_cylinder_week7_1/mlp_metrics.json)

### Week 7.2 — Sparse-sensor state estimation

**Problem:** Estimate a wake from noisy velocity sensors.<br>
**CFD / data:** Retained FlowMLLab D2Q9–TRT LBM Re110 trajectory.<br>
**Learning method:** POD–DMD dynamics and a Kalman filter; no neural network in this estimator.

![Reference and causal state estimates at the final test frame](results/week07_2_state_estimation/state_estimation_fields.png)

A validation-selected rank-8 Kalman filter assimilates 32 noisy transverse-velocity
sensors on the retained Re110 wake. Mean test relative L2 is **2.18%**, versus
**3.16%** for matched sensor-only POD reconstruction and **4.36%** for open-loop
DMD. Its nominal 95% marginal intervals cover only **55.1%** of sampled values;
the overconfidence is retained as a model failure.
[Protocol, all baselines and limits](results/week07_2_state_estimation/README.md)

### Week 8 — Gas dynamics and SciML

**Problem:** Predict and invert compressible-flow relations.<br>
**CFD / data:** Exact gas-dynamics relations and numerical root finding; no spatial CFD run.<br>
**Learning method:** MLP inverse maps versus interpolation and radial-basis-function baselines.

![Gas-dynamics model evidence, blind errors and matched-budget comparisons](results/gas_dynamics_week8/week8_model_evidence.png)

Preserve physical branches while comparing exact solvers, interpolation and
learned inverse maps under matched budgets.
[Benchmarks and validity limits](results/gas_dynamics_week8/README.md)

### Week 9 — Rarefied micro-step and micro-nozzle

**Problem:** Predict geometry-dependent step fields and pressure-dependent nozzle fields.<br>
**CFD / data:** Author-supplied DSMC archives; the nozzle uses modified Bird-family exports with a documented boundary defect.<br>
**Learning method:** Step: geometry/coordinate MLP. Nozzle: shock-aligned POD with polynomial or tanh-MLP coefficient maps. Article and experimental DeepONet results are distinguished in the linked reports.

![Held-out H44 micro-step: DSMC and independent teaching-model contours](results/mahdavi_deeponet/step_independent_contours/held_out_H44_independent.png)

The independent H44 teaching model uses geometry and coordinates only.
[Step evidence and provenance](results/mahdavi_deeponet/README.md)

![Micro-nozzle 25-kPa DSMC fields, selected registered POD model and absolute errors](results/nozzle_transport/nozzle_P25_fields.png)

![Micro-nozzle 25-kPa profiles comparing DSMC, interpolation and learned branches](results/nozzle_transport/nozzle_P25_profiles.png)

The selected registered-POD polynomial model and trained neural branches are
compared with the original interpolation baseline. The displayed transverse
velocity on the symmetry plane is the prescribed **V = 0** boundary condition,
not a learned accuracy result. Raw exports have a documented symmetry defect;
these are historical-holdout regression results, not fresh blind validation.
[Nozzle report](results/nozzle_transport/README.md)
· [Raw boundary audit](results/nozzle_transport/symmetry_boundary_audit.png)

### Week 10 — DSMC cavity and molecular shocks

**Problem:** Reconstruct rarefied cavity and mono/diatomic shock data.<br>
**CFD / data:** Author-supplied article DSMC tables, not new particle-solver runs.<br>
**Learning method:** POD–polynomial operators and interpolation; no neural network in the displayed reproduction.

![DSMC cavity, monatomic and diatomic shocks, and Maxwell-equilibrium reproduction](results/aescte_dsmc/week10_dsmc_reproduction_summary.png)

The article-data experiment retains **1.281% maximum primary cavity NRMSE**
and **1.018% maximum shock-profile relative L2 error**, with the data contract
and regeneration workflow available for inspection.
[Reproduction evidence](results/aescte_dsmc/README.md)
· [Data contract](data/aescte_dsmc/README.md)

### Week 10.1 — Ab initio collision DeepONet

**Problem:** Compute cylinder flow using a collision-angle surrogate.<br>
**CFD / data:** Author-supplied DS2V-based DSMC research runs.<br>
**Learning method:** DeepONet supplies scattering-angle tables, not whole-field predictions; the CPU lab is a separate Lennard-Jones analog.

![Jäger Ar–Ar cylinder temperature: Exact and DeepONet at different output times](results/abinitio_deeponet_cylinder/temperature_exact_deeponet.png)

Author-supplied research fields from the later DeepONet collision-angle package,
related to [Roohi, Shoja-sani and Stefanov, PoF 38, 057123 (2026)](https://doi.org/10.1063/5.0328463).
DSMC generates these fields using exact-derived or DeepONet-derived angle tables;
this is not a whole-field neural prediction or reproduction of the article's MLP.
Shared colors, **different times and sampling windows**: qualitative comparison only.
[All four colored fields and provenance](results/abinitio_deeponet_cylinder/README.md)
· [Surface pressure and heat flux](results/abinitio_deeponet_cylinder/README.md#surface-pressure-and-heat-flux)
· [Week 10.1 lecture companion](lectures/week10_1_abinitio_collision_deeponet.md)

[Run the CPU collision-map lab](notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb):
solve a reduced Lennard-Jones scattering problem, check analytic limits and
audit a fitted surrogate using transport integrals. This executable analog is
separate from the Jäger research fields above; it does not reproduce the
article's potential or network.

### Week 11 — Shock and vortex identification

**Problem:** Identify shocks/vortex cores and test reconstruction before detection.<br>
**CFD / data:** Archived ShockVortexML compressible fields for the lead figure; FlowMLLab D2Q9–TRT LBM for the wake extension. The lead archive's exact producing-solver revision is not established here.<br>
**Learning method:** Frozen joint shock/vortex segmentation checkpoint; a separate reconstruction U-Net versus direct-mask prediction comparison.

![Week 11 real airfoil field and learned shock and vortex masks](results/week11_research/airfoil_2.png)

Six fresh fixed-checkpoint forward passes on existing airfoil and cylinder fields from
Roohi's [ShockVortexML research](https://github.com/Ehsan-Roohi/ShockVortexML).
ML-only outputs; previously inspected development-test cases, not human-validated accuracy.
[All six full-size figures and provenance](results/week11_research/README.md) ·
[Notebook and lecture](notebooks/week11/README.md). Synthetic controls remain as the warm-up.

**Real-field extension:** [U-Net reconstruction followed by vortex identification](results/week11_reconstruction/README.md)
compares interpolation, reconstructed velocity plus swirling strength, and direct
mask prediction on the same retained LBM wake cases. Three training seeds, loss
histories and saved-checkpoint audits are included. These are weak-reference
vortex scores on coarse incompressible data, not shock accuracy or a blind test.

### Week 12 — DSMC moment reconstruction

**Problem:** Reduce cavity heat-flux sampling noise.<br>
**CFD / data:** Author-supplied multi-seed DSMC with an independent finite-sample reference.<br>
**Learning method:** Archived observation-conditioned estimator in the lead figure; a separate 64×32 tanh patch MLP in the Noise2Noise lab.

![Week 12 real DSMC heat flux reference, observation and reconstruction](results/week12_research/cavity_qy_hero.png)

Existing author-supplied DSMC cavity results associated with
[Roohi, arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637).
Eight seeds, both heat-flux components, 80 recomputed errors; no new DSMC or neural training.
Mean reference NRMSE for qy: Raw(3) 17.61%, Raw(10) 9.80%, conditioned estimator 4.34%.
[Full comparisons, profiles and all-seed errors](results/week12_research/README.md) ·
[Notebook and lecture](notebooks/week12/README.md). The independent reference still has sampling noise.

The companion [Noise2Noise-style MLP lab](results/week12_noise2noise/README.md)
trains on raw DSMC observations, compares seven estimators and reports both
held-out seeds. Its training results are documented separately from the
archived research reconstruction shown above.

### Week 13 — Rectangular-cavity PINN research audit

**Problem:** Solve square and deep lid-driven cavities.<br>
**CFD / data:** Displayed fields are PINN solutions, not CFD output. OpenFOAM finite-volume and Nektar++ spectral/hp cavity calculations form a separate reference-validation campaign; they are not yet certified matched references for these displayed deep-cavity PINNs.<br>
**Learning method:** Streamfunction PINN with hard wall constraints, trained using Adam then SSBroyden2 in float64.

The final module develops streamfunction PINNs for square and deep lid-driven
cavities. Compare **Re = 100 and 400**, **H/L = 1 and 2**, and **Adam followed
by SSBroyden2** using retained float64 A100 runs, exact wall constraints and
independent residual checks. Square cases have frozen near-matched CFD gates;
the displayed deep-cavity cases still require matched CFD field validation.

[Lecture](lectures/week13_rectangular_cavity_pinn.pdf) ·
[Notebook and results](notebooks/week13/README.md) ·
[Reproduction protocol](qa/WEEK13_PINN_MATRIX_PROTOCOL.md)

Preparatory reading covers [PINN residuals and hard boundary constraints](lectures/week04_2_pinn_cavity.pdf),
including [McDevitt's DeepPlasma cavity code](https://github.com/cmcdevitt2/DeepPlasma/tree/fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b/LDC),
used with his permission. The earlier [Re=100 qualification and CFD comparison](results/week04_2_pinn_cavity/README.md)
provides supporting evidence for this module.

**Re = 100**

Square cavity, D = H/L = 1:

![Reynolds 100 square cavity PINN speed, streamfunction and streamlines](results/week13_rectangular_pinn/re100-d1/fields.png)

![Reynolds 100 square cavity PINN training and held-out residual histories](results/week13_rectangular_pinn/re100-d1/loss.png)

Deep cavity, D = H/L = 2:

![Reynolds 100 deep cavity PINN speed, streamfunction and streamlines](results/week13_rectangular_pinn/re100-d2/fields.png)

![Reynolds 100 deep cavity PINN training and held-out residual histories](results/week13_rectangular_pinn/re100-d2/loss.png)

**Re = 400**

Square cavity, D = H/L = 1:

![Reynolds 400 square cavity PINN speed, streamfunction and streamlines](results/week13_rectangular_pinn/re400-d1/fields.png)

![Reynolds 400 square cavity PINN training and held-out residual histories](results/week13_rectangular_pinn/re400-d1/loss.png)

Deep cavity, D = H/L = 2:

![Reynolds 400 deep cavity PINN speed, streamfunction and streamlines](results/week13_rectangular_pinn/re400-d2/fields.png)

![Reynolds 400 deep cavity PINN training and held-out residual histories](results/week13_rectangular_pinn/re400-d2/loss.png)

Loss curves show momentum **residual RMS**, not CFD field error: blue is the
Adam warm-up (steps 1–1000), orange is SSBroyden2 continuation, dashed black is
the held-out full-domain residual, and dotted green is the held-out top-corner
residual. Decreasing training loss alone does not establish convergence of the
physical solution; the retained held-out discrepancies remain visible.

## Reuse and contribute

The installable Python package, numerical solvers, notebooks, and teaching
materials are open source. See [contribution guidelines](CONTRIBUTING.md), the
[roadmap](ROADMAP.md), and [source/attribution policy](THEORY_SOURCE_POLICY.md).
The [branch and release procedure](docs/BRANCH_AND_RELEASE_POLICY.md) describes
evidence promotion and reproducible notebook HTML.
Student submissions are not included.

[Read the software manuscript](manuscript/FlowMLLab_v1.1.0_Original_Software_Article.pdf)
· [Citation metadata](CITATION.cff)
· [Workshop, support, and consulting details](docs/RESULTS_GUIDE.md#workshops-support-and-consulting)

Current release: **v1.5.1** · [GitHub release](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/v1.5.1)
· [Release notes](RELEASE_NOTES_v1.5.1.md).
Version-specific Zenodo DOI: [10.5281/zenodo.22651906](https://doi.org/10.5281/zenodo.22651906).
For earlier versions and their archived records, see the
[release history](https://github.com/Ehsan-Roohi/FlowMLLab/releases).
The [all-versions DOI](https://doi.org/10.5281/zenodo.22074169) resolves to the latest published archive.

**Ehsan Roohi** · University of Massachusetts Amherst · [roohie@umass.edu](mailto:roohie@umass.edu)

Copyright © 2026 Ehsan Roohi. [MIT License](LICENSE).
