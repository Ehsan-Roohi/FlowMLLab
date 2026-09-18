# Notebook guide

Every notebook below opens directly from GitHub in Google Colab. Each first
code cell clones the repository and installs the tested package, so a fresh
Colab runtime needs no manual uploads. Choose a Colab GPU runtime only for
notebooks that explicitly require CUDA (Track 6). Weeks 14 and 15 are
full-checkout CPU modules with their own data guides; their notebooks run from
a local clone (see the linked setup pages).

## One-click launcher, in course order

| Module | Notebook | Launch |
| --- | --- | --- |
| Week 1 | Python for CFD and AI | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| Week 1 | TensorFlow for AI in fluids | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/02_tensorflow_for_ai_fluids.ipynb) |
| Week 1 | Cavity CFD and Ghia validation | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |
| Week 1.1 | AI-assisted scientific software and executable acceptance gates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb) |
| Week 2 | Supervised learning and rarefaction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) |
| Week 2.1 | Probabilistic UQ for CFD surrogates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) |
| Week 3 | Maxwellian noise and ML | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) |
| Week 3 | Mini DSMC cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) |
| Week 4 | Lab 1: CFD data production | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) |
| Week 4 | Lab 2: scalar and field surrogates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) |
| Week 4 | Lab 3: POD-DeepONet cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) |
| Week 4.1 | Classical POD-Galerkin/POD-DEIM | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) |
| Week 4.2 | Lab 4: Stokes-to-Navier-Stokes correction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) |
| Week 4.2 | Lab 4 addendum: 51 x 51 grid validation | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) |
| Weeks 5 to 6 | P0 setup and data audit | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Weeks 5 to 6 | P1 Reynolds-number generalization | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P1_Re_Generalization.ipynb) |
| Weeks 5 to 6 | P2 physics-guided DNN/PINN objectives | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P2_Physics_Guided_DNN.ipynb) |
| Weeks 5 to 6 | P3 POD study | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P3_POD_Study.ipynb) |
| Weeks 5 to 6 | P4 uncertainty and data sufficiency | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P4_Uncertainty_Study.ipynb) |
| Weeks 5 to 6 | P5 rarefied cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P5_Rarefied_Cavity.ipynb) |
| Weeks 5 to 6 | P6 Fokker-Planck closure (CUDA GPU) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) |
| Week 5 companion | Sparse sensing and dynamics (cylinder wake) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb) |
| Week 7 | Lattice-Boltzmann cylinder wakes | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) |
| Week 7 companion | Modal forecasting | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb) |
| Week 7.1 | Rarefied hypersonic-cylinder operator learning | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) |
| Week 7.2 | Sparse-sensor cylinder-wake state estimation | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb) |
| Week 7.3 | Self-supervised pretraining and label efficiency (cylinder wake) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb) |
| Week 7.4 | Diverse-wake pretraining and lift decoding | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb) |
| Week 8 | Lab 1: exact gas dynamics before ML | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| Week 8 | Lab 2: gas-dynamics SciML evidence | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) |
| Week 9 | Lab 1: micro-step zonal-loss DeepONet | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) |
| Week 9 | Lab 2: shock-aligned micro-nozzle DeepONet | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb) |
| Week 9 | Lab 3: moving-throat nozzle data-alignment audit | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb) |
| Week 10 | DSMC cavity and mono/diatomic shock reproduction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) |
| Week 10.1 | Classical collision map and surrogate audit | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb) |
| Week 11 | Shock and vortex identification | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Shock_Vortex_Identification.ipynb) |
| Week 11 | Lab 2: reconstruction followed by identification | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb) |
| Week 11 | Hydrofoil vapor-cloud detection | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) |
| Week 12 | Noisy DSMC moment reconstruction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb) |
| Week 13 | Rectangular-cavity PINNs | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb) |
| Week 14 | RANS, inverse PINN and neural closures (local checkout) | [Notebook](week14/W14_pyCALC_RANS_PINN_NN.ipynb) and [setup](week14/README.md) |
| Week 15 | Geometry-aware neural operators (local checkout) | [Notebook](week15/W15_Geometry_Operators_Step_Audit.ipynb) and [OpenFOAM data guide](week15/README.md) |

## Weekly laboratories

- `week01/`: Python, TensorFlow, and validated continuum cavity CFD.
- `week01_1/`: a vendor-neutral, specification-driven audit of code proposed by
  a person or coding agent. The lab verifies second-order vorticity on four
  manufactured grids, rejects an executable axis-swap bug, audits the accepted
  `Re=100` cavity field, and requires a complete AI-use disclosure and bounded
  claim.
- `week02/`: supervised-learning foundations and a rarefied-flow surrogate.
- `week02_1/`: the incremental probabilistic-UQ lecture/lab pair. It connects Gaussian observation models, exact Bayesian regression, POD--Gaussian-process fields, proper scores, validation scaling, and blind under-coverage between Weeks 2 and 3.
- `week03/`: Maxwellian sampling/noise and a mini DSMC cavity.
- `week04/`: CFD data production, scalar/field surrogates, a POD-DeepONet laboratory, the classical ROM lab, and a Stokes-to-Navier-Stokes correction lab (with a 51 x 51 addendum) using constant/diverse lid families, development-only selection, three-seed evidence, physical diagnostics, and explicit same-grid limitations.
- `week05_06/`: the combined two-week guided-project pack (P0 to P6). Week 5 establishes the setup, baseline, controlled modification, and checkpoint; Week 6 completes the same selected track, physical validation, reproducibility package, and final report. The sparse-sensing companion lab uses the Week 7 cylinder wakes and can be taken after Week 7.
- `week07/`: D2Q9 BGK/TRT cylinder flow, a concise collide--stream--boundary algorithm walkthrough, physically gated force and Strouhal diagnostics, an executed three-grid study with a retained formal asymptotic/GCI failure, Reynolds-regime classification, strong temporal baselines, and separate one-step/recursive audits of a four-frame multi-scale CNN on a retained held-out interpolation case.
- `week07_1/`: incremental rarefied hypersonic-cylinder operator lab using a compact author-released derivative of 20 DSMC Mach cases, frozen whole-case splits, a strong structured field-interpolation baseline, reviewed Fusion-DeepONet anatomy, a fast CPU teaching analog, and empirical ensemble-coverage checks.
- `week07_2/`: causal POD-space Kalman filtering of the retained Re110 LBM wake with validation-selected sensors and covariance inflation, matched sensor-only/open-loop/persistence baselines, and an explicit marginal-coverage failure.
- `week07_3/`: masked-autoencoder pretraining on the unlabelled Re90/Re100 LBM wakes (PyTorch, CPU), zero-shot, linear-probe, fine-tuned and from-scratch completion of the Re110 wake against the number of labelled frames, gappy-POD baselines with matched information, validation-only selection on Re105, and a retained classical-baseline win.
- `week08/`: two CPU labs that start from exact branch-aware gas dynamics, then compare bracketed roots, interpolation, and physics-guided MLP evidence across five inverse problems, edge holdouts, dimensional scaling, and application workloads.
- `week09/`: three CPU labs based on the Roohi--Mahdavi micro-step and micro-nozzle studies. Lab 1 uses two author-permitted, checksummed derivatives of nine real DSMC height fields with file-level case separation; Lab 2 reproduces shock-centered POD and fresh full-field predictions from checksummed derivatives of 15 public DSMC snapshots before evaluating three held-out pressures; Lab 3 is a short data-alignment audit on a quasi-1D moving-throat family.
- `week10/`: one complete CPU article-reproduction lab with 14 rarefied-cavity fields, mono/diatomic shock profiles, the DSMC algorithm, full provenance audit, log-Knudsen synthesis, POD-branch operators, physical interpretation, and retained numerical gates.
- `week10_1/`: a CPU Lennard-Jones collision-map teaching model with analytic scattering checks, transport integrals and localized surrogate errors.
- `week11/`: manufactured shock/vortex controls and a small pixel classifier; a matched U-Net reconstruction-then-identification comparison on retained LBM wakes; and hydrofoil vapor-cloud detection on retained CFD fields.
- `week12/`: additive-moment algebra, prior-plus-observation reconstruction on a synthetic analog, a fresh Noise2Noise-style fit on real DSMC cavity observations, and an audit of the archived research reconstruction.
- `week13/`: streamfunction PINNs for square and deep cavities: a small CPU training exercise, the retained Re=1000, D/W=2.2 field beside Nektar++ CFD, and the four-case A100 residual audit.
- `week14/`: Davidson-based RANS closure module (separate environment; see its README).
- `week15/`: geometry-aware operator audit on 130 OpenFOAM step-flow fields with whole-geometry and family holdouts (see its README for data).

The Week-9 evidence contract is intentionally asymmetric. Lab 1 uses real
micro-step DSMC fields under a specific author publication permission, and its
coordinate MLP is a new independently trained teaching baseline. Lab 2 uses
real public nozzle DSMC full fields and centerlines under CC BY 4.0, reproduces
the 15-case POD audit, and generates a fresh six-output full-field result.

The Week-8 labs are synchronized to checksummed evidence from the author's
`GasDynamicsSciML` repository. Lab 1 links to all nine detailed classical
notebooks in `Introduction-to-Compressible-Flows`; Lab 2 keeps the full model
retraining optional and uses immutable CSV evidence for the classroom path.
The SU2 diamond-airfoil work is a clearly labelled multidimensional-CFD bridge,
not an accepted nine-case dataset: only the sharp-wall alpha-zero Euler case is
a qualified teaching reference at the frozen source commit.

The Week-7 notebook labels the retained low-cost run as qualitative. It links
every field plot to force, density, Mach-number, relaxation-time, blockage, and
reference diagnostics. Before ML, students audit a retained `Re=100`
three-grid sequence (`D/dx=12,18,27`) with fixed nondimensional physics and
statistical convergence. Fine-pair tolerances pass, but the formal
asymptotic/GCI gate fails and is retained. The subsequent D40 release decision
is recorded in [evidence status](../qa/REMAINING_EVIDENCE.md); a waived release
gate does not establish asymptotic grid convergence.
Machine-readable retained evidence is in `../results/cylinder_lbm/`.
The executed 1080p complete-Re blind animation and its field-error/baseline
evidence are in `../results/cylinder_ml/`.
The corrected one-step CNN, cubic-extrapolation comparison, downstream
diagnostics, and failed 50-step recursive audit are retained separately in
`../results/cylinder_cnn/`.

The Week-7.1 notebook is intentionally additive. It does not mix the continuum
LBM labels with the rarefied DSMC archive. The 1.4 GB source ZIP is reduced by a
deterministic, checksummed builder to 44,500 finite teaching points. The retained
result is scientifically useful precisely because the direct Mach-field
interpolation baseline outperforms the trained 3x96 tanh MLP on all six aggregate
field comparisons. See the [paper-to-course audit](../qa/WEEK71_PAPER_PARITY.md)
for historical split errors, normalization caveats and the separate requirements
for full paper reproduction.

The additive `week04/W4_1_Classical_ROM_Cavity.ipynb` lab comes after the
original Week-4 sequence without modifying it. It implements dynamic centered
POD--Galerkin and POD--DEIM for the same lid-driven cavity, validates the added
snapshot FOM against the fixed 65x65 archive and Ghia centerlines, performs grid
and time-step refinement, freezes rank on `Re=300`, opens `Re=175,275,375` once,
and reports offline/online timing plus break-even query count. Its executed
machine-readable evidence is in `../results/cavity_rom/`.

The executed Week-4 operator result is stored in `../results/pod_deeponet/`. Start with `week04/W4_Lab3_DeepONet_Cavity_Student.ipynb`; set its regeneration switches only after reading the frozen protocol. The notebook distinguishes the valid advantage (fast repeated full-field inference with retained benchmark fidelity) from the invalid claim that a neural surrogate makes Ghia data more accurate.

Continue with `week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb` to test an actual
Stokes field as a low-fidelity input. Its retained evidence is in
`../results/stokes_refined/`; the original test cases are regression tests, not
a newly blind benchmark.

The Week-1 cavity notebook first reproduces the manuscript's Ghia velocity and Botella--Peyret pressure validations. The Week-3 DSMC notebook first validates the executed HS--NTC solver directly against Mohammadzadeh wall-pressure data. Both store paper-ready PNG/PDF files and metric JSON under `../results/article_figures/`. See [`../ARTICLE_FIGURE_MAP.md`](../ARTICLE_FIGURE_MAP.md) for every notebook-to-figure contract.

## Research-project notebooks

- `week05_06/P0_Project_Setup.ipynb`: environment, dataset audit, baseline recovery, and project card.
- `week05_06/P1_Re_Generalization.ipynb`: Reynolds interpolation/extrapolation and failure localization.
- `week05_06/P2_Physics_Guided_DNN.ipynb`: wall-weighted or divergence-penalized learning.
- `week05_06/P3_POD_Study.ipynb`: POD rank, basis choice, and coefficient learnability.
- `week05_06/P4_Uncertainty_Study.ipynb`: seed variability, data sufficiency, and error indicators.
- `week05_06/P5_Rarefied_Cavity.ipynb`: noisy particle labels and Knudsen generalization.
- `week05_06/P6_FP_Cavity_Closure.ipynb`: offline and closed-loop Fokker–Planck closure evaluation.

Each project notebook includes:

- prerequisites and a concept map;
- a reproducible repository bootstrap;
- physical and mathematical definitions before code;
- prediction prompts before decisive computations;
- validation-only model-selection rules;
- a visible blind-test gate;
- metric interpretation and common failure modes;
- troubleshooting guidance;
- required deliverables and a report outline; and
- track-specific further reading.

Run notebooks in order. Restart and run all before submission. A notebook with stale out-of-order state is not a reproducible result.

## Modal companions

- [Week 5 sparse sensing](week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb)
- [Week 7 modal forecasting](week07/W7_Lab2_Modal_Forecasting.ipynb)
- [Retained protocol, metrics and figures](../results/modal_labs/README.md)
