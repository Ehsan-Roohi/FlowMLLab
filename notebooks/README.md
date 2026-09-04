# Notebook guide

Each notebook opens directly from GitHub, clones the complete FlowMLLab release,
and installs the tested package in its first code cell. Choose a Colab GPU runtime
only for notebooks that explicitly require CUDA.

## One-click Colab launcher

| Module | Notebook | Launch |
| --- | --- | --- |
| Week 1 | Python for CFD and AI | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| Week 1 | TensorFlow for AI in fluids | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/02_tensorflow_for_ai_fluids.ipynb) |
| Week 1 | Cavity CFD and Ghia validation | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |
| Week 2 | Supervised learning and rarefaction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) |
| Week 2.1 | Probabilistic UQ for CFD surrogates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) |
| Week 3 | Maxwellian noise and ML | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) |
| Week 3 | Mini DSMC cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) |
| Week 4 | CFD data production | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) |
| Week 4 | Scalar and field surrogates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) |
| Week 4 | POD-DeepONet cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) |
| Week 4.1 | Classical POD-Galerkin/POD-DEIM | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) |
| Weeks 5–6 | P0 setup and data audit | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Weeks 5–6 | P1 Reynolds-number generalization | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P1_Re_Generalization.ipynb) |
| Weeks 5–6 | P2 physics-guided DNN/PINN objectives | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P2_Physics_Guided_DNN.ipynb) |
| Weeks 5–6 | P3 POD study | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P3_POD_Study.ipynb) |
| Weeks 5–6 | P4 uncertainty and data sufficiency | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P4_Uncertainty_Study.ipynb) |
| Weeks 5–6 | P5 rarefied cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P5_Rarefied_Cavity.ipynb) |
| Weeks 5–6 | P6 Fokker-Planck closure | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) |
| Week 7 | Lattice-Boltzmann cylinder wakes | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) |
| Week 8 | Exact gas dynamics before ML | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| Week 8 | Gas-dynamics SciML evidence | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) |
| Week 9 | Micro-step zonal-loss DeepONet | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) |
| Week 9 | Shock-aligned micro-nozzle DeepONet | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb) |
| Week 10 | DSMC cavity and mono/diatomic shock reproduction | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) |

## Weekly laboratories

- `week01/`: Python, TensorFlow, and validated continuum cavity CFD.
- `week02/`: supervised-learning foundations and a rarefied-flow surrogate.
- `week02_1/`: the incremental probabilistic-UQ lecture/lab pair. It connects Gaussian observation models, exact Bayesian regression, POD--Gaussian-process fields, proper scores, validation scaling, and blind under-coverage between Weeks 2 and 3.
- `week03/`: Maxwellian sampling/noise and a mini DSMC cavity.
- `week04/`: CFD data production, scalar/field surrogates, and a 27-cell POD-DeepONet laboratory with development-only selection, a visible blind-test gate, three-seed uncertainty, Ghia checks, physical diagnostics, and timing.
- `week05_06/`: the original combined two-week guided-project pack. Week 5 establishes the setup, baseline, controlled modification, and checkpoint; Week 6 completes the same selected track, physical validation, reproducibility package, and final report.
- `week07/`: D2Q9 BGK/TRT cylinder flow, a concise collide--stream--boundary algorithm walkthrough, physically gated force and Strouhal diagnostics, an executed three-grid study with a retained formal asymptotic/GCI failure, Reynolds-regime classification, strong temporal baselines, and separate one-step/recursive audits of a four-frame multi-scale CNN on a retained held-out interpolation case.
- `week08/`: two CPU labs that start from exact branch-aware gas dynamics, then compare bracketed roots, interpolation, and physics-guided MLP evidence across five inverse problems, edge holdouts, dimensional scaling, and application workloads.
- `week09/`: two CPU research-to-classroom labs based on the Roohi--Mahdavi micro-step and micro-nozzle studies. The first uses two author-permitted, checksummed derivatives of nine real DSMC height fields with file-level case separation; the second reproduces shock-centered POD and fresh full-field predictions from checksummed derivatives of 15 public DSMC snapshots before evaluating three held-out pressures.
- `week10/`: one complete CPU article-reproduction lab with 14 rarefied-cavity fields, mono/diatomic shock profiles, the DSMC algorithm, full provenance audit, log-Knudsen synthesis, POD-branch operators, physical interpretation, and retained numerical gates.

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

The Week-7 notebook labels the retained low-cost run as qualitative.  It links
every field plot to force, density, Mach-number, relaxation-time, blockage, and
reference diagnostics. Before ML, students audit a retained `Re=100`
three-grid sequence (`D/dx=12,18,27`) with fixed nondimensional physics and
statistical convergence. Fine-pair tolerances pass, but the formal
asymptotic/GCI gate fails and is retained; `D/dx=40` is the next declared run.
Machine-readable retained evidence is in `../results/cylinder_lbm/`.
The executed 1080p complete-Re blind animation and its field-error/baseline
evidence are in `../results/cylinder_ml/`.
The corrected one-step CNN, cubic-extrapolation comparison, downstream
diagnostics, and failed 50-step recursive audit are retained separately in
`../results/cylinder_cnn/`.

The additive `week04/W4_1_Classical_ROM_Cavity.ipynb` lab comes after the
original Week-4 sequence without modifying it.  It implements dynamic centered
POD--Galerkin and POD--DEIM for the same lid-driven cavity, validates the added
snapshot FOM against the fixed 65x65 archive and Ghia centerlines, performs grid
and time-step refinement, freezes rank on `Re=300`, opens `Re=175,275,375` once,
and reports offline/online timing plus break-even query count.  Its executed
machine-readable evidence is in `../results/cavity_rom/`.

The executed Week-4 operator result is stored in `../results/pod_deeponet/`. Start with `week04/W4_Lab3_DeepONet_Cavity_Student.ipynb`; set its regeneration switches only after reading the frozen protocol. The notebook distinguishes the valid advantage—fast repeated full-field inference with retained benchmark fidelity—from the invalid claim that a neural surrogate makes Ghia data more accurate.

The Week-1 cavity notebook first reproduces the manuscript's Ghia velocity and Botella--Peyret pressure validations. The Week-3 DSMC notebook first validates the executed HS--NTC solver directly against Mohammadzadeh wall-pressure data; the earlier empty digitization exercise has been removed. Both store paper-ready PNG/PDF files and metric JSON under `../results/article_figures/`. See [`../ARTICLE_FIGURE_MAP.md`](../ARTICLE_FIGURE_MAP.md) for every notebook-to-figure contract.

## Research-project notebooks

- `week05_06/P0_Project_Setup.ipynb`: environment, dataset audit, baseline recovery, and project card.
- `week05_06/P1_Re_Generalization.ipynb`: Reynolds interpolation/extrapolation and failure localization.
- `week05_06/P2_Physics_Guided_DNN.ipynb`: wall-weighted or divergence-penalized learning.
- `week05_06/P3_POD_Study.ipynb`: POD rank, basis choice, and coefficient learnability.
- `week05_06/P4_Uncertainty_Study.ipynb`: seed variability, data sufficiency, and error indicators.
- `week05_06/P5_Rarefied_Cavity.ipynb`: noisy particle labels and Knudsen generalization.
- `week05_06/P6_FP_Cavity_Closure.ipynb`: offline and closed-loop Fokker–Planck closure evaluation.

Each project notebook now includes:

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
