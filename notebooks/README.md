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
| Week 3 | Maxwellian noise and ML | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) |
| Week 3 | Mini DSMC cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) |
| Week 4 | CFD data production | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) |
| Week 4 | Scalar and field surrogates | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) |
| Week 4 | POD-DeepONet cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) |
| Week 4.1 | Classical POD-Galerkin/POD-DEIM | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) |
| Week 5 | Lattice-Boltzmann cylinder wakes | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05/W5_Lattice_Boltzmann_Cylinder_Student.ipynb) |
| Project | P0 setup and data audit | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P0_Project_Setup.ipynb) |
| Project | P1 Reynolds-number generalization | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P1_Re_Generalization.ipynb) |
| Project | P2 physics-guided DNN | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P2_Physics_Guided_DNN.ipynb) |
| Project | P3 POD study | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P3_POD_Study.ipynb) |
| Project | P4 uncertainty and data sufficiency | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P4_Uncertainty_Study.ipynb) |
| Project | P5 rarefied cavity | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P5_Rarefied_Cavity.ipynb) |
| Project | P6 Fokker-Planck closure | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/P6_FP_Cavity_Closure.ipynb) |

## Weekly laboratories

- `week01/`: Python, TensorFlow, and validated continuum cavity CFD.
- `week02/`: supervised-learning foundations and a rarefied-flow surrogate.
- `week03/`: Maxwellian sampling/noise and a mini DSMC cavity.
- `week04/`: CFD data production, scalar/field surrogates, and a 27-cell POD-DeepONet laboratory with development-only selection, a visible blind-test gate, three-seed uncertainty, Ghia checks, physical diagnostics, and timing.
- `week05/`: D2Q9 BGK/TRT cylinder flow, force and Strouhal validation, Reynolds-regime classification, and a complete-Re holdout comparison between harmonic POD and a small neural POD model.

The Week-5 notebook labels the retained low-cost run as qualitative.  It links
every field plot to force, density, Mach-number, relaxation-time, blockage, and
reference diagnostics, and provides a separate qualification/refinement path.
Machine-readable retained evidence is in `../results/cylinder_lbm/`.

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

- `P0_Project_Setup.ipynb`: environment, dataset audit, baseline recovery, and project card.
- `P1_Re_Generalization.ipynb`: Reynolds interpolation/extrapolation and failure localization.
- `P2_Physics_Guided_DNN.ipynb`: wall-weighted or divergence-penalized learning.
- `P3_POD_Study.ipynb`: POD rank, basis choice, and coefficient learnability.
- `P4_Uncertainty_Study.ipynb`: seed variability, data sufficiency, and error indicators.
- `P5_Rarefied_Cavity.ipynb`: noisy particle labels and Knudsen generalization.
- `P6_FP_Cavity_Closure.ipynb`: offline and closed-loop Fokker–Planck closure evaluation.

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
