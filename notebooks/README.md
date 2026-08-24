# Notebook guide

## Weekly laboratories

- `week01/`: Python, TensorFlow, and validated continuum cavity CFD.
- `week02/`: supervised-learning foundations and a rarefied-flow surrogate.
- `week03/`: Maxwellian sampling/noise and a mini DSMC cavity.
- `week04/`: CFD data production, scalar/field surrogates, and a 27-cell POD-DeepONet laboratory with development-only selection, a visible blind-test gate, three-seed uncertainty, Ghia checks, physical diagnostics, and timing.

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
