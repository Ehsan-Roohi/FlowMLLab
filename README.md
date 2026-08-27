# FlowMLLab

[![FlowMLLab CI](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22074237.svg)](https://doi.org/10.5281/zenodo.22074237)

**FlowMLLab** is an open-source framework for reproducible CFD-to-scientific-machine-learning experiments. It integrates transparent continuum and particle solvers, case-wise data partitions, non-neural baselines, coordinate networks, POD-DeepONet models, physical diagnostics, machine-readable evidence, and release checks.

The repository also contains the complete tutorial and lecture layer developed for the six-week graduate course **MIE 690A: AI in Fluid Mechanics**, University of Massachusetts Amherst, Summer 2026. The reusable modules, datasets, validators, and figure builders are the software core; the notebooks are documented examples of that framework.

The course treats scientific machine learning as a controlled computational-physics experiment:

1. generate or audit numerical data;
2. define a physically meaningful learning problem;
3. separate development, validation, and blind physical cases;
4. compare with a transparent non-neural baseline;
5. evaluate both statistical and physical fidelity; and
6. retain failure modes, configuration, and machine-readable evidence.

Start with [START_HERE.md](START_HERE.md). It gives the installation check, recommended order, expected runtimes, and a first 20-minute validation exercise.

## Workshops, support, and consulting

FlowMLLab remains free and open source. Optional professional services are available for universities, research laboratories, instructors, and engineering teams:

- **Live workshops:** a two-hour introduction, a one-day intensive, or a multi-session program covering validated CFD, scientific machine learning, POD--DeepONet, and DSMC.
- **Technical onboarding and support:** environment setup, benchmark reproduction, dataset qualification, notebook adaptation, and troubleshooting.
- **Research consulting:** design and review of CFD-to-SciML workflows, physical validation strategies, neural-operator studies, and rarefied-flow applications.
- **Custom extensions:** integration of new physical cases, institutional datasets, validation targets, or research-lab workflows.

For workshop, support, or consulting inquiries, contact [Ehsan Roohi](mailto:roohie@umass.edu?subject=FlowMLLab%20workshop%2C%20support%2C%20or%20consulting). These services are optional and do not affect access to the MIT-licensed software or educational materials.

If FlowMLLab supports your teaching or research, you can also support its continued open-source maintenance through the **Sponsor** button at the top of this repository.

## What is included

| Resource | Contents |
| --- | --- |
| `flowmllab/` | Installable Python package, scientific asset checks, repository QA, and figure-generation CLI |
| `pyproject.toml` | Versioned package metadata, locked core dependencies, optional ML/test environments, and `flowmllab` entry point |
| `lectures/` | Five lecture/guide PDFs covering Weeks 1–6, plus editable LaTeX/PPTX sources where available |
| `notebooks/week01`–`week04` | Ten guided laboratories: Python, TensorFlow, validated cavity CFD, supervised learning, rarefaction, Maxwellian sampling, DSMC, scalar/field surrogates, POD-DeepONet, and an additive classical cavity-ROM study |
| `notebooks/P0`–`P6` | Seven expanded research-project notebooks with conceptual notes, frozen decision gates, physical diagnostics, troubleshooting, deliverables, and further reading |
| `common/` | Shared CFD, surrogate, POD, kinetic, and QA utilities |
| `data/` | Fixed 11-case cavity reference dataset and numerical-quality table |
| `results/article_validation/` | Re=1000 pressure-recovery solutions and independent Botella--Peyret reference data |
| `results/dsmc_validation/` | Four HS--NTC wall-pressure solutions and Mohammadzadeh Fig. 3 DSMC markers |
| `results/article_figures/` | Paper-ready PNG/PDF validation figures and machine-readable error summaries |
| `results/pod_deeponet/` | Model-selection, blind-case, Ghia-centerline, timing, and full-field POD-DeepONet evidence |
| `results/cavity_rom/` | FOM reproduction and refinement, leakage-free POD/DEIM selection, blind trajectories, portable model, timing, break-even count, and summary figure |
| `advanced/fp_closure/` | Bounded educational workflow for exact and learned Fokker–Planck closure testing |
| `references/` | Annotated reading guide and BibTeX database |
| `qa/` | Release validator for notebook syntax, required assets, and reproducibility anchors |

## Tutorial and example sequence

The recommended path is cumulative:

- **Week 1 — Numerical foundations:** Python/NumPy/TensorFlow fundamentals, finite differences, residuals, and validation of lid-driven-cavity centerlines against Ghia et al.
- **Week 2 — Supervised learning and model validity:** features, targets, scaling, losses, optimization, case-wise splits, interpolation versus extrapolation, and rarefied-flow nondimensionalization.
- **Week 3 — Particle and kinetic descriptions:** Maxwellian sampling, macroscopic moments, sampling-error scaling, DSMC algorithmic structure, noisy labels, and averaging.
- **Week 4 — Surrogates and operator learning:** audited CFD fields, scalar baselines, coordinate DNNs, and a restricted POD-DeepONet with complete-case selection, three-seed blind tests, Ghia validation, physical diagnostics, and measured inference cost.
- **Week 4.1 — Classical and hyper-reduced ROM:** an additive notebook for the same cavity, with dynamic centered POD-Galerkin, nonlinear-cost diagnosis, POD-DEIM, convergence checks, frozen blind tests, and offline/online break-even accounting.
- **Weeks 5–6 — Controlled research project:** select one track—including POD, physics-guided learning, uncertainty, rarefied flow, or learned closure—freeze the protocol, open blind cases once, document a failure/tradeoff, and produce a reproducible research summary.

The full module-to-evidence mapping is in [COURSE_MAP.md](COURSE_MAP.md).
The exact manuscript-figure ownership and reproduction commands are in [ARTICLE_FIGURE_MAP.md](ARTICLE_FIGURE_MAP.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"       # core framework and QA
flowmllab smoke --root .
flowmllab qa --root .
flowmllab figures all --root .
flowmllab rom --root .               # optional Week-4.1 full regeneration
```

Install the neural-network stack with `python -m pip install -e ".[ml,test]"`.
The exact core and ML dependency versions are recorded in `pyproject.toml` and
`requirements.txt`. The `flowmllab` commands are the supported programmatic
entry points; notebooks are tutorials that call the same shared modules.

For Google Colab, upload the notebook and the files named in its **Required files** section. Project notebooks also contain a repository bootstrap cell, so they can be launched from this checkout without manually copying `common/` or `data/` files.

TensorFlow is needed for the neural-network training cells. The numerical audit, interpolation, POD, data checks, and notebook syntax validation can be run without a GPU. Track 6 requires a CUDA-capable runtime for its final closed-loop run; its fast-mode configuration is only a smoke test, not research-resolution evidence.

## Common benchmark and data contract

The shared dataset contains full `u`, `v`, `p`, `psi`, and `omega` fields on a fixed grid for

`Re = 100, 150, 175, 200, 225, 250, 275, 300, 350, 375, 400`.

The streamfunction–vorticity solver is an educational reference. Re = 100 and 400 include Ghia centerline checks; all cases retain numerical residuals and a zero-mean pressure gauge. A smoother surrogate is not automatically more physically accurate than the CFD labels.

The reference dataset SHA-256 is recorded in `common/reproducibility.txt` and checked by the release validator.

The Week-1 cavity notebook opens with both velocity and pressure validation. The Week-3 DSMC notebook opens with a direct comparison of the package solver against Mohammadzadeh *et al.* at the same $Re=1.5$, $Kn=0.1$, $Ma=0.09$ condition. Both notebooks generate the paper-facing figures through `common/article_validation.py`; optional switches expose full numerical generation.

## Classical cavity-ROM validation result

Open `notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb` for the additive
post-Week-4 lab, or run `flowmllab rom --root .` to regenerate
its frozen evidence.  The snapshot-enabled FOM reproduces the accepted
`Re=100` and `400` archive fields to below `8e-16` relative error and retains
the existing Ghia checks.  Independent grid and time-step refinements decrease
monotonically.

A development-only rule at `Re=300` selects POD rank 16 and DEIM dimension 16.
Across untouched `Re=175,275,375` trajectories, maximum-in-time velocity error
is **0.4941%** for POD--Galerkin and **0.6338%** for POD--DEIM; final vorticity
error remains below **0.56%**, wall error is exactly zero, and discrete
divergence is near round-off.  In the recorded CPU run, standard POD--Galerkin
does not materially accelerate this efficient small-grid FOM because it still
evaluates the full nonlinear field, whereas POD--DEIM is about **9.2x** faster.
Including offline snapshot and basis cost gives a recorded break-even near
**8 queries**.
Exact values and environment metadata are in `results/cavity_rom/`.

## POD-DeepONet validation result

Run `python common/run_pod_deeponet_validation.py` to reproduce the CPU study. The development-only rule selects a rank-3 POD trunk and a `(32,32)` tanh branch. The untouched `Re = 175, 275, 375` fields have three-seed ensemble velocity errors of **0.4525%**, **0.0718%**, and **0.0928%**. Wall error is exactly zero by the declared output transform and the discrete divergence norm remains at round-off because the trunk is built from divergence-free CFD snapshots.

This result does **not** claim that the network improves the Ghia benchmark. At `Re = 100` and `400`, POD-DeepONet preserves the centerline fidelity of the educational CFD solver. Its demonstrated advantage is amortized field evaluation after training: approximately 0.8 ms for the three-seed ensemble versus approximately 8 s for a fresh CPU CFD solve in the recorded run. Exact machine-readable values, protocol, seeds, fields, and the comparison figure are in `results/pod_deeponet/`.

## Rules that apply to every project

- Split complete physical cases when the claim concerns a new Reynolds number, Knudsen number, wall speed, seed, or operating condition.
- Fit scalers and select models using development data only.
- Freeze architecture, rank, loss weights, stopping rules, and rejection thresholds before blind testing.
- Compare neural models with a transparent baseline using the same allowed information.
- Report local physics diagnostics alongside aggregate error norms.
- Treat a negative result as valid when the comparison is controlled and reproducible.
- Do not describe reduced teaching budgets as production convergence evidence.

## Student work and permissions

This repository contains instructor-developed teaching materials, common numerical assets, and starter notebooks. Student submissions are not included. Any student result reproduced in the associated manuscript remains subject to explicit student permission and attribution.

## Associated manuscript

**FlowMLLab: An open-source framework for reproducible computational-fluid-dynamics and scientific-machine-learning experiments**

The manuscript prepared for *AI Thermal Fluids* documents the software architecture,
numerical and ML workflow, validation contract, representative continuum and kinetic
applications, classical reduced-order-model baselines, limitations, and reuse pathway.

## Citation and contact

Use [CITATION.cff](CITATION.cff) when citing the release.

Version 1.1.0 is the current citable release. Its version-specific Zenodo DOI is
recorded in [CITATION.cff](CITATION.cff) and here after the archival record is
published. The GitHub release and archival record contain only FlowMLLab. They do
not include materials from any separate research repository.

Ehsan Roohi  
Department of Mechanical and Industrial Engineering  
University of Massachusetts Amherst  
roohie@umass.edu

Copyright © 2026 Ehsan Roohi. Released under the MIT License; see [LICENSE](LICENSE).
