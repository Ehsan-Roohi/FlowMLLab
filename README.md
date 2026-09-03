# FlowMLLab

[![FlowMLLab CI](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml/badge.svg)](https://github.com/Ehsan-Roohi/FlowMLLab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22126785.svg)](https://doi.org/10.5281/zenodo.22126785)

## Validated CFD to scientific ML—from first Colab to reproducible benchmark

Start with a physics-checked circular-cylinder teaching wake, compare LBM with an
autonomous phase-stable surrogate on a fresh unseen Reynolds case, and verify the evidence chain. Continue through the 20-minute cavity
exercise to physics-checked POD--DeepONet benchmarks when ready.

<p align="center">
  <a href="https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb"><img src="https://img.shields.io/badge/Run-20--minute_Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Run the 20-minute FlowMLLab Colab"></a>
  <a href="demo/README.md"><img src="https://img.shields.io/badge/Explore-blind--case_demo-146C94" alt="Explore the validated blind-case demo"></a>
  <a href="notebooks/README.md"><img src="https://img.shields.io/badge/Open-all_18_notebooks-315A7D" alt="Open all 18 FlowMLLab notebooks"></a>
  <a href="lectures/README.md"><img src="https://img.shields.io/badge/Open-all_7_lectures-6B4C9A" alt="Open all 7 FlowMLLab lectures"></a>
</p>

## Featured: circular-cylinder vortex shedding

The optional Week 7 extension advances from the steady cavity to an unsteady circular-cylinder
wake. A D2Q9 lattice-Boltzmann solver generates the educational CFD labels. A phase-stable
learned decoder is initialized from four true fields and then predicts all 277 future
fields of the fresh unseen `Re=95` case with no future CFD input.

The learned cylinder models are evaluated against the retained educational
`D/dx=12` LBM labels. A separate `Re=100` three-grid study audits the CFD itself;
it passes statistical and practical fine-pair gates but does **not** establish a
positive-order asymptotic/GCI sequence.

<p align="center">
  <img src="results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.webp" alt="Animated fresh Re=95 comparison of circular-cylinder LBM vorticity and an autonomous phase-stable learned decoder" width="100%">
</p>

<p align="center"><em>The comparison plays automatically and loops. Relative to the educational LBM labels, the 277 autonomous future frames have 4.281% global vorticity relative L2 error, 5.194% worst-frame error, and 0.264% Strouhal error.</em><br><a href="results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4">Open the full-resolution MP4</a> · <a href="results/cylinder_phase/phase_stable_validation.png">Inspect validation and spectra</a></p>

### Cavity benchmark

<p align="center">
  <a href="demo/README.md">
    <img src="assets/flowmllab_blind_demo.gif" alt="Animated comparison of retained FlowMLLab blind POD-DeepONet predictions and validated CFD fields" width="100%">
  </a>
</p>

<p align="center"><em>Three retained test cases from the versioned evidence. Velocity and zero-mean pressure are direct outputs of separate POD--DeepONet heads.</em></p>

| Audited result | Retained evidence / decision |
| --- | ---: |
| ML-independent `Re=100` LBM grid study | `D/dx=12,18,27`; all statistical gates **PASS**; formal asymptotic/GCI gate **FAIL** |
| Fine-pair CFD sensitivity, `D/dx=18→27` | $\overline C_D$: **2.142%**; $St$: **0.512%**; $L_r/D$: **1.361%** — all declared practical limits pass |
| Finest CFD versus independent 2-D reference bands | $\overline C_D$ **10.335% above** the upper bound; $St$ **6.908% above** the upper bound; $L_r/D$ **inside** its band |
| Fresh unseen `Re=95`, 277-frame autonomous vorticity error versus educational LBM labels | **4.281%** global; **5.194%** worst frame |
| Fresh unseen `Re=95` Strouhal error versus educational LBM labels | **0.264%** |
| Retained `Re=105` one-step cylinder vorticity error versus educational LBM labels | **0.815%** relative $L_2$; cubic baseline **1.053%** |
| Mean cylinder-wake profile error versus educational LBM labels at `2D,4D,6D,8D` | **0.804%** relative $L_2$ |
| Three-seed velocity error on retained cavity cases | **0.0727%–0.4455%** relative $L_2$ |
| Direct zero-mean pressure prediction | **0.1073%–0.2506%** relative $L_2$ |
| Repeated three-field evaluation versus a fresh recorded CPU CFD solve | **approximately $5.3\times10^3\times$ faster** |
| Reproducible learning and research entry points | **18 Colab notebooks + 7 lectures** (6 PDFs; Weeks 5–6 share one guide) |

### Choose a starting point

| Your goal | Start here |
| --- | --- |
| See the featured result immediately | [Play the fresh `Re=95` autonomous cylinder video](results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4) |
| Complete the first evidence chain | [Run the 20-minute Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Reproduce the software release | Follow [START_HERE.md](START_HERE.md), then run `flowmllab qa --root .` |
| Adopt material in a course or workshop | Use [COURSE_MAP.md](COURSE_MAP.md) and the [notebook launcher](notebooks/README.md) |
| Contribute a bounded improvement | Review [ROADMAP.md](ROADMAP.md) and [CONTRIBUTING.md](CONTRIBUTING.md) |

**FlowMLLab** is an open-source framework for reproducible CFD-to-scientific-machine-learning experiments. It integrates transparent continuum and particle solvers, case-wise data partitions, non-neural baselines, coordinate networks, POD-DeepONet models, physical diagnostics, machine-readable evidence, and release checks.

The repository contains a seven-lecture learning path: the complete tutorial and lecture layer developed for the original six-week graduate course **MIE 690A: AI in Fluid Mechanics**, University of Massachusetts Amherst, Summer 2026, plus the Week-7 cylinder extension. The reusable modules, datasets, validators, and figure builders are the software core; the notebooks are documented examples of that framework.

The course treats scientific machine learning as a controlled computational-physics experiment:

1. generate or audit numerical data;
2. define a physically meaningful learning problem;
3. separate development, validation, and blind physical cases;
4. compare with a transparent non-neural baseline;
5. evaluate both statistical and physical fidelity; and
6. retain failure modes, configuration, and machine-readable evidence.

Start with [START_HERE.md](START_HERE.md). It gives the installation check, recommended order, expected runtimes, and a first 20-minute validation exercise.

## What is included

| Resource | Contents |
| --- | --- |
| `flowmllab/` | Installable Python package, scientific asset checks, repository QA, and figure-generation CLI |
| `demo/` | Read-only Streamlit explorer for the retained POD--DeepONet blind cases |
| `pyproject.toml` | Versioned package metadata, bounded compatible dependencies, optional ML/test environments, and `flowmllab` entry point |
| `lectures/` | Seven lectures delivered in six PDFs: Weeks 1–4, a shared Weeks 5–6 project guide, and the separate Week-7 cylinder lecture, with editable sources where available |
| `notebooks/week01`–`week04` | Ten guided laboratories for Weeks 1–4, including the additive Week-4.1 classical ROM lab |
| `notebooks/week05_06` | The original combined Weeks 5–6 project pack: seven expanded notebooks (`P0`–`P6`) with PINNs/physics-guided learning, POD, uncertainty, rarefied flow, FP closure, frozen decision gates, and final-project evidence |
| `notebooks/week07` | The Week-7 cylinder-wake LBM and neural-surrogate extension |
| `common/` | Shared CFD, surrogate, POD, kinetic, and QA utilities |
| `data/` | Fixed cavity reference data plus the documented, checksummed cylinder-CFD release contract |
| `results/article_validation/` | Re=1000 pressure-recovery solutions and independent Botella--Peyret reference data |
| `results/dsmc_validation/` | Four HS--NTC wall-pressure solutions and Mohammadzadeh Fig. 3 DSMC markers |
| `results/article_figures/` | Paper-ready PNG/PDF validation figures and machine-readable error summaries |
| `results/pod_deeponet/` | Model-selection, blind-case, Ghia-centerline, timing, and full-field POD-DeepONet evidence |
| `results/cavity_rom/` | FOM reproduction and refinement, leakage-free POD/DEIM selection, blind trajectories, portable model, timing, break-even count, and summary figure |
| `results/cylinder_ml/` | archived Re=100 phase/POD failure baseline and its diagnostics |
| `results/cylinder_cnn/` | four-frame CNN validation/retained-interpolation video, strong polynomial baselines, one-step and rollout audits, frozen weights, and regeneration script |
| `results/cylinder_phase/` | autonomous phase-stable validation, fresh-test metrics, Strouhal spectra, and LBM comparison video |
| `results/cylinder_grid_convergence/` | three-grid Re=100 CFD fields, histories, statistical gates, retained formal grid-independence failure, protocol, and convergence figure |
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
- **Weeks 5–6 — Combined guided-project pack:** Week 5 covers POD, physics-guided objectives/PINNs, neural-operator extensions, project selection, and the frozen checkpoint. Week 6 continues the same selected track through Fokker–Planck/hybrid methods where assigned, a-posteriori testing, uncertainty, reproducibility, and the final report.
- **Week 7 — Cylinder wakes with LBM and learned prediction:** derive the D2Q9 collide--stream--boundary loop, distinguish wake regimes, run a controlled `Re=100` three-grid study and retain its formal asymptotic failure, validate forces and gated Strouhal estimates, retain POD/CNN failure evidence, and audit a phase-stable decoder over 277 autonomous future fields on a fresh Reynolds test.

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
flowmllab cylinder --root .          # verify Week-7 regimes, grid study, and learned evidence
```

Install the neural-network stack with `python -m pip install -e ".[ml,test]"`.
The exact core and ML dependency versions are recorded in `pyproject.toml` and
`requirements.txt`. The `flowmllab` commands are the supported programmatic
entry points; notebooks are tutorials that call the same shared modules.

Every public notebook now has an **Open in Colab** badge and a first-code-cell
bootstrap that clones this repository and installs the tested package automatically.
Use the [one-click notebook launcher](notebooks/README.md); no manual upload of
`common/`, `data/`, or package files is required.

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

## Cylinder LBM teaching module

Week 7 adds an installable D2Q9 cylinder solver with transparent BGK theory and
a more robust TRT default, Zou--He inflow, a convective outlet, periodic
transverse boundaries, Bouzidi interpolated bounce-back on an analytical
circular wall, momentum-exchange forces, gauge
pressure, vorticity, recirculation length, and Strouhal diagnostics.  The
retained quick sweep covers `Re = 5, 20, 40, 100, 180`, so students see attached
flow, a steady symmetric wake, and periodic vortex shedding before using ML.

Open `notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb` for the full
lesson. It keeps complete Reynolds cases together. The original phase/POD
surrogate is retained as a failure baseline because its downstream vortices
diffuse. The corrected path uses four consecutive LBM fields, a three-scale
fully convolutional residual predictor, and field/gradient/vorticity/divergence
losses. The design follows the predictive structure of Lee & You (JFM 2019)
without claiming a reproduction of their research-scale study.

The archived POD comparison used `Re=60,80,90,110,120,140` and withheld
`Re=100` from fitting. Once its diffuse wake was inspected to redesign the
method, `Re=100` correctly became validation rather than blind. Its 16.70%
vorticity error is retained rather than hidden.

[![Archived Re=100 POD failure comparison](results/cylinder_ml/blind_re100_lbm_vs_neural_poster.png)](results/cylinder_ml/blind_re100_lbm_vs_neural.mp4)

The corrected CNN was selected with complete-case `Re=100` validation and then
tested on the historically frozen `Re=105` interpolation case. That case is now
open and is retained evidence, not a fresh blind case for future tuning. With
dimensionless frame spacing `dt*=0.1042`, its one-step vorticity error is
**0.815%**, versus **1.053%** for the strongest matched non-neural baseline
(cubic extrapolation) and **11.01%** for persistence. Mean downstream
vorticity-profile error at `2D,4D,6D,8D` is **0.804%** versus **1.581%** for
cubic extrapolation; complex space-time spectral incoherence is **0.00313%**
versus **0.01273%**. Station enstrophy ratios remain within `0.9993–1.0037`.

[Play the full retained `Re=105` LBM-versus-CNN video](results/cylinder_cnn/re105_lbm_vs_multiscale_cnn.mp4).

This is a teacher-forced one-step prediction from four previous true LBM
frames, not an autonomous rollout. A separate recursive audit is retained: the
CNN mean vorticity error is **8.814%** at 10 steps, crosses the declared 15%
gate at 20 steps (**20.222%**), and reaches **90.724%** at 50 steps. The
one-step result is therefore claimed as low-error; long autonomous rollout is
explicitly a failed gate for that architecture.

[Inspect the retained rollout audit](results/cylinder_cnn/re105_retained_rollout.png).

The long-horizon correction is a separate phase-stable learned decoder, not a
renaming of the one-step CNN. Harmonic order is selected only on complete-case
`Re=100` validation. After freezing that choice, `Re=95` is opened once as the
fresh test and `Re=105` remains a retained historical test. Starting from four
true fields, the decoder produces 277 future fields with zero future CFD inputs.
All declared gates pass: global/worst-frame vorticity errors are **6.037%/7.959%**
on validation, **4.281%/5.194%** on the fresh test, and **4.625%/6.240%** on the
retained test; Strouhal errors are **0.343%**, **0.264%**, and **0.253%**.

[Inspect the phase-stable validation figure](results/cylinder_phase/phase_stable_validation.png),
[play the fresh-test comparison](results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4),
or download the checksummed CFD inputs from the
[versioned `cylinder-cfd-v1` release](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1).

The committed `quick` evidence is a classroom regime check, not a
grid-converged external-cylinder DNS result.  Week 7 now places a separately
executed grid-verification study before the ML interpretation.
It fixes `Re=100`, lattice Mach number, nondimensional domain, boundaries,
startup, and sampling while refining the analytical cylinder through
`D/dx = 12, 18, 27`. The fine-pair sensitivity limits pass, but the sequence
does not establish a positive asymptotic order; the formal GCI/grid-independence
gate therefore remains FAIL. The report includes statistical gates,
medium-to-fine changes, and the failed order/GCI decision for mean drag,
Strouhal number, and recirculation length. The declared next resolution is
`D/dx=40`. A future pass would establish
only grid insensitivity for those outputs at the fixed domain; blockage/domain
and external-reference validation remain separate gates.

Run `python qa/run_cylinder_grid_independence.py --regenerate --workers 3` to
regenerate the full study. Run `flowmllab cylinder --root .` to verify both the
retained CFD evidence and learned-model evidence without repeating expensive
solves. The existing neural models still target the low-cost 12-node CFD
dataset and are therefore labelled educational rather than high-fidelity DNS.

## POD-DeepONet validation result

Run `python common/run_pod_deeponet_validation.py` to reproduce the CPU study. The multi-output model uses separately scaled POD trunks so pressure cannot distort the divergence-free velocity basis. The development-only rule selects a rank-3 velocity trunk with a `(32,32)` tanh branch and a rank-3 pressure trunk with an 8-neuron branch acting on `log(Re)`. At `Re = 175, 275, 375`, three-seed ensemble velocity errors are **0.4455%**, **0.0727%**, and **0.0947%**, while direct zero-mean pressure errors are **0.2506%**, **0.1245%**, and **0.1073%**. Wall error is exactly zero and discrete divergence remains at round-off.

This result does **not** claim that the network improves the Ghia benchmark or independently validates the pressure-recovery method used to create the labels. At `Re = 100` and `400`, POD-DeepONet preserves the centerline fidelity of the educational CFD solver. Its demonstrated advantage is amortized three-field evaluation after training: approximately 1.0 ms for the three-seed ensemble versus approximately 5.3 s for a fresh CPU CFD solve in the recorded run. Exact machine-readable values, protocol, seeds, fields, and the comparison figure are in `results/pod_deeponet/`.

<p align="center">
  <a href="results/pod_deeponet/pod_deeponet_ghia_validation.png">
    <img src="results/pod_deeponet/pod_deeponet_ghia_validation.png" alt="FlowMLLab POD-DeepONet validation against CFD and Ghia data" width="100%">
  </a>
</p>

<p align="center"><em>Complete scientific validation figure: Ghia centerlines, blind fields, physical diagnostics, and measured cost.</em></p>

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

[Read the public FlowMLLab v1.1.0 original-software manuscript (PDF)](manuscript/FlowMLLab_v1.1.0_Original_Software_Article.pdf).
This is an author manuscript and is not a journal version of record. Cite the software
using the version-specific Zenodo DOI below.

## Workshops, support, and consulting

FlowMLLab remains free and open source. Optional professional services are available for universities, research laboratories, instructors, and engineering teams:

- **Live workshops:** a two-hour introduction, a one-day intensive, or a multi-session program covering validated CFD, scientific machine learning, POD--DeepONet, and DSMC.
- **Technical onboarding and support:** environment setup, benchmark reproduction, dataset qualification, notebook adaptation, and troubleshooting.
- **Research consulting:** design and review of CFD-to-SciML workflows, physical validation strategies, neural-operator studies, and rarefied-flow applications.
- **Custom extensions:** integration of new physical cases, institutional datasets, validation targets, or research-lab workflows.

For workshop, support, or consulting inquiries, contact [Ehsan Roohi](mailto:roohie@umass.edu?subject=FlowMLLab%20workshop%2C%20support%2C%20or%20consulting). These services are optional and do not affect access to the MIT-licensed software or educational materials.

If FlowMLLab supports your teaching or research, you can also support its continued open-source maintenance through the **Sponsor** button at the top of this repository.

## Citation and contact

Use [CITATION.cff](CITATION.cff) when citing the release.

Version 1.2.0 is the current source release. The latest archived release remains
v1.1.0 until a new Zenodo record is minted; its version-specific DOI is
[10.5281/zenodo.22126785](https://doi.org/10.5281/zenodo.22126785) and is also
recorded in [CITATION.cff](CITATION.cff). The GitHub release and archival record
contain only FlowMLLab. They do not include materials from any separate research
repository.

Ehsan Roohi  
Department of Mechanical and Industrial Engineering  
University of Massachusetts Amherst  
roohie@umass.edu

Copyright © 2026 Ehsan Roohi. Released under the MIT License; see [LICENSE](LICENSE).
