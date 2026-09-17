# Lectures and guides

## Week 14 - Davidson-based turbulence closure module

[Lecture PDF](week14_rans_pinn_nn.pdf) / [editable notes](source/week14_rans_pinn_nn.md)
/ [executed notebook and run guide](../notebooks/week14/README.md).
Twelve pages: inverse PINN, three closure coefficients, original NN training,
coupled RANS, and the limits found in the source-package audit. All figures are
newly plotted from attributed data and retained executions.

| File | Main topics | Companion notebooks |
| --- | --- | --- |
| [week01_numerical_foundations.pdf](week01_numerical_foundations.pdf) | Continuity and component momentum, nondimensionalization, streamfunction/vorticity derivations, finite differences, wall conditions, coupled iteration, convergence, Ghia validation | `notebooks/week01/` |
| `week01_1_ai_assisted_scientific_software.pdf` | Specification, manufactured-solution verification, physical gates, provenance, adversarial axis tests, human-agent authority and disclosure | `notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb` |
| `week02_supervised_learning_rarefaction.pdf` | Neurons, MLPs, losses, optimization, scaling, case-wise splits, rarefaction | `notebooks/week02/` |
| `week02_1_probabilistic_uq.pdf` | Observation models, exact Bayesian regression, POD--Gaussian-process fields, proper scores, leakage-free calibration, retained blind under-coverage | `notebooks/week02_1/Probabilistic_UQ_CFD.ipynb` |
| `week03_kinetic_dsmc.pdf` | Distribution functions, Maxwellian moments, sampling error, DSMC | `notebooks/week03/` |
| `week04_cavity_surrogates_deeponet.pdf` | Data qualification, scalar/field surrogates, DeepONet, physical metrics | `notebooks/week04/` |
| [week04_2_pinn_cavity.pdf](week04_2_pinn_cavity.pdf) | Historical PINN foundations reading, now preparatory material for Week 13 | Five assignments in the lecture; no new trained PINN claim |
| [week04_2_stokes_to_navier_stokes.pdf](week04_2_stokes_to_navier_stokes.pdf) | Actual Stokes input, Reynolds-speed consistency, constant/diverse lids, POD correction network, validation-only selection and bounded regression evidence | `notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb` |
| [week04_2_grid51_validation.pdf](week04_2_grid51_validation.pdf) | Fresh 51 × 51 CFD and Stokes solves, retrained correction, 25-to-51 grid comparison, speed/streamline/vorticity plots | [51 × 51 validation notebook](../notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) |
| `week05_06_project_guide.pdf` | Six project tracks, frozen protocols, POD, uncertainty, rarefied cavity, FP closure | `notebooks/week05_06/P0_Project_Setup.ipynb` through `P6_FP_Cavity_Closure.ipynb` |
| `week07_cylinder_lbm_neural_surrogate.pdf` | Circular-cylinder physics; concise D2Q9 BGK/TRT algorithm; curved-wall boundaries; force and gated Strouhal diagnostics; executed three-grid study with retained formal asymptotic/GCI failure; POD/CNN failure analysis; and leakage-controlled 277-frame phase-stable validation | `notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb` |
| `week07_1_hypersonic_rarefied_cylinder.pdf` | Rarefaction and DSMC cylinder fields; parameter-to-field operators; whole-case splitting; reviewed Fusion-DeepONet topology; strong Mach-field interpolation; deep-ensemble diagnostics; retained baseline win; and explicit claim boundaries | `notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb` |
| `week07_2_cylinder_state_estimation.pdf` | POD-space linear-Gaussian modeling; causal predict-update filtering; sparse sensors; matched information baselines; validation-only covariance inflation; retained interval under-coverage | `notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb` |
| `week08_gas_dynamics_sciml.pdf` | Exact Rayleigh, Fanno, oblique-shock, nozzle-shock, and shock-tube physics; branch-aware inversion; exact/interpolation/MLP decision rules; edge generalization; dimensional scaling; and a qualified SU2 CFD bridge | `notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb` and `W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb` |
| `week09_rarefied_deeponet_case_studies.pdf` | Independent DSMC verification; concise DSMC algorithm; DeepONet/POD-trunk formulation; micro-step Knudsen and height cases; fresh full-field nozzle back-pressure predictions; and a clearly bounded throat-location reference | `notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb` and `W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb` |
| `week10_dsmc_data_driven_surrogates.pdf` | Independent DSMC qualification; move--collide--sample algorithm; rarefied-cavity log-Knudsen synthesis; monatomic and diatomic shock operators; translational--rotational relaxation; interpolation/extrapolation; and complete article-result reproduction | `notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb` |

Editable sources are included in `source/` where they were available. The PDF is the authoritative rendered lecture for this release; Week 2.1 also includes an editable PPTX and its deterministic JavaScript builder.

The Week-4 correction companion and executed Lab 4 are rebuilt together by
[`../qa/build_week04_2_stokes_correction.py`](../qa/build_week04_2_stokes_correction.py).
The builder reads retained numerical evidence, regenerates the plots, executes
all notebook code cells sequentially and writes the ten-page Times-style PDF.
The historical PINN reading remains at `source/week04_2_pinn_cavity.md` with its
builder at `../qa/build_week04_2_lecture.py` for existing Week-13 references.

## Weeks 11 through 13 (working course, after v1.4.1)

| Lecture | Notebook | Editable source |
| --- | --- | --- |
| [Week 11: shock, vortex and vapor-cloud identification](week11_shock_vortex_identification.pdf) | [CPU controls](../notebooks/week11/W11_Shock_Vortex_Identification.ipynb) · [U-Net reconstruction comparison](../notebooks/week11/W11_Lab2_Reconstruction_and_Identification.ipynb) · [Hydrofoil vapor-cloud detection](../notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) | [Lecture notes](source/week11_shock_vortex_identification.md) |
| [Week 12: DSMC moment reconstruction](week12_dsmc_moment_reconstruction.pdf) | [CPU lab](../notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb) | [Lecture notes](source/week12_dsmc_moment_reconstruction.md) |
| [Week 13: rectangular-cavity PINN research audit](week13_rectangular_cavity_pinn.pdf) | [Evidence-audit notebook](../notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb) | [Continuous-text lecture](source/week13_rectangular_cavity_pinn.md) |
| [Week 14: RANS and neural closures](week14_rans_pinn_nn.pdf) | [Executed lab](../notebooks/week14/README.md) | [Notes](source/week14_rans_pinn_nn.md) |
| [Week 15: geometry-aware neural operators](week15_geometry_generalization.pdf) | [Executed lab and data](../notebooks/week15/README.md) | [Notes](source/week15_geometry_generalization.md) |

Each has eight lecture-note pages, worked concepts, an executed classroom figure
and assessment prompts. Research attribution and synthetic-teaching scope are
explicit; neither notebook claims to reproduce the original research model.
Week 13 is a separate final-course research module built from retained A100
runs; its deep-cavity cases are not called field-validated without matched raw CFD.

## Week 10.1 reading companion

[Ab initio collision DeepONet](week10_1_abinitio_collision_deeponet.md) is a
supplement to Lecture 10, with [colored research-field comparisons](../results/abinitio_deeponet_cylinder/README.md)
and a citation to [Roohi et al., PoF 38, 057123](https://doi.org/10.1063/5.0328463).
It is a Markdown reading note, not an additional PDF or training notebook.

Recommended teaching pattern for each meeting:

1. physical framing and a prediction question;
2. a short derivation or algorithm walkthrough;
3. guided notebook work;
4. benchmark/baseline/physical comparison; and
5. an exit prompt asking what evidence would falsify the conclusion.

## Rebuilding the Week-1 foundations lecture

The expanded 23-page [editable LaTeX source](source/week01_numerical_foundations.tex)
includes the proof of the original lecture's Eq. (15), an optional curl-of-momentum
derivation, and the distinction between inner Poisson sweeps and outer time steps.
The sign convention and wall formulas match the introductory Week-1 notebook.
The [finite-difference and code companion source](source/week01_code_walkthrough.tex)
is included in the same PDF: pages 14–17 cover worked finite differences and
Poisson iteration, pages 18–21 explain the existing solver functions, page 22
shows algorithm diagrams, and page 23 maps all sixteen original sections.
See the [coverage audit](source/week01_coverage.md) for what was restored or
clarified and an immutable link to the unchanged original PDF.

From the repository root, regenerate the figure and numerical record with:

```bash
python qa/build_week01_foundations_figure.py
```

This runs only the existing introductory Re=100 solver functions and benchmark
arrays, without executing the notebook's separate retained pressure-validation
demonstrations. It requires NumPy and Matplotlib. The figure and single-grid
diagnostics are retained in `source/week01_assets/`; they are not a claim of
grid independence or a new research validation.

Compile with a standard LaTeX installation (including the packages named in the source):

```bash
cd lectures/source
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp week01_numerical_foundations.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp week01_numerical_foundations.tex
cp /tmp/week01_numerical_foundations.pdf ../week01_numerical_foundations.pdf
```
