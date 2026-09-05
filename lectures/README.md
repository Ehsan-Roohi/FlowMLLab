# Lectures and guides

| File | Main topics | Companion notebooks |
| --- | --- | --- |
| `week01_numerical_foundations.pdf` | Python, TensorFlow, CFD fields, finite differences, convergence, Ghia validation | `notebooks/week01/` |
| `week02_supervised_learning_rarefaction.pdf` | Neurons, MLPs, losses, optimization, scaling, case-wise splits, rarefaction | `notebooks/week02/` |
| `week02_1_probabilistic_uq.pdf` | Observation models, exact Bayesian regression, POD--Gaussian-process fields, proper scores, leakage-free calibration, retained blind under-coverage | `notebooks/week02_1/Probabilistic_UQ_CFD.ipynb` |
| `week03_kinetic_dsmc.pdf` | Distribution functions, Maxwellian moments, sampling error, DSMC | `notebooks/week03/` |
| `week04_cavity_surrogates_deeponet.pdf` | Data qualification, scalar/field surrogates, DeepONet, physical metrics | `notebooks/week04/` |
| `week05_06_project_guide.pdf` | Six project tracks, frozen protocols, POD, uncertainty, rarefied cavity, FP closure | `notebooks/week05_06/P0_Project_Setup.ipynb` through `P6_FP_Cavity_Closure.ipynb` |
| `week07_cylinder_lbm_neural_surrogate.pdf` | Circular-cylinder physics; concise D2Q9 BGK/TRT algorithm; curved-wall boundaries; force and gated Strouhal diagnostics; executed three-grid study with retained formal asymptotic/GCI failure; POD/CNN failure analysis; and leakage-controlled 277-frame phase-stable validation | `notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb` |
| `week07_1_hypersonic_rarefied_cylinder.pdf` | Rarefaction and DSMC cylinder fields; parameter-to-field operators; whole-case splitting; reviewed Fusion-DeepONet topology; strong Mach-field interpolation; deep-ensemble diagnostics; retained baseline win; and explicit claim boundaries | `notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb` |
| `week08_gas_dynamics_sciml.pdf` | Exact Rayleigh, Fanno, oblique-shock, nozzle-shock, and shock-tube physics; branch-aware inversion; exact/interpolation/MLP decision rules; edge generalization; dimensional scaling; and a qualified SU2 CFD bridge | `notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb` and `W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb` |
| `week09_rarefied_deeponet_case_studies.pdf` | Independent DSMC verification; concise DSMC algorithm; DeepONet/POD-trunk formulation; micro-step Knudsen and height cases; fresh full-field nozzle back-pressure predictions; and a clearly bounded throat-location reference | `notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb` and `W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb` |
| `week10_dsmc_data_driven_surrogates.pdf` | Independent DSMC qualification; move--collide--sample algorithm; rarefied-cavity log-Knudsen synthesis; monatomic and diatomic shock operators; translational--rotational relaxation; interpolation/extrapolation; and complete article-result reproduction | `notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb` |

Editable sources are included in `source/` where they were available. The PDF is the authoritative rendered lecture for this release; Week 2.1 also includes an editable PPTX and its deterministic JavaScript builder.

## Weeks 11 and 12 (working course, after v1.4.1)

| Lecture | Notebook | Editable source |
| --- | --- | --- |
| [Week 11: shock and vortex identification](week11_shock_vortex_identification.pdf) | [CPU lab](../notebooks/week11/W11_Shock_Vortex_Identification.ipynb) | [Lecture notes](source/week11_shock_vortex_identification.md) |
| [Week 12: DSMC moment reconstruction](week12_dsmc_moment_reconstruction.pdf) | [CPU lab](../notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb) | [Lecture notes](source/week12_dsmc_moment_reconstruction.md) |

Each has eight lecture-note pages, worked concepts, an executed classroom figure
and assessment prompts. Research attribution and synthetic-teaching scope are
explicit; neither notebook claims to reproduce the original research model.

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
