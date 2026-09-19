# Lectures and guides

One row per lecture, in course order. The PDF is the authoritative rendered
lecture for each release; editable sources live in `source/` where available.
Page counts are those of the current files.

| Week | Lecture (pages) | Main topics | Companion notebooks | Editable source |
| --- | --- | --- | --- | --- |
| 1 | [Numerical foundations](week01_numerical_foundations.pdf) (27) | Continuity and momentum, nondimensionalization, streamfunction/vorticity derivations, finite differences, wall conditions, coupled iteration, convergence, Ghia validation | `notebooks/week01/` | [LaTeX](source/week01_numerical_foundations.tex), [code walkthrough](source/week01_code_walkthrough.tex), [pressure appendix](source/week01_pressure_derivation.tex) |
| 1.1 | [AI-assisted scientific software](week01_1_ai_assisted_scientific_software.pdf) (9) | Specification, manufactured-solution verification, physical gates, provenance, adversarial axis tests, human-agent authority and disclosure | `notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb` | [Markdown](source/week01_1_ai_assisted_scientific_software.md) |
| 2 | [Supervised learning and rarefaction](week02_supervised_learning_rarefaction.pdf) (21) | Neurons, MLPs, losses, optimization, scaling, case-wise splits, rarefaction | `notebooks/week02/` | [LaTeX](source/week02_lecture.tex) |
| 2.1 | [Probabilistic UQ](week02_1_probabilistic_uq.pdf) (14) | Observation models, exact Bayesian regression, POD–Gaussian-process fields, proper scores, leakage-free calibration, retained blind under-coverage | `notebooks/week02_1/Probabilistic_UQ_CFD.ipynb` | [PPTX](source/week02_1_probabilistic_uq.pptx) and its [builder](source/build_week02_1_probabilistic_uq.mjs) |
| 3 | [Kinetic theory and DSMC](week03_kinetic_dsmc.pdf) (34) | Distribution functions, Maxwellian moments, sampling error, DSMC | `notebooks/week03/` | [PPTX](source/week03_lecture.pptx) |
| 4 | [Cavity surrogates and DeepONet](week04_cavity_surrogates_deeponet.pdf) (10) | Data qualification, scalar/field surrogates, DeepONet, physical metrics | `notebooks/week04/` | [LaTeX](source/week04_lecture.tex) |
| 4.2 | [Stokes-to-Navier-Stokes correction](week04_2_stokes_to_navier_stokes.pdf) (10) and [51 x 51 addendum](week04_2_grid51_validation.pdf) (3) | Actual Stokes input, Reynolds-speed consistency, constant/diverse lids, POD correction network, validation-only selection, grid refinement | `notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb`, `W4_Lab4_Grid51_Validation.ipynb` | built by [`qa/build_week04_2_stokes_correction.py`](../qa/build_week04_2_stokes_correction.py) and [`qa/build_week04_2_grid51.py`](../qa/build_week04_2_grid51.py) |
| 5 to 6 | [Project guide](week05_06_project_guide.pdf) (60) | Six project tracks, frozen protocols, POD, uncertainty, rarefied cavity, FP closure | `notebooks/week05_06/P0` to `P6` | [LaTeX](source/weeks05_06_project_guide.tex) |
| 5 (companion) | [Modal sensing](week05_modal_sensing.pdf) (4) | Gappy POD, sensor placement, SINDy bridge | `notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb` | [Markdown](source/week05_modal_sensing.md) |
| 7 | [Cylinder LBM and neural surrogates](week07_cylinder_lbm_neural_surrogate.pdf) (18) | Circular-cylinder physics; D2Q9 BGK/TRT algorithm; curved-wall boundaries; force and gated Strouhal diagnostics; three-grid study with retained formal GCI failure; POD/CNN failure analysis; 277-frame phase-stable validation | `notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb` | [LaTeX](source/week07_cylinder_lbm_neural_surrogate.tex) |
| 7 (companion) | [Modal forecasting](week07_modal_forecasting.pdf) (4) | POD coefficient forecasting and DMD baselines | `notebooks/week07/W7_Lab2_Modal_Forecasting.ipynb` | [Markdown](source/week07_modal_forecasting.md) |
| 7.1 | [Rarefied hypersonic cylinder](week07_1_hypersonic_rarefied_cylinder.pdf) (16) | Rarefaction and DSMC cylinder fields; parameter-to-field operators; whole-case splitting; Fusion-DeepONet topology; strong Mach-field interpolation; ensemble diagnostics; retained baseline win | `notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb` | [builder](source/build_week07_1_hypersonic_rarefied_cylinder.py) |
| 7.2 | [Cylinder-wake state estimation](week07_2_cylinder_state_estimation.pdf) (4) | POD-space linear-Gaussian modeling; causal predict-update filtering; sparse sensors; matched information baselines; validation-only covariance inflation; retained interval under-coverage | `notebooks/week07_2/W7_2_Cylinder_Wake_State_Estimation.ipynb` | [Markdown](source/week07_2_cylinder_state_estimation.md) |
| 7.3 | [Self-supervised pretraining and label efficiency](week07_3_masked_pretraining.pdf) (6) | Masked autoencoders; self-supervised pretraining on unlabelled wakes; zero-shot, linear probe, fine-tuning and from-scratch under one budget; gappy POD as the matched baseline; label-efficiency curves and label saving; retained classical win | `notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb` | [Markdown](source/week07_3_masked_pretraining.md) |
| 7.4 | [Diverse-wake pretraining and lift decoding](week07_4_diverse_wake_pretraining.pdf) (4) | Trajectory-held representation transfer; target-label budgets; POD and random-encoder baselines; source-label accounting and limitations | `notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb` | [Markdown](source/week07_4_diverse_wake_pretraining.md) |
| 8 | [Gas dynamics and SciML](week08_gas_dynamics_sciml.pdf) (12) | Exact Rayleigh, Fanno, oblique-shock, nozzle-shock and shock-tube physics; branch-aware inversion; exact/interpolation/MLP decision rules; edge generalization; dimensional scaling; qualified SU2 CFD bridge | `notebooks/week08/` | [LaTeX](source/week08_gas_dynamics_sciml.tex) |
| 9 | [Rarefied DeepONet case studies](week09_rarefied_deeponet_case_studies.pdf) (18) | Independent DSMC verification; DSMC algorithm; DeepONet/POD-trunk formulation; micro-step Knudsen and height cases; full-field nozzle back-pressure predictions; bounded throat-location reference | `notebooks/week09/W9_Lab1_...`, `W9_Lab2_...` | [LaTeX](source/week09_rarefied_deeponet_case_studies.tex) |
| 9 Lab 3 | [Nozzle data alignment](week09_3_nozzle_data_alignment.pdf) (2) | Branch/trunk data contract, silent coordinate-pairing bugs, isentropic reference checks | `notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb` | [LaTeX](source/week09_3_nozzle_data_alignment.tex) |
| 10 | [DSMC data-driven surrogates](week10_dsmc_data_driven_surrogates.pdf) (13) | Independent DSMC qualification; move–collide–sample algorithm; rarefied-cavity log-Knudsen synthesis; monatomic and diatomic shock operators; translational–rotational relaxation; article-result reproduction | `notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb` | [LaTeX](source/week10_dsmc_data_driven_surrogates.tex) |
| 10.1 | [Ab initio collision DeepONet](week10_1_abinitio_collision_deeponet.md) (reading note) | Collision-angle surrogates and DSMC cylinder contours; see [colored research-field comparisons](../results/abinitio_deeponet_cylinder/README.md) and [Roohi et al., PoF 38, 057123](https://doi.org/10.1063/5.0328463) | `notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb` | Markdown note (no PDF) |
| 11 | [Shock, vortex and vapor-cloud identification](week11_shock_vortex_identification.pdf) (17) | Velocity-gradient diagnostics, pixel classifiers and thresholds, U-Net reconstruction before identification, hydrofoil vapor-cloud detection | `notebooks/week11/` (three notebooks) | [Markdown](source/week11_shock_vortex_identification.md) |
| 12 | [DSMC moment reconstruction](week12_dsmc_moment_reconstruction.pdf) (13) | Additive moments, prior-plus-observation reconstruction, Noise2Noise training on real DSMC, support monitoring | `notebooks/week12/W12_DSMC_Moment_Reconstruction.ipynb` | [Markdown](source/week12_dsmc_moment_reconstruction.md) |
| 13 | [Rectangular-cavity PINNs](week13_rectangular_cavity_pinn.pdf) (12) | Streamfunction PINNs, hard wall constraints, Adam then SSBroyden2, deep-cavity fields, residual audits | `notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb` | [Markdown](source/week13_rectangular_cavity_pinn.md) |
| 13 (preparatory) | [PINN foundations reading](week04_2_pinn_cavity.pdf) (12) | Nondimensional residuals, soft/hard constraints, streamfunction lifting, analytic checks; originally numbered 4.2 | Five assignments in the lecture; no trained-PINN claim | [Markdown](source/week04_2_pinn_cavity.md) |
| 14 | [RANS, PINN and neural closures](week14_rans_pinn_nn.pdf) (8) | Inverse PINN, three closure coefficients, original NN training, coupled RANS, source-package audit limits | `notebooks/week14/W14_pyCALC_RANS_PINN_NN.ipynb` ([setup](../notebooks/week14/README.md)) | [Markdown](source/week14_rans_pinn_nn.md) |
| 15 | [Geometry-aware neural operators](week15_geometry_generalization.pdf) (24) | DeepONet, Geom-DeepONet, Geo-FNO, SMART, GeoTransolver and DoMINO; learning-rate sensitivity; reverse-flow topology; retrospective double-step transfer | `notebooks/week15/W15_Complete_Geometry_Generalization.ipynb` ([data guide](../notebooks/week15/README.md)) | [Markdown](source/week15_geometry_generalization.md) |

Recommended teaching pattern for each meeting:

1. physical framing and a prediction question;
2. a short derivation or algorithm walkthrough;
3. guided notebook work;
4. benchmark/baseline/physical comparison; and
5. an exit prompt asking what evidence would falsify the conclusion.

## Rebuilding lectures

The Week 4.2 companion and executed Lab 4 are rebuilt together by
[`qa/build_week04_2_stokes_correction.py`](../qa/build_week04_2_stokes_correction.py):
it reads the retained numerical evidence, regenerates the plots, executes the
notebook code cells in order and writes the PDF. The historical PINN reading
is built by [`qa/build_week04_2_lecture.py`](../qa/build_week04_2_lecture.py).

The Week 1 foundations lecture is compiled from
[`source/week01_numerical_foundations.tex`](source/week01_numerical_foundations.tex),
which includes the code walkthrough and the pressure-derivation appendix. From
`lectures/source`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp week01_numerical_foundations.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp week01_numerical_foundations.tex
cp /tmp/week01_numerical_foundations.pdf ../week01_numerical_foundations.pdf
```

Its figure and single-grid diagnostics are regenerated from the repository root with
`python qa/build_week01_foundations_figure.py` (NumPy and Matplotlib only); the
outputs are retained in `source/week01_assets/` and are not a claim of grid
independence. The [coverage audit](source/week01_coverage.md) records what the
rewrite restored or clarified. Earlier editions of this lecture are kept under
[`versions/`](versions/) for reference only; the main PDF supersedes them.
