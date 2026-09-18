# Course map: concept → computation → evidence

Every module links to its notebook. The [notebook launcher](notebooks/README.md) and the [lecture index](lectures/README.md) list the same modules with Colab links and page counts.

| Module | Conceptual focus | Guided computation | Evidence required before moving on |
| --- | --- | --- | --- |
| [1A](notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) | Eulerian fields, nondimensionalization, boundary conditions | Annotate lid-driven cavity variables and scales | Explain what is prescribed, solved, and derived |
| [1B](notebooks/week01/02_tensorflow_for_ai_fluids.ipynb) | Python/NumPy/TensorFlow for scientific work | Arrays, slicing, finite differences, tensors, gradients | Derivative/residual calculation and one-neuron update |
| [1C](notebooks/week01/03_cavity_ghia.ipynb) | Numerical convergence versus validation | Streamfunction–vorticity cavity and Ghia comparison | Residual, centerlines, streamlines, and benchmark errors |
| [1.1](notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb) | AI-assisted scientific software and accountable trust | Freeze an executable diagnostic specification; verify against an analytic streamfunction; reject an axis-swap bug; audit the accepted `Re=100` cavity field | Observed order, analytic error, physical gates, data SHA-256, retained accept/reject JSON, manual scientific review, AI-use disclosure, and bounded claim |
| [2A](notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) | Features, targets, scaling, and losses | Build a rarefied-flow regression dataset | Explicit feature/target table and split definition |
| [2B](notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) | Knudsen number and model validity | Classify continuum, slip, transition, and free-molecular regimes | Explain why nondimensional inputs encode physical validity |
| [2C](notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb) | Baseline before neural model | Polynomial/interpolation versus DNN | Interpolation/extrapolation comparison and limitation statement |
| [2.1](notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) | Observation models, Bayesian prediction, proper scores, and calibration | Exact Bayesian velocity-profile update; POD--GP cavity fields; validation-only interval scaling | Complete-case split; interpolation baseline; NLL/CRPS; blind coverage and width; retained under-coverage; physical diagnostics |
| [3A](notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb) | Maxwellian distributions and macroscopic moments | Sample molecular velocities and recover mean/T | Error-versus-sample-size plot and expected sampling slope |
| [3B](notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) | DSMC logic | Move, index, collide, reflect, sample | Map every algorithmic step to its physical role |
| [3C](notebooks/week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb) | Noisy field estimation | Mini particle cavity and averaging | Mean fields, uncertainty discussion, and transient/noise distinction |
| [4A](notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) | Data qualification | Generate/audit the 11-Re cavity family | Accepted-case table, data hash, and numerical diagnostics |
| [4B](notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) | Scalar and coordinate surrogates | `(Re,x,y) → (u,v,p)` with case-wise holdout | Blind errors plus wall, divergence, pressure, and centerline checks |
| [4C](notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) | Operator learning with an interpretable trunk | Executed scalar-branch POD-DeepONet for the parametric cavity | Development-only selection; all three blind fields and seeds; wall/divergence checks; Ghia-fidelity table; measured CFD/inference cost; explicit scalar-branch limitation |
| [4.1](notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb) | Classical dynamical ROM and nonlinear cost | Centered POD--Galerkin and POD--DEIM for the same transient cavity | Exact recovery of accepted FOM fields; grid/time refinement; validation-only rank freeze; all blind trajectories; wall/divergence/vortex checks; offline, online, and break-even cost |
| [4.2](notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) | Multi-fidelity correction | Actual matched Stokes field plus Reynolds number to Navier-Stokes streamfunction correction; [51 × 51 refinement](notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) | Constant/diverse lid formulas; complete-case splits; three-seed ensemble; family-transfer tests; fresh 51-node labels and retraining; 25-to-51 CFD grid shift and OOD limits |
| [5A](notebooks/week05_06/P3_POD_Study.ipynb) | POD and reduced-order learning | SVD/POD basis and neural or interpolated coefficients | Energy, representation error, learning error, and blind reconstruction |
| [5B](notebooks/week05_06/P2_Physics_Guided_DNN.ipynb) | Physics-guided objectives and PINNs | Wall/divergence-weighted loss and PDE-residual concepts | Matched ablation with a predeclared tolerance and a justified model choice |
| [5C](notebooks/week05_06/P0_Project_Setup.ipynb) | Research protocol | Freeze question, baseline, split, metric, and failure threshold | Signed/frozen project card before blind testing |
| [6A](notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) | Fokker–Planck closure | Exact coefficient generation and neural surrogate | Offline coefficient errors by physical block |
| [6B](notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) | A-posteriori testing | Deploy learned closure inside solver | Stability, high-order moments, fields, centerlines, and runtime |
| [6C](notebooks/week05_06/README.md) | Reproducibility and communication | Restart/run-all, save metrics, make one-slide summary | Complete evidence bundle and explicit limitation |
| [7A](notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) | Lattice-Boltzmann mechanics | Derive D2Q9 equilibrium; compare transparent BGK with robust TRT; identify every boundary operation | Mass/density stability, `Ma`, `tau`, no-slip mask, and reproducible configuration |
| [7B](notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) | Cylinder-wake regimes and CFD verification | Run `Re=5,20,40,100,180`; then perform the fixed-physics `D/dx=12,18,27` study at `Re=100` | Correct regime classification; statistical gates; acoustic-mode rejection; retained formal asymptotic/GCI failure; declared next refinement; literature bands; and separate grid/domain/validation decisions |
| [7C](notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb) | Educational unsteady field learning | Reynolds/phase POD failure baseline, four-frame multi-scale CNN, and a phase-stable learned decoder | Case-wise split; one-step field/spectral/downstream checks; retained failed CNN recursion; validation-only harmonic selection; fresh `Re=95` 277-frame rollout; and separately gated vorticity/Strouhal evidence |
| [7.1](notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) | Rarefied hypersonic-cylinder operator learning | Audit 20 DSMC Mach cases; compare a 3x96 tanh MLP with structured Mach-field interpolation | Whole-case split; historical interpolation/extrapolation errors; interpolation wins all six aggregate comparisons; legacy normalization names are not verified units; [paper-parity limits](qa/WEEK71_PAPER_PARITY.md), not full-paper reproduction |
| [7.2](notebooks/week07_2/README.md) | Linear-Gaussian state estimation and information contracts | Fit training-only POD dynamics and sensor maps; compare causal Kalman filtering with open-loop DMD, persistence and matched sensor-only reconstruction | Validation-selected rank/sensors/inflation; five held noise seeds; 2.18% mean test error; retained 55.1% nominal-95% marginal coverage failure; previously inspected trajectory, not new-Re or grid-independent evidence |
| [7.3](notebooks/week07_3/README.md) | Self-supervised pretraining and label efficiency | Pretrain a masked autoencoder on the unlabelled Re90/Re100 wakes; complete masked Re110 frames zero-shot, with a linear probe, fine-tuned and from scratch for k = 1 to 128 labelled frames; gappy POD with transferred, target-only and pooled bases as matched baselines | Every choice on the Re105 validation trajectory; three mask seeds; fine-tuning beats from-scratch at every k and most at small k; retained win of gappy POD (about 1.4% zero-shot) over the network (about 18%); previously inspected trajectory, not new-physics evidence |
| [7.4](notebooks/week07_4/README.md) | Representation transfer to external force labels | Pretrain across eleven Reynolds cases; adapt frozen neural/POD probes to four target trajectories | Matched target labels; validation-only selection; source-label accounting; mean gains over random features; fixed-rank POD comparison; one encoder seed and coarse-CFD limits |
| [8A](notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) | Exact compressible-flow references | Rayleigh, Fanno, oblique-shock, nozzle-shock, shock-tube, shock-polar, interacting-wave, and Taylor--Maccoll computations | Declared domain and branch; exact/bracketed/ODE reference; forward-substitution residual; limiting behavior |
| [8B](notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) | Branch-aware gas-dynamics SciML | Expose hidden-branch regression failure; compare bounded MLPs with interpolation across five inverse tasks | Frozen blind errors; matched coverage; physical bounds and residuals; edge-holdout test; explicit exact/interpolation/MLP decision |
| [8C](notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) | Dimensional scaling and CFD bridge | Generalized two-to-five-input shock tube; 100,000-state workload; qualified SU2 diamond-airfoil workflow | Matched offline budget; storage and timing protocol; source hashes; no unverified SU2 case promoted to a training label |
| [9A](notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) | Geometry-dependent operator learning | Map real DSMC micro-step height and coordinates to velocity through a branch--trunk representation | File-separated 5/2/2 geometry split; validation-only loss selection; no held-out flow patches; retained paper evidence kept separate |
| [9B](notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) | Physics-guided zonal objectives | Balance reverse-flow and main-flow errors with separately normalized regional losses | Validation-only loss-weight selection; global/local tradeoff; historical 44% and 67% teaching tests, no longer untouched after inspection |
| [9C](notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb) | Shock-aligned rarefied-flow operators | Audit 15 public DSMC nozzle cases; compare physical and shock-centered POD; fit full-field POD trunks and neural branches | Source hashes and CC BY attribution; 8-to-2 mode POD audit; frozen 16/25/30 kPa tests; 2-D density/$U$/Mach/pressure errors; shock-location error |
| [9.3](notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb) | Branch/trunk data contracts for operator learning | Deliberately break and repair the coordinate pairing of a quasi-1D moving-throat nozzle family; verify the isentropic reference | Two intentional contract failures caught before training; mass-flow, sonic and area-Mach checks on the reference; no surrogate accuracy claim |
| [10A](notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) | DSMC solver qualification and provenance | Audit cavity/shock tables, run metadata, hashes, shapes, and molecular models | Mesh/time/particle/sample checklist; exact case inventory; machine-readable manifest |
| [10B](notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) | Rarefied-cavity parameter synthesis | Reproduce complete held-out fields at $Kn=0.05$ and $0.5$ for two lid speeds | Shared contour scales, normalized RMSE denominators, profiles, and higher-moment diagnosis |
| [10C](notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) | Mono/diatomic shock operators | Fit POD trunks and Mach branches; compare interpolation and one-sided extrapolation | Density/velocity/temperature profiles; translational overshoot; rotational lag; fixed error gates |
| [10.1](notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb) | Classical scattering, surrogate audit and a separate research companion | Solve a reduced Lennard-Jones collision map on CPU; inspect author-supplied Jäger cylinder fields separately | Analytic deflection checks and transport-integral errors; Lennard-Jones is not the article potential; asynchronous research contours do not establish speedup or convergence |

## Research applications and final PINN module

| Module | Concept | Executable exercise | Evidence and limits |
| --- | --- | --- | --- |
| [11](notebooks/week11/README.md) | Shock/core identification; shear versus rotation; hydrofoil vapor-cloud detection | Manufactured shock/core controls; retained HJ evidence; U-Net reconstruction; original cavitation model replay and optional adaptation | Complete-case splits; weak-reference limits; 158 cavitation frames, including errors and inspected nontraining cases |
| [12](notebooks/week12/README.md) | Additive moments, prior-plus-observation reconstruction, support | Synthetic warm-up plus fresh Noise2Noise-style patch-MLP training on real DSMC cavity observations | Four fitting seeds, two selection seeds, two evaluation seeds; high-budget reference excluded from fitting; mixed qx/qy baseline results; already-inspected archive, not a fresh blind or paper-model reproduction |
| [13](notebooks/week13/README.md) | Physics-informed neural networks: streamfunction formulation, hard wall constraints, autodiff residuals, Adam then quasi-Newton; residual versus field error | Build and train a small cavity PINN on CPU and judge it by training loss, held-out residual and the Week 1 CFD reference; inspect the retained Re=1000, D/W=2.2 field beside Nektar++; audit the four retained A100 cases (Re=100/400, D=1/2) | Exact walls and continuity by construction; a training loss that falls while the held-out residual does not; frozen square-case CFD gates (3.3% and 10.6% interior error); deep cases residual-audited only; the training/held-out residual gap interpreted |
| [14](notebooks/week14/README.md) | RANS closure, inverse PINN and local-feature NN | Reproduce Davidson's small coefficient network; compare interpolation; inspect source solver restarts | Correct baseline identity, source hashes, residual gates and paper-stage distinction; full end-to-end paper reproduction is not established |
| [15](notebooks/week15/README.md) | Geometry-aware neural operators: Geo-DeepONet, FNO and U-FNO on separated step flow | Audit 130 sampled OpenFOAM fields across 51 geometry masks; recompute velocity, centred-pressure and reverse-flow diagnostics for retained predictions; compare whole-geometry and family holdouts, including a fixed-domain ordinary DeepONet | Hash-verified data and predictions; recomputed metrics match the source report; three-seed holdout scores retained; historical g011 geometry is a development example, not an unopened test |

Each includes a PDF lecture and an executed CPU notebook. Week 11 additionally
audits frozen-checkpoint research masks; these are not new ground-truth accuracy
measurements. Week 12 follows
Week 11 because noise-sensitive derivatives connect feature detection to field
reconstruction; it can also be taught directly after Weeks 3 and 10.
Week 13 is an independent research audit that returns to continuum cavity
flow after students understand optimization, validation and evidence boundaries.

The [PINN foundations reading](lectures/week04_2_pinn_cavity.pdf), originally
numbered 4.2, is preparatory material within Week 13. It covers nondimensional
residuals, soft/hard constraints, McDevitt's streamfunction lifting and analytic
checks. The [initial Re=100 qualification](results/week04_2_pinn_cavity/README.md)
supports the final module's CFD comparison and restart protocol. The new Week-4
Lab 4 covers Stokes-to-Navier-Stokes correction.

## Suggested adoption modes

### One-day workshop

Use Modules 1C, 2A, 2C, and a short version of 4B. The learning objective is to distinguish a validated numerical label from a convenient training target and to compare a neural model with interpolation.

### Six-week core course (Weeks 1 to 6)

Use Weeks 1 to 6 in order. Weeks 5 and 6 form one combined guided-project pack: Week 5 establishes the controlled modification and checkpoint; Week 6 completes the selected track and final evidence. Advanced Track 6 remains instructor-approved. This is the sequence taught in Summer 2026.

### Extended course (Weeks 7 to 15)

Take the extension weeks after the core; none replaces the Weeks 5 to 6 project.

- Weeks 7, 7.1, 7.2, 7.3 and 7.4: unsteady external flow (LBM cylinder), the rarefied hypersonic cylinder with operator learning, sparse-sensor state estimation, self-supervised reconstruction, and diverse-wake transfer to lift with an explicit target-label budget.
- Week 8: compressible-flow branches, exact-to-ML comparisons and a qualified multidimensional-CFD bridge.
- Weeks 9 and 9.3: geometry-dependent and shock-aligned rarefied-flow operators, and the data-alignment audit that precedes any operator fit.
- Weeks 10 and 10.1: an end-to-end DSMC article reproduction and the collision-map surrogate companion.
- Weeks 11 and 12: feature identification (shocks, vortices, vapour clouds) and noisy-moment reconstruction.
- Weeks 13 to 15: physics-informed networks for the cavity, RANS closures, and geometry-aware neural operators. Week 14 needs a separate environment (see its README); Week 15 runs from a full checkout.

### Full semester

Expand each row into a lecture/lab pair. Add grid/time-step studies, multi-parameter or geometry-varying data, a dedicated neural-operator unit (Weeks 9 and 15), the Week 2.1 probabilistic-UQ increment, and a research-resolution final project.

## Assessment philosophy

Assess evidence rather than software completion. Recommended final-project categories are:

1. scientific question and matched baseline;
2. split and model-selection discipline;
3. numerical metrics;
4. physical validation;
5. failure/limitation analysis;
6. reproducibility; and
7. scientific communication.

No category should require the ML method to outperform the baseline.

## Further development

See the [theory coverage and proposal matrix](THEORY_GAP_MATRIX.md) for the
original proposal history, including additional sampling diagnostics.
[Week 7.2 state estimation](notebooks/week07_2/README.md) and
[Week 7.3 label efficiency](notebooks/week07_3/README.md) are implemented;
PINN verification is covered in the final [Week 13 module](notebooks/week13/README.md).
Use the current course tables above to distinguish available modules from
future proposals, and the linked results reports to assess their validation status.
