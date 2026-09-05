# Course map: concept → computation → evidence

| Module | Conceptual focus | Guided computation | Evidence required before moving on |
| --- | --- | --- | --- |
| 1A | Eulerian fields, nondimensionalization, boundary conditions | Annotate lid-driven cavity variables and scales | Explain what is prescribed, solved, and derived |
| 1B | Python/NumPy/TensorFlow for scientific work | Arrays, slicing, finite differences, tensors, gradients | Derivative/residual calculation and one-neuron update |
| 1C | Numerical convergence versus validation | Streamfunction–vorticity cavity and Ghia comparison | Residual, centerlines, streamlines, and benchmark errors |
| 2A | Features, targets, scaling, and losses | Build a rarefied-flow regression dataset | Explicit feature/target table and split definition |
| 2B | Knudsen number and model validity | Classify continuum, slip, transition, and free-molecular regimes | Explain why nondimensional inputs encode physical validity |
| 2C | Baseline before neural model | Polynomial/interpolation versus DNN | Interpolation/extrapolation comparison and limitation statement |
| 2.1 | Observation models, Bayesian prediction, proper scores, and calibration | Exact Bayesian velocity-profile update; POD--GP cavity fields; validation-only interval scaling | Complete-case split; interpolation baseline; NLL/CRPS; blind coverage and width; retained under-coverage; physical diagnostics |
| 3A | Maxwellian distributions and macroscopic moments | Sample molecular velocities and recover mean/T | Error-versus-sample-size plot and expected sampling slope |
| 3B | DSMC logic | Move, index, collide, reflect, sample | Map every algorithmic step to its physical role |
| 3C | Noisy field estimation | Mini particle cavity and averaging | Mean fields, uncertainty discussion, and transient/noise distinction |
| 4A | Data qualification | Generate/audit the 11-Re cavity family | Accepted-case table, data hash, and numerical diagnostics |
| 4B | Scalar and coordinate surrogates | `(Re,x,y) → (u,v,p)` with case-wise holdout | Blind errors plus wall, divergence, pressure, and centerline checks |
| 4C | Operator learning with an interpretable trunk | Executed scalar-branch POD-DeepONet for the parametric cavity | Development-only selection; all three blind fields and seeds; wall/divergence checks; Ghia-fidelity table; measured CFD/inference cost; explicit scalar-branch limitation |
| 4.1 | Classical dynamical ROM and nonlinear cost | Centered POD--Galerkin and POD--DEIM for the same transient cavity | Exact recovery of accepted FOM fields; grid/time refinement; validation-only rank freeze; all blind trajectories; wall/divergence/vortex checks; offline, online, and break-even cost |
| 5A | POD and reduced-order learning | SVD/POD basis and neural or interpolated coefficients | Energy, representation error, learning error, and blind reconstruction |
| 5B | Physics-guided objectives and PINNs | Wall/divergence-weighted loss and PDE-residual concepts | Matched ablation with a predeclared tolerance and a justified model choice |
| 5C | Research protocol | Freeze question, baseline, split, metric, and failure threshold | Signed/frozen project card before blind testing |
| 6A | Fokker–Planck closure | Exact coefficient generation and neural surrogate | Offline coefficient errors by physical block |
| 6B | A-posteriori testing | Deploy learned closure inside solver | Stability, high-order moments, fields, centerlines, and runtime |
| 6C | Reproducibility and communication | Restart/run-all, save metrics, make one-slide summary | Complete evidence bundle and explicit limitation |
| 7A | Lattice-Boltzmann mechanics | Derive D2Q9 equilibrium; compare transparent BGK with robust TRT; identify every boundary operation | Mass/density stability, `Ma`, `tau`, no-slip mask, and reproducible configuration |
| 7B | Cylinder-wake regimes and CFD verification | Run `Re=5,20,40,100,180`; then perform the fixed-physics `D/dx=12,18,27` study at `Re=100` | Correct regime classification; statistical gates; acoustic-mode rejection; retained formal asymptotic/GCI failure; declared next refinement; literature bands; and separate grid/domain/validation decisions |
| 7C | Educational unsteady field learning | Reynolds/phase POD failure baseline, four-frame multi-scale CNN, and a phase-stable learned decoder | Case-wise split; one-step field/spectral/downstream checks; retained failed CNN recursion; validation-only harmonic selection; fresh `Re=95` 277-frame rollout; and separately gated vorticity/Strouhal evidence |
| 7.1 | Rarefied hypersonic-cylinder operator learning | Audit 20 DSMC Mach cases; formulate `M_inf -> q(x,y)`; inspect the reviewed Fusion-DeepONet topology; fit a CPU separable teaching analog | Author-release provenance and hashes; whole-case split; strong structured field interpolation; blind interpolation/extrapolation errors; ensemble coverage; retained neural-baseline failure; no full-paper reproduction claim |
| 8A | Exact compressible-flow references | Rayleigh, Fanno, oblique-shock, nozzle-shock, shock-tube, shock-polar, interacting-wave, and Taylor--Maccoll computations | Declared domain and branch; exact/bracketed/ODE reference; forward-substitution residual; limiting behavior |
| 8B | Branch-aware gas-dynamics SciML | Expose hidden-branch regression failure; compare bounded MLPs with interpolation across five inverse tasks | Frozen blind errors; matched coverage; physical bounds and residuals; edge-holdout test; explicit exact/interpolation/MLP decision |
| 8C | Dimensional scaling and CFD bridge | Generalized two-to-five-input shock tube; 100,000-state workload; qualified SU2 diamond-airfoil workflow | Matched offline budget; storage and timing protocol; source hashes; no unverified SU2 case promoted to a training label |
| 9A | Geometry-dependent operator learning | Map real DSMC micro-step height and coordinates to velocity through a branch--trunk representation | File-separated 5/2/2 geometry split; validation-only loss selection; no held-out flow patches; retained paper evidence kept separate |
| 9B | Physics-guided zonal objectives | Balance reverse-flow and main-flow errors with separately normalized regional losses | Validation-only loss-weight selection; global/local tradeoff; untouched 44% and 67% teaching tests |
| 9C | Shock-aligned rarefied-flow operators | Audit 15 public DSMC nozzle cases; compare physical and shock-centered POD; fit full-field POD trunks and neural branches | Source hashes and CC BY attribution; 8-to-2 mode POD audit; frozen 16/25/30 kPa tests; 2-D density/$U$/Mach/pressure errors; shock-location error |
| 10A | DSMC solver qualification and provenance | Audit cavity/shock tables, run metadata, hashes, shapes, and molecular models | Mesh/time/particle/sample checklist; exact case inventory; machine-readable manifest |
| 10B | Rarefied-cavity parameter synthesis | Reproduce complete held-out fields at $Kn=0.05$ and $0.5$ for two lid speeds | Shared contour scales, normalized RMSE denominators, profiles, and higher-moment diagnosis |
| 10C | Mono/diatomic shock operators | Fit POD trunks and Mach branches; compare interpolation and one-sided extrapolation | Density/velocity/temperature profiles; translational overshoot; rotational lag; fixed error gates |

## Suggested adoption modes

### One-day workshop

Use Modules 1C, 2A, 2C, and a short version of 4B. The learning objective is to distinguish a validated numerical label from a convenient training target and to compare a neural model with interpolation.

### Six-week intensive course

Use all modules in order. Weeks 5–6 form one combined guided-project pack: Week 5 establishes the controlled modification and checkpoint; Week 6 completes the selected track and final evidence. Advanced Track 6 remains instructor-approved.
Use Weeks 7--10 only after the original six-week sequence. Week 7 extends to unsteady external flow; Week 7.1 contrasts the continuum cylinder with rarefied hypersonic DSMC and operator learning; Week 8 extends to compressible-flow branches, exact-to-ML comparisons, and a qualified multidimensional-CFD bridge. Week 9 provides geometry-dependent and shock-aligned rarefied-flow operators. Week 10 closes the sequence with an end-to-end DSMC article-reproduction experiment. None replaces the Weeks 5–6 project.

### Full semester

Expand each row into a lecture/lab pair. Add grid/time-step studies, multi-parameter or geometry-varying data, a dedicated neural-operator unit, the Week-2.1 probabilistic-UQ increment, and a research-resolution final project.

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
