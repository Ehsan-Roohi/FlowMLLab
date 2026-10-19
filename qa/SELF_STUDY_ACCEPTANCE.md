# Self-study acceptance review — draft, 19 October 2026

This is an editorial and mathematical acceptance record for the revised teaching
materials based on `c9562b8`. It is **not** a student pilot, a learning-outcome
validation, a fresh full-course execution, or publication approval. The original
long-form lectures remain the teaching source; the short exit checks below are
acceptance probes, not replacement lectures. Review included revised lecture
sources, extracted PDF text, notebook Markdown and task contracts, the three
original audit lesson maps, and implementation ledgers. Scientific execution and
visual PDF acceptance have separate reviewers and records.

## Readiness rule and workload assumptions

A self-study lesson should let a learner identify its prerequisites, the physical
quantity and units, what is fitted or loaded, a worked calculation, an exercise
with a judgeable answer, and what the evidence cannot establish. A correct plot
or a passing helper call alone is insufficient. Before Run All, the learner
should answer: **what is fitted now, what is retained evidence, and what information
is available at evaluation?** Afterward, change one condition and explain the
result without consulting the reference answer.

Times below are **editorial planning estimates of active student work**, including
reading, hand calculation, bounded notebook activity and interpretation. They are
not measured completion times, compute benchmarks or guarantees. They assume
basic mechanical/fluid engineering, calculus, introductory linear algebra and
Python/NumPy readiness except for Week 1's Python introduction. Installation,
external downloads, unattended solver/training time and GPU queues are excluded.
A novice missing a prerequisite needs additional preparation. The Week 5–6
projects are alternatives: choose one track, not all six. Optional extensions and
the geometry–time bridge are not additional mandatory weeks.

The table's evidence statements describe inspected instructional content and
reviewed contracts; they do not mean a representative student completed it.
For Weeks 1 and 1.1, baseline entry-route content was checked for scope and
prerequisites; no new independent full lesson audit is claimed here.

## Lesson-level acceptance evidence

| Lesson / route | Concept and observable exit check | Execution and evidence boundary | Estimated active study |
|---|---|---|---:|
| 1 Python, TensorFlow, cavity | Arrays and finite differences lead to a cavity solution; explain why grid/boundary and Ghia checks answer different questions. Three labs are separate prerequisites, not one short Run All. | CPU teaching exercises; solver regeneration is distinct from reading stored validation. Baseline scope check. | 6–10 h across three labs |
| 1.1 AI-assisted scientific software | Explain a derivative or indexing defect and show the numerical check that catches it; an AI-generated implementation is a proposal until verified. | Local numerical audit, not a general guarantee about generated software. Baseline scope check. | 2–3 h |
| 1.2 pressure–velocity | Two volumes, one shared flux: `D=[1,-1]^T`, `G=[-1,1]`; with q*=2 and d=.5 obtain p'=(-4,0), qnew=0. Explain gauge and shared-face cancellation before assembling a sparse solve. | Nondimensional algebra and bounded CPU comparison; higher-Re campaign separate. PDF contains worked correction. | 3–5 h |
| 2 supervised rarefaction | Follow the neuron/gradient example; distinguish scaling, interpolation and extrapolation in Kn and accommodation. State that the teaching Q* map is synthetic. | CPU regression fits; small data loss does not validate gas physics. | 3–5 h |
| 2.1A scalar UQ | Two observations give posterior mean 1, variance 1/216; add known observation variance only for a new observation. Identify what CFD discrepancy is omitted. | Study after Week 2; Gaussian/scalar differentiation prerequisites. Hand calculation precedes helper. | 1.5–2.5 h |
| 2.1B field UQ | After POD, solve 2×2 covariance system: latent variance 5/7, noisy variance 27/28. Track coefficient-squared covariance units and compare NLL/CRPS with units and coverage. | Study after Week 4/POD, not as an unexplained pre-POD jump. Retain under-coverage; pointwise coverage is not simultaneous field coverage. | 2–3 h |
| 3 Lab1 Maxwellian | Compute velocity moments and explain why subtracting an estimated mean biases the finite-N temperature estimator; distinguish the historical control from the conditional iid Bessel correction. | CPU sampling and small ML; N=0/N=1 and mask conventions matter. No new kinetic-accuracy claim. | 3–4.5 h |
| 3 Lab2 mini-DSMC | Connect movement, wall interaction, collision and sampling to moment fields; separate pooled and instantaneous temperatures and sampling variance from bias. | CPU core; optional GPU/figure-validation work is a separate advanced extension with additional cost. | 4–6 h core |
| 4 Lab1 CFD production | Before a surrogate, explain how small residual can coexist with poor reference agreement; distinguish pressure consistency and independently validated pressure. | Load or produce labels under stated quality gates; the gate does not prove continuum accuracy. | 2–3 h |
| 4 Lab2 scalar/field surrogates | Identify actual u,v outputs and explain why training-loss stopping is convergence control, not held-out validation. Compare a non-neural interpolation baseline. | Small fits; pressure is not silently promised by this lab's output interface. | 3–4 h |
| 4 Lab3 DeepONet | Explain branch/trunk dimensions, frozen train-only POD and shuffled-query control; distinguish the model fitted in memory from stored figures. | `retained_demo` or `fresh_experiment`; fresh evaluation now consumes the fitted bundles. Public-held-out cases are already exposed. | 3.5–5 h |
| 4.1 classical ROM, optional | Rank-one projection of (2,.5) leaves residual .5; coefficient 1.8 adds .2 in-subspace error, total sqrt(.29). DEIM pivot perturbation demonstrates amplification and missing-basis error separately. | SVD, ODE and vorticity prerequisites; optional CPU ROM, not a mandatory beginner project unit. | 4–6 h |
| 4.2 Stokes correction | Use Stokes linearity to distinguish a speed-rescaling control from changed lid shape; explain a matched-grid pressure-consistency comparison. | Stored/fresh paths separately labeled; past public evaluation is not prospectively blind. | 3–5 h |
| 4.2 grid51 | Distinguish surrogate-to-solver error from two-grid discrepancy; no theorem that a surrogate cannot accidentally be closer to continuum truth. | Short evidence audit; two grids do not establish continuum discretization error. | 1–1.5 h |
| 4.2 PINN reading | Derive hard-wall lifting and explain why exact walls/divergence do not prove momentum or velocity accuracy. | Conceptual preparation for Week 13; retained qualification is a separate run. | 1.5–2.5 h |
| 5–6 P0 | Freeze question, metric, threshold and baseline; record prior exposure to shared Re cases. | Common setup, then choose one project; a genuinely unopened case needs a separate prospective protocol. | 1.5–2.5 h |
| P1 Reynolds generalization | Hold the question fixed while changing training support; interpret interpolation versus extrapolation against the linear baseline. | Controlled CPU project; report all case errors and failed threshold checks. | 6–9 h plus report |
| P2 physics-guided DNN | Relate Gaussian wall weighting and derivative chain-rule scaling to the actual code; distinguish a physical derivative from a derivative of a standardized feature. | One factor at a time; training penalties are not physical validation. | 7–12 h plus report |
| P3 POD | Distinguish representation floor from coefficient error; select the lowest eligible rank only after pressure/physics gates, within declared velocity tolerance. | Mandatory interpolation control; selection uses development evidence. | 6–9 h plus report |
| P4 uncertainty | Explain sample ensemble SD with K−1 and why seed spread is not a calibrated predictive interval or model-discrepancy estimate. | Keep each seed; no inference of statistical confidence from a tiny ensemble. | 7–10 h plus report |
| P5 rarefied cavity | Separate shared stochastic bias from seed variance; explain why cache identity includes solver/config/cases rather than filename alone. | Stochastic data generation may be costly; active work estimate excludes waiting. | 9–14 h plus report |
| P6 closure, instructor-assigned | Before 16→9 closure, derive OU variance balance with gamma=2 s^-1, D=8 m²s^-3: V(.5)=4.6767 m²s^-2. Explain second/third moments and coefficient versus closed-loop field evidence. | Advanced kinetic/GPU track; not a mandatory beginner Week 6 exercise. | 12–18 h plus report and compute |
| 5 modal sensing, after 7 | Near-parallel sensors amplify .01 observation error into unit coefficient error; improve conditioning and explain why representation residual remains. | Study after Week 7 modal dynamics despite directory name; paired noise and fixed sensor budget. | 3–4.5 h |
| 7 LBM cylinder | Relate BGK/TRT, force history, phase, one-step and rollout; separate continued field time from archived force-history window. | Retained core and explicit continuation; no phase claim from mismatched windows, no new grid-independent result. | 4–6 h |
| 7 modal forecasting | Convert discrete eigenvalue magnitude/phase to growth/frequency using physical time step; predict recursively without reset. | Same retained Re trajectory; DMD and longer-history MLP have different initialization information. | 2–3 h |
| 7.1 hypersonic cylinder | Explain the missing solid/support mask and why three fields require independent output directions rather than one scalar bottleneck. | Default small MLP versus optional fusion builder are different routes. TOV/P/source coordinate units remain unqualified. | 3–4.5 h |
| 7.2 state estimation | Scalar update 2→2.5 at gain .5; distinguish stored row state from column Kalman state and transpose the dynamics/covariance consistently. | Causal filtering uses current sensors; open-loop and smoothing use different information. Preserve 55.10% nominal-95% under-coverage. | 2.5–4 h |
| 7.3 masked pretraining | Define hidden target/label and compare gappy POD with frozen probe/fine-tuning; initialization alone is not isolated when optimization budgets differ. | Fresh demonstration and retained result stay distinct; failure of the neural arm remains evidence. | 3–4.5 h |
| 7.4 diverse wakes | Count source labels separately and distinguish field-completion from force-decoding objectives. | Obtain data before optional `RUN_FULL`; fresh outputs separate from retained. Source download/time is additional. | 2–3.5 h |
| 8 Lab1 exact gas dynamics | Numeric area-ratio inverse: A/A*=1.6875 has M=2 and M≈.3722444862; bracket, substitute, and select physical branch. Sonic endpoint control tests understanding. | CPU exact relations/root solves, not ML or full nozzle reproduction; gamma and all ratios dimensionless. | 2.5–4 h |
| 8 Lab2 SciML evidence | Match branch/domain and data budget before comparing neural and RBF interpolation; forward residual checks an approximation even when norm is small. | Retained comparison plus bounded local computations; external chapter/SU2 routes are separate. | 3–4 h |
| 9 Lab1 microstep | Explain zonal versus global prediction and inspect common-scale true/predicted/error fields, including separated-flow support. | Source-unit fields; no invented SI or pressure-ratio conversion. Published target-assisted method and independent adaptation differ. | 3–4.5 h |
| 9 Lab2 nozzle | Connect shock alignment, baseline interpolation and fit warnings; report convergence warnings instead of suppressing them. | Fresh small fits versus historical headline evidence; archived pressure/density units remain declared source units. | 3–4.5 h |
| 9 Lab3 alignment audit | Explain why correct shape can hide wrong row/grid alignment; reconstruct the two negative controls. | Default audit is not training reproduction; opt-in command names the cost/result needed for headline reproduction. | 1–2 h |
| 10 DSMC surrogate | Compare linear profile interpolation with POD/polynomial under one shock threshold and degree contract; keep component/denominator normalization explicit. | Paired comparisons on retained cases; not a new DSMC solver validation. | 3–4.5 h |
| 10.1 collision map | Connect scattering angle to impact parameter/energy and transport integral; distinguish audit probes outside the training domain from qualified predictions. | Lennard–Jones analogue, not ab-initio full-flow reproduction; bounded audit CPU. | 2–3.5 h |
| 11 shock/vortex | Compute Dice=2/3 and IoU=.5 for intersection3, masks4/5; explain why rotation, shear and compression require distinct diagnostics and overlapping masks. | Manufactured controls and retained weak-reference evidence; mask agreement is not independent physical truth. | 3–4.5 h |
| 11 reconstruction | Compare interpolation→diagnostic, U-Net reconstruction→same diagnostic and direct segmentation under matched inputs. | `audit` default does not fit; explicit small/full experiment choices. Execution time is not study time. | 2–3.5 h |
| 11 cavitation | Distinguish native vapor support, raster display and pressure-based detection; compare alpha/pressure on the same field. | Retained evidence and synthetic noise sensitivity; no camera-noise or independent experiment claim. | 2.5–4 h |
| 12 moment reconstruction | Explain additive-first moment correction, divergence counterexample and conditional independence required by Noise2Noise; retain simplest qx baseline win. | Archive units and original block/RNG ancestry remain unknown; no fabricated noise independence or physical heat-flux validation. | 3.5–5 h |
| 13 PINN | Analytic derivative/metric preflight at D=1 and2.2 precedes optimization; frozen quasi-Newton objective; report train/held-out residual and reference field error separately. | Small CPU teaching fit then retained research audit. Checkpoint55118 field and65711 continuation history must not be merged. | 4–6 h |
| 14 turbulence closure | Expand channel flux balance; nonlinear coefficient correction must be learned beyond endpoint lifting. Explain .001 flux uncertainty amplified by division through small k'. | Small inverse/supervised teaching fits; long inverse and coupled CFD remain retained/separate. Boundary data are not a universal identifiability guarantee. | 4–6 h |
| 15 geometry operators | CPU core: compute SDF, branch/trunk contraction and a Fourier operation; distinguish unseen geometry from family transfer and test query-set sensitivity. | Local core does not require 342-field archive; research verification optional. Pressure normalization and author/proxy distinctions remain explicit. | 2–3 h core; 4–6 h extended reading |
| 15 historical step audit | Interpret old held-out evidence and provenance without replacing the complete core or promising new blind tests. | Historical optional companion, not an additional mandatory lab. | 1.5–2.5 h |
| 16 supersonic design | Derive weighted Euler source[0,0,p,0], convert Cp using2.268 at M1.8, solve Taylor–Maccoll, compare ridge/MLP/GP, verify current checkpoint bytes/inference. | Two sessions plus3–5h independent; new proposal is unverified until CFD; core grades retained recomputation separately. No PLdB conclusion. | 5.5–7.5 h total |
| 17 attention | Hand QKV weighted result≈1.0094; constant V disproves weight-as-importance. Demonstrate 2D/3D dimensions and permutation equivariance. | Random kernel and saved trained heads at80/120/160 separately labeled. | 3–5 h total |
| 18 causal decoding | Shift all positions, trace embedding/position/4 width6 heads/24→48→24 FF; compute natural-log CE≈.7675 and PPL≈2.154. | Small character-model training; full overlapping validation-window CE is not unique-character document likelihood or CFD accuracy. | 3–5 h total |
| 19 representation/sensors | Invert patches; calculate attention memory576× ratio; distinguish orthogonal POD floor from gappy noise amplification. | Frozen saved basis; unregularized least-squares core. Sensor selection never uses evaluation truth. | 3–5 h total |
| 20 sensor learning | Remove value-coordinate tokens together; count actual trainable parameters; train with validation-only earliest-best selection. Show4% floor+3% dynamics=5%, not7%. | 120-step teaching fit and retained release comparison differ; augmentation is not architecture alone. | 3–5 h total |
| 21 forecasting | Plot121 uninterrupted predictions across50/71 boundary; separate phase origin from accumulating drift and global norm from mean per-frame ratios. | Observed initialization156:160; no reset at210; exact saved representation. Strict checkpoint agreement remains failed separately from scientific tolerance. | 3–5 h total |
| 22 transfer | Count44/84/154 labels including initialization; preserve signed paired gain, untrained controls and each representation floor. Explain source-POD access and why updates≠FLOPs. | Retained targets already inspected; explicit source-information/compute rubric points. No broad foundation-model claim. | 3–5 h total |
| Optional geometry–time bridge | Compute168/42/42 windows, masked means6.5 vs22/3, future-poison prediction invariance plus score sensitivity, circular2° rather than358°, undefined constant-signal phase. | Manufactured arrays only; no model fit/CFD/checkpoint/download. Four substantive exercises; not a paper reproduction. | 150–180 min |

## A completed run can be a scientifically unsuccessful teaching example

The independent execution reviewer reports that the default Week13 CPU fit
(1000 Adam updates and up to300 L-BFGS iterations) completed but produced about
**60.8% interior velocity error**, with held-out residual **5.648→10.28**. This is
a teaching failure of accuracy/generalization, not a verified accurate PINN.
Exact wall/continuity constraints and a falling training loss do not reverse that
conclusion. Do not tune the budget to conceal it. The PartA exercise should reward
a correct diagnostic comparison of training, held-out and field errors, including
a supported negative conclusion; computational completion does not demonstrate
that a student learned the lesson. These are the execution reviewer's fresh local
sequential-cell results, not a hosted Jupyter/Colab test or a human pilot.

## Grading depth and a human acceptance procedure

Weeks 17–22 allocate 15/100 to the three short helper tasks and 25/100 to the
integrated fourth task; 25 derivation, 25 interpretation and 10 reproducibility
remain separate. The reference checks can therefore certify at most the 40 coding
points. Week 18's block trace is a derivation deliverable. Week 22 now separately
assigns interpretation points to source access, compute accounting and paired-gain
scope. Each week supplies a worked reason and a misconception discriminator.
The optional bridge's 15 numerical/10 interpretation points per exercise similarly
prevent an implementation-only pass from certifying the scientific explanation.

For the other lessons, use their existing assignment criteria and the table's
exit checks. A response that only quotes the displayed metric is incomplete;
a response that names the physical quantity, information contract and limiting
counterexample supplies the evidence needed for an interpretation judgment.
Do not silently convert this review table into a new mandatory graded curriculum.

**Still required for human acceptance:** two representative Python-ready students
independently attempt selected lessons without seeing solutions; record active
time, stopping points and mistaken interpretations. Two graders independently
score three good/borderline/incorrect reports for each sampled assessment, compare
criterion-level scores and adjudicate disagreement. Include early UQ, a project,
one kinetic/units-limited lesson, Week13 or16, and Transformer18/22. No such
student or grading-agreement exercise was performed in this revision. Teaching
acceptance remains provisional until that evidence is available.

## Limits that must survive final approval

- Shared public-held-out cases are previously exposed; reframing them honestly
  does not create a new blind benchmark.
- Week7.1/9/12/15 source-unit and provenance gaps require source recovery. Clear
  labels bound the exercise; they do not supply the missing physical evidence.
- Current strict checkpoint reproduction fails although scientific tolerance
  passes. Keep `passed=False`, `strict_passed=False`, `scientific_passed=True`.
- Bounded local execution is not hosted Colab acceptance, full HPC reproduction,
  all-platform numerical identity, an independent CFD validation or learner success.
- Final visual review and exact TNR/black-text checks are separate gates. Text
  extraction confirms content presence, not readable equations or unclipped figures.
