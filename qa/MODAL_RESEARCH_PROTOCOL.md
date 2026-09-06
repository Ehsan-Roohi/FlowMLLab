# Research qualification before publication

Status: proposed qualification program, NOT research results. Keep main and the
front page unchanged. The earlier teaching benchmark is frozen at ca570ae.

## Gate 1 - continue, do not replace, the failed grid study

Run Re100 with D/dx=40, 800x320 cells, U=0.05, a 20D x 8D periodic-transverse
domain and cylinder center at x/D=5. Retain the original 100D/U duration and
discard the first 45D/U, so the only primary independent change is resolution.
TRT/Bouzidi, initial perturbation and startup protocol remain unchanged. Tau
is recomputed from Re, U and D, not tuned against a reference answer.

Use retained grids18/27 plus new40 for generalized unequal-ratio Richardson/GCI.
Keep the original gates: fine-pair changes 3/2/5% and GCI 5/3/8% for Cd/St/Lr.
Require statistical convergence on every contributing grid. Non-monotone or
non-positive-order sequences FAIL; do not drop an inconvenient grid or relax a
threshold after seeing results. Another refinement is a new declared experiment.

`qualify_modal_cfd.py` records source/data hashes before the run, refuses existing
output directories and cannot mark research_ready even if the grid gate passes.
An 18/27/40 fixed-domain pass would not retroactively validate 12-node ML labels.

## Gate 2 - distinguish confinement, compressibility and sampling

The solver models a transverse periodic array, not isolated-cylinder far-field
boundaries. The existing 8D height means 12.5% blockage. Do NOT use an isolated
cylinder drag/Strouhal band as a matched validation of that confined setup.

After Gate1, predeclare production-scale comparisons before inspecting results:

- Domain: at fixed accepted grid, Re and U, compare 30D x 20D (8D upstream)
  with 40D x 30D (12D upstream), preserving periodic transverse boundary type.
  This combined domain enlargement screens total domain sensitivity, not a
  separate causal estimate of inlet, outlet and lateral effects. Isolate each
  dimension in follow-up if it fails. Initial screening limits: Cd/St <2%, Lr <5%.
- Mach: repeat the selected geometry at U=0.025 versus0.05 with fixed Re and
  spatial resolution. Lattice relaxation/time step changes consistently. This
  screens combined acoustic-scaling/compressibility sensitivity; it is not an
  independent pure time-discretization test. Same screening limits as domain.
- Sampling: run at least200D/U; inspect post-transient blocks and require at
  least16 shedding cycles, <2% drag block change and <10% lift-RMS block change.
  Failure requires a longer declared run, not cherry-picked phases.
- External: use an independent solver/measurement with matched Reynolds,
  geometry, blockage and boundary conditions, or establish the isolated-domain
  limit before using isolated-cylinder reference bands. Record reference
  uncertainties and exact source; no matched external reference is approved yet.

These are declared engineering screening criteria, not a universal certificate
of research quality. Grid/domain/Mach interactions may require another grid
study on the final selected geometry. No mesh is accepted for training solely
because an unrelated fixed-domain grid gate passed.

## Gate 3 - a fresh, frozen ML experiment

Only after CFD qualification, generate new long trajectories with identical
accepted numerical settings across complete Reynolds cases. Proposed roles:
training Re80/100/120; validation Re90/110; test Re95/105. Reynolds values were
studied previously, so describe this as fresh qualified trajectories, not
never-studied flow physics. Seal test trajectories until hyperparameters freeze.

Use training-only POD/scaling and compare matched-rank, matched-history linear
models, DMD and neural forecasts. Include strong interpolation baselines for
parameter reconstruction; they are not substitutes for future-time forecasting.
Evaluate autonomous forecasts over at least10 shedding cycles, with no reset.
Predeclare field error, worst-case error, phase, spectrum, physical diagnostics
and all training seeds. No average may hide a divergent case.

For sensing, use identical measurement/noise budgets, enough random layouts
to characterize their distribution, and complete independent CFD cases as the
statistical unit. Artificial sensor-noise seeds are not independent flow cases.

SINDy is not a front-page candidate now. Include transient/off-attractor data,
report sparsity-versus-error and long-horizon stability, and test alternative
initial conditions. The earlier 18/20-term model is weakly sparse and is not a
discovery of unique governing equations. A higher-dimensional or stability-
constrained model needs its own frozen validation experiment.

## Publication rule

Publish a research claim only after all relevant gates pass AND the scientific
contribution is clear beyond using existing algorithms. A positive result is
not guaranteed. Retain failed experiments and all reported baselines. Select
front-page figures for a verified, accurately scoped finding, not attractiveness.
No new release or DOI during this qualification stage.

## Commands from repository root

```bash
python qa/qualify_modal_cfd.py plan
python qa/qualify_modal_cfd.py smoke --output tmp/modal-cfd-smoke
python qa/qualify_modal_cfd.py run --output tmp/modal-cfd-D40
python qa/qualify_modal_cfd.py inspect --output tmp/modal-cfd-D40
```

The production run is CPU/memory-bandwidth bound, not GPU accelerated. The
current NumPy implementation is serial: requesting many cluster CPU cores
will not automatically parallelize a single case. Time a short full-size pilot
before selecting a batch time limit. A smoke run is never CFD validation.

Local timing pilot (2026-09-05): 100 steps on the production-size800x320 grid
took53.90 seconds, without startup ramp and with no physical-validation claim.
Linear extrapolation to80000 steps is approximately12 hours on this machine;
startup costs, machine load and cluster hardware make this only a rough estimate.
Production has NOT been launched. Confirm a Unity login and suitable batch
resources before submitting; no SSH host alias is configured in this workspace.

## Primary methodological sources

- NASA Glenn, [Examining Spatial (Grid) Convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html):
  distinguish asymptotic grid verification from physical validation.
- PySINDy, [original SINDy examples](https://pysindy.readthedocs.io/en/latest/examples/sindy-original-example/30f4dcd/original_paper.html)
  and [Trapping SINDy](https://pysindy.readthedocs.io/en/stable/examples/sindy-addl-examples/eba3fdb/example.html):
  motivate transient wake information and explicit stability checks, not copied code/results.
