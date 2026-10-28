# Week 1.2: pressure-velocity coupling in a Re=100 cavity

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb) | [Lecture](../../lectures/week01_2_pressure_velocity.pdf) | [Executed comparison](../../results/week01_2_pressure_velocity/README.md)

An extension after Week 1 (and optionally Week 1.1), connecting finite-volume conservation to pressure correction. The baseline case is Re=100; separate retained experiments compare Re=400 and 1000.

[Worked student study: convergence versus mesh accuracy](../../docs/reviews/student_cavity_20261026.md)
includes a 12-case student report table, selected fresh Python checks,
physical velocity profiles, and exercises on pressure-solve work and numerical
diffusion. Submitted measurements and fresh author-side runs are labeled
separately. Section 11 of the notebook displays the comparison without running
new CFD. Two additional reading notes explain momentum-coefficient normalization
and old-time data in PIMPLE.

[Expanded comparison and teaching supplement](COMPARISON_AND_TEACHING.md)
adds three algorithm flowcharts, a primary-source literature review, a measured
15-configuration one-step coupling experiment, and a staged higher-Re and
transient-accuracy plan. The lecture contains 59 pages. A smaller nonlinear
step defect is distinguished explicitly from a smaller physical time error.

The illustrated lesson locates stored and interpolated velocities on separate
pressure and momentum volumes, follows one face's momentum flux, and draws
the SIMPLE, PISO and PIMPLE loops. Short arithmetic examples connect the
diagrams to pressure correction before the matrix notation.

## Learning outcomes

- Derive a pressure equation from finite-volume continuity and momentum.
- Distinguish a physical time step, a coupling iteration and a linear solve.
- Explain SIMPLE under-relaxation, PISO's additional momentum corrections,
  and PIMPLE's outer nonlinear iterations.
- Check face-flux mass conservation and pressure-gauge invariance.
- Compare the same steady discrete solution with different iteration costs.
- Use an independent streamfunction-vorticity formulation and Ghia profiles
  as evidence while preserving their different spatial discretizations.

## Shared Python implementation

[pressure_velocity.py](../../flowmllab/pressure_velocity.py) builds a MAC grid:
pressure `p[j,i]` is at cell centers; `u[j,i]` is on vertical faces;
`v[j,i]` is on horizontal faces. All arrays use y-then-x ordering.
The unit-square lid speed and density are 1; nu=0.01. Normal wall velocities
are exactly zero. Tangential wall values use half-cell diffusion distances.
The discontinuous moving-lid/stationary-wall corners are not smoothed.

Momentum: `A q = b - G p`. Define `d=1/diag(A)` and
`H(q)=b-offdiag(A)q`. Continuity gives the positive pressure operator
`L=-D d G`. Every shared face flux enters neighboring cells with opposite
signs. The closed-cavity pressure matrix has a constant nullspace; one cell is
pinned for the sparse solve and the reported pressure is shifted to zero mean.

**SIMPLE:** assemble and under-relax steady momentum; predict velocity;
solve `L p'=-D q*`; correct face velocities with the full correction;
update `p <- p + alpha_p p'`; repeat. Alpha_u=0.7, alpha_p=0.3.

**PISO:** assemble backward-Euler momentum with immutable old-time velocities;
predict velocity; recompute H, solve pressure and correct velocities twice.
Recomputing off-diagonal momentum contributions is essential: two linear
solver sweeps on one pressure equation are not two PISO corrections.

**PIMPLE:** repeat momentum assembly/prediction and the two PISO corrections
three times within one physical time step. Old-time velocities are fixed
across those outer loops. One outer loop is exactly the implemented PISO limit.

The recorded FV schemes use central face convection and central diffusion.
First-order upwind is an optional teaching control, not part of the retained
comparison. PISO/PIMPLE use backward Euler (first-order time accuracy), dt=0.05,
and march until the same steady FV momentum/continuity tolerances pass.
Sparse LU via SciPy solves linear systems; no multigrid or packaged CFD solver.

## What is compared

Two matched spacings: 32 and 64 FV cells per side, versus 33 and 65 nodes for
the unchanged streamfunction-vorticity reference. Physical contour scales,
Ghia velocity centerlines, zero-mean pressure, vorticity and streamlines are
shown. FV steady momentum Linf and cell divergence Linf are logged against
measured CPU wall time. SIMPLE iteration count is not physical time.

The independent reference uses `common/w4utils.py` unchanged: explicit central
vorticity transport, a DST streamfunction Poisson solve, and steady momentum
pressure recovery. It has a different residual definition; its update residual
is not overlaid as if it were the same FV momentum defect.

## Scope and next experiment

Agreement among the three FV methods establishes their common discrete steady
limit; it does not independently validate continuum accuracy. Ghia validates
velocity profiles, not pressure. Two grids establish a refinement trend, not
asymptotic order. PIMPLE's possible benefit at larger time steps is not claimed
by a steady run. The next study should refine dt for startup or an oscillating
lid, then compare methods at matched temporal error. A transient reference
pressure recovery must include acceleration terms.

The retained FV errors against Ghia increase slightly from 32 to 64 cells,
while the FD reference errors decrease. This nonmonotone FV trend is reported
explicitly. A further mesh study and another velocity reference are needed
before claiming asymptotic accuracy; its cause has not been established.

## Run and reproduce

Run all notebook cells to view the eight executed comparisons. In the fresh-run
cell, set `RUN_NEW=True`, leave `METHOD="all"` and `CELLS=32` to recompute
SIMPLE, PISO and PIMPLE at Re=100. Individual method selection is also supported.
The results use real Python computations, with no embedded encoded source.

The notebook and its all-three-method fresh-run branch were executed locally
without errors. The Colab launcher is provided; hosted Colab execution has not
been tested. [Validation evidence](../../results/week01_2_pressure_velocity/validation.json)
records the environment, numerical tests, notebook execution and preserved
Week-1 source hashes.

For development, install the project test extras and `pdfplumber`, then run
`python qa/validate_week01_2.py` after rebuilding the results and notebook.
The PDF builder uses the installed Windows Times New Roman font files under
`C:/Windows/Fonts`. The retained PDF and PNG figures embed the rendered font;
fresh plots in Colab use an available serif fallback when that font is absent.

## Exercises

1. Sum cell divergences and show cancellation of all interior face fluxes.
2. Add a constant to pressure; show that every velocity correction is unchanged.
3. Compare one and two PISO corrections using the one-step momentum defect.
4. Set PIMPLE outer_correctors=1 and compare every field with PISO.
5. Change only SIMPLE relaxation and compare cost at the same final tolerance.
6. Compare central and upwind convection without attributing spatial diffusion
   to the pressure-coupling algorithm.

## Reading and provenance

This is an independently written educational solver with explicit attribution:

- [PySIMPLE](https://github.com/VishalKandala/PySIMPLE): educational staggered FV
  pressure correction and separation of solver/diagnostics.
- [NIST FiPy Stokes cavity](https://pages.nist.gov/fipy/en/latest/generated/examples.flow.stokesCavity.html): pressure-correction derivation and grid-placement discussion. Its Stokes example does not validate this finite-Re solver.
- [OpenFOAM algorithm control](https://www.openfoam.com/documentation/user-guide/6-solving/6.3-solution-and-algorithm-control), [icoFoam source](https://github.com/OpenFOAM/OpenFOAM-10/blob/master/applications/solvers/incompressible/icoFoam/icoFoam.C), and [pimpleFoam source](https://github.com/OpenFOAM/OpenFOAM-10/blob/master/applications/solvers/incompressible/pimpleFoam/pimpleFoam.C): correction-loop semantics.
- [pyOpenFOAM PIMPLE source](https://github.com/alanZee/pyOpenFOAM/blob/master/src/pyfoam/solvers/pimple.py): another inspectable Python loop organization; not a reference truth for our results.
- [kangluosee/CFD-Python](https://github.com/kangluosee/CFD-Python): an educational cavity notebook collection. The audited PISO-titled notebook had its second correction commented out, so its name alone was not used as algorithm verification.
- Ghia, Ghia and Shin (1982), *Journal of Computational Physics* 48, 387-411,
  DOI [10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4): benchmark velocity tables.

No external implementation is copied into this module. Validation records
retain configuration, stopping condition, source hashes and field hashes.

## Higher-Re experimental investigation

The notebook now includes a separate, opt-in Re=400/1000 investigation with
explicit student deliverables, error targets, refinement, coupling controls,
retained evidence and failure reporting. It uses
`flowmllab.pressure_velocity_experiments.ExperimentalConfig`; the original
Re=100 configuration, solver and retained results remain unchanged.

See [measured higher-Re results](../../results/week01_2_higher_re/README.md).
Run `python qa/run_week01_2_higher_re.py --cells 32` to reproduce the first
comparison. Refine with `--cells 64 --methods simple`, then decide whether
128 cells are justified. Central/upwind comparisons must use the same mesh.
This is a 2D steady experimental study, not a turbulence or transient-accuracy
qualification. There is no universal maximum Re for a coupling algorithm.

## Before running higher Reynolds numbers

**Runtime and hardware warning.** Higher Reynolds numbers generally need finer meshes
and more iterations to resolve the flow. These runs can take substantially longer
and use more memory; a stronger CPU and more RAM may be needed. Doubling cells per
side quadruples the number of 2-D cells, and total work can grow by more than four
times because convergence and linear solves also change. Start at 32 cells, run
one method, measure its time, then decide whether to refine. Do not launch every
128-cell case at once. Cached figures below require no new CFD solve. Fresh runs
are opt-in; there is no guaranteed completion time on a student's computer.


### Why is streamfunction-vorticity absent from the higher-Re comparison?

The Re=400/1000 extension originally measured only SIMPLE, PISO and PIMPLE.
That was a study-scope omission, not a limitation of streamfunction-vorticity.
The existing Week-1 `common/w4utils.py::run_cavity(Re=..., N=...)` accepts these
Reynolds numbers. It solves streamfunction and vorticity, then recovers pressure
separately. In two-dimensional incompressible flow this is an appropriate
independent formulation for the same cavity problem.

A fair extension must actually run that solver, check time-step stability and
steady convergence, compare matching physical grid spacing and centerlines,
and retain its own vorticity-update residual. That residual is different from
the FV momentum defect: their curves must not be ranked as the same quantity.
A small update alone cannot establish steady accuracy. Section 10 now supplies actual higher-Re streamfunction-vorticity runs and a
sequential four-method comparison. See the independent-formulation report.

**Reading the figures:** signed fields share limits across methods. A symmetric
logarithmic color scale resolves small interior values and large wall gradients.
Black curves and arrows follow computed velocity on every field panel, including
pressure and vorticity. Corner seeds make resolved small recirculation easier
to see; they cannot create a missing vortex or compensate for a coarse mesh.
Streamlines on a pressure plot are velocity trajectories, not pressure contours.
Use the colorbar values when comparing signed fields.

## Independent formulation at Re=400 and 1000

Section 10 compares fresh streamfunction-vorticity and SIMPLE/PISO/PIMPLE runs
at matched spacing, with Ghia centerline errors, separate native convergence
plots, sequential CPU timings and flow fields with streamlines.

[Report and retained fields](../../results/week01_2_formulations/README.md).
Fresh FD calculations are optional through `RUN_STREAMFUNCTION`. Start at 32
cells and allow more time and memory for finer grids. FD pressure is recovered
after solving velocity; the residual definitions differ between formulations.

```sh
python qa/run_week01_2_formulations.py
python qa/run_week01_2_formulations.py --cells 64 --methods simple streamfunction-vorticity
python qa/run_week01_2_formulations.py --cells 128 --methods streamfunction-vorticity
python qa/report_week01_2_formulations.py
```

Verify archives and execute the retained-results notebook with
`python qa/validate_week01_2_formulations.py` after the runs finish.

## Equations and algorithm guide

The notebook and 26-page lecture define continuity, both momentum components, all matrix symbols, pressure correction and the SIMPLE/PISO/PIMPLE loops. Re=100, 400 and 1000 timing comparisons include the independent streamfunction-vorticity method. Native FD and FV residuals appear on separate axes.

[Teaching revision validation](../../results/week01_2_formulations/teaching_revision_validation.json) records the notebook controller audit and PDF checks.

## Lecture and algebra examples

The lecture develops scalar cell mass balance and face momentum before introducing matrix operators. It derives the pressure-correction stencil, explains SIMPLE relaxation, and separates PISO inner corrections from PIMPLE outer coefficient updates.

Pages 13-14 calculate two PISO corrections; pages 15-16 explain PIMPLE and calculate a nonlinear coefficient example. The notebook runs both algebraic examples and shows their mass and momentum defects. These examples illustrate coupling and are distinct from the retained cavity simulations.

The references include Patankar, Versteeg and Malalasekera, Ferziger et al., Moukalled et al., the original Issa PISO paper and official algorithm documentation.

Editable source: [LaTeX lecture](../../lectures/source/week01_2_pressure_velocity.tex) and [core equations](../../lectures/source/week01_2_pressure_velocity_core.tex). Build with `python qa/build_week01_2_assets.py` followed by `python qa/build_week01_2_lecture.py`; Tectonic and licensed Times New Roman are required. [Lesson validation](../../results/week01_2_formulations/coupling_lesson_validation.json).

## Figures and companion explanations

The lecture has 20 numbered figures with captions; the notebook uses the same identifiers and adds 13 S-numbered supplements. Stored velocities, interpolated momentum-face values and north-face transport are shown separately at full width. Physical fields and centerline curves are redrawn from hash-checked archived arrays; recorded runtimes are unchanged. The end of the lecture and notebook links free SIMPLE/PISO YouTube lectures and CFD Direct readings for pressure correction and PIMPLE.

To reproduce the figures and their notebook guide:

```bash
python qa/refresh_week01_2_figures.py
python qa/integrate_week01_2_figure_guide.py
python qa/sync_week01_2_teaching.py
python qa/build_week01_2_lecture.py
python qa/validate_week01_2_figure_guide.py
```

The figure validator executes the inexpensive interpolation examples and retained-result tables. Fresh CFD solves remain opt-in.

The teaching sequence starts with the cavity physics, then defines control volumes and contrasts finite volume with finite differences before integration. Momentum notation proceeds from one scalar row and its known source to vectors and operators. Pressure correction, relaxation and residual checks precede SIMPLE; the PISO and PIMPLE sections define their changing fields and fixed data before the loop equations. Video links are blue in the lecture PDF.
