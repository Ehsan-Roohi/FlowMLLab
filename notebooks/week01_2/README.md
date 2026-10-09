# Week 1.2: pressure-velocity coupling in a Re=100 cavity

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb) | [Lecture](../../lectures/week01_2_pressure_velocity.pdf) | [Executed comparison](../../results/week01_2_pressure_velocity/README.md)

An independent extension after Week 1 (and optionally Week 1.1). All original
Week-1 notebooks, source modules and retained results remain unchanged.
The initial qualified case is **Re=100 only**.

[Expanded comparison and teaching supplement](COMPARISON_AND_TEACHING.md)
adds three algorithm flowcharts, a primary-source literature review, a measured
15-configuration one-step coupling experiment, and a staged higher-Re and
transient-accuracy plan. The lecture now contains 18 pages. A smaller nonlinear
step defect is distinguished explicitly from a smaller physical time error.

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
