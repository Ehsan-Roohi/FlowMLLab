# Week 1.4 - Executed spectral and DG teaching experiments

[Student notebook](../../notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb) · [Lecture](../../lectures/week01_4_spectral_dg.pdf) · [Execution record](../../qa/week01_4_execution.json)

These outputs are recomputed by the standalone CPU notebook. `execution.json` records settings, package versions and numerical tables; CSV files retain the same data. Alongside interpolation, manufactured pressure and DG transport, the notebook solves **actual steady incompressible cavities at Re=100 and 400**, with Chebyshev degrees N=16,24,32. Each case starts from a zero algebraic guess; no cached CFD field or Ghia fitting is used.

| Evidence | Meaning |
| --- | --- |
| [Table A](table_A.csv) | Smooth and corner interpolation; maximum error on 2001 common points |
| [Table P](table_P.csv) | Dirichlet Poisson on a unit square; common 151x151 field errors and three repeated assembly/solve timings |
| [Table D](table_D.csv) | Equal 64-coefficient DG allocations for a localized periodic target |
| [Table F](table_F.csv) | Central/upwind transport of a square pulse; retained extrema and mass drift |
| [Table T](table_T.csv) | Fixed-space time refinement with SSPRK3 |
| [Table H](table_H.csv) | Fixed-degree h-refinement of a smooth profile |
| [Table C](table_C.csv) | Six actual cavities: settings, steady convergence, node/off-grid continuity, work and Ghia scores |
| [Table G](table_G.csv) | Every original Ghia sample, computed value, signed difference and predeclared flag |

All transport cases declare their speed, flux, cell count, degree, actual time step and final time in the notebook/record. The default speed is 1 and final time is 0.35. The square pulse has initial mass 0.3. Signed errors and overshoots are retained. No Mach or Reynolds number applies to these scalar model problems.

For the cavities, the open-edge lid speed and side length are one, the remaining walls are stationary, and viscosity is 1/Re. Upper corner velocities follow the stationary side walls. The pressure gauge is p(0.5,0.5)=0. `cavity_re{Re}_N{N}.npz` retains the computed fields, differentiation-based diagnostics and Newton history. Newton iterations are not physical time steps.

The core continuity RMS uses a fixed 101×101 grid in [0.05,0.95]². It is distinct from the tiny collocation residual and is not a full-domain norm. Corner singularities leave a visible spatial defect. Ghia provides centreline velocity samples, not a pressure-field reference. Re=400's previously flagged printed v value at x=0.9063 remains in the unfiltered score; the screened score omits only that entry. The Re=100 error does not decrease monotonically, so no universal spectral accuracy or speedup claim is made.

The nine independent tests cover polynomial derivatives, manufactured Poisson, DG conservation/energy/time refinement, a manufactured full Navier-Stokes velocity-pressure field, the exact Newton Jacobian, cavity boundary/gauge conditions, Ghia error bounds, spatial continuity refinement and explicit nonconvergence. Timings describe this CPU execution. The DG solver is scalar transport; the actual cavity solver is Chebyshev collocation.
