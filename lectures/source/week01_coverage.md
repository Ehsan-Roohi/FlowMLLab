# Week 1: preservation and explanation audit

The 23-page edition is a reorganization and expansion, not a verbatim copy.
The first 13-page revision retained the sixteen original topic areas but
shortened some introductory definitions and replaced several diagrams with
prose. It was therefore not accurate to describe it as preserving every detail.
This follow-up restores those explanations and visual teaching aids, and adds
an extended finite-difference and code tutorial without changing the notebook.

[Original PDF, unchanged at baseline commit dfe1e86946b41833295fb3095a9f76ad21e61f77](https://github.com/Ehsan-Roohi/FlowMLLab/blob/dfe1e86946b41833295fb3095a9f76ad21e61f77/lectures/week01_numerical_foundations.pdf)

| Original section | Expanded edition |
| --- | --- |
| 1. Purpose | Page 1: learning question and all three companion notebooks |
| 2. What is a fluid? | Page 2: fluid/solid distinction, examples, density, pressure, viscosity, temperature glossary |
| 3. Continuum viewpoint | Page 2: scales, fields, and computational interpretation |
| 4. Eulerian description | Page 2: fixed positions and particle-following contrast |
| 5. Acceleration/material derivative | Page 2: chain-rule derivation and local/convective terms |
| 6. Continuity | Page 3: general and constant-density equations |
| 7. Navier–Stokes momentum | Page 3: vector statement, both components, and physical meanings |
| 8. Reynolds number | Page 4: scaling, interpretation, regime tendencies, dimensional versus nondimensional viscosity |
| 9. Boundary conditions | Pages 9 and 19: no slip, impermeability, wall vorticity, corners |
| 10. Streamfunction/vorticity | Pages 5–7: definitions, proof of original Eq. (15), optional transport derivation |
| 11. Finite differences | Pages 8 and 14–17: Taylor derivations, worked errors, stencil, slices, four-unknown system |
| 12. Residuals/convergence | Pages 11 and 21: change diagnostic, equation defects, stability, ML-loss connection |
| 13. Benchmarks | Page 12: Couette, Poiseuille, and cavity; Couette calculation on page 5 |
| 14. Ghia validation | Pages 12 and 21: centerlines, benchmark sampling, relative norms |
| 15. Pressure recovery | Pages 13 and 22: equation, consistent boundary data, gauge, diagram |
| 16. Learning outcomes | Page 13 plus checkpoints throughout |

## Visual teaching aids

- Streamlines: replaced by a fresh execution of the existing solver, accompanied
  by both centerline comparisons (page 12); the old image remains in the original.
- Five-point stencil: redrawn with physical and array directions and neighbor
  labels (page 16).
- Solver diagram: redrawn with the actual boundary-update order, inner Poisson
  sweeps, and fixed-count outer time steps (page 22).
- Pressure diagram: redrawn to include boundary data and pressure gauge (page 22).
- The original cover's notebook progression is retained as the three-notebook
  reading route, rather than as identical cover artwork.

## Clarifications rather than silent carry-over

The original solver diagram suggested tolerance-based termination, but the
actual notebook runs to a fixed step count. The revision explains this behavior.
It also distinguishes a small update from a small equation defect, clarifies
that 0.01 is the dimensionless diffusion coefficient at Re=100, and states the
pressure boundary conditions needed in addition to the Poisson right-hand side.
Equation numbering changes; the original Eq. (15) is explicitly identified in
its proof. All original reference works are retained.

## New code explanation

The tutorial covers `build_grid`, `laplacian`,
`apply_vorticity_boundary_conditions`, `solve_streamfunction_poisson`,
`compute_velocity`, `advance_vorticity`, `run_cavity`,
`centerline_profiles`, and `ghia_errors`. Code excerpts retain the original
operations; line wrapping and explanatory prose are added. It also explains
`meshgrid`, `[j,i]` indexing, slices, `.copy()`, in-place mutation, array shapes,
fixed sweep counts, step reporting, return dictionaries, and interpolation.

The notebook and solver algorithm are unchanged. The existing single-grid CFD
figure/record are retained from the earlier fresh execution. The new arithmetic
examples (cubic slopes, quartic curvature, a quadratic 2D Laplacian, a four-node
Poisson system, and a single vorticity update) were checked independently.

## Pressure follow-up: 27-page edition

The core coverage map above still applies. Pages 24–27 add the reason for
pressure elimination, a full componentwise pressure-Poisson derivation,
dimensional scaling, wall-normal momentum conditions, compatibility, gauge,
and analytic sign checks. Sources include the historical Thom (1933) paper
and the directly relevant Ghia et al. (1982) cavity benchmark; the notebook's
Jacobi/Euler algorithm is explicitly distinguished from Ghia's CSI-MG method.
The prior 23-page PDF is preserved in `lectures/versions/week01_23pages.pdf`;
the current edition has its own 27-page filename in that same directory.
