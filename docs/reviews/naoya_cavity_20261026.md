# From a student cavity study to a reproducible comparison

Naoya Miyamoto's three reports, received on 26 October 2026, cover the
streamfunction-vorticity foundations in Week 1 and pressure-velocity coupling
in Week 1.2. They do not provide evidence for the later machine-learning units.
The useful contribution is a worked comparison of coupling cost and mesh error,
supported by a plain-language description of the algorithms.

## What the reports contribute

| Report | Useful material | How it is used |
| --- | --- | --- |
| IS HW, 3 pages | Streamfunction and vorticity definitions; Poisson solve; finite differences; a flowchart connecting vorticity, streamfunction and velocity | The definitions agree with the existing Week 1 lesson. A y-derivative denominator needs correction on unequal spacings; a manufactured-field check is added to Week 1. |
| SIMPLE and PIMPLE Algorithms (5), 9 pages | Predictor, continuity imbalance, pressure correction, and nested PIMPLE loops; two readable flowcharts | The sequence is retained as a reading aid. Coefficient normalization and time-level notation need the corrections below. |
| Lid-Driven Cavity - Re 400 and 1000, 13 pages | Twelve central-convection cases: three methods, two Reynolds numbers, two grids; residuals, pressure-solve counts and profile errors | The reported scalars are transcribed separately from fresh calculations. The refinement lesson is added to the Week 1.2 notebook. |

The report PDFs are the private review inputs. Public teaching material consists
of the attributed summary, corrected explanations, selected numerical values
and fresh reproducible figures. Source-file hashes and page references are
recorded in the [provenance record](../../results/week01_2_student_review/provenance.json).

## Read convergence and accuracy as different measurements

The report uses a 5% profile-error target. It is a teaching acceptance threshold,
not a universal CFD standard. Its reported errors, rounded to two decimal
places, are the same for the three coupling methods at a fixed Reynolds number
and grid:

| Re | Cells per side | Reported u relative L2 error | Reported v relative L2 error |
| --- | ---: | ---: | ---: |
| 400 | 32 | 3.61% | 8.43% |
| 400 | 64 | 0.83% | 1.85% |
| 1000 | 32 | 8.67% | 14.15% |
| 1000 | 64 | 2.59% | 3.46% |

At Re=400 the vertical-velocity norm excludes the published anomalous Ghia
sample at x/L=0.9063. Its original value remains visible as an unscored marker
in the benchmark plots; no replacement value is invented. At Re=1000 all
tabulated samples are used. The relative norm is

$$
E_f=\frac{\sqrt{\sum_{j\in S}(f_{\mathrm{computed},j}-f_{\mathrm{reference},j})^2}}
{\sqrt{\sum_{j\in S}f_{\mathrm{reference},j}^2}},
$$

where S is the stated set of scored reference positions. Computed profiles are
interpolated to those positions. u is sampled on the vertical centerline and v
on the horizontal centerline.

Every reported final momentum defect is below 1e-6 and every reported final divergence
below 1e-9, yet the 32-cell cases fail at least one 5% profile target. Small
equation defects show that the discrete equations have been solved closely.
They do not show that the grid represents the continuous flow accurately.
At the same Reynolds number and grid, converged SIMPLE, PISO and PIMPLE target
the same spatial equations. Changing the coupling loop primarily changes the
work needed to reach that solution. Refinement changes the spatial equations.

For the same grid, increasing Re reduces viscous smoothing relative to inertia;
steeper wall gradients can need more resolution. Two grids demonstrate a trend,
not an asymptotic convergence order, grid independence, or the physical stability
of a three-dimensional flow.

The final defects summarize the end of each run. The report's plotted momentum
histories start at larger values and decay; a statement that these defects were
small throughout would misread those histories.

## Use the correct spacing for each derivative

On page 2 of IS HW, the central approximation of du/dy uses 2 Delta x in the
denominator. The denominator must be 2 Delta y. Equal square-grid spacings hide
this distinction. Week 1 now includes a small independent check with
psi=x^2 y^3, so u=3x^2 y^2, v=-2xy^3 and omega=-2y^3-6x^2y.
At (x,y)=(0.4,0.6), with Delta x=0.1 and Delta y=0.2, the correct central
differences give omega=-1.008. Reusing Delta x in the y derivative gives -1.584.
This manufactured field satisfies incompressible continuity and checks the
derivative implementation; it is not a no-slip cavity solution.

## Count the work performed by the loops

In the submitted central-convection runs, SIMPLE uses one pressure solve per
iteration, PISO two, and PIMPLE six. The PIMPLE count is consistent with three
outer passes and two inner corrections per pass in the course configuration.
These are algorithm-level pressure solves, not iterations inside a linear solver.
PIMPLE's outer loop updates momentum coefficients while the inner loop corrects
pressure and velocity. Physical old-time fields stay fixed until the step ends.
The [CFD Direct algorithm controls](https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution)
describe this distinction; FlowMLLab implements the loops directly in Python.

For Re=1000 on 32 cells, the report records 610 pressure solves for SIMPLE,
4740 for PISO and 14640 for PIMPLE, with the same rounded velocity errors.
That observation does not establish a universal method ranking. A pressure
solve is also not a complete cost model: momentum factorization, mesh size,
linear-solver implementation and host matter. Report wall time separately.
The submitted timings are single observations with no attached machine
specification, raw execution log or repeated timing sample. They remain
**student-reported timings**, not independently reproduced speedups.

## Correct coefficient normalization before comparing formulas

The report writes a velocity-correction sensitivity as volume divided by the
momentum diagonal. The course code uses the reciprocal of a volume-normalized
diagonal. Those expressions agree when pressure and equation units agree.
For the course's dimensionless pressure p and a velocity control volume V_e,
write the integrated momentum correction as

$$
\widehat a_e u'_e=-V_e\frac{\partial p'}{\partial x}.
$$

Here a hat marks the coefficient before division by volume, and u'_e and p'
are additive corrections. Dividing the row by V_e defines

$$
a_e=\frac{\widehat a_e}{V_e},\qquad
d_e=\frac{V_e}{\widehat a_e}=\frac{1}{a_e},\qquad
u'_e=-d_e\frac{p'_E-p'_P}{h}.
$$

h is the distance between neighboring pressure centers P and E. With dimensional
pressure p_d, the corresponding acceleration uses p_d/rho, where rho is constant
density. Do not transfer a coefficient between dimensional, integrated and
volume-normalized equations without these conversions. The code's matrix D
denotes divergence; d_e denotes the face correction sensitivity.

## Keep old-time data separate from the current predictor

The central-stencil example on page 6 of the algorithm report labels the west
and south transported neighbors with the old time n, while the other transported
neighbors carry the predictor star. That is inconsistent with the implicit row
and the coefficient formulas printed immediately below it. All transported
neighbor unknowns in that row must use the same predictor level.

For the actual finite-volume course formulation, backward Euler adds
1/Delta t to the normalized momentum diagonal and u_e^n/Delta t to its source:

$$
\left(a_e^{(s,k)}+\frac{1}{\Delta t}\right)u_e^*
=\sum_{g\in\mathrm{NB}(e)}a_{eg}^{(k)}q_g^*
+b_e^{(s,k)}+\frac{u_e^n}{\Delta t}-\frac{p_E^k-p_P^k}{h}.
$$

The superscript s identifies the spatial row, n the completed physical time
level, k the current outer estimate, and a star the unknown momentum predictor.
NB(e) lists neighboring velocity locations, q_g their velocity components,
and b_e contains known boundary/source contributions. The same construction
applies to the vertical-velocity row. Convection coefficients may be updated
from the current outer estimate; the old-time source remains unchanged across
all outer passes of that physical step. An advective finite-difference stencil
is not the conservative MAC finite-volume implementation used by the code.

## What the upwind discussion can support

The report's last page compares central and first-order upwind profiles
qualitatively. It does not tabulate an upwind error or a complete run
configuration. First-order upwind adds numerical diffusion and can smooth
gradients; that is a reason to measure the effect, not proof of its magnitude.
The fresh study includes a matched Re=400, N=32 SIMPLE upwind case and its
central counterpart. Their actual profiles and errors are provided below.
No higher-order scheme or new physical model is introduced by this review.

## Fresh checks and classroom use

[Fresh numerical evidence and figures](../../results/week01_2_student_review/README.md)
are kept separate from the [transcribed report table](../../results/week01_2_student_review/submitted_summary.csv).
The fresh suite checks all three coupling methods on N=32 at each Reynolds
number, SIMPLE on N=64 at each Reynolds number, and the matched upwind case.
It does not independently rerun the four N=64 PISO/PIMPLE report cases.

1. Identify which reported cases pass the equation-defect targets and which
   pass both profile-error targets. Explain why those decisions differ.
2. Compute pressure solves per reported iteration. Relate 1, 2 and 6 to the loops.
3. Compare the fresh coarse-grid method curves, then coarse and fine SIMPLE
   curves. Locate the regions that change instead of only quoting an error norm.
4. Compare central and upwind at matched Re, grid, stopping target and method.
5. State what additional grid, time-step, configuration and timing evidence
   would be needed before claiming an accuracy order or a general speedup.

The current pressure-Poisson, vorticity and pressure-correction derivations in
Weeks 1 and 1.2 remain the authoritative equations. In particular, a vorticity
formulation still uses momentum conservation, and pressure recovery still needs
consistent boundary conditions and a pressure gauge.
