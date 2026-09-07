# Specification: boundary volume-flux diagnostic

Complete and commit this document BEFORE supplying it to a coding agent.
This is a student template; the existing `SCIENTIFIC_SPEC.md` describes the
instructor's separate derivative-verification example.

## Physical problem

Compute net outward volume flux per unit depth around a rectangular 2-D
velocity field. Explain its connection to incompressible mass conservation
and why zero global flux does not imply zero local divergence.

Student explanation: [complete]

## Inputs and outputs

Function: `net_volume_flux(x, y, u, v) -> float`.
Coordinates: finite, strictly increasing 1-D arrays with at least two points.
Velocities: finite arrays of shape `(len(y), len(x))`; `field[y,x]` semantics.
Output: signed outward boundary integral using composite trapezoidal quadrature.
Write the four signed boundary integrals explicitly: [complete]

## Units and nondimensionalization

For x,y in metres and u,v in m/s, volume flux per depth has units m squared/s.
With x*=x/L and u*=u/U, Q*=Q/(U L). Multiplication by a constant density gives
mass flux per unit depth. This function does not integrate variable density.
State the units and characteristic U,L for every proposed test: [complete]

## Validity

Axis-aligned rectangular domains, collocated boundary velocities, fixed mesh.
Composite trapezoidal quadrature; increasing nonuniform coordinates allowed.
No moving-boundary, compressible mass-flux, curved or unstructured-grid claim.
Explain discretization error for nonlinear boundary traces: [complete]

## Frozen numerical acceptance criteria

Use absolute error <=1e-12 for the exactly integrable normalized reference
cases in ASSIGNMENT.md. Existing Week 1.1 acceptance gates must still pass.
For your additional nonuniform/rectangular test, derive its exact reference
and justify a numerical tolerance before implementation: [complete]

## Failure cases

Raise ValueError for incompatible shapes, nonfinite inputs, repeated or
decreasing coordinates, or fewer than two points in a direction.
Test wrong outward signs and swapped axes with non-square domains.
Explain how the boundary perturbation exposes a function that always returns
zero: [complete]

## Verification plan and revision history

List separate unit, regression, invariant and independent-reference tests.
Record specification commit, later revisions and reasons in PROCESS_LOG.md.
Do not select thresholds after viewing candidate results.
