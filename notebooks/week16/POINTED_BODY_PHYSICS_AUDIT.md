# Pointed-body enthalpy defect: source-supported interpretation

The retained eight CFD cases fail the declared full-field 10% maximum total-enthalpy deviation criterion. The numerical neural/CFD error comparisons remain reproducible, but they do not override this failed physical check. No nodes or regions have been excluded and no threshold has changed.

## What is observed

Across all eight cases, the maximum excess in total enthalpy is 17.15–18.70%, always at the shared trailing-tip/axis node (x/L=1, r/L=0). Between two and seven of 81,399 volume nodes exceed 10%; every such node is on the axis, including the nose and immediate tail region. The 99th-percentile absolute deviation is 0.92–1.60%. Density and pressure are positive and the stagnation-density bound passes.

For case 000, velocities at both shared tip nodes are effectively zero (components below 5e-14 m/s). The trailing-tip specific total energy is 399239.40 J/kg; with zero velocity this yields h0=1.4E, 17.15% above the freestream value 477104.52 J/kg. The freestream is Mach 1.8, 288.15 K and 101325 Pa.

## What the official SU2 v8.5.0 source establishes

1. [CFVMFlowSolverBase.hpp, lines 1130–1133](https://github.com/su2code/SU2/blob/v8.5.0/SU2_CFD/include/solvers/CFVMFlowSolverBase.hpp#L1130) delegates the Euler-wall boundary to the symmetry-plane implementation.
2. [CFVMFlowSolverBase.inl, BC_Sym_Plane, lines 1170–1330](https://github.com/su2code/SU2/blob/v8.5.0/SU2_CFD/include/solvers/CFVMFlowSolverBase.inl#L1170) obtains possibly modified normals, projects normal conservative momentum out of the old solution, and removes the normal momentum residual. The stationary-grid projection itself does not modify density or total-energy entries. Grid-motion energy handling is a separate branch.
3. [CGeometry.cpp, ComputeModifiedSymmetryNormals, lines 2857–3008](https://github.com/su2code/SU2/blob/v8.5.0/Common/src/geometry/CGeometry.cpp#L2857) handles both Euler walls and symmetry planes. At shared nodes, normals for nonparallel markers are orthogonalized; nearly parallel markers are treated separately. A curved body meeting a straight axis at a sharp tip can therefore impose two independent momentum constraints in the meridional plane.

For an ideal gas, specific total enthalpy can be written

h0 = gamma E - (gamma-1)|u|²/2.

Removing a velocity component u_n while holding density and total energy fixed increases h0 by (gamma-1)u_n²/2. This algebra establishes that the implemented projection can introduce a local enthalpy excess. Zero velocities at both retained shared tip nodes and the axis localization are consistent with that mechanism.

## What is not established

The source inspection does not prove that boundary projection alone quantitatively accounts for the converged 17–19% error. The complete residual, axisymmetric source discretization, mesh and iteration history also affect the eventual solution. Calling the defect a proven harmless singular-node error would require additional controlled evidence. Conversely, these localized maxima do not demonstrate a domain-wide energy failure. The accurate statement is a localized physical-plausibility failure with a source-supported boundary-treatment hypothesis.

The 10% gate remains failed. Near-field waveform and drag comparisons remain numerical comparisons with these CFD labels. Independent mesh/solver studies or a boundary/discretization correction, followed by unchanged acceptance tests, would be needed before making stronger validation claims.

## Retrieved source SHA-256

- CFVMFlowSolverBase.hpp: ff00be37735ec54fab8969bd1db716ace6a3bb4b8fc68745179ee0e4501d9f65
- CFVMFlowSolverBase.inl: b8e54f986ce075052e54586c6f12026c785262f96875426bb6e918ae374b3c46
- CGeometry.cpp: af479d996b6493cb2ac11c031b17f99511806ac94afd2d7ee05c5a4f6d54e79f

## Bounded recovery assessment

The parallel-normal tolerance is a compiled constant, not a configuration parameter: `PARALLEL_TOLERANCE = 0.001` in `ComputeModifiedSymmetryNormals`. The test is `abs(1 - abs(n_i dot n_j)) < 0.001`. For unit normals this corresponds to an acute angle below approximately 2.563 degrees (or the corresponding antiparallel orientation). Outside that branch, the shared normals are orthogonalized. The separate minimum-area constant is 1e-12. Changing these constants would be a solver modification, not an ordinary case setup.

No supported weak-Euler-wall switch was found in the official v8.5.0 configuration registration, configuration template or boundary-condition documentation. `MARKER_EULER` and `MARKER_SYM` are marker lists; switching between them does not select a different implementation. The template explicitly identifies their implementation as identical. No-slip heat-flux or isothermal markers change the physical boundary and are not a substitute. The symmetry/Euler projection remains in the shared boundary routine for the available time-integration choices, so merely switching implicit to explicit integration does not remove that operation.

Primary configuration sources:

- [v8.5.0 CConfig.cpp, marker registration](https://github.com/su2code/SU2/blob/v8.5.0/Common/src/CConfig.cpp#L1530).
- [v8.5.0 config_template.cfg, boundary-condition definitions](https://github.com/su2code/SU2/blob/v8.5.0/config_template.cfg).
- [Official boundary-condition documentation](https://su2code.github.io/docs_v7/Markers-and-BC/).

The reviewed official documentation describes slip walls and symmetry markers but provides no verified special setup that removes the pointed body/axis shared-node constraint while retaining this geometry. This is a bounded finding from those sources, not proof that no alternative exists anywhere in SU2.

The immediate recommendation is therefore to retain the failed gate and its diagnostic evidence rather than announce a configuration-only repair. A controlled single-case solver comparison or an officially supported upstream correction would be the appropriate next decision point before regenerating a campaign. A smooth endpoint parameterization, such as replacing the sine factor with its square, defines a different geometry family and different CFD labels; it would require a fresh full dataset and independent checks. It must not be presented as recovery of the existing model's original labels. No such campaign, solver modification, tolerance adjustment or node exclusion was performed in this investigation.

## Official earlier-version comparison identified

A subsequent inspection of official tags identifies one bounded alternative: **SU2 v8.0.1**. Although v7.5.0, v7.5.1 and v8.0.1 already delegate Euler-wall handling to the symmetry routine, their symmetry implementation differs from v8.5.0. The older routine constructs a reflected primitive state and supplies its upwind boundary flux to the residual; it does not perform the later strong projection of `solutionOld` momentum. The reflected velocity reverses its normal component, preserving its magnitude in the reflected state. This distinction is a change in symmetry implementation, not the date of Euler/symmetry unification.

| Official tag inspected | Boundary implementation |
|---|---|
| v7.5.0, v7.5.1, v8.0.1 | Reflected primitive state and upwind boundary residual |
| v8.1.0, v8.2.0, v8.3.0, v8.4.0, v8.5.0 | Strong normal-momentum projection in the shared symmetry routine |

The official tag listing contains no intermediate v8.0.2 tag. Thus v8.0.1 is the latest listed stable tag before this implementation transition among the inspected releases.

Primary sources:

- [v8.0.1 reflected-state implementation](https://github.com/su2code/SU2/blob/v8.0.1/SU2_CFD/include/solvers/CFVMFlowSolverBase.inl#L1093).
- [v8.1.0 strong-projection implementation](https://github.com/su2code/SU2/blob/v8.1.0/SU2_CFD/include/solvers/CFVMFlowSolverBase.inl#L1111).
- [Official v8.0.1 release](https://github.com/su2code/SU2/releases/tag/v8.0.1).
- [Official Linux 64-bit binary asset](https://github.com/su2code/SU2/releases/download/v8.0.1/SU2-v8.0.1-linux64.zip).

A **single-case comparison using that official binary with the same mesh, geometry, physical conditions and compatible configuration** is a defensible next experiment. Preserve both results and apply the same full-field thermodynamic and convergence gates. Record binary and configuration hashes, and disclose any required configuration compatibility changes. This compares released solver versions; since other code also changed between versions, an improved result would not isolate boundary treatment as the sole cause. It would not retroactively validate the old labels or authorize substituting a new dataset under the old model identity. No such comparison was launched during this source audit.
