# Deep-cavity PINN refinement protocol

This protocol freezes the Re=100, D/W=5 refinement comparison before the
additional runs are inspected. Both runs start from the same checkpoint and
repeat the same low-learning-rate Adam and SSBroyden2 schedule. The control
changes only the optimizer schedule; the targeted run additionally inserts
40,000 deterministic residual anchors in the lower 60% of the cavity.

The accepted CFD reference is the Nektar++ 32 x 160 element, polynomial-order
6 solution. The 64 x 320 repeat is a grid-independence check. OpenFOAM remains
an independent second-order finite-volume cross-check.

Selection is based on fields, not training loss alone. A refinement is useful
only if it:

1. reduces the global velocity relative L2 error from the frozen 3.8221%
   baseline by at least 10%;
2. keeps the primary and secondary vortex-center displacement below 0.01 W
   and their streamfunction-magnitude error below 2%;
3. improves the signed lower-vortex extrema without degrading the upper-flow
   metrics; and
4. does not rely on corner-local pressure errors for model selection.

All checkpoint, sampling, optimizer, source-hash, and parameter-box metadata
must be retained. Failure to meet these criteria is reported as a negative
result rather than hidden by selecting a favorable loss value.
