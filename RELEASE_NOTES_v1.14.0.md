# FlowMLLab v1.14.0 - Cavity validation, PINN failures and controlled optimizer studies

The new lessons connect frozen numerical fields to explicit student experiments.
Successful supervised prediction and unsuccessful label-free PINN trials are
reported separately, with source, signed errors and independent physical checks.

- **All seven Ghia Reynolds numbers:** 42 completed label-free PINN runs with
  three seeds and two corner collars. None passes every predeclared criterion.
  All seeds and failures are retained. Fourteen of 21 nominal collar pairs used
  identical collocation points, so zero differences cannot show collar robustness.
- **One sampling intervention:** 12 paired trials at Re=100 and Re=7500 use fixed,
  area-weighted corner/wall/interior strata and match each original run's
  objective/gradient work. Interior velocity error improves in 2/12 pairs and
  worsens in 10/12, including all six Re=7500 pairs. This is a measured negative
  result, not an advertised accuracy improvement.
- **D/W=5 optimizer comparison:** three matched seeds compare Adam followed by
  L-BFGS with SOAP followed by compact self-scaled Broyden. Both use the authors'
  Gaussian lid, depth envelope, residual scaling, initial weights and samples,
  with 10,000 objective/gradient calls per arm. SOAP improves interior vorticity
  error in all three pairs but velocity in only one and pressure in none.
  Interior velocity NRMSE remains approximately 97-117%; neither finite-budget
  route accurately recovers this deep-cavity reference.
- **Teaching and reproduction:** two additive CPU evidence notebooks, lecture
  extensions, attributed byte-verified author source, physical fields and
  signed profiles, complete-state recovery tests and measured chunk-cost evidence.
  Numerical tables retain absolute and relative weak-flow errors.
- **Earlier lessons preserved:** the original 52 pages of the affected lectures
  and preceding notebook exercises remain intact. The high-Re POD-DeepONet
  gallery, solver-cost lesson and Week 15 capacity study are retained.

The sampling experiment includes a disclosed evaluation repair: all twelve
training endpoints were frozen before any target access; an earlier partial
pass scored four cases before a cross-platform sample-regeneration check failed.
The repaired inference-only adapter loads the exact saved state and freezes all
twelve native projections before its own scoring pass. No fitting, resampling
or endpoint selection followed the partial target access. This does not certify
cross-platform optimizer continuation.

Data integrity does not imply physical accuracy. CFD is a retained discrete
reference with documented mesh sensitivity; the square-cavity archive has no
pressure truth. Fresh case-specific PINNs do not demonstrate held-out-Re
operator generalization.

Protocols:
[baseline](docs/CAVITY_PINN_CAMPAIGN_PROTOCOL.md),
[sampling comparison](docs/CAVITY_PINN_STRATIFIED_PROTOCOL.md),
[D5 optimizer comparison](docs/DEEP_CAVITY_ABLATION_PROTOCOL.md).
Numerical and teaching admission receipts are retained under qa/.
