# Teaching review and changes

Two automated review agents inspected the Week 16 material independently: one focused on NASA geometry and numerical evidence, the other on the learning pipeline and notebook execution. This is an internal instructional review, not journal peer review.

| Review finding | Change |
|---|---|
| Cone, NASA geometry and training body could be confused | Separate purpose, geometry, Mach number and claim scope in both guides and lecture |
| Pressure conventions differ | Explain and apply dp/p_inf = gamma M^2 Cp/2 |
| Original NASA alignment needs traceability | Preserve both original macros and unmodified records |
| LAVA could be mistaken for an exact answer | Label it as independent archived CFD; identify experimental records separately |
| Optimizer required an ignored local file | Read baseline metrics from distributed numerical_evidence.zip when raw file is absent |
| Fixed MLP seed did not fix PCA randomness | State refit variability; reproduce the audit from frozen predictions |
| Average error concealed a worse individual prediction | Include all case errors and the 15.80% worst waveform error |
| Notebook needed a short execution route | Student cells use distributed compact files; SU2/Gmsh regeneration is separate |
| Mesh sensitivity could be overstated | Distinguish residual convergence, last-two-grid change and formal GCI; document fixed two-cell cap |
| Chinese aircraft reproduction could be overstated | Add paper-by-paper comparison and missing input list |

## Suggested 100-minute class

1. 15 minutes: derive pressure normalization and compare the three geometries.
2. 20 minutes: inspect NASA records, uncertainty and coordinate transforms.
3. 20 minutes: compare computed signatures and convergence/refinement evidence.
4. 25 minutes: fit the surrogate, inspect the fixed audit and explain train/test boundaries.
5. 20 minutes: propose a constrained shape and discuss why CFD recomputation is required.

## Questions to assess understanding

- Why does solving a 2D planar outline fail to represent an axisymmetric body?
- Why is the LAVA solution not an exact reference?
- Can a pressure peak agree while the waveform L2 error is large? Explain using a shifted shock.
- Why does NASA CFD validation not experimentally validate this neural network?
- Why can an optimizer exploit a surrogate with a small average prediction error?
- What additional model and evidence are needed before reporting ground PLdB?
