# Nektar++ bounded independent-control pilot

This is a CPU-preempt transient qualification extension, not retained steady CFD evidence.
The cavity has width 1, depth 5, lid speed 1 and Reynolds number based on width.
The lid is discontinuous at its corners, as in the initial qualification; no smoothing
or spectral vanishing viscosity is introduced silently.

## Matrix and budget

All four tasks use Re=100 and 8 by 40 quadrilateral elements:

| Task | Polynomial order | dt | Initial end time |
|---|---:|---:|---:|
| 0 | 4 | 0.0005 | 20 |
| 1 | 6 | 0.0005 | 20 |
| 2 | 8 | 0.0005 | 20 |
| 3 | 6 | 0.00025 | 20 |

Re=500 production is deliberately not part of this pilot. Each task gets one CPU,
12 GB, a two-hour allocation and at most 20 requeues. Each solver invocation
advances one nondimensional time unit. Only a clean solver exit, expected final
Time metadata and finite full-field ASCII VTU export permit accepting a chunk.
An incomplete attempt is never reused; four incomplete attempts on one chunk
stop for review. This can lose up to one chunk on abrupt preemption. It is not
crash-exact continuation of the multistep integrator.

Before submission, inspect the short-time uninterrupted versus restarted
comparison at all requested orders/time steps. The gate JSON checks operational
and short-time field agreement only. Startup error can accumulate over many
restarts, so the pilot should additionally compare one longer uninterrupted
segment with the chunked path before the chunking schedule is accepted for
steady-state production. Scripts and the gate are snapshotted per task so later
repository changes do not alter requeued task code.

## Scientific continuation/acceptance review

Time 20 is a bounded inspection point, not a convergence target. Inspect energy,
divergence, wall conditions, CFL, field evolution and all lower vortices. A global
velocity norm or tiny algebraic residual alone cannot establish convergence of
the weak lower eddies.

Before retaining a steady result, require all of the following on a predeclared
common evaluation grid and identical conventions:

- Stable vortex count and signs; each matched vortex centre changes by less
  than 0.002 cavity widths over at least three equally spaced late-time samples.
- Each resolved vortex streamfunction extremum changes by less than 1% over
  those samples. Relative errors below the independently estimated reconstruction
  uncertainty are not meaningful and must be marked unresolved.
- Reconstruct streamfunction independently (velocity integration and a
  vorticity/Poisson method), with path/closure uncertainty reported locally at
  each eddy; uncertainty should be below one tenth of that eddy's amplitude.
- Compare orders 4/6/8 and dt versus dt/2, including each lower-eddy amplitude
  and centre, not only a domain-wide velocity norm. Report differences even
  when a target tolerance is missed; do not alter tolerances after observing them.
- Audit lid-corner treatment and width-based Re against the reference paper and
  the recovered Fluent case before interpreting cross-solver differences.

If these checks fail, retain the diagnostics and make a bounded continuation or
mesh study decision. Do not relabel the transient pilot as a converged benchmark.
