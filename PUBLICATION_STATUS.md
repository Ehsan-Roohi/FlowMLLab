# Week 16 publication status

This development branch contains the v1.9.0 reference extension. Main and the final release have not been updated.

The eight fresh teaching-body CFD computations and retained-checkpoint audit passed their declared numerical and aggregate error gates. All eight retain their actual meshes, configurations, fields and solver histories in workflow 35936740632. This is a retrospective numerical test, not experimental neural validation.

**NASA validation is blocked.** Read-only inspection of the residual-converged coarse field exposed an incorrect low-pressure stagnation region on the small flat nose cap. The failed medium-grid field also contains a severe near-axis density anomaly. Positivity and residual convergence alone were therefore insufficient. The existing mesh family does not adequately resolve the nose, and is excluded from accepted validation even if its off-body pressure curve meets a tolerance. Its original evidence remains retained as numerical pilot evidence.

A replacement mesh must resolve the upstream nose region and cap, scale cap resolution with refinement, pass local physical checks, and then pass the unchanged experimental and mesh-sensitivity thresholds. No failed or incomplete result is a validated release.

Original NASA records, paper references, portable model and instructional sources remain available in this branch. Full Beihang aircraft reproduction and ground-level PLdB are not claimed.
