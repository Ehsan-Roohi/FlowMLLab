# Common numerical audit: rotating hexagon

Experiment and curation: Ehsan Roohi. Implementation and audit: AI-assisted.

## Scope

Three retained implementations from one sequential conversation are evaluated. The labels `extra-high`, `medium`, and `light` are user-reported settings, not independently verified model identities. These are individual code specimens, not representative samples or a model leaderboard. The first retained output underwent iterative correction before delivery; later outputs could see earlier conversation content. The original intermediate drafts were not retained here. No additional model generations were made for this audit.

The audit is independent of solver-reported diagnostics, but was written by the same assistant after inspecting the retained outputs; it is not a blinded third-party review. No statistical claims or official benchmark scores are made.

## Reproduction

Run from this directory with Node.js 18 or later:

```sh
node tests/audit.cjs
```

The audit requires no packages or network access. It verifies that each tested physics script is byte-identical to the first script extracted from its retained HTML fragment. `manifest.json` contains SHA-256 checksums. `src/adapters.js` only normalizes parameters and state access; it does not modify the physics equations. Every solver keeps its own internal step policy and contact correction.

## Shared conditions

Unit mass 1 kg; hexagon circumradius 1 m; ball radius 0.065 m; initial angle 0.12 rad; initial position (0.08, 0.36) m; initial velocity (0.72, 0.10) m/s. All long tests last 60 simulated seconds, calling `advance(1/2400)` on every implementation. Default parameters: omega=0.7 rad/s, g=9.81 m/s², restitution=0.85. Identical external output intervals do not imply equal internal work or equal computation cost. Earlier in-chat self-reported numbers used different measurement definitions and sometimes different call intervals, so they are not used to rank these outputs.

## External measurements

1. **Free flight:** compare position after 0.05 s with the closed-form ballistic solution before contact.
2. **Isolated collision:** prescribe a contact at t=0.0005 s, away from vertices, with no gravity; evaluate at t=0.001 s. Test omega=0, 1.3, and -2 rad/s. Compute the wall velocity at the analytic contact point and enforce the expected restitution and tangential-velocity laws. The expected answer does not call a solver collision helper. Report both velocity and position error.
3. **Containment:** independently compute `max(0, n_i dot x + r - apothem)` using all six outward normals, after every returned step. This is sampled post-correction penetration, not maximum internal pre-correction penetration. Zero here cannot establish continuous non-penetration.
4. **Static elastic conservation:** with omega=0, e=1, g=9.81, measure the largest absolute change in `K + m g y`.
5. **Rotating elastic conservation:** with omega=3, e=1, g=0, measure the largest absolute change in `K - omega L`, where `L=m(x vy-y vx)`. This Jacobi invariant accounts for moving-wall work: an elastic collision satisfies delta K = omega delta L. Laboratory kinetic energy alone need not be constant.

Five long scenarios also check finite states and sample containment: default, static elastic, rotating elastic without gravity, stationary zero-restitution contact (g=20), and reverse rotation (omega=-3). Default and reverse cases are not assigned energy conservation scores: gravity is present and moving walls exchange energy while collisions dissipate energy. A full independent impulse-work audit for those cases is outside this comparison.

## Results

Maximum isolated-collision velocity error and maximum conservation drift over each 60 s conservative run:

| Retained output | Collision velocity error (m/s) | Static energy drift (J) | Static energy drift (% initial) | Rotating Jacobi drift (J) |
|---|---:|---:|---:|---:|
| extra-high | 3.488292e-14 | 7.970910e-08 | 2.099929e-06 | 1.068648e-07 |
| medium | 4.461494e-14 | 7.444903e-09 | 1.961353e-07 | 5.725630e-09 |
| light | 4.116215e-04 | 1.472614e-01 | 3.879588e+00 | 9.220715e-02 |

All three free-flight position errors are below 1e-15 m. All 15 long scenario/output combinations remain finite and have zero positive sampled penetration after returned steps. This geometry result does not expose hidden position corrections.

The Light specimen has a measurable analytic collision timing/position error and accumulates substantially greater conservation drift. The Extra High and Medium specimens locate isolated collisions much more accurately. Medium has the smaller drift in the two selected long conservative cases; this is a result about these implementations and these tests, not evidence that a reasoning setting is generally better.

## Limits

- One retained output per reported setting, with shared conversation context.
- No verified backend model identifier, token budget, latency comparison, or first-attempt success statistic.
- No blinded grading, repeated generations, statistically significant ranking, or exhaustive robustness proof.
- Resting corners involve multiple contact constraints and numerical stabilization; the isolated collision tests cover a single wall away from a corner.
- A common external step interval can affect a solver's trajectory and cost. No time-step convergence claim is made.
- Values in `results/audit.json` are authoritative for this run. The live page is a demonstration using the same extracted solvers and independent geometric/conservation readouts; it does not regenerate the stored audit.
