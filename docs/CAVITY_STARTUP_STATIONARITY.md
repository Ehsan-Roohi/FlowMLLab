# When has cavity startup settled?

The [Week 1.2 startup movies](../results/cavity_startup/README.md) end at dimensionless physical time t⋆ = tU/L = 10. They show how circulation develops after an impulsive lid start; they do not establish a steady solution. A vortex can look almost stationary while its strength, pressure field or surrounding flow continues to change.

The separate [longer continuation preview, t⋆ = 0–100](https://ehsan-roohi.github.io/FlowMLLab/cavity-continuation/), extends the same native trajectories while preserving the original 0–10 lesson. Its common movie endpoint does not imply that all seven flows have become steady.

This guide explains how to continue those native solutions and test temporal stationarity. It specifies a workflow, not a claim that a particular Reynolds number has passed. Use each run's current receipts and completed-window records for its actual outcome.

## The physical problem and the evidence required

The domain is a unit square. The top lid moves right with U = 1, the other walls are stationary, and every wall is impermeable. The fluid starts from rest, with no lid ramp. Re = UL/ν takes the seven established benchmark values 100, 400, 1000, 3200, 5000, 7500 and 10000. Here L is the cavity width and ν is kinematic viscosity. Time, velocity and pressure are expressed in units L/U, U and ρU², respectively, where ρ is density.

Continuation retains the original MAC grid, central convection, timestep, PIMPLE coupling settings and boundary treatment. It imports the final native staggered velocity components and cell-centered pressure at their actual physical time. It does not reconstruct a state from a picture or restart the clock at zero. Pressure retains the solver's zero area-mean gauge, so pressure changes are comparable without adding an arbitrary constant.

Three different conclusions require different evidence:

| Conclusion | What supports it | What remains separate |
|---|---|---|
| **Fixed-grid stationarity candidate** | Three consecutive complete physical windows pass all frozen field, residual and conservation gates | Accuracy under changes in timestep, coupling and grid |
| **Tighter-coupling confirmation** | A distinct native-state fork changes PIMPLE 3 outer / 2 pressure corrections to 6 / 4 and passes one additional complete window, including agreement with the parent's final window means | Timestep and spatial discretization accuracy |
| **Numerically resolved flow behavior** | Controlled timestep and mesh studies agree adequately in the quantities being reported, with suitable benchmark comparisons | Three-dimensional stability or experimental validity, unless separately studied |

A finer run is a numerical comparator, not an exact solution. A small continuity residual alone says that mass conservation is satisfied; it does not establish temporal stationarity or accurate velocity and pressure fields.

## What the frozen stationarity policy measures

The production thresholds below were selected as engineering gates before continuation. They are not tolerances taken from the cited papers and are not adjusted after looking at a trajectory. All quantities are dimensionless.

The solver evaluates the gates at **every completed physical step**. For velocity RMS, it combines all stored native u and v degrees of freedom with equal weight; L∞ is the largest absolute component. Pressure RMS uses all cell centers. These are full-field tests, not tests at a few probes.

| Quantity | Requirement throughout each complete window |
|---|---|
| Velocity change divided by Δt⋆ | RMS < 10⁻⁶ and L∞ < 10⁻⁵ |
| Velocity departure from the field at the window's start | Maximum RMS < 10⁻⁶ and maximum L∞ < 10⁻⁵ |
| Zero-mean pressure departure from its window-start field | Maximum RMS < 10⁻⁶ and maximum L∞ < 10⁻⁵ |
| Kinetic-energy range within the window | Maximum minus minimum < 10⁻⁷ |
| Streamfunction minimum and maximum | Range of each extremum < 10⁻⁷ |
| Unrelaxed steady momentum defect | Maximum L∞ < 10⁻⁷ |
| Unrelaxed time-step momentum defect | Maximum L∞ < 10⁻⁷ |
| Discrete velocity divergence | Maximum L∞ < 10⁻⁸ |
| Wall-normal velocity | Maximum L∞ < 10⁻⁸ |
| Absolute area-mean pressure | Maximum < 10⁻¹⁰ |
| Difference between consecutive full-window mean velocity fields | RMS < 10⁻⁶ |
| Difference between consecutive full-window mean pressure fields | RMS < 10⁻⁶ |

The two momentum defects answer different questions. The **steady** defect evaluates convection, pressure and diffusion without a time-derivative term. The **time-step** defect includes the discrete temporal term and checks whether the current transient step has solved its momentum equations accurately. Both use the unrelaxed discrete equations; neither is simply the velocity update divided by the timestep. A small time-step defect can coexist with genuine, well-resolved unsteady motion.

Kinetic energy is one half of the cell-area sum of the squared cell-centered velocity magnitude. Streamfunction extrema describe changes in circulation strength, but their values alone do not verify every corner vortex. The window-start comparisons catch excursions that an end-to-end comparison could miss. Consecutive window means help expose slow drift that individual small updates might conceal.

The nonoverlapping windows have duration **20** for Re ≤ 3200 and **50** for Re ≥ 5000. They lie on an absolute-time lattice starting at t⋆ = 0. A continuation beginning at t⋆ = 10 therefore has an incomplete first interval; that interval is recorded but cannot count as a passing window. The first complete window has no preceding complete-window mean, and those two mean-comparison entries are explicitly unavailable. Later complete windows compare against their predecessor. Any failed complete window resets the consecutive-pass count; three consecutive passes are required.

Thus, reaching t⋆ = 25, 50 or 100 is a stage boundary, not a stopping criterion by itself. In particular, a high-Re case starting its continuation at t⋆ = 10 cannot provide three complete 50-unit windows by t⋆ = 100.

## Continue a verified solution on CPU

The following commands run from the repository root in an environment with FlowMLLab installed. They use the published Re = 100 seed as an example; the same command structure applies to the other six Reynolds numbers with matching seed files and policies. A GPU is not required. Fine grids and long physical intervals can still take substantial CPU time; use a small assigned case before attempting the whole collection.

Create the policy **once**, before starting the continuation:

```bash
python qa/continue_cavity_startup.py policy --reynolds 100 --output results/cavity_startup/student_stationarity/policy_re100.json
```

Inspect and retain that policy. The command refuses to overwrite an existing policy. Start from the closed, checksum-verified native seed:

```bash
python qa/continue_cavity_startup.py run --seed results/cavity_startup/fine128/re100.npz --seed-receipt results/cavity_startup/fine128/re100.json --policy results/cavity_startup/student_stationarity/policy_re100.json --output results/cavity_startup/student_stationarity/re100 --target-time 25 --stop-on-candidate --progress
```

Resume in the **same output directory**, keeping the seed, policy, numerical source and runtime contract unchanged. For the next stages, repeat the command with `--target-time 50` and then `--target-time 100`. These targets are absolute physical times, not additional run durations. `--stop-on-candidate` stops early only after the frozen complete-window requirements pass. It does not perform the separate confirmation experiment.

Inspect these files after each closed stage:

| File or directory | Use |
|---|---|
| `status.json` | Current physical time, candidate flag, consecutive passes and stop reason |
| `windows.json` | Complete-window metrics, failed gates and excluded partial initial window |
| `diagnostics.csv` | Recorded physical-time histories, residuals and signed fixed-location probes |
| `restart.json` and `restarts/` | Verified current and previous complete recoveries |
| `snapshots/` | Immutable native velocity and pressure fields at recorded physical times |

Diagnostic rows are recorded every 0.05 physical-time unit, native snapshots every 0.5, and complete checkpoints every 1 unit and at stage end. The initial imported state is also saved. These schedules use absolute time, and every gate still sees every solver step. A policy cadence must be an integer number of the unchanged timestep; the production examples with Δt⋆ = 0.01, 0.005 or 0.0025 satisfy that requirement.

Recovery includes native fields, unfinished-window anchors, extrema, sums, previous means, diagnostic history and source/policy identity. The current recovery is verified before an older superseded recovery is retired, while the previous complete generation is retained. Native snapshots are not discarded as checkpoints. The frozen limits are 1000 native snapshots and 100000 diagnostic rows; reaching a limit produces an explicit stop rather than silent data removal. Do not delete a writer lock or start a second writer to bypass an admission failure.

## Confirm a candidate with tighter coupling

Use this step only after the parent has closed at a full-window boundary with three verified passing windows. The parent must have no active writer or launcher. Its candidate flag is necessary but does not replace the integrity and window checks performed by the confirmation driver.

```bash
python qa/confirm_cavity_stationarity.py --parent results/cavity_startup/student_stationarity/re100 --output results/cavity_startup/student_stationarity/re100_confirm_6_4 --progress
```

The new directory must be distinct from and outside the parent directory. The driver imports the parent's actual native field and actual time, changes only the two PIMPLE correction counts from 3 / 2 to 6 / 4, and starts new window accumulators. It preserves the original parent contract, checkpoint hashes and settings in an explicit native-fork receipt. This is a continuation from a candidate, not another flow starting from rest.

The default target is one additional complete 20- or 50-unit window. All strict window gates must pass, and the new full-window mean velocity and pressure must also agree with the parent's final candidate-window means within the frozen RMS tolerances. Read `confirmation.json`: `complete` and `independent_confirmation_passed` are separate fields. A completed experiment can fail confirmation. The child's ordinary three-window `candidate` flag is not the confirmation result.

For a shorter stage, supply `--target-time` with an absolute time between the parent time and the one-window endpoint. Repeating the same command resumes the complete accumulated state. The parent identity must remain unchanged, so do not resume or modify the parent while its confirmation is in progress. Confirmation at 6 / 4 does not by itself demonstrate timestep or mesh accuracy.

## Compare physical behavior at matched times

For a controlled timestep study, keep Re, grid, boundary conditions and coupling settings fixed, and generate each trajectory from the same rest-start problem with its declared timestep. For a mesh study, keep the timestep and other controls fixed while changing the grid. Changing the timestep or grid inside an existing baseline checkpoint would mix two different numerical histories; it is not the controlled comparison used here.

Compare u along x/L = 0.5, v along y/L = 0.5, and zero-mean pressure along both centerlines at exact common physical times. Preserve signs and sample coverage. The analyzer uses each grid's native cell-center coverage and spatial interpolation only where required for the centerline or a coarse/fine difference. It does not invent a wall pressure value or substitute a nearest available time. Fine-to-coarse sampling is a comparison operation, not an alteration of either scientific trajectory.

For an instructor-supplied campaign manifest and a verified local mirror containing `runs/` and `policies/`, the existing analysis and plotting commands are:

```bash
python qa/analyze_cavity_settling.py --campaign /path/to/campaign.json --data-root /path/to/campaign --seed-directory results/cavity_startup/fine128 --profile-times 10 25 50 100 --output /path/to/analysis/current.json
python qa/plot_cavity_settling.py --analysis /path/to/analysis/current.json --output /path/to/analysis/figures --profile-times 10 25 50 100
```

Replace these placeholder paths with the supplied manifest, mirror and analysis locations; a single student continuation directory is not a campaign manifest. Missing cases and unavailable requested times remain explicit. An absent t⋆ = 50 profile is not replaced by a t⋆ = 49.5 profile. Use the numerical comparator's settings in each legend rather than calling it truth.

The central, upper-left, lower-left and lower-right probes record signed u, v, pressure and vorticity at fixed cell centers. Their actual coordinates are recorded and can differ slightly between grids. They are **not tracked vortex centers**. A probe's sign change may mean that a vortex moved past it, changed strength, or that another structure entered that location. Establish vortex motion from the surrounding native field and streamfunction, not from a probe label alone.

## What if the flow does not become steady?

A failed stationarity test can indicate continuing startup, insufficient numerical convergence within each step, inadequate spatial or temporal resolution, or persistent physical unsteadiness. Those possibilities require different checks. Do not relax the frozen thresholds or add damping merely to obtain a steady-state label.

The two-dimensional cavity has published periodic solutions as well as steady benchmarks; [Bruneau and Saad (2006), *The 2D lid-driven cavity problem revisited*](https://doi.org/10.1016/j.compfluid.2004.12.004), discusses both and investigates the first Hopf bifurcation. [Auteri, Parolini and Quartapelle (2002), *Numerical Investigation on the Stability of Singular Driven Cavity Flow*](https://doi.org/10.1006/jcph.2002.7145), localized a first Hopf bifurcation near Re = 8018 for their treatment of the impulsively started, singular-lid problem.

These studies make sustained unsteadiness at Re = 10000 a plausible outcome, not a guaranteed finding of this discretization. Boundary treatment and numerical resolution matter. A persistent oscillation must be tested against timestep, coupling and grid changes before it is interpreted physically. Report an unresolved case as unresolved; report demonstrated persistent unsteadiness with its evidence. There is no requirement that all seven cases eventually qualify as steady, and a 2D trajectory does not establish the stability of a real 3D cavity.

## Five student investigations

1. **Follow startup beyond the movie.** For one assigned Reynolds number, compare t⋆ = 10, 25 and 50 wherever those exact frames exist. Overlay the two centerline velocity profiles and both zero-mean pressure profiles, and show instantaneous streamlines at those times on matched axes. State which corner structures and profile extrema changed. Record unavailable frames rather than filling the gaps.

2. **Separate an endpoint from a window.** Choose one complete recorded window. Tabulate its maximum velocity derivative, maximum window-start departure, energy range, streamfunction-extremum ranges and both momentum defects against the frozen thresholds. Compare its field means with the preceding full window when available. Explain how similar endpoint fields could conceal an excursion or oscillation, and why the saved diagnostic rows alone need not capture the every-step maximum recorded in `windows.json`.

3. **Measure numerical sensitivity.** At the same Re and exact time, compare two timesteps on one mesh, then two meshes at a common timestep. Plot signed physical profiles and report velocity relative L2 and pressure RMS differences with the comparator identified. State what was held fixed, where coarse/fine interpolation was used, and whether the difference is localized near a wall or changes the core circulation. Do not call a two-grid difference an exact error estimate.

4. **Test the coupling decision.** For an eligible closed candidate, inspect the 3 / 2 parent and the one-window 6 / 4 confirmation. Report every failed gate, the new-window versus parent-window mean differences, the actual physical interval and the recorded work. If no candidate is available, identify which gates prevent this experiment instead of manufacturing a confirmation. Explain why a passed confirmation leaves the mesh study necessary.

5. **Interpret a corner probe.** Plot signed upper-left and central velocity or vorticity versus actual physical time. Mark the recorded probe coordinates and inspect native streamfunction fields during a noticeable change. Decide whether the evidence supports circulation growth, movement past the probe, a repeatable oscillation, or an unresolved numerical effect. If estimating a period, use several observed cycles and show timestep/mesh sensitivity; otherwise state that the time coverage is insufficient.

Submit the exact settings and receipt identities with the physical profiles, time histories and gate table. The objective is an evidence-based classification of the flow and its numerical limitations, not a visually motionless final frame.
