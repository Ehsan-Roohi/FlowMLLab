# SPARTA pressure-driven backward-facing-step pilot

This is the first **solver-transfer, boundary and cost pilot**, before making a
larger FlowMLLab geometry dataset. It is not a validated replacement for Bird's
DSMC data and is not a new neural-network benchmark. No ML test archive is read.

The case uses unmodified upstream SPARTA commit
`95b9abaa8bd548991cc3c3f1c58b34722f7ade74`, compiled for **CPU MPI**. A GPU allocation
alone does not make this executable use GPUs.

## Physical specification and what is still provisional

The source is the user-supplied *Analysis of the Rarefied Flow at Micro-Step using
a DeepONet Surrogate Model with a Physics-Guided Zonal Loss Function*, Roohi and
Mahdavi, `Step-19.pdf`, sections 3.1–3.2 and Figure 1. The manuscript itself is not
redistributed here. These are **micrometres**, not nanometres; all walls are 300 K.

| Quantity | Pilot value | Basis |
|---|---:|---|
| Total length L | 85.47 micrometres | Manuscript |
| Downstream height H | 17.094 micrometres | L/H = 5 |
| Step position | 0.3 L | Manuscript |
| Step height h/H | 0.50 | First transfer case |
| Kn | 0.01 | Geometry-study regime |
| Pin/Pout | 2 | Manuscript |
| Inlet and wall temperature | 300 K | Manuscript |
| Gas | N2, mass 4.65e-26 kg | Manuscript |
| Collision model | VHS, d=4.17e-10 m, omega=0.74, Tref=273 K | Manuscript |
| VSS angular parameter | alpha=1 | Implements VHS, not alpha=1.4 or 1.6 |
| Wall interaction | Fully diffuse, full accommodation | Manuscript |
| Kn reference state | Outlet density, 300 K | **Explicit pilot convention** |
| Incoming outlet particles | 300 K | **Explicit pilot convention** |
| N2 internal modes | 2 rotational DOF, probability 0.2; vibration disabled | **Explicit pilot convention** |

The manuscript gives `Kn=lambda/H` but does not say whether lambda uses inlet,
outlet or another reference density. Its original Bird input is not available
in this package. The rotation-relaxation settings and exact reservoir algorithm
are also not established. Thus this is a documented candidate transfer case,
not a claim of exact Bird parity. Confirm these conventions before comparing
or pooling solver data.

For this monodisperse VHS gas we use the same mean-free-path convention as
SPARTA's current `compute lambda/grid`:

```
lambda = (T/Tref)^(omega-1/2) / (sqrt(2)*pi*dref^2*n)
n_out = (T/Tref)^(omega-1/2) / (sqrt(2)*pi*dref^2*Kn*H)
p_out = n_out*kB*T; p_in = 2*p_out
```

This gives approximately **Pout=32.081 kPa, Pin=64.163 kPa**. These pressures are
derived from the declared convention, not recovered Bird deck values. The
viscosity quoted by the manuscript is not separately imposed on top of its
diameter/omega collision parameters.

Both x boundaries are open: outgoing particles are removed, and incoming
particles are supplied by `fix emit/face ... subsonic P 300`. The streaming
velocity is taken from the adjacent cell. This is a pressure-driven reservoir
model, not a vacuum outlet or a prescribed mass-flow inlet. Specifying the
temperature of injected outlet particles does not clamp every particle in an
outlet cell to that temperature. Pressure-only `T=NULL` is deliberately not the
default; it requires separate boundary validation before replacing this setup.

Official references:
[pressure emission](https://sparta.github.io/doc/fix_emit_face.html),
[thermal moments](https://sparta.github.io/doc/compute_thermal_grid.html),
[mean free path and collision time](https://sparta.github.io/doc/compute_lambda_grid.html),
[surface orientation](https://sparta.github.io/doc/read_surf.html).

## Numerical pilot

Defaults: **1000 x 200 cells**, nominal **20 particles per fluid cell at outlet
reference density**, one seed `20260905`, **40,000 warmup + 20,000 sampling
steps**, ten nonoverlapping sampling blocks. Initial density is the mean of
inlet and outlet references, with zero streaming velocity. Particle count then
evolves through the pressure boundaries. Approximately 5.1 million particles
are initialized; this is not a fixed-particle simulation.

The timestep is the smaller of 0.2 times the inlet reference mean collision time
and a thermal cell-transit bound. Actual sampled lambda and tau are checked in
the report. The particle weight uses **2D unit-depth cell area**, not L cubed.
The step is aligned with the Cartesian mesh, its normals point into the fluid,
and an in-solver `grid/check ... error` checks particle placement.

For MPI, `create_grid ... block * * 1` explicitly starts with compact processor
blocks. `balance_grid rcb cell` then runs **before** `read_surf`. With
`global gridcut 0.0`, a dispersed partition cannot acquire nearby ghost cells,
so surface inside/outside marking fails. Balancing after surface import is too
late; a one-rank run cannot expose this error. This corrects Unity job 64010511.

**This pilot does not meet the manuscript's stated cell size < lambda/3
criterion.** At the reference inlet the largest cell size is about one mean
free path. The point is to establish executable provenance, boundary behavior,
stationarity and measured cost before committing to finer runs. A spatial,
timestep, particle-count and sampling convergence study remains necessary.
For fixed parameters, roughly 3000 x 600 cells reaches the *reference*
lambda/3 threshold; the observed local minimum lambda may require more.
That larger run is not automatically submitted.

## Unity execution

Use a full **FlowMLLab commit SHA**, not `main`, in the download URL and argument:

```bash
(set -o pipefail; curl -fsSL --retry 3 https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/FLOW_COMMIT/cases/sparta_step/submit_unity.sh | bash -s -- FLOW_COMMIT) || echo SPARTA_SUBMIT_FAILED_TERMINAL_REMAINS_OPEN
```

The final `|| echo` also guards a login shell that already has `set -e` enabled.
The command runs in child shells, preserves the login shell and HOME, and does
not edit existing Conda environments or existing FlowMLLab checkouts. It creates
a unique directory under:

```
/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step/runs/
```

The defaults deliberately reproduce the successful prior Unity SPARTA workflow:

1. **Build/preflight:** CPU, one node, 8 allocated cores, 16 GB, 45 minutes.
   Load `openmpi/5.0.3` inside the job, fetch the pinned source, build with that
   MPI, and execute a real 2-rank step smoke test with output validation.
2. **Pilot:** `afterok` build, one CPU node, 16 MPI ranks, 32 GB, up to 24 hours.
   Verify the executable hash and MPI launcher path; feed the deck through stdin.
3. **Collector:** `afterok` pilot, one CPU, 8 GB, 30 minutes. Parse all outputs,
   compute diagnostics and checksums; only then write `PIPELINE_COMPLETE`.

Every downstream job uses `--kill-on-invalid-dep=yes`. Job IDs are saved after
each successful submission; a partial submission remains visible if a later
`sbatch` fails. The status command checks **all three exact IDs**, never just the
most recent run folder or a vanished `squeue` entry. Module loads stay inside
batch jobs. Conda and module MPI are not mixed. A writable job-local TMPDIR is
used; neither HOME nor a GPU/GRES setting is repurposed.

The account is `pi_roohie_umass_edu`. Optional submission overrides are supported
after the SHA: `--module openmpi/5.0.3 --ranks 16 --account ACCOUNT`.
Set `FLOW_SPARTA_BASE` to select a different scratch base.

Status, without changing shell options:

```bash
(B=/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-sparta-step; O=$(cat "$B/LATEST_SPARTA_STEP_PILOT") && python3 -I "$O/code/pilot.py" status --out "$O") || echo SPARTA_STATUS_FAILED
```

## Output contract

Each run preserves the case settings, source revisions, binary checksum and
linked libraries, MPI launcher/compiler, Slurm logs, generated deck and geometry.
`pilot/` also contains:

- `grid.final.gz`: accumulated **post-warmup** cell moments at the exact final step.
- `grid.block.*.gz`: raw, nonoverlapping block moments. A zero-initialized dump
  at the warmup boundary may exist; the parser excludes it by explicit timestep.
- `fields.csv.gz`: fluid cells only, SI coordinates, cell area, particle count,
  n, rho, u/v/w, rotational and thermal temperature, pressure, momentum density.
- `axial_profiles.csv`: mass flux per unit depth and area-weighted density/pressure.
  Mass flux uses directly accumulated rho*u, not a product of separate averages.
- `flux.blocks`: inlet/outlet insertion AND escape counts. Net rates subtract
  backflow on the proper boundary and are normalized per unit depth.
- `wall.running.*.gz`: collision-derived step-wall pressure, shear and kinetic
  energy flux, resolved by surface segment. `boundary.blocks` stores the box
  boundary moments, including upper/lower walls; these are not cell-center proxies.
- `restart.final`, `report.json`, `SHA256SUMS`.

Thermal temperature subtracts the sampled bulk velocity. No field is smoothed.
Block spread is reported as a diagnostic, not as an independent-sample confidence
interval. Additional block lengths/seeds are needed for uncertainty estimates.

Structural corruption, missing final steps, particles in solid cells, duplicate
cells, nonfinite data, invalid fluid area or stuck particles make collection fail.
Physical diagnostics include mass imbalance, half-to-half drift, adjacent-cell
pressure, local cell/lambda and dt/tau. Their tolerances (5%, 5%, 10%, 1/3, 1/4)
are provisional screening criteria; a failed physical screen is recorded rather
than hidden or confused with a crashed executable. `training_data_approved` and
`bird_parity_validated` remain **false**, even when every job completes.

Review the pilot before increasing geometry count. First settle the original
Bird pressure/Kn and internal-energy conventions, inspect raw velocity and
recirculation profiles, then refine the simulation and establish uncertainty.
The current package does not claim that comparison has already passed.

## Local checks

Only Python's standard library, a C++ compiler, make and MPI are needed. No
TensorFlow, CUDA, user-site packages or GPU smoke test is involved.

```bash
python3 -I cases/sparta_step/verify_local.py --binary /absolute/path/to/spa_serial
```

This executes real SPARTA decks for several step heights, validates output
geometry and particle accounting, checks an equal-pressure equilibrium control,
and confirms malformed output is rejected. Submission tests replace `sbatch`
with a mock and verify dependency/failure handling without submitting jobs.
See `VALIDATION.md` for the checks actually performed for this revision.

An MPI regression workflow builds the same pinned SPARTA source and checks one,
two and four ranks. For the two- and four-rank runs it first reproduces the old
ghost-cell failure as a negative control, then runs all three corrected step
geometries through final output validation:

```bash
python3 -I cases/sparta_step/verify_local.py --binary /absolute/path/to/spa_mpi --mpi-launcher mpirun --ranks 2 --geometry-only --check-legacy-failure
```

This verifies parallel initialization and short-run output integrity. It does
not establish converged DSMC fields or Bird parity.
