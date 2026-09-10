# Re500 deep-cavity numerical verification campaign

Development work, not retained validated course evidence. No release or DOI.
All cases are two-dimensional Nektar++ 5.9.0 incompressible Navier-Stokes,
velocity-correction IMEX2, spectral/hp dealiasing, Re=U_lid W/nu=500,
W=1, H=5, nu=0.002. No neural network is trained in this campaign.
The source is the accepted p6, 16x80-element t=220 state. Source data and existing
Re1000/OpenFOAM runs are read-only. Element counts below are not point counts.

| Array | Label | Elements | Polynomial degree | dt | Isolated question |
|---:|---|---|---:|---:|---|
| 0 | base | 16x80 | 6 | 0.00025 | Matched continuation/checkpoint control |
| 1 | h32 | 32x160 | 6 | 0.00025 | h refinement |
| 2 | h64 | 64x320 | 6 | 0.00025 | Third h level |
| 3 | p8 | 16x80 | 8 | 0.00025 | p refinement |
| 4 | p10 | 16x80 | 10 | 0.00025 | Third p level |
| 5 | dt2 | 16x80 | 6 | 0.000125 | Half time step |
| 6 | dt4 | 16x80 | 6 | 0.0000625 | Quarter time step |
| 7 | corner_h4 | 22x80, nonuniform x | 6 | 0.00025 | Split first/last base x cells four ways |

Every mesh retains the source vertices. The corner case preserves all middle
elements and the same classical lid expression `(x>0)*(x<1)`. It reduces the
width affected by finite-p endpoint projection; it does **not** change the
physical problem to a smoothed lid. A single corner-refinement comparison is a
diagnostic, not a complete limit study of boundary regularization.

## Mandatory execution gates

Each case maps using FieldConvert interpfield (or an exact same-space copy),
then maps back to the source space. Pointwise field agreement and four weak-vortex
strength/centre checks must pass. The mapped restart retains t=220 explicitly.
Before production, a 0.02-time solve is compared against two 0.01-time solves.
Equal end clocks, finite fields, no-slip walls, central lid, and CFL<0.8 are
required. Full-lid projection errors are reported, not hidden or accepted as
proof of classical-boundary accuracy. These gates establish operability only.

Reference: [official FieldConvert modules](https://doc.nektar.info/userguide/latest/user-guidese26.html).
No clipping of interpolated velocity or fitting to the article is used.

## Temporal and scientific acceptance

All new cases use the same 0.25 physical-time checkpoint/restart cadence; the
base case therefore controls for the change from the older one-unit cadence.
At t=220,230,240,..., reconstruct element-local polynomial velocities. Locate
four alternating main vortex centres through u=v=0, rejecting saddle points.
Integrate the polynomial analytically from the left and bottom separately and
report path disagreement. Clockwise vorticity is u_y-v_x; native FieldConvert
spectral vorticity is additionally retained for independent extraction checks.

Stop only after each of the four main vortices has <0.1% change in both centre
streamfunction and vorticity, and <0.001W centre movement, over each of two
consecutive ten-time-unit windows. This is a temporal gate, not a certificate of
spatial accuracy. The domain used by the existing seed detector excludes x<0.1
and x>0.9; tiny corner/side eddies are **not** automatically certified.

After temporal settling, compare each vortex across h16/h32/h64, p6/p8/p10,
dt/dt2/dt4, and base/corner_h4. Report both relative and absolute differences,
the two streamfunction paths, and native spectral vorticity consistency.
Target <1% for strengths/vorticity and <0.001W for centres on the final pair;
require a decreasing sensitivity trend. Do not claim an observed order/GCI if
three levels are non-monotone or outside an asymptotic regime. A combined refined
h/p/dt case is a follow-up if separable tests reveal competing errors.
No article value is used in the stop or mapping criteria.

The article remains a numerical comparison, not exact ground truth:
[Cheng & Hung (2006), Table 2](https://doi.org/10.1016/j.compfluid.2005.08.006).
Existing Re1000 continuation and the running OpenFOAM fine-grid case remain
separate evidence. A successful Re500 study does not certify other Reynolds numbers.

## Bounded resource use and retention

cpu-preempt, 1 CPU/48 GB per case, at most three cases concurrently. Two-hour
allocations use 6600 seconds internally and at most 24 requeues (50 allocated
CPU-hours per case upper bound). End-time budget is t=280, not assumed convergence;
the script stops earlier if the temporal gate passes. Four incomplete attempts
at the same step halt for review. Shell/browser disconnection is irrelevant to
the submitted batch jobs. Scheduler preemption uses Slurm requeue where provided;
all restarts accept only clean, hash-checked checkpoints.

Keep two recent quarter-time checkpoints, full ten-time-unit snapshots including
spectral vorticity, all sessions/logs/metrics and hash manifests. Delete only newly
generated accepted intermediate heavy files inside this campaign, after newer
checkpoints exist. Original sources and other campaigns are never pruned.
Incomplete attempts are not accepted as restarts and remain available for diagnosis.

## Local tests

`python -m unittest discover -s qa -p 'test_week13_independence.py' -v`

On shared filesystems with busy-directory teardown, set
`FLOWML_KEEP_TEST_FIXTURES=1` and `TMPDIR` to a project test directory. This
retains the small test fixtures for audit; all assertions, including intermediate
file-deletion/retention safety tests, still run unchanged.

These cover manufactured polynomial integration down to amplitude 1e-14,
vorticity sign, nested meshes, unchanged physical lid, temporal gates, and
retention path safety. They do not replace the compute-node solver gates.
