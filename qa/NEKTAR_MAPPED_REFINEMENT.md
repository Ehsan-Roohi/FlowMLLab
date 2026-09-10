# Re1000 mapped h-refinement

Only Re1000 is in this bounded CPU-preempt campaign. Re500 is excluded at the
author's request. This is numerical qualification, not retained benchmark evidence.

| Case | Elements | Polynomial degree | dt | Initial time | Bounded end |
|---|---|---|---|---|---|
| Fine | 32 x 160 | 6 | 0.00025 | 260 | 360 |
| Coarse time-control | 16 x 80 | 6 | 0.00025 | 260 | 360 |

Both use the same accepted Re1000 checkpoint, D/W=5 and stationary lid endpoints.
The fine case uses the official Nektar++
[interpfield module](https://doc.nektar.info/userguide/latest/user-guidese26.html)
on nested meshes; the control copies the original field. No source data is edited.
Physical time is preserved explicitly in the mapped field metadata.

Before advancing, a roundtrip to the source grid must pass max field-difference
tolerance 1e-8 + 1e-6 times the field maximum (pressure constant removed), and
each main vortex must retain strength within 0.1% and centre within 0.001W.
A separate 80-step smoke solve must advance time by 0.02, export finite fields,
keep CFL below 0.8, stationary-wall speed and central-lid error below 1e-7.
Full-lid projection error is reported, not silently removed. These gates check
restart operability, not physical or mesh convergence.

The production runner writes one-unit accepted checkpoints with primitive fields,
spectral vorticity, physical times and SHA256 hashes. Incomplete attempts are never
restart sources. The two-hour job reserves time for orderly interruption and
requeues itself, bounded to 30 requeues; scheduler preemption uses Slurm requeue.
The output lock excludes concurrent writers. A failed numerical gate stops the job.

Code must be pushed first, fetched into a separate clean checkout, and pinned via
FLOWML_REFINE_COMMIT. FLOWML_REFINE_ROOT must name a new project campaign directory.
Submit qa/unity_week13_nektar_refinement.sbatch from that checkout. Never change
the checkout while its jobs are running.

Compare the two runs at equal physical times, and require each vortex's strength
change below 0.1% and centre movement below 0.001W over two successive ten-unit
windows before calling it temporally settled. Inspect side/corner eddies separately;
the current main-vortex extractor excludes x/W below 0.1 and above 0.9. One refined
mesh does not establish grid independence or reconcile the paper discrepancy.
