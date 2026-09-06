# Pilot-budget CPU campaign

This replaces the archived 39-run refinement proposal for the user's explicitly
requested full CPU geometry sweep. It does not claim mesh independence or approve
the resulting data for training. The successful GPU benchmark established an
execution route; its deliberately frequent compressed outputs are not identical
to production output cadence, so its speed ratio is not a production guarantee.

31 independent fresh cases cover the 15 existing heights (16, 21, 25, 30, 33, 36,
40, 44, 50, 58, 60, 64, 67, 70, 75 percent), with two seeds per height and three
at 50 percent. Every case uses 1000x200 cells and 16 CPU MPI ranks. Outlet PPC is
17–20, reduced only as needed to keep the initial inventory estimate at or below
5.1 million. Physical open-boundary inventory can fluctuate. No refined mesh or
double-particle run is included. Each independent geometry starts fresh, not from
the shared benchmark restart.

Use the successful CPU pilot timestep 3.589201579675864e-11 seconds. Warmup is
80,000 steps (2.87136 microseconds); sampling is 120,000 steps (4.30704 microseconds),
six times the pilot sampling interval. Twelve nonoverlapping 10,000-step blocks
are retained, with cell fields sampled every ten steps. Nonoverlap does not imply
statistical independence. The timestep remains subject to each case's diagnostics.

Each CPU job requests 16 GiB and 24 hours. Fresh and changed-rank restart smoke
tests run first on an allocated CPU node, then release the entire geometry array.
The existing CPU pilot binary, MPI library and launcher are verified before each
job. No CUDA module is required. The collector requires every hashed completion
receipt and never marks training data approved.

Unity's CPU partition QOS default allows 1,000 CPU cores per account, verified live
on 6 September 2026. At submission, sum all existing running AND pending account
jobs (conservatively including separate dependent cylinder stages), reserve three
additional cores, then cap the array at floor((1000 - reserved - 3)/16). Never
request extra ranks merely to exhaust the quota. This is a submission-time budget;
new unrelated jobs submitted later can change account demand. Slurm enforces its
own account quota. Source: https://docs.unity.rc.umass.edu/documentation/jobs/

Checkpoints use atomic warm-restart receipts and checksums, preserving incomplete
attempts. Slurm requeues can resume safely; after timeout or a failed allocation,
`resume` can resubmit only incomplete cases after all recorded jobs are terminal.
Resume is explicit, not an infinite automatic retry loop. Resumed particles get
one settling block followed by an entirely new sampling window; interrupted
averages are never stitched together. A cut before warmup completes starts fresh.

```
python3 -I "$OUT/code/cpu_campaign.py" status --out "$OUT"
python3 -I "$OUT/code/cpu_campaign.py" resume --out "$OUT"
```
