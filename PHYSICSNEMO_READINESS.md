# A concrete route to a PhysicsNeMo contribution

Reviewed 2026-09-04. This is a proposal and readiness assessment, not an
existing NVIDIA collaboration. No outreach has been sent in this work.

## Proposed contribution

Cylinder licensing update (2026-09-05): Ehsan Roohi confirmed ownership and
authorized the committed Week-7.1 dataset's CC BY 4.0 release; see its
[data license](data/hypersonic_cylinder/DATA_LICENSE.md). That resolves the
cylinder permission gate, not the nozzle defects or a PhysicsNeMo recipe's
scientific/integration gates. No external submission is implied.

A reproducible rarefied micro-nozzle benchmark and a pressure/geometry-to-field
neural-surrogate recipe, including DSMC provenance, explicit boundary metadata,
fair interpolation baselines, local shock metrics, and a compact teaching run.
Do not claim it is the first DSMC example without an upstream review.
FlowMLLab can remain a CPU teaching project while a separate PyTorch recipe
targets PhysicsNeMo; wholesale migration is unnecessary.

## What NVIDIA's public material supports

The [AeroJEPA recipe](https://docs.nvidia.com/physicsnemo/26.08/physicsnemo/examples/cfd/external_aerodynamics/aerojepa/README.html)
downloads SuperWing, trains, runs inference and reports field/aerodynamic
metrics. Its default is a small single-GPU tutorial, while the paper-scale
configuration is a distinct, expensive run. A small honest recipe is therefore
a plausible contribution; a tutorial run should not be sold as paper-scale
reproduction.

The [contribution guide](https://github.com/NVIDIA/physicsnemo/blob/main/CONTRIBUTING.md)
welcomes discussion through an issue and explicitly links a collaboration
proposal form. Every PR should link an issue, undergo review, and carry the
required sign-off. Obtain feedback on scope before a large simulation campaign.
An accepted example is a concrete milestone; it does not by itself imply a
commercial agreement, research funding or a jointly authored paper.

The [Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)
currently states that new applications are not accepted. The page also retains
older call content describing up to 30,000 H100 hours. Do not use the earlier
September 30 deadline or 32,000 A100-hour claim as a current funding plan.

## Milestones and exit criteria

| Stage | Concrete result required | Current state |
|---|---|---|
| 1. Data integrity | Exact solver/export lineage, units, cell versus node convention, corrected symmetry export and checked wall/mass-flow diagnostics | Blocked by the nonzero source V on symmetry and missing exact producing run |
| 2. Fair scientific benchmark | Frozen case splits; physical, shock-aligned and learned baselines; fresh independent cases; paper comparison with identical metrics | Historical-case improvement exists; fresh validation and article checkpoint comparison remain |
| 3. Runnable recipe | PyTorch model, config, dataset loader, train/evaluate commands, checkpoint, CPU smoke test and an actual recorded GPU run | Not implemented or GPU-validated in this change |
| 4. Technical discussion | Short proposal with downloadable evidence and one bounded request for feedback | Can be drafted after stages 1–2; no message sent |
| 5. Upstream contribution | Maintainer-approved scope, linked issue, tested PR and documentation | Not started |

Start with a small additional DSMC pilot that tests the 30 kPa neighborhood
and one unseen geometry. Check solver uncertainty with repeated sampling,
choose independent test cases before training, then use error-versus-cost
curves to decide how much data to produce. A launch criterion should include
shock-location error and boundary/conservation diagnostics, not just a global
percentage or visually sharp contours.

Suggested recipe layout: `conf/`, `src/data.py`, `src/model.py`, `train.py`,
`evaluate.py`, `tests/`, a dataset card and a README. Store geometry and operating
conditions as prediction inputs; target-derived field patches and target shock
positions belong only in evaluation diagnostics. Keep tutorial and research
configurations visibly separate. This describes proposed work, not files
already implemented in this branch.

## Public technical route

After the data-integrity and benchmark gates are met, use the public
collaboration form linked in the contribution guide. This technical assessment
does not require personal networking information. Hardware-support availability
must be checked directly: historical Hardware Grant descriptions do not
establish that applications are open today.

Proposed technical ask: review a licensed DSMC benchmark and a runnable neural
recipe for possible inclusion as a rarefied-flow example. Attach exact failure
cases and measured GPU performance, and ask for a short scope discussion.
Neither acceptance nor a 3–6-month timeline can be guaranteed.
