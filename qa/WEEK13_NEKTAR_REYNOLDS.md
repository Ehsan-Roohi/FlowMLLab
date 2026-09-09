# Week 13 Nektar++ Reynolds campaign

This production campaign holds the rectangular geometry (`D/W=5`), boundary
convention, discretization and nondimensionalization fixed while varying only
`Re = 100, 500, 1000`.

All cases use 16 by 80 quadrilateral elements, polynomial order 6, IMEX2,
`dt=0.00025`, spectral/hp dealiasing and one-unit immutable restart chunks to
`tU/W=80`. Each accepted chunk retains primitive fields and vorticity computed
by Nektar++ itself with `FieldConvert -m vorticity`; finite-difference
post-processing is not accepted as the campaign vorticity evidence.

The jobs are restartable on `cpu-preempt`. A clean completion means only that
the requested horizon and artifact-integrity gates passed. Publication still
requires stationarity/unsteadiness classification, temporal and p/h refinement,
matched Cheng--Hung quantities and an explicit corner-boundary sensitivity.

Submit from a clean, pushed commit:

```bash
export FLOWML_NEKTAR_ROOT=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909
export FLOWML_NEKTAR_PRODUCTION_ROOT=/project/pi_roohie_umass_edu/FlowMLLab_nektar_reynolds_20260909
export FLOWML_NEKTAR_EQUIVALENCE_GATE=/project/pi_roohie_umass_edu/FlowMLLab_nektar_qualification_20260909/restart-equivalence-gate.json
sbatch qa/unity_week13_nektar_reynolds.sbatch
```

Do not relabel these runs as validated merely because they finish.
