# Course improvement verification — 2026-09-06

- Week 10.1: six physical/code tests pass; evidence regenerated locally.
- Full suite: 125 tests, three skipped, no failures.
- Release QA: pass; 28 notebooks and 298 code cells parsed.
- Weeks 10.1, 7.1 and 9 Lab 2 execute successfully with the socket-free IPython backend; 360 retained files unchanged during execution.
- Standard Jupyter execution is configured in notebook-html CI, but cannot run in the local socket-restricted environment. Its remote result must be checked separately.
- Week 7.1 classroom MLP evidence regenerated in `week71_reproduced_metrics.json`; this is not a reproduction of the paper checkpoint or the unspecified Unity modal evidence.

## Outstanding research inputs

The exact nozzle producing-run lineage, raw moments and exporter have not been recovered in this change. See `docs/NOZZLE_DATA_NOTE.md`. The author decisions are recorded separately in `qa/AUTHOR_DECISIONS_2026-09-06.md`; no source correction, new paper-level model reproduction, submitted corrigendum, preprint deposit, or collection of collaborator signatures is implied.
The cylinder data license is already resolved in `data/hypersonic_cylinder/DATA_LICENSE.md`; do not list it as pending.
The D40 teaching data and temporal audit remain available; failed grid-independence criteria are preserved.

## Scope of the scattering audit

The imported package uses a Lennard-Jones teaching potential, not the paper's potential/checkpoint. Gamma(3/2) sampling represents random Maxwellian pairs, conditional on energy bounds; it is not collision-event sampling. The historical function/JSON names remain compatible, but narrative labels are corrected. Reference transport integrals are numerical finite-grid quadratures. At T*=6 the surrogate transport error is 0.7179%, so the earlier claim "below 0.7%" is not retained.
