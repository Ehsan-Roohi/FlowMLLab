# Week 10.1 executable lab: classical collision map and its audit

`W10_1_Collision_Map_Surrogate_Audit.ipynb` turns the Week 10.1 reading companion into a
CPU lab (about one minute). It uses the reduced Lennard-Jones potential as a transparent
teaching analog of the ab initio angle-prediction problem in Roohi, Shoja-Sani and Stefanov,
Phys. Fluids 38, 057123 (2026). It does not use the Jaeger Ar-Ar potential, the article's
network, any checkpoint, or a DSMC run.

Verification built into the lab (all in `tests/test_scattering_lab.py`):
- Rutherford analytic deflection reproduced within the tested tolerance of 1e-6 rad.
- Rigid-sphere limit reproduced by a steep power-law potential within 0.02 rad.
- Lennard-Jones Omega(2,2)* against the Hirschfelder-Curtiss-Bird table within the tested tolerance of 1 % for T* = 1 to 10.

Retained evidence: `results/scattering_lab/` (metrics.json and two figures), regenerated with
`PYTHONPATH=. python qa/run_scattering_lab.py --output tmp/scattering-new` (refuses existing directories).

The random-pair audit uses Gamma(3/2) energies, truncated to the stated energy range; it is not weighted by collision rate. Regenerated transport-integral surrogate errors at T*=1.5, 3, 6 are 0.00954%, 0.1733%, and 0.7179%. Historical JSON keys containing `collision_weighted` are retained for compatibility.
