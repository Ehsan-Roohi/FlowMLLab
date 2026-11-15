# FlowMLLab v1.12.0 — Cavity validation and deep-cavity assessment

Three separate notebooks extend the cavity teaching route. Existing notebook cells and outputs remain intact, with links to the new labs.

- **Week 4 Lab 5:** all seven Ghia Reynolds numbers (100, 400, 1000, 3200, 5000, 7500, 10000), original centreline samples, disclosed anomaly sensitivity, and explicit mesh/time-step qualification exercises.
- **Week 4 Lab 6:** fifteen retained CPU fits compare Adam, L-BFGS, staged optimization, 2/5-layer networks and wall weighting across three seeds. Whole-Re splits isolate validation and tests. The improved network beats its Adam baseline but loses to log-Re linear interpolation on both held-out cases.
- **Week 13 D/W=5:** author-supplied CFD/PINN fields at Re=100, 250, 500 and 1000; signed profiles; pressure-gauge and weak-flow diagnostics; 128 reproduced regional quantities and historical workflow trade-offs.
- Four existing lectures gain fourteen pages with theory, physical figures and specific engineering exercises. Their 120 original pages remain intact.
- Shared reference utilities, analytical-gradient tests, frozen model/data hashes, build sources and a literature-backed development roadmap accompany the lessons.

The profile experiment is not a PDE/full-field validation. Newly qualified fields at Re=3200/5000/7500/10000 and a fresh controlled deep-cavity optimizer ablation are not included. Retained D5 workflows have unequal training histories. All three new notebooks passed execution in fresh Jupyter kernels on CPU, along with sixteen targeted tests and the full course release checks. Earlier notebooks retain their existing outputs and have preservation checks; this is not a fresh hosted Run All certification of every older notebook.

The [Zenodo concept DOI](https://doi.org/10.5281/zenodo.22074169) identifies the archive series. This release preserves subsequent repository updates as well as the cavity additions.
