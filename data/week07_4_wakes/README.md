# Week 7.4 diverse-wake LBM data

Sixteen complete D2Q9-TRT trajectories support the MAPA-inspired Week 7.4
pretraining study.  Cases are split by complete Reynolds-number trajectory,
never by random frames:

- development: Re = 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 130;
- validation: Re = 105;
- untouched test: Re = 95, 115, 125, 135.

Each archive contains 251 cropped 32 x 78 transverse-velocity fields,
instantaneous lift and drag, snapshot times, perturbation amplitude,
and deterministic seed.  `manifest.json` records the solver settings, crop,
roles, sizes and SHA-256 hashes.  Regenerate with:

```bash
python qa/generate_week07_4_wakes.py --workers 4
```

These are quick educational LBM trajectories (D = 6 lattice nodes, periodic
transverse boundary), not grid-independent DNS.  The attempted Re = 140 run
became non-physical and was rejected by the solver gate; it is not part of the
retained data.  The short histories do not pass the solver's Strouhal-resolution
gate, so Week 7.4 decodes the stored instantaneous lift rather than claiming a
validated shedding frequency.
