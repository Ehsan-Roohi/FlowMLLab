# NASA SEEB-ALR original reference files

These are unchanged NASA source files, recovered on 2026-09-23. `source_manifest.json` records original URLs, archive members, byte counts and SHA-256 hashes. NASA's archive states that its files are in the public domain. The five extracted workshop files were checked byte-for-byte against their archive members.

- `SEEB-ALR-as-built.stp`: original geometry, including the as-built nose and sting.
- `NASA_README.txt`: original flow conditions and reference information.
- `shiftavg_195_219-221.39.out` and `shiftavg_553_578-580.out`: experimental records.
- Corresponding `.mcr` files: NASA coordinate transformations and uncertainty-band construction.
- `Sozer_LAVA_AUSMPWplus_minmod_inviscid_axisymmetric_curvilinear_330K_SEEB.plt`: submitted numerical reference with multiple extraction zones.

See [the teaching guide](../../../notebooks/week16/NASA_REFERENCE_GUIDE.md) for geometry inspection, units, pressure normalization, error definitions and exercises. Experimental data and the LAVA CFD solution must be labelled separately. These source files alone do not demonstrate that a new SU2 run or neural model has passed validation.
