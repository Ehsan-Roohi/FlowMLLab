# Unity Bird-source inventory — 2026-09-06

This inventory records read-only inspection of the author's Unity storage. It distinguishes files that were found from producing-run lineage that still needs to be demonstrated.

## Micro-nozzle

The working directory is `/work/pi_roohie_umass_edu/Nozzle`.

It contains fifteen full-domain Tecplot derivatives named `P=<kPa>full.dat` for back pressures 15, 16, 18, 19, 20, 22--30 and 33 kPa. Multiple scripts in the same directory consume the glob `P=*full.dat`; examples label these inputs as DSMC Tecplot files and define held-out back pressures such as 15, 25 and 33 kPa. The directory also contains later transformed and predicted files, so an unqualified `P=*full.dat` glob matches more than the fifteen selected inputs.

No Bird-2D executable, Fortran source, input deck or explicit Bird marker was
found within two levels of this Unity directory. Author-supplied laptop archives
subsequently yielded a 2009 modified Bird-family `DSMC2` source and a matching
25-kPa input. The 100-by-30 main and 30-by-40 buffer cell grids generate the
same 101-by-31 and 31-by-41 nodal zone dimensions as the article exports. The
source's output schema lacks the article export's `QX`, `QY` and `Txy` fields,
so it remains a close solver-family candidate rather than the exact producer.

The full-domain files are deterministic derivatives of the article repository's
half-domain exports: all non-`Y` values in the first two zones match exactly at
float32 precision, `Y` is translated by -92 micrometres, and two Tecplot mirror
zones are appended. Those mirror zones share `QY`, `V` and `Txy` without the
required sign reversal. The similarly named `/project/pi_roohie_umass_edu/Nozzle_Heat`
directory and recovered 7-kPa/300-K and 2026 GHS archives are later wall-heat-
flux/model studies, not the producing archive for the article snapshots.

## Argon cylinder

The source and derived fields are under:

- `/project/pi_roohie_umass_edu/cylinderNew`
- `/project/pi_roohie_umass_edu/cylinderNew/structured`
- `/work/pi_roohie_umass_edu/Argon-Nitrogen-cylinder`

`cylinderNew` contains triangulated Tecplot fields. `Allconvert.py` explicitly converts the `*GridM*.dat` finite-element fields to 400 by 400 ordered POINT files named `structured/StructuredGridM*.dat`. The structured directory has Mach 5--15 data and intermediate Mach values, model checkpoints and figure scripts. Several `*_Extra.py` scripts reserve `StructuredGridM15.dat` as the manual test file; `40_Extra.py` loads `models/deeponet_v29_fourier_residual.h5` before fine tuning.

The work directory also contains `Ar-Bird Prop-Kn-Part1.rar` and Part 2. A read-only archive listing shows Knudsen-number case directories with Bird-style products `DS2FF.DAT`, `DS2SU - Copy.DAT`, `DS2VD.DAT` and aerodynamic-coefficient files. The archives were listed in place and were not extracted or modified.

## Diatomic nitrogen cylinder

`/project/pi_roohie_umass_edu/cylinderNew/CylinderNitrogen` contains Tecplot finite-element fields `Mach5.dat` through `Mach14.dat`, nonequilibrium DeepONet scripts and surface/contour figures for density, pressure, translational and rotational temperature and related quantities.

The work directory contains four explicitly named Bird archives:

- `Ar-Bird Prop-Kn-Part1.rar`
- `Ar-Bird Prop-Kn-Part2.rar`
- `N2-Bird Prop-Kn-Part1.rar`
- `N2-Bird Prop-Kn-Part2.rar`

This closes the earlier statement that the Knudsen and diatomic source branches were unlocated. Paper-level reproduction still requires mapping a particular archive case, preprocessing configuration and checkpoint to each published figure or table.

## Validation policy

New DSMC simulations are not required for the GitHub/course release. Their only additional value would be a genuinely blind test of generalization. An archived Bird case provides equivalent evidence only when records show that it was not used for training, hyperparameter selection, checkpoint selection or debugging.
