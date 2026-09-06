# Unity Bird-source inventory — 2026-09-06

This inventory records read-only inspection of the author's Unity storage. It distinguishes files that were found from producing-run lineage that still needs to be demonstrated.

## Micro-nozzle

The working directory is `/work/pi_roohie_umass_edu/Nozzle`.

It contains fifteen canonical Tecplot snapshots named `P=<PR>full.dat` for pressure ratios 15, 16, 18, 19, 20, 22--30 and 33. Multiple scripts in the same directory consume the glob `P=*full.dat`; examples label these inputs as DSMC Tecplot files and define held-out pressure ratios such as 15, 25 and 33. The directory also contains later transformed and predicted files, so an unqualified `P=*full.dat` glob matches more than the fifteen canonical inputs.

No Bird-2D executable, Fortran source, DS2 input deck or explicit Bird marker was found within two levels of this directory. The maintainer identifies the canonical snapshots as output of the two-dimensional Bird code, but the exact solver/input/exporter chain has not yet been independently reconstructed. The similarly named `/project/pi_roohie_umass_edu/Nozzle_Heat` directory is a later GHS wall-heat-flux design study and is not the producing archive for these snapshots.

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
