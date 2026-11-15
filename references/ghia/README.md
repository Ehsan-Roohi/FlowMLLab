# Seven Reynolds numbers: published square-cavity samples

Ghia, U., Ghia, K.N. and Shin, C.T. (1982), *High-Re solutions for incompressible
flow using the Navier-Stokes equations and a multigrid method*, JCP 48, 387-411.
[Original article](https://doi.org/10.1016/0021-9991(82)90058-4), Tables I and II.

[Raw sample table](published_samples.csv) contains 17 u and 17 v entries at each
of **Re=100,400,1000,3200,5000,7500,10000**. Velocities are divided by lid speed;
coordinates by cavity width. The lid moves in +x. u is sampled on x/L=0.5 and v
on y/L=0.5. These centreline samples are not full-field training data.

The canonical implementation is [flowmllab/ghia.py](../../flowmllab/ghia.py).
Week 4 and the pressure-velocity experimental wrapper now import this single
table. Existing low-Re CFD data and frozen historical studies retain their scope.
The transcription was cross-checked against the original tables and the public
[u transcription](https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219)
and [v transcription](https://gist.github.com/Chaztikov/efbbd7cdbb28d4ad30fd2f7405a0d1a3).
The numerical samples retain their original reported precision.

## Printed-entry sensitivity policy

| Re | Component | Coordinate | Printed value | Predeclared concern |
| ---: | --- | ---: | ---: | --- |
| 400 | v | x/L=0.9063 | -0.23827 | isolated table anomaly |
| 3200 | u | y/L=0.4531 | -0.86636 | suspected decimal error |
| 10000 | u | y/L=0.5 | +0.03111 | suspected missing minus sign |

All three remain in the raw data and in unfiltered scores. Screened scores omit
these entries without inventing corrected values. Plots show a cross at a flagged
printed entry. This is a disclosed sensitivity analysis, not a claim of an
author-issued erratum. Masks were fixed before this teaching model was trained.

## Interpreting an error

Compare predictions at the original sample coordinates using interpolation;
never extrapolate an interior-only field to a wall. Report the norm of the error
divided by the norm of the reference profile, as well as absolute RMS in lid-speed
units and signed differences. A pointwise percentage at a zero crossing is not
used. Reference-only curves do not certify a solver or a learned 2D flow field.

[Classroom study and results](../../results/cavity_ghia_allre/README.md) ·
[Research improvements and sources](../../docs/CAVITY_RESEARCH_ROADMAP.md).
