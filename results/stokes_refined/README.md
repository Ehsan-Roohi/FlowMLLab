# Week 4 Lab 4: Stokes-to-Navier-Stokes correction evidence

This directory contains the retained evidence used by the executed Week-4 Lab 4
notebook and its ten-page lecture companion. The task is

`matched Stokes field + Reynolds number -> Navier-Stokes streamfunction correction`.

The Stokes solution is computed for the same cavity and lid profile as each
Navier-Stokes case. It is a physical low-fidelity input, not a pretrained neural
network. Pressure is recovered from momentum after the velocity prediction and
is not a network output.

## Dataset families

The constant-lid family uses `U_lid(x)=U_ref`. The diverse family uses a
positive, max-normalized three-mode sine expansion. Both families share the
same Reynolds-number distribution, with `L=1`, `nu=0.0025`, and
`U_ref=Re*nu/L`. The shape-OOD group activates five modes at larger amplitude;
the Reynolds-OOD group uses `450 < Re < 600`.

The expanded stage contains 64 training and 16 validation cases per family.
The six original test cases per group are unchanged from the pilot and are
therefore regression tests rather than a newly blind benchmark.

## Selected model

- training-only POD of the Stokes inputs;
- training-only POD of the Navier-Stokes minus Stokes correction;
- features `[Re/400, log(Re/400), normalized Stokes POD coefficients]`;
- two 96-neuron tanh hidden layers and a linear rank-24 output;
- derivative-aware quadratic objective;
- Adam followed by L-BFGS from the validation-selected Adam checkpoint; and
- a fixed mean ensemble of seeds 7, 17, and 27.

## Mean velocity relative L2 error

| Training family | Test family | Original POD | Refined, same data | Refined, 64 train |
| --- | --- | ---: | ---: | ---: |
| constant | constant | 2.523% | 0.185% | 0.107% |
| constant | diverse | 26.505% | 20.157% | 19.920% |
| diverse | constant | 5.270% | 6.859% | 2.044% |
| diverse | diverse | 4.032% | 4.828% | 0.917% |

The expanded diverse model records 8.428% on shape OOD and 10.204% on Reynolds
OOD. The result supports training-boundary diversity; it does not support
universal transfer from a constant lid to unseen boundary shapes.

## Files and limits

- `comparison.csv`: old/same-data/expanded comparison on unchanged tests.
- `expanded/`: selected configurations, three-seed checkpoints, histories,
  predictions, and the expanded dataset manifest.
- `pressure_metrics.csv`: pressure-recovery consistency metrics.
- `velocity_pressure_fields.npz`: plotted velocity and recovered-pressure arrays.
- `vortex_comparison.csv`: original-grid primary and lower-right streamfunction
  extrema plus interior vorticity errors for all six same-family tests per lid.
- `week04_2_*.png`: regenerated lecture/notebook figures.

All reported fields use an `n=25` grid and the same educational finite-difference
solver. No mesh-independent accuracy, independent pressure validation, or new
network-method novelty is claimed.
The displayed velocity and pressure contours use linear interpolation between
retained grid samples for legibility. All field errors are computed on the
original `n=25` arrays before display interpolation.
The speed/streamline and vorticity comparisons are in the
[constant-lid figure](../../figures/Cavity_constant_streamlines_vorticity.png) and
[diverse-lid figure](../../figures/Cavity_diverse_streamlines_vorticity.png). The latter's lower-right
streamfunction maximum is a grid-resolved recirculation candidate. Matching its
node does not establish mesh-independent capture of a small secondary vortex.
