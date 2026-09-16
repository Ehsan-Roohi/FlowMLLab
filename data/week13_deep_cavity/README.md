# Retained deep-cavity PINN field

User-supplied Unity export received September 16, 2026 (UTC): Re=1000,
depth/width=2.2, tri=0; checkpoint `restart-55118.ckpt`.

`field.npz` contains x, y, u, v, p, psi and omega on a 301-by-661 grid.
`audit.json` is preserved unchanged. `manifest.json` records the file hashes.
The originating author model is Chris's `CavityTrapREDepthSSB20.py`; that
private source and its weights are not distributed here. Author-supplied
research data remain separately attributed; repository MIT terms do not
relicense the originating private source.

The notebook independently recomputes psi and omega from the velocity export.
No raw CFD reference, optimizer history or final model weights accompanied
these files. The full-grid finite-difference divergence maximum is about 4.33;
it must not be omitted or mistaken for the model's automatic-differentiation
residual. Field finiteness and boundary checks are not convergence proof.

`nektar_cavity_t120.vtu` is the raw Nektar++ CFD export used for the selected
comparison. `nektar_cfd_on_pinn_grid.npz` is its duplicate-consolidated linear
mapping to the PINN grid; `comparison.json` records provenance and metrics.
The two-row `cfd_pinn_fields.png` uses common speed and pressure scales. Its
3.59% velocity relative L2 is near-matched evidence, not final validation:
the lid mismatch is retained and one last-time difference does not establish
mesh independence or a steady asymptote.

## Separately recovered continuation history

`loss_continuation.dat` contains the step and two training momentum MSE columns
from Unity `training/data/loss.dat` in the same project. Identical stage-boundary
duplicates were removed (81 source rows, 74 unique steps). `resume.json`
identified checkpoint `restart-65711.ckpt`, phase 11, Adam done 5000 and an
external parameter warm restart with inverse Hessian reset. The original test
columns duplicated training values and are not reported as independent tests.
This later history is not matched to field checkpoint 55118 and is not the
complete from-scratch training history.
