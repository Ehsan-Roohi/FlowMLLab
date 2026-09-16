# Week 11 - Shock and vortex identification

[Lecture PDF](../../lectures/week11_shock_vortex_identification.pdf) ·
[Executable notebook](W11_Shock_Vortex_Identification.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Shock_Vortex_Identification.ipynb)

Allow 75-90 minutes after Weeks 1, 2, 7 and 8. CPU only; use Restart and Run All.
The lab verifies rotation/shear/compression controls, trains a small two-output
MLP, freezes validation thresholds and compares complete held-out analytic cases
against a physical baseline. Outputs remain in memory; results/ is not overwritten.

Research source: Ehsan Roohi, *Physics-audited joint neural segmentation of shocks
and vortex cores: cross-solver transfer and controlled airfoil--cylinder studies*,
author-supplied 2026 manuscript, [ShockVortexML](https://github.com/Ehsan-Roohi/ShockVortexML).
The warm-up is an original manufactured-field teaching analog, **not** a
gas-dynamically consistent shock simulation or the paper's network.
The final section adds [real research evidence](../../results/week11_research/README.md):
six fresh forward passes from the frozen task-preserving Harmonized Joint (HJ)
shock-repair checkpoint on existing airfoil/cylinder fields, three times each,
with native masks and source hashes. HJ is a custom shared-encoder, multi-branch
encoder-decoder with specialist shock, vortex-core, wake/shear and expansion
paths; it is not a standard U-Net. No new CFD, training or human-ground-truth
accuracy is claimed. All synthetic exercises remain available.

[Airfoil Supplementary Movie S2](https://www.youtube.com/watch?v=j9rO5j3sudA)
and [cylinder Supplementary Movie S8](https://www.youtube.com/watch?v=hh3K40KRBUQ)
use the same fixed task-preserving model family. The linked videos are qualitative
research predictions, not an independent validation set.

## Real-field reconstruction extension

[Lab 2: reconstruction followed by identification](W11_Lab2_Reconstruction_and_Identification.ipynb)
adds a separate matched-input comparison of interpolation, U-Net velocity reconstruction
followed by swirling strength, and direct U-Net mask prediction on existing
author LBM wakes. Read the [frozen protocol and attribution](RECONSTRUCTION_PROTOCOL.md).
This is vortex-reference agreement on a coarse retained dataset, not independent
shock accuracy. Full retraining is optional and writes only to a new scratch folder.

## Hydrofoil vapor-cloud detection extension

[Cavitation notebook](W11_Cavitation_Cloud_Detection.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) ·
[Model, inference and adaptation code](../../flowmllab/cavitation_detection.py) ·
[Results gallery](../../results/week11_cavitation/README.md)

This is the existing author machine-vision experiment for hydrofoil vapor clouds,
packaged with the original model and 158 retained CFD snapshots from seven cases.
The final multi-method section compares 16 shared moving-case frames using the
alpha-input model, fixed pressure threshold, pressure-only 3×3 model, pressure
U-Net and pressure topology U-Net, replaying all three original pressure seeds.
Each figure uses identical fields and times; pressure inference receives no alpha.
[Comparison data and weights](../../data/week11_cavitation_methods/README.md).

Run All reproduces the three-class predictions and pooled metrics, draws the
vapor field beside its detection, compares a raster-topology baseline and checks
zero-vapor/noisy inputs. The final section of the lecture develops the same example.
Allow 30-45 minutes for discussion; the CPU inference itself is shorter.

Inputs are CFD vapor-fraction rasters and geometry, not camera photographs.
Labels describe attached cavity and disconnected vapor in 2-D; native weak
references are not human ground truth. Four cases trained the model; the other
three were previously inspected. Full case identities, hashes, failures and
training provenance are retained. Default execution leaves the supplied data
and weights unchanged. The optional 1000-update final-stage adaptation starts
from the supplied parent checkpoint and returns a separate model in memory.

Next: [Week 12 - noisy DSMC moment reconstruction](../week12/README.md).
