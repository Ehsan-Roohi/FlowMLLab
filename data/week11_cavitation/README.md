# Hydrofoil vapor-cloud detection data and weights

Author: Ehsan Roohi. Derived teaching subset from the existing
`native_alpha20_v6` hydrofoil cavitation experiment (2026-09-12).
The author requested publication of this code and its results in FlowMLLab.
These author-supplied derived arrays and model weights accompany the repository
under its [MIT license](../../LICENSE). No third-party paper, camera dataset,
raw Fluent archive or private directory tree is included.

Each case NPZ contains the existing 192-by-384 vapor-fraction rasters (`alpha`),
solid mask (`wall`), physical coordinates in metres (`x`, `y`), stored times in
seconds, exact raster-mapped native weak references, and original predictions.
There are **158 frames in seven whole cases**. No resizing, lossy quantization,
new CFD calculation, relabeling or output repair is introduced by packaging.

Native references threshold vapor fraction at **alpha_v >= 0.20** and use the
original cell-face graph and actual wall contact. Class 0 is background; 1 is
attached; 2 is disconnected in 2-D; 255 marks solid or uncertain boundary/crop
support excluded from scores. These are algorithmic references, not human labels.

TRAIN: Cases 13, 14, 16 and Case1LES (86 frames). Validation: Case 19 (27).
Nontraining: Cases 24 and 23 (27 and 18). All cases were previously inspected;
none is a new blind test. Adjacent frames are not independent case replicates.

`detector_weights.npz` stores the original v6 state dictionary as numeric arrays.
`parent_weights.npz` stores native_v4 for optional replay of the final adaptation.
Both load with `allow_pickle=False`. The v6 model has 126,275 parameters and uses
three channels: vapor fraction, solid, and clipped pixel distance to solid.
Its output is raw three-class argmax with no topology-based output correction.

The [manifest](manifest.json) contains file hashes, original source/checkpoint
hashes, case roles, class counts, training configuration and original scores.
The full native CFD meshes are not included, so raw Fluent conversion and native
teacher construction cannot be regenerated from this subset alone. Exact native
reference arrays, input preprocessing, inference, evaluation and the final-stage
training loop can be reproduced.

Run [the notebook](../../notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb)
or inspect [the model code](../../flowmllab/cavitation_detection.py).
