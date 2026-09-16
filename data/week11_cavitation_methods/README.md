# Original alpha and pressure methods: shared-frame comparison

This author-provided subset contains 16 retained CFD frames from the September
13, 2026 hydrofoil experiments: Plunging3 (7) and Oscillation3 (9). It accompanies
the existing September 12 alpha-input lab. Raw Fluent archives are not included.
The manifest records source array, checkpoint and script hashes.

| Method | Original campaign | Actual input | Actual output |
| --- | --- | --- | --- |
| Alpha context U-Net | multicase_v1 | Vapor fraction, wall and wall distance | Background / attached / disconnected |
| Fixed pressure threshold | Original pressure baseline | Absolute pressure and configured vapor pressure | Total cavity only |
| Pressure 3×3 model | combined_static_dynamic_v6 | Local pressure patches and wall | Continuous vapor fraction |
| Pressure U-Net | pressure_unet_v8 | Pressure field and wall | Continuous vapor fraction |
| Pressure topology U-Net | pressure_topology_unet_v9 | Pressure field and wall | Vapor fraction plus three topology classes |

The four pressure scales are 10, 100, 1000 and 10000 Pa, transformed with asinh.
Pressure deficit is computed in the original float32 operation order:
`pressure_gauge + operating_pressure - configured_vapor_pressure`.
Both retained cases use 103088 Pa operating pressure and 3169 Pa configured
vapor pressure. The raw gauge field and the converted deficit are both supplied.
The fixed baseline is strictly `deficit < 0`. No alpha channel or reference
mask enters pressure-model inference.

All ten saved models are numeric NPZ arrays loaded without pickle: one
alpha model and three seeds (11, 22, 33) for each learned pressure method.
Frame files retain original archived class masks or thresholded binary masks.
The replay checks these arrays exactly; it does not claim bitwise equality of
all underlying continuous predictions across arbitrary hardware.

For common total-cavity Dice, topology classes 1 and 2 are united; continuous
vapor predictions are thresholded at 0.20. The reference is CFD alpha >= 0.20,
restricted to valid native-reference support outside solid. Both trajectories
have no valid attached-reference pixels, so these examples cannot validate
attached-class generalization. Empty-versus-empty is undefined. The overlays
retain wall mistakes even though solid pixels are excluded from the Dice.

The alpha model is the original moving-case continuation of the earlier
context U-Net, not the September 12 checkpoint used in the preceding lab.
Its training cases were WithPorousLES, WithoutPorousLES and Plunging2, with
Oscillation2 for validation. The pressure models also used Case13 and Case1LES
for training and Case14 for validation. The two displayed cases were excluded
from optimization but inspected in earlier development. Inputs, model sizes
and training histories differ; this is not a matched-budget architecture ranking.
These are related hydrofoil geometries, not evidence of novel-shape transfer.

The original 3×3 protocol prose says validation BCE; its executable training
loop actually selects minimum macro validation MAE. The manifest explicitly
records this correction. Original pressure-noise results are retained in
`retained_pressure_patch_noise_metrics`; the public comparison replays clean
inputs. U-Net variants remain diagnostic comparisons, including their failures.

The illustrations use the first seed and each case's maximum valid CFD cavity
area, selected independently of model scores. All 16 frames and all three seeds
remain in the reported tables. CFD alpha is shown as the common visual background;
it is not an input to the pressure methods.

[Executable notebook](../../notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb) ·
[Inference code](../../flowmllab/cavitation_methods.py) ·
[Figures and metrics](../../results/week11_cavitation/README.md) ·
[Machine-readable provenance](manifest.json).
