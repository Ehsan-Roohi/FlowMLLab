# Real DSMC observations for Week 12 Noise2Noise

Source: Ehsan Roohi, *Geometry-native machine learning reconstruction of DSMC
moment fields with support monitoring*, [arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637).
Existing author-supplied `JCP2(1).zip`, condition `S2_kn0p085_u350`.
These fields were produced in the author's DSMC research before integration
into FlowMLLab. No synthetic Gaussian noise is added to this real-data section.

- `observations.npz`: eight seeds, qx/qy, native 100x100 raw-three-block fields.
- `evaluation_only.npz`: independent finite-budget reference and corresponding
  raw-ten-block observations, not available to the fit/selection functions.
- `manifest.json`: SHA-256 checks, preserved source summary and declared split.

Training seeds 26082101-104; validation 105-106; evaluation 107-108.
This split was declared for a teaching fit of an already-inspected archive;
it is not a new blind research test. Every seed represents the same condition.
Spatial points and overlapping patches are not independent realizations.

The source documents distinct observation/reference seeds, a 260-block
reference, and disjoint Raw(3)/Raw(10). This metadata is preserved, not silently
upgraded to independent reconstruction of RNG streams or solver block lineage.
Zero conditional noise mean for finite-sample central heat-flux estimates is
not established. The finite-budget reference is not exact truth.

No archived neural predictions, model checkpoints or solver code are included.
Values retain archive units. Array indices are mapped to [0,1] for plotting;
these are not independently verified physical coordinates. No spatial
resampling, denoising or normalization is applied during extraction.

## Rights and attribution

Included for the author's requested teaching extension. This notice records
source and purpose, not a new blanket license for upstream research data.
The software's MIT license must not be assumed to cover this research subset.
External redistribution requires checking the applicable data rights.

Noise2Noise is cited as a method from Lehtinen et al. (ICML 2018),
[arXiv:1803.04189](https://arxiv.org/abs/1803.04189).
The lab implementation is original FlowMLLab code; no code or weights from
NVlabs' CC BY-NC implementation were copied.

Re-extract to a fresh directory using
`python qa/prepare_week12_noise2noise_data.py --archive PATH --output NEW_DIRECTORY`.
Normal notebook execution only reads these files.
