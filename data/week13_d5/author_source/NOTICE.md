# Author-supplied deep-cavity source

These five files preserve the exact source of the retained Roohi–McDevitt
D/W=5 experiment. They are included for scientific reproducibility under the
author's request to publish the paper's data and code. The companion study is
[the Roohi–McDevitt preprint](https://arxiv.org/abs/2610.38814v2).

The paper credits the streamfunction lifting, amplitude envelope and optimizer
implementation to C. J. McDevitt. The balanced-sampling and execution wrappers
are project extensions. Preserve this attribution and the original file bytes.
`PROVENANCE.json` records their SHA256 identities.

The repository's MIT license does **not** relicense inherited source in this
folder. No additional upstream license is asserted here. New FlowMLLab adapter,
audit, plotting and teaching code outside this folder follows the repository
license. The public dense optimizer in DeepPlasma is not substituted for the
compact optimizer supplied here.

Use the new controlled driver rather than starting a historical wrapper:

```bash
python qa/run_deep_cavity_ablation.py \
  --source-dir data/week13_d5/author_source \
  --output /your/allocated/workspace/d5-cost-adam \
  --method adam_lbfgs --seed 12345 --points 8192 \
  --warm-evaluations 50 --qn-evaluations 10 --chunk-size 8192 \
  --device cuda --purpose cost_smoke
```

This short run measures implementation and cost; it is not the completed
research training budget or an accuracy claim. The [controlled-study protocol](../../../docs/DEEP_CAVITY_ABLATION_PROTOCOL.md)
describes the matched comparison and restart safeguards.
