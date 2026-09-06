# Recovered Fusion-DeepONet execution

This is an executable revision of Unity `33fusion.py`, SHA-256
`a2015510a33bcd7e0dca4e9f2cf16f03b700ab59fe49fa5e1da1d0e2a35778ed`.
It preserves its Mach 5/7/9 training split, Mach 10 evaluation and six-layer,
512-wide BatchNormalization/Dropout architecture. It is **not** an authenticated
reproduction of the old 256-wide checkpoints or the paper's Mach-15 experiment.

Install the repository plus `tensorflow-cpu==2.16.2`, pandas, scikit-learn and
matplotlib. From the repository root:

```bash
PYTHONPATH=. python examples/cylinder_fusion/recovered_fusion.py train --data-dir /path/to/AllMachNNCylinder --model-dir /path/to/new-fusion-bundle
PYTHONPATH=. MPLBACKEND=Agg python examples/cylinder_fusion/recovered_fusion.py replay --data-dir /path/to/AllMachNNCylinder --model-dir /path/to/new-fusion-bundle
```

Training requires a new output directory and never runs implicitly. The completed
bundle contains all `.keras` members and JSON affine transforms, source/data/model
hashes, sampling settings and TensorFlow version. Replay verifies member hashes,
loads with `compile=False`, and uses only the saved transforms; it does not need
training data. No executable Lambda deserialization is needed: the equivalent
built-in Dot layer replaces the original reduction.

Replay writes `predictions.npz` and `stagnation.png`. Ensemble mean ±2 standard
deviations describes member spread; it is not calibrated coverage. The inherited
nearest-to-y=0 extraction is retained, including its limitations near the solid.

A one-member, one-epoch run on synthetic Tecplot data is an execution smoke test,
not a physical benchmark. Default training remains five members and 500 epochs;
run it on an allocated compute node. Original weights and historical metrics are
unchanged. Old weights lacking their paired preprocessing fail closed.
