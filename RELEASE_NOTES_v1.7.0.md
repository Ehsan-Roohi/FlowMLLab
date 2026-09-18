# FlowMLLab v1.7.0 - Self-supervised wake learning

This release adds paired modules to the Week 7 LBM sequence, bringing the
course to 42 notebooks and 26 lecture PDFs. Existing Weeks 14 and 15 retain
their identities.

## Week 7.3: a retained classical-baseline win

Masked-autoencoder pretraining, frozen probing, fine-tuning and training from
scratch are compared with gappy POD on low-rank wake reconstruction. POD wins;
the lecture explains why. The review corrects visible-token probe pooling,
sample-standard-deviation reporting and notebook execution dependencies.

## Week 7.4: diverse-wake representation transfer

Sixteen compact LBM trajectories span Re60-135. Eleven development trajectories
provide pretraining fields, Re105 provides validation, and four Reynolds cases
provide separate target-label pools and scored windows. A frozen pretrained
encoder, an identical random encoder and POD-32 coefficients feed ridge probes.

The retained mean lift NRMSE is 12.31% for pretrained features with 32 target
labels versus 16.02% for random features with 128: better mean accuracy with
one-quarter as many target labels. POD-32 leads through k=32; pretrained
features have lower mean error at k=64 and 128. Per-trajectory records and
label-draw variation are retained alongside the aggregate figures.

These are target-label budgets. Ridge selection also uses 2761 development
and 251 validation lift labels. There is one encoder initialization, one
initial-condition realization per Reynolds number and a fixed POD rank.
The comparison does not establish statistical significance, optimal-POD
superiority, independent initial-condition generalization or forecasting skill.
Coarse D=6 LBM fields are educational data, not grid-independent force references.
Lift labels are generated with fields, so no actual CFD label-cost saving is
claimed. The module is MAPA-inspired, not a reproduction of its neural-interface
benchmark. Test cases are now inspected retained evidence.

## Reproduction

```bash
python -m pip install -e '.[test,reconstruction]'
python qa/generate_week07_4_wakes.py --workers 4
python qa/run_week07_4_protocol.py
python qa/build_week07_4_materials.py
python -m pytest -q tests qa
flowmllab smoke --root .
flowmllab qa --root .
```

Data manifests and result files record hashes and the split. Regeneration
skips existing trajectory archives; use a new `--output` directory to regenerate
all solver data. The new data occupy about 36 MB, omitting unused vorticity.

Zenodo assigns the version-specific DOI after archival. Until then, the citation
uses the existing all-versions DOI; the v1.6.1 record remains unchanged.
