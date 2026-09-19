# FlowMLLab v1.8.1 - Reproducibility and teaching corrections

This maintenance release implements the independent repository review of the
v1.8.0 Week 15 edition. It does not generate new CFD, retrain a model, alter a
frozen checkpoint, or replace a historical prediction.

## Week 15 portability and scientific wording

- The flagship verification notebook now runs from a fresh clone plus the
  frozen Zenodo LR-sweep archive. Machine-local paths and an unnecessary
  PyTorch import were removed.
- The repository now links all four frozen Week 15 evidence archives and
  explains which packages have and have not received an independent audit.
- The three data-backed learning-rate and transfer figures used by the lecture
  are included with portable provenance paths.
- The lecture and README distinguish the historical, ordinary holdout and
  post-audit data splits; raw and filtered reverse-flow IoU; and the unresolved
  pressure nondimensionalization label.
- The SMART paper title, seed dependence at Re=25, the upper-edge learning-rate
  selection and the DoMINO velocity/IoU tradeoff are stated explicitly.
- The regenerated 24-page Week 15 PDF uses the corrected source and figures.

## Course portability

- Week 7.4 downloads and hash-verifies its frozen release data when absent, so
  the Colab path no longer fails in its first data cell.
- Week 4 locates the repository independently of the notebook working
  directory, and the Week 7 LBM mathtext expression works with current
  Matplotlib.
- Student notebook runs write regenerated evidence to ignored `student_runs/`
  directories rather than overwriting retained release evidence.
- The release QA command prints a concise summary and writes its detailed JSON
  report under `output/qa/`.
- Version, notebook/PDF counts, page references, dependency descriptions and
  terminology were reconciled across the root documentation.

## Verification

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests qa
python -m flowmllab.cli smoke --root .
python qa/validate_course_release.py
```

The large v1.8.0 evidence archives remain canonical on
[Zenodo](https://doi.org/10.5281/zenodo.22840293); v1.8.1 changes the teaching
and verification layer around those frozen assets, not the assets themselves.
