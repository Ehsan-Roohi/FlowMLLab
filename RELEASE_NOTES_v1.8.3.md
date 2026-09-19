# FlowMLLab v1.8.3 — verified release repair

This release supersedes v1.8.2. It closes the remaining functional and release
integrity defects found by a fresh-clone audit; it does not add new CFD or model
training results.

## Week 15

- Replaced the duplicated/clipped footer in all six comparison figures with one
  readable line and rebuilt the 24-page lecture PDF.
- Regenerated `figure_provenance.json` from the published PNG files.
- Re-executed the flagship notebook with the hash-verified Zenodo LR archive;
  all 15 code cells, 342 retained fields, and the six figure hashes pass.

## Release integrity

- Synchronized `pyproject.toml`, the package version, `CITATION.cff`, README,
  results guide, release notes, and `.zenodo.json` at v1.8.3.
- Added ignore rules for documented notebook downloads and generated outputs.
- Evidence workflows now upload artifacts for review instead of committing
  regenerated binary evidence directly to `main`.
- The wheel and source distribution are built and checked, and the GitHub
  release carries both distributions, their hash list, and the Week 15 lecture
  PDF.

## Verification

The tag is created only after a clean-clone run of the Week 15 flagship
notebook, the full test suite, `flowmllab qa`, and
`qa/validate_course_release.py`.
