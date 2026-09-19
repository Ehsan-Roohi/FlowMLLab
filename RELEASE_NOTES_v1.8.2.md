# FlowMLLab v1.8.2 — repaired reproducibility release

This maintenance release supersedes v1.8.1, whose published tag contained five
truncated notebook files.  The scientific scope and the frozen release data are
unchanged; this release repairs the affected teaching/reproducibility artifacts.

## Repairs

- Restored intact, valid notebooks for Week 1, Week 3, Week 4, and the two
  Week 15 activities.
- Restored the accompanying Week 15 lecture/evidence files required by those
  notebooks.
- Normalized the tracked Week 15 CSV to LF line endings so validation does not
  dirty a fresh checkout.

## Verification

In a fresh clone, v1.8.2 was verified with:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests qa
python -m flowmllab.cli qa --root .
python qa/validate_course_release.py
```

Results: all 43 notebooks parse as valid JSON; 240 tests passed and 13 were
skipped for optional dependencies; both release QA commands passed.

The v1.8.0 evidence archives remain canonical on
[Zenodo](https://doi.org/10.5281/zenodo.22840293).  v1.8.2 corrects repository
artifacts and does not claim new simulation, model-training, or experimental
results.
