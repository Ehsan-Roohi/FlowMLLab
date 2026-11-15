# Rebuild the cavity extension

Authoritative teaching sources are `qa/build_cavity_expansion_lessons.py` and
`lectures/source/cavity_expansion.json`. The new scientific helpers live in
`flowmllab/ghia.py`, `ghia_surrogate.py`, `cavity_course.py` and `deep_cavity_audit.py`.

Run from the repository root, in this order:

1. If changing the CPU study, run `python -m qa.run_ghia_profile_study`. It retrains
   all predeclared seeds and writes selection before held-out scoring. Do not
   tune this frozen protocol against the now-disclosed teaching test results.
2. The D5 fields are already supplied; ordinary students do not need the private
   author archive. Rebuilding that subset is an author task using
   `qa/build_deep_cavity_course_data.py --source <assessment_bundle_r1>`.
3. Rebuild any earlier notebook/lecture with its existing builder first.
4. Run `python -m qa.build_cavity_expansion_lessons`. It replaces only tagged
   navigation cells in earlier notebooks, keeps their cells/outputs, and builds
   and executes the three separate new cavity notebooks. It does not certify a fresh hosted Run All of older cells.
5. Run `python -m qa.build_cavity_expansion_lectures` with licensed Times New Roman
   available (`FLOWMLLAB_TNR_DIR` can point to the font directory). It appends the
   new sections to the four existing PDFs and records the exact preserved base
   page count. Repeated assembly replaces its own sections instead of duplicating them.
6. Run `python qa/update_course_inventory.py --write`, the targeted numerical
   tests, and render/inspect every changed PDF page and physical figure.

The new extension uses its own PDF metadata key, so older reading-guide pages
are retained. Always run the final extension assembly after a base builder; a
base-only PDF rebuild cannot contain later extension sources automatically.

No full-field Ghia cases are manufactured from centreline samples. Coverage is
reported programmatically; high-Re runs require separate numerical qualification.
No training checkpoint is deleted, no HPC job is created by these build commands,
and no public release is made by generating a PDF.
