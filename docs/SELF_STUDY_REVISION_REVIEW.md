# Self-study revision: integration review

This revision adds worked examples, clearer experiment contracts and assessment
guidance to the existing course. It preserves the introductory material rather
than replacing it with summaries. It is a teaching revision, not a new scientific
validation campaign or a new numbered software/DOI release.

## Origin and preservation

The supplied review was prepared against main commit
`c9562b85e60c36458d1ea45d227c21e9c211834c`. Integration used that same base.
The source package contained 227 files; all matched its SHA-256 manifest.
All 33 PDFs in the separate lecture archive exactly matched the source package.

Archive SHA-256:

- Review package: `d4f9b9af38c601cc08bd5827518d434f43b3d93819e55c63410aef073f360cc2`
- Lecture package: `ebaad2296b4260c700b55eaa29383c6b99a3e3c26c028cfb5f518a48d30ff123`

Independent integration checks confirmed:

- Week 1 numerical-foundations and Week 1.1 scientific-software PDFs and student
  notebooks remain byte-identical to the base. Their sources are not modified.
- Week 1.2 adds a hand-solvable two-cell conservative pressure-correction example;
  its solver implementation, higher-Re extension and retained experiments remain.
- No tracked `data/` or `results/` file is changed.
- All 49 supplied Python sources and 512 notebook code cells were parsed.
  Changed code cells carry no stale saved outputs (73 cells checked).
- All 33 incoming PDFs (424 pages) were audited for text bounds and fonts; no
  text extended outside a page. First/middle/last-page contact sheets were
  visually inspected, with full-size checks of selected equations and figures.
  Two null-character extraction candidates in extensible parentheses were
  visually confirmed to render correctly. This was not a visual reading of
  every page.
- The generated inventory and 316 primary-document targets passed validation:
  34 active lecture PDFs, 51 student notebooks, one historical student companion,
  and one separately listed optional geometry/time module.
- An independent Windows/Python 3.12 package/QA run passed 284 tests, with 11
  optional-runtime tests skipped. Revised Week 14 and Week 15 notebooks also
  completed in separate fresh Jupyter kernels (8 and 17 code cells).

## Substantive changes

The revised material distinguishes a fresh DeepONet experiment from historical
tables; makes cache settings and solver identity explicit; selects POD rank using
validation and declared physical gates; and clarifies finite-sample temperature,
uncertainty, boundary masking and physical units. Transformer exercises now have
consistent array contracts, worked interpretation and a point allocation that
does not equate automated coding checks with complete scientific reasoning.

The geometry/time supplement uses manufactured arrays. It teaches grouped
splitting, masked errors, autonomous versus teacher-forced evaluation and phase
identifiability. It does not claim a CFD or source-paper reproduction.

Integration additionally separates notebook execution from PDF rendering via
`qa/build_transformer_course.py --notebooks-only`; CI notebook checks therefore
do not require Microsoft font installation. The inventory uses portable POSIX
paths on Windows as well as Linux. PDF authoring still requires genuine locally
installed Times New Roman, with no font binaries redistributed.

Execution verification targets the freshly executed output tree using
`qa/verify_transformer_course.py --notebook-root build/ci-course`; source notebooks
are allowed to have cleared stale outputs. Retained-evidence checks run separately
with `--evidence-only`, followed by all original execution/answer assertions on
the new notebook outputs.

The general notebook CI gate now includes Weeks 1.2, 4 DeepONet, 14, 15, 16
and both optional bridge editions, in addition to its previous six lessons.
Week 14's strict executed-cell check targets the fresh output notebook. Source
validation accepts cleared stale outputs; scientific hashes and negative gates
are still checked in both modes. The Week 16 source contract now includes the
added Taylor–Maccoll cell. Missing optional notebook/PDF index links, the old
Week 20 task title and the optional module's incomplete installation command
were corrected during integration. A NumPy archive handle in the scientific
check was closed explicitly for Windows temporary-directory cleanup.

## Evidence boundaries

The upstream review reports 50 regression tests and 338 sequential Python cell
executions. Those are supplied evidence, not independently repeated Jupyter or
Colab runs. GitHub Actions attached to the integration pull request provides
fresh test and notebook-execution results for the integrated commit.

The short Week 13 PINN is an algorithm/diagnostic exercise and can have large
velocity error or worsening held-out residual. Strict checkpoint agreement and
the separately declared scientific tolerance remain distinct; failure of the
strict criterion must remain visible. Public held-out cases are not newly blind.
No thresholds were relaxed to approve this teaching revision.

Student pilot timings, independent grader agreement, hosted Colab acceptance,
new unseen physical cases and full research training remain unmeasured. The
per-lesson procedure and acceptance limits are in
[SELF_STUDY_ACCEPTANCE.md](../qa/SELF_STUDY_ACCEPTANCE.md).
