# Teaching explanations and diagnostic exercises

The current teaching update adds reading guides to 21 notebooks and integrates
corresponding typeset sections into 17 lecture PDFs. Twenty-seven diagnostic code
cells were adapted from Aryan Mohta's student submission, then checked in the
current course environment. Submitted execution outputs were not copied as
author-verified results. Source-cell attribution is recorded in
[the provenance ledger](../qa/student_feedback_cell_sources.json).

## Main changes

- Distinguish a coordinate MLP, a branch/trunk model and field interpolation.
- Identify retained evidence, fresh computation and historical holdouts.
- Keep published blind answers closed with the live assessment switch.
- Display newly generated Week 10 figures from the fresh result directory.
- Show both global and reverse-flow validation errors of the unweighted baseline.
- Localize errors instead of relying on global averages or contours alone.
- Compare noisy-field diagnostics after recomputing all derivative features.
- Include feature ablation, negative controls, threshold sensitivity, connectivity
  baselines and solid-region false positives.
- Explain source-label budgets, compute budgets, optimizer limits and missing
  low-energy support in transport integrals.
- Write optional Stokes training to a new student directory; reject retained or
  already existing destinations. Resolve training inputs from the package root.
- Report physically impossible Rayleigh targets explicitly; the supersonic
  inversion no longer imposes an arbitrary Mach-30 cutoff.

The optional DeepONet training experiment with conflicting numerical summaries
was not adopted. A poor transfer result alone does not isolate an architectural
cause. Network download cells and unverified causal conclusions were also omitted.

## Validation and limits

[The execution record](../qa/feedback_execution_validation.json) lists 16 default
notebook paths completed in independent local CPU kernels, with timings and the
actual package versions. These runs include all accepted diagnostic cells.
[Structural validation](../qa/feedback_structural_validation.json) covers notebook
schema, code syntax, portable math delimiters, overlay consistency, PDF fonts and
safe output guards. The gas-dynamics unit suite also passes.

The long Week 7.3 training and optional full CFD/DSMC or model regenerations were
not repeated. Stale Week 7.3 saved outputs were cleared. No hosted Colab Run All,
new blind research result or new physical-validation campaign is claimed.
Retained scientific datasets and result packages were not rewritten.

## Rebuilding the teaching material

The committed overlay JSON and TeX sections are public build inputs; the private
submission and email are not required. After an original notebook builder, run:

```bash
python qa/build_student_feedback_lessons.py
python qa/validate_student_feedback.py
```

After an original lecture builder, reapply the typeset sections:

```bash
python qa/build_feedback_lecture_sections.py
python qa/update_course_inventory.py --write
```

The PDF builder requires licensed Times New Roman and Tectonic. Set
`FLOWMLLAB_TNR_DIR` or `TECTONIC_BIN` if they are not discoverable. It preserves
the base lecture pages and navigation, continues page numbering and replaces its
own prior appendix rather than accumulating duplicates.

To refresh outputs explicitly, use the notebook runner with filenames, for
example `python qa/execute_feedback_notebooks.py W10_1_Collision_Map_Surrogate_Audit.ipynb`.
Each run uses a separate CPU kernel; optional heavy training remains governed by
the notebook's own switches.
