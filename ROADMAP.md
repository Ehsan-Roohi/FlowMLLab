# FlowMLLab roadmap

FlowMLLab grows through small, testable changes that preserve the released
scientific protocols. New examples are welcome, but reproducibility and physical
validation take priority over increasing the number of methods.

## Current state (v1.6 line)

- Fifteen weekly modules with 41 notebooks and 25 lecture PDFs; every notebook
  is listed in the [launcher](notebooks/README.md) and every lecture in the
  [lecture index](lectures/README.md), and the release gate checks both lists.
- Weeks 1 to 10 are student-run laboratories; Weeks 11 to 15 combine a CPU
  teaching exercise with audits of retained research evidence.
- Python 3.10 to 3.13 CI runs the package tests, the core smoke test and the
  full release gate on every push.

## Next

Ordered by what most improves a student's experience:

1. Execute Weeks 1 to 4 and P0 end to end in a fresh Colab runtime after every
   change to `common/` and record the runtimes in `START_HERE.md`.
2. Bring the remaining audit-style modules (Weeks 11, 12, 14 and 15) to the
   shape of Weeks 4.2 and 13: learning goals, definitions before use, one
   model the student trains, results, interpretation, claim boundary, exercise;
   provenance in one cell per notebook and one notice per lecture.
3. Add the missing definitions and worked examples to the lecture notes of
   Weeks 7.2, 9.3, 10.1, 11, 12, 14 and 15 (Dice/IoU, Kalman update, DeepONet
   branch/trunk, wall units, scattering integral), and re-plot the Week 12
   field panels with a robust colour scale.
4. Consolidate the duplicated cavity solvers and `relative_l2` helpers into
   `flowmllab/` with one argument convention, and remove the `sys.path`
   insertions from notebooks.
5. Move large retained datasets to release assets fetched on demand so that
   the Colab bootstrap clone stays small.
6. Only then add new modules. The label-efficiency lab proposed here is now
   [Week 7.3](notebooks/week07_3/README.md) (masked-autoencoder pretraining on
   the Week 7 LBM wake, linear probe, fine-tuning, error versus number of
   labelled frames, gappy-POD baselines); its open extensions are its exercises
   (structured masks, a vorticity label, noisy sensors, larger budgets).

## Good first contributions

| Contribution | Acceptance criterion |
| --- | --- |
| Test the 20-minute Colab from a clean account | Report every cell runtime and any manual step; do not alter the blind protocol |
| Verify Windows installation | Record Python version, command transcript, and `flowmllab smoke` result |
| Verify Apple-silicon installation | Record chip/Python details and separate core from optional TensorFlow findings |
| Improve demo accessibility | Check keyboard navigation, color contrast, labels, and alternative text without changing evidence |
| Add a notebook runtime report | Use a fresh runtime and the template in the reproducibility issue form |
| Correct or clarify documentation | Link the exact confusing passage and propose the smallest accurate correction |
| Add a validated physical case proposal | Define reference data, case-wise split, baseline, physics diagnostics, and computational budget before code |

Maintainers will convert these entries into scoped `good first issue` tickets.
For a scientific extension, open a proposal before investing in a large
implementation so the validation contract can be agreed first.

## Not on the roadmap

- replacing retained evidence with visually preferable untracked output;
- calling a pointwise random split a new-physics generalization test;
- adding a model without a matched non-neural or exact baseline; or
- expanding the repository only to increase its apparent method count.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the scientific-change contract.

Week 7.4 is implemented as the independent [diverse-wake transfer lab](notebooks/week07_4/README.md), paired with the classical-baseline lesson of Week 7.3. Future work requires new held-out trajectories, multiple encoder seeds, validation-selected POD rank, and independent initial-condition/geometry shifts.
