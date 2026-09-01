# FlowMLLab roadmap

FlowMLLab grows through small, testable changes that preserve the released
scientific protocols. New examples are welcome, but reproducibility and physical
validation take priority over increasing the number of methods.

## Next release: v1.2

- deploy the read-only blind-case explorer and link it directly from the README;
- make a clean package distribution pass an isolated wheel/sdist check;
- document Windows and Apple-silicon installation outcomes;
- add contributor-reported notebook runtimes without changing the frozen results;
- collect course and laboratory adoptions with permission; and
- keep Python 3.10--3.12 CI, repository QA, and scientific evidence gates green.

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
