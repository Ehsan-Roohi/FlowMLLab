# Contributing to FlowMLLab

FlowMLLab welcomes reproducibility reports, documentation corrections, tested
software fixes, and well-scoped extensions to the existing CFD-to-SciML
workflows. Please do not post private student work or unpublished restricted
datasets in a public issue.

For a first contribution, start with the scoped entries in
[ROADMAP.md](ROADMAP.md). Usage questions and examples belong in GitHub
Discussions once enabled; reproducible faults and bounded extension proposals
belong in the matching issue form.

## Before opening an issue

Use the matching issue form and include the FlowMLLab version or commit, Python
version, operating system, exact command or notebook, and the complete error.
For a numerical discrepancy, also include the dataset hash, physical case,
solver/model settings, seed, and the expected and observed metric.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
flowmllab smoke --root .
flowmllab qa --root .
```

## Scientific-change contract

- Preserve complete-case train/validation/blind partitions.
- Compare a changed model with the declared non-neural or exact baseline.
- Retain physical diagnostics and machine-readable evidence with aggregate errors.
- Update `ARTICLE_FIGURE_MAP.md` when a manuscript-facing figure or command changes.
- Do not replace an accepted output with a visually preferable untracked run.

## Pull requests

Keep each pull request focused. Explain the problem, the change, the validation
performed, and any scientific or compatibility limitation. Run the full commands
above before requesting review. New dependencies require a clear justification;
large generated files should be committed only when they are part of the declared
reproducibility record.

By contributing, you agree that your contribution is provided under the MIT
License used by this repository.
