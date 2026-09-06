# Packaging and public deployment

This document records the remaining maintainer steps. It does not claim that a
PyPI package or public demo exists before those services confirm deployment.

## Fall 2026 stabilization policy

The immutable teaching baseline is **v1.4.1**, commit
`757eb4dfaf5b0cfc0c68b3ebca8057c7ef652747`, archived as
[10.5281/zenodo.22348207](https://doi.org/10.5281/zenodo.22348207).
The moving `main` branch already contains post-tag changes: it is not the exact
DOI snapshot, even while package metadata still reports 1.4.1. Cite the full
commit for work using either a working branch or a post-release notebook.

- For the remainder of the term, changes proposed for `main` are bounded bug
  fixes, provenance corrections and maintenance of existing teaching material.
  New modules and experimental campaigns stay on a development branch.
- `codex/modal-sensing-labs` is the current development branch for modal labs,
  CFD qualification and SPARTA/V5 follow-up. Presence of executable code is
  not evidence of a qualified solver or an accepted research result.
- The old tag already contains a basic SPARTA pilot and V5 tooling; later GPU
  and campaign tooling also reached `main`. This policy does not rewrite that
  history or pretend these files were absent. Preserve tags, old evidence and
  DOI records; label their validation limits explicitly.
- Plan the next DOI-bearing software release at the next term-level review,
  not after individual fixes or runs. No release or DOI is created by this
  stabilization update. A scientifically corrected research dataset has its
  own versioned correction procedure, independent of software-release cadence.
- Promote new scientific evidence only after frozen acceptance checks, source
  lineage, baseline comparisons, test isolation and required author permissions
  have been reviewed. A completed scheduler job alone is insufficient.

These are maintainer review rules, not a claim that server-side branch
protection has been configured. See [execution status](qa/EXECUTION_STATUS.md)
and the [nozzle correction sequence](docs/NOZZLE_DATA_NOTE.md).

## Evaluation terminology

**Held-out** means excluded from fitting and model/hyperparameter selection in
the recorded protocol. **Inspected historical holdout** means its result has
already been examined; it remains useful for reproducibility and regression,
but must not be presented as a newly sealed evaluation. **Sealed fresh test**
means the relevant evaluation has not yet been inspected and all decisions and
acceptance criteria are fixed before unsealing.

Publication alone does not retroactively invalidate a correctly held-out
measurement. Adaptive tuning after inspecting a case does invalidate its use
as a fresh independent test for those decisions. Retain the original protocol,
record reuse, and obtain a new sealed test for a new confirmation claim. In
legacy materials, read "blind" together with the case history rather than as a
guarantee of continuing secrecy. None of these terms implies grid convergence
or validation against independent physics measurements.

## Python package

The `Python package check` workflow builds an sdist and wheel, validates their
metadata, installs the wheel in an isolated environment, and runs both the CLI
version check and `flowmllab smoke` outside a repository checkout. The fixed
cavity archive is included as package data with the same release SHA-256; full
repository QA, figure regeneration, and notebooks still require a checkout.

Publishing should remain disabled until the package name is reserved and the
maintainer has reviewed the built distributions.

When that gate is satisfied, use PyPI Trusted Publishing rather than a long-lived
API token. Configure the GitHub repository, workflow filename, and a protected
`pypi` environment in the PyPI publisher settings before adding the publish job.

## Interactive demo

Deploy `demo/streamlit_app.py` from the public repository using Python 3.12.
Streamlit Community Cloud will find the lightweight dependency file beside the
entry point. After the deployment passes its three blind cases, replace the
README's local demo link with the confirmed public URL.

## Repository settings after approval

- Topics: `scientific-machine-learning`, `computational-fluid-dynamics`, `cfd`,
  `deeponet`, `neural-operators`, `reduced-order-model`, `pod`, `dsmc`,
  `reproducibility`, `teaching-materials`.
- Enable Discussions with `Q&A`, `Show and tell`, `Teaching`, and
  `Reproducibility` categories.
- Set the repository homepage to the confirmed public demo URL.
- Pin FlowMLLab on the owner's GitHub profile.
- Create scoped issues from the `Good first contributions` table in
  [ROADMAP.md](ROADMAP.md).
