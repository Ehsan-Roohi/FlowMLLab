# Packaging and public deployment

This document records the remaining maintainer steps. It does not claim that a
PyPI package or public demo exists before those services confirm deployment.

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
