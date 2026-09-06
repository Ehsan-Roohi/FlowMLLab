# Branch and release procedure

Use a topic branch for changes, review the diff and relevant verification before
promotion to main. An explicitly authorized maintainer update may go directly to
main; use fast-forward updates, never force-push shared history. This document
describes the workflow, not an assertion that GitHub branch protection is enabled.

Before a release:

1. Run the package tests and `python qa/validate_course_release.py`.
2. Execute the affected notebooks. For the published HTML set, run
   `python qa/check_stabilization_notebooks.py --publish-html`; the runner checks
   that retained data and results have not changed. CI attaches generated HTML
   and its execution report for review; it does not silently commit generated files.
3. Review new evidence, its source revision, dependencies, checksums and scope.
   Preserve failed scientific criteria and distinguish a teaching example from
   an independently validated research result. A passing unit test does not
   certify a simulation dataset.
4. Review data-specific license scope. Do not infer collaborator consent from
   a software license. Author decisions and manuscript corrections must be
   recorded separately.
5. Update version metadata and change notes together. Tag the reviewed commit;
   do not move old release tags or replace historical datasets without a new
   version and change record. Publishing a package or DOI is a distinct action
   from pushing code to main.

For reserved tests, freeze model and preprocessing selection before opening or
scoring the test. Record first use, model hash, test hash, metric definitions and
results, including failures. Checksums prove identity, not secrecy or physical
accuracy. After feedback or tuning, call that case a regression test.
