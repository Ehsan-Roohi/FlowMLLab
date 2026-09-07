# Week 1.1 - Specification, Verification, and Trust

## Your change to FlowMLLab

Add `net_volume_flux(x, y, u, v)` in a new file
`flowmllab/student_mass_balance.py`. Use a coding agent of your choice to
produce the first implementation. You, the investigator, review and correct it.
Running the reference notebook alone does not complete this assignment.

The function integrates outward normal velocity around a rectangular 2-D
domain: right minus left contributions from u, plus top minus bottom
contributions from v. Use composite trapezoidal quadrature with the supplied
coordinates. Do not reuse the reference divergence RMS as the new feature:
boundary-integrated flux and local incompressibility are different diagnostics.
Global cancellation can hide local errors.

## 1. Specify, commit, then request code

Create a branch in your own fork. Copy the supplied `SPEC.md` to
`submissions/week01_1/SPEC.md` and complete every field before requesting code.
Commit only the completed specification first. Record that commit SHA in
`PROCESS_LOG.md`; a timestamp alone does not establish the sequence.
Do not modify retained `results/` or the instructor's acceptance criteria.

## 2. Ask a coding agent to implement the specification

Supply the committed specification and request the function plus tests.
Keep the exact prompt and response, the first generated patch and the final
human-reviewed patch. Record actual tool/model identifiers; use `not exposed`
if a tool does not disclose its model. Never invent a model or a correction.
Do not include credentials or private unrelated context in the public log.
If an agent is unavailable, obtain an instructor-approved alternative and
record the exception; silently replacing the exercise with manual coding is
not equivalent to this assignment.

## 3. Run four distinct evidence levels

| Level | Required evidence |
| --- | --- |
| Unit test | Reject wrong shapes, nonfinite data and non-increasing coordinates; check the scalar return and outward signs. |
| Numerical regression | Run the existing Week 1.1 tests and compare the retained acceptance record; preserve the cavity archive hash. |
| Physical invariant | A constant velocity field and the closed Re=100 cavity have zero net outward volume flux within 1e-12. Add 0.1 to u on the entire right boundary of the cavity: flux must increase by 0.1 on the unit square. This is a deliberately altered field, not a new CFD solution. |
| Baseline/reference | For u=x, v=y on the unit square, exact flux is 2. For u=x squared, v=0, it is 1. Test rectangular and nonuniform coordinates against independently derived boundary integrals. |

Run the instructor checks from the repository root:

```sh
python qa/check_week01_1_candidate.py flowmllab/student_mass_balance.py
python -m unittest tests.test_scientific_software -v
```

Save complete command output and exit codes under your submission, including
failed attempts. The candidate checker is an initial acceptance suite, not a
proof for arbitrary geometry or arbitrary velocity fields. Add your own tests
and explain their independence from the candidate implementation.

## 4. Review every changed line

Use the final diff with line numbers. In `LINE_REVIEW.md`, cover every changed
code/test line or contiguous range: physical meaning, units, signs, coordinate
axis, quadrature, input handling and test independence where applicable.
Explain any line that has no scientific effect. Re-review changed lines after
each correction. One general comment is not a line-by-line review.

## 5. Submit a GitHub pull request

Submit a PR to your own fork or the instructor-designated teaching repository.
Include the specification-first commit, candidate module, tests, process log,
line review, evidence outputs and `REPORT.md`. Do not publish a PR to the
upstream course repository unless the instructor requests it.

The report title must be **What the agent produced, what the physicist corrected, and why**.
Explain observed results, corrections, retained failures and the claim limits.
If no correction was necessary, document the checks supporting that conclusion
instead of fabricating an error. Submission requires a genuine implementation
diff and all four evidence levels, not merely a successful notebook run.
