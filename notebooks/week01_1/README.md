# Week 1.1 - AI-Assisted Scientific Software: Specification, Verification, and Trust

This incremental module sits between numerical foundations and supervised
learning. It teaches students to turn a scientific change request into an
executable specification and to audit code proposed by either a person or a
coding agent.

- [Scientific specification](SCIENTIFIC_SPEC.md)
- [Executed notebook](W1_1_AI_Assisted_Scientific_Software.ipynb)
- [Lecture](../../lectures/week01_1_ai_assisted_scientific_software.pdf)
- [Machine-readable evidence](../../results/week01_1_scientific_software/acceptance_record.json)
- [One-click Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb)

The lab does not call a remote model. It provides a vendor-neutral audit harness
for candidate code and therefore runs deterministically on CPU. The retained
reference result verifies second-order vorticity convergence, then checks the
same implementation against the accepted `Re=100` FlowMLLab cavity field.

## Required submission

The assessed task is specified in [ASSIGNMENT.md](ASSIGNMENT.md): add a boundary
volume-flux diagnostic using a coding agent. Complete and commit [SPEC.md](SPEC.md)
before requesting code. Use [PROCESS_LOG.md](PROCESS_LOG.md),
[LINE_REVIEW.md](LINE_REVIEW.md) and [REPORT.md](REPORT.md) for the submission.
The notebook above is the worked preparation, not the completed student assignment.

1. Specification-first commit and actual coding-agent prompt/model log.
2. A new boundary-flux implementation and its original and corrected diffs.
3. Unit, regression, invariant and independent-reference outputs, including failures.
4. Line-by-line review of every changed code and test line.
5. The report titled "What the agent produced, what the physicist corrected, and why".

Do not tune a threshold after opening the final cavity result. If the frozen
contract fails, retain the failure and revise the method in a new, versioned
attempt.
