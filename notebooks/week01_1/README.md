# Week 1.1 - AI-assisted scientific software

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

1. A completed AI-use disclosure, including `No coding agent used` when applicable.
2. A candidate implementation or controlled modification.
3. The complete acceptance JSON, including failed gates.
4. One manual code-review finding that is scientific rather than stylistic.
5. A bounded claim stating exactly what the evidence does and does not support.

Do not tune a threshold after opening the final cavity result. If the frozen
contract fails, retain the failure and revise the method in a new, versioned
attempt.
