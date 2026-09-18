# Week 14 attribution and reuse notice

This module adapts, for teaching, work that is not FlowMLLab's own:

| Component | Source | Status in this repository |
| --- | --- | --- |
| RANS method, `pyCALC-RANS` solver, PINN and NN scripts, correction tables | Lars Davidson (Chalmers), [pyCALC-RANS report and code-update notes](https://www.cfd-sweden.se/lada/Using-Physical-Informed-Neural-Network-PINN-and-NN-improve-a-k-omega-turbulence-model.html); Davidson (2026), DOI 10.1080/14685248.2026.2665148; preprint [arXiv:2511.12493](https://arxiv.org/abs/2511.12493) | Not redistributed. `qa/prepare_week14_source.py` downloads a pinned public archive into a local folder and checks its SHA-256; solver checkpoints stay outside the published tree. |
| Channel-flow DNS statistics at Re_tau = 5200 | Lee and Moser (2015), DOI 10.1017/jfm.2015.268 | Compact attributed teaching extract only. |
| Explanations, plots, diagnostic utilities, classroom orchestration, and the executed evidence under `results/week14_validation/` | FlowMLLab (Ehsan Roohi, AI-assisted) | MIT licence, as the rest of FlowMLLab. |

Permission to adapt Davidson's materials for this course was reported by the
instructor; the written terms are not embedded in this repository. FlowMLLab's
MIT licence therefore does not extend to the upstream solver, scripts,
checkpoints or DNS data. Anyone redistributing those components, or building a
derived work on them, must check the original terms first.

When citing results from this module, cite Davidson for the method and solver,
Lee and Moser for the DNS, and FlowMLLab only for the teaching adaptation and
the audit protocol.
