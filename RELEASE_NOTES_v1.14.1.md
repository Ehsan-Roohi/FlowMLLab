# FlowMLLab v1.14.1 - Integrated lessons and reliable course checks

Teaching explanations use direct, present-tense language throughout the lessons.
Exercises, executable code, retained outputs and scientific limitations remain intact.

Course checks execute the current published notebooks in fresh Jupyter kernels and
verify that tracked lesson content and numerical evidence remain unchanged. The
optional Week 7 live continuation is explicitly disabled in the retained-evidence CI
profile. Weeks 7-10 compile from the repository root using XeLaTeX and verified
Times New Roman fonts. The Week 15 check covers every current code cell and requires
the complete archived evidence.

The [v1.14.0 cavity studies](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/v1.14.0)
are included without new training or changed numerical results: 42 label-free Ghia
PINN runs, 12 matched sampling trials and six controlled deep-cavity optimizer runs.
Their accuracy failures and mixed gains remain explicit and separate from successful
supervised POD-DeepONet prediction. The two cavity notebooks and both expanded lecture
PDFs are included in the release assets.

[Course inventory](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.14.1/docs/COURSE_INVENTORY.md)
and [learning paths](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.14.1/START_HERE.md).
