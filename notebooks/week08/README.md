# Week 8 - Gas dynamics and scientific machine learning

Week 8 connects the author's two verified gas-dynamics repositories to the
FlowMLLab evidence workflow without duplicating their complete scope.

| Notebook | Purpose | Launch |
| --- | --- | --- |
| `W8_Lab1_Exact_Gas_Dynamics_Student.ipynb` | Exact Rayleigh, Fanno, oblique-shock, nozzle-shock, and shock-tube references; branch and residual checks; links to all nine classical chapter notebooks | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| `W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb` | Branch-failure demonstration; matched interpolation/MLP comparison; edge holdout; dimensional scaling; application timing | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) |

Both notebooks run on CPU. They use the exact relations in
`flowmllab.gas_dynamics` and the frozen evidence in
`results/gas_dynamics_week8/`.

The nine detailed classical notebooks remain in
[`Introduction-to-Compressible-Flows`](https://github.com/Ehsan-Roohi/Introduction-to-Compressible-Flows).
The complete five-problem research pipeline remains in
[`GasDynamicsSciML`](https://github.com/Ehsan-Roohi/GasDynamicsSciML).

The SU2 diamond-airfoil repository is treated only as a bridge to
multidimensional CFD. At the frozen Week-8 commit, one alpha-zero Euler case is
a qualified teaching reference and the other eight cases remain unverified.
