# Start here

This page is the shortest reliable path from a fresh checkout to a meaningful scientific result.

## 1. Choose your mode

| Mode | Use it when | First notebook | One-click run |
| --- | --- | --- | --- |
| Complete beginner | You know fluid mechanics but have limited Python experience | `notebooks/week01/01_python_for_cfd_ai_fluids.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| Python-ready | You can use NumPy and Matplotlib | `notebooks/week01/03_cavity_ghia.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |
| Scientific-ML ready | You already understand CFD validation and supervised learning | `notebooks/week05_06/P0_Project_Setup.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Instructor adoption | You are planning a course or workshop | `COURSE_MAP.md`, then `lectures/` |

Do not begin with Track 6 unless you already understand case-wise splitting, scaling, offline versus closed-loop validation, and GPU troubleshooting.

The complete [notebook launcher](notebooks/README.md) links directly to all 18
Colab notebooks. Their first code cells obtain the repository and package, so a
fresh Colab runtime does not require manual file uploads.

## 2. Create an environment

Python 3.12 is the compatibility target used for the release.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
flowmllab smoke --root .
flowmllab qa --root .
```

For neural-network training, install `python -m pip install -e ".[ml,test]"`.
If TensorFlow is unavailable on your platform, continue with the data-audit,
interpolation, POD, and plotting sections. Use Google Colab for
TensorFlow-specific cells.

## 3. Run the 20-minute evidence chain

Open `notebooks/week05_06/P0_Project_Setup.ipynb` and run it top to bottom.

You should be able to answer all five questions before selecting a project:

1. Which Reynolds numbers are development cases and which are withheld?
2. What is the dataset hash and why is it recorded?
3. What is the difference between solver residual and benchmark error?
4. Why does the wall metric omit the two moving-lid corners?
5. Does field interpolation reproduce the withheld Re = 275 case within the expected teaching-data tolerance?

The notebook writes a project card only after the dataset audit and baseline succeed.

## 4. Follow the weekly order

Each week has the same learning loop:

**predict → derive or inspect → run → validate → interpret → retain evidence**.

Do not skip the interpretation cells. A notebook is complete only when you can explain why the output is credible, where it may fail, and which evidence would change your conclusion.

Typical student runtimes are approximate:

| Unit | CPU/GPU | Typical runtime |
| --- | --- | --- |
| Week 1 Python/TensorFlow warmups | CPU | 10–30 min each |
| Week 1 cavity validation | CPU | 10–40 min, depending on solver settings |
| Week 2 surrogate | CPU or Colab | 20–45 min |
| Week 3 Maxwellian lab | CPU | 20–40 min |
| Week 3 mini DSMC | CPU/GPU | 30–120 min depending on grid and particle budget |
| Week 4 surrogate labs | CPU or Colab | 20–90 min each |
| Optional Week 4.1 cavity ROM | CPU | <5 min with frozen evidence; about 1 min to regenerate validation |
| Project Tracks 1–4 | Colab recommended | 1–4 h including controlled sweeps |
| Project Track 5 | CPU/GPU | several hours for all stochastic cases |
| Project Track 6 | CUDA GPU | smoke test first; final study is substantially longer |
| Week 7 cylinder LBM | CPU | 15–45 min for the quick five-regime path; longer for qualification/refinement |

## 5. Keep blind cases blind

Before the first blind cell, save:

- the physical case lists;
- architecture/rank/loss candidates;
- the validation selection rule;
- the random seed or seed list;
- the planned metrics and failure threshold; and
- one sentence predicting the blind outcome.

If you tune after seeing a blind result, rename that case as development data and create a new untouched test. Do not keep the word “blind” for a case that influenced a decision.

## 6. Minimum evidence for a final result

A defensible project contains:

- one matched non-neural or exact baseline;
- all development and blind cases listed explicitly;
- aggregate numerical errors;
- at least two physical diagnostics;
- one failure, limitation, or tradeoff;
- runtime/cost when acceleration is claimed;
- a saved configuration and machine-readable metrics; and
- a notebook that restarts and runs in order.

Use the checklist at the end of each project notebook before writing the one-slide research summary.
