# Start here

This page is the shortest reliable path from a fresh checkout to a meaningful scientific result.
The course has a six-week core (Weeks 1 to 6) and extension weeks (7 to 15); the
[course map](COURSE_MAP.md) shows how they fit together.

## 1. Choose your mode

| Mode | Use it when | First notebook | One-click run |
| --- | --- | --- | --- |
| Complete beginner | You know fluid mechanics but have limited Python experience | `notebooks/week01/01_python_for_cfd_ai_fluids.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| Python-ready | You can use NumPy and Matplotlib | `notebooks/week01/03_cavity_ghia.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |
| AI-assisted research-software ready | You want to qualify agent-proposed scientific code before using it in a result | `notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb) |
| Scientific-ML ready | You already understand CFD validation and supervised learning | `notebooks/week05_06/P0_Project_Setup.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Gas-dynamics ready | You know Mach number and perfect-gas relations and want exact-to-ML comparisons | `notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| Rarefied-cylinder ready | You want real DSMC fields, operator-learning anatomy, and a hard baseline test | `notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) |
| Neural-operator ready | You want geometry-aware and shock-aligned rarefied-flow case studies | `notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) |
| DSMC article reproduction | You want complete rarefied-cavity and mono/diatomic shock data with executable checks | `notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) |
| Week 2.1 probabilistic ML | You want posterior prediction, proper scores, and CFD uncertainty calibration | `notebooks/week02_1/Probabilistic_UQ_CFD.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) |
| Instructor adoption | You are planning a course or workshop | `COURSE_MAP.md`, then `lectures/` |

Project Track 6 (Weeks 5 to 6, Fokker-Planck closure, CUDA GPU) assumes case-wise splitting, scaling, offline versus closed-loop validation, and GPU troubleshooting; take it only after the earlier tracks.

The complete [notebook launcher](notebooks/README.md) links directly to the
Colab notebooks. Their first code cells obtain the repository and package, so a
fresh Colab runtime does not require manual file uploads.

## 2. Create an environment

Python 3.12 is the reference reproducibility target. The package and Colab
entry points support Python 3.10 through 3.13.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
flowmllab smoke --root .
flowmllab qa --root .
```

For TensorFlow training, install `python -m pip install -e ".[ml,test]"`.
For PyTorch notebooks, install
`python -m pip install -e ".[reconstruction,test]"`; install both extras when a
module compares TensorFlow and PyTorch implementations.
If TensorFlow is unavailable on your platform, continue with the data-audit,
interpolation, POD, and plotting sections. Use Google Colab for
TensorFlow-specific cells.

## 3. Run the 20-minute evidence chain

After completing the Week 4 foundations, open
`notebooks/week05_06/P0_Project_Setup.ipynb` and run it top to bottom. P0 is the
20-minute evidence chain for the Weeks 5-6 project track, not the first notebook
for a complete beginner.

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

Typical compute times on a laptop CPU are approximate. Class time is longer,
because every notebook asks for predictions and interpretation between the
cells; each week's README states it (75 to 90 minutes for Weeks 11 to 13).

| Unit | CPU/GPU | Typical runtime |
| --- | --- | --- |
| Week 1 Python/TensorFlow warmups | CPU | 10–30 min each |
| Week 1 cavity validation | CPU | 10–40 min, depending on solver settings |
| Week 1.1 scientific-software audit | CPU | <1 min; no remote model or TensorFlow required |
| Week 2 surrogate | CPU or Colab | 20–45 min |
| Week 3 Maxwellian lab | CPU | 20–40 min |
| Week 3 mini DSMC | CPU/GPU | 30–120 min depending on grid and particle budget |
| Week 4 surrogate labs | CPU or Colab | 20–90 min each |
| Optional Week 4.1 cavity ROM | CPU | <5 min with frozen evidence; about 1 min to regenerate validation |
| Project Tracks 1–4 | Colab recommended | 1–4 h including controlled sweeps |
| Project Track 5 | CPU/GPU | several hours for all stochastic cases |
| Project Track 6 | CUDA GPU | smoke test first; final study is substantially longer |
| Week 7 cylinder LBM | CPU | <2 min with retained evidence; the complete three-grid regeneration is intentionally an instructor/assignment run |
| Week 7.1 rarefied hypersonic cylinder | CPU | <2 min; 20-case data audit, field-interpolation baseline, and five-member teaching analog |
| Week 8 exact gas dynamics and SciML evidence | CPU | <7 min for both labs; full five-problem model regeneration remains optional |
| Week 9 micro-step and micro-nozzle DeepONet cases | CPU | about 6–8 min for both labs, including the compact 2-D FlowMLLab nozzle run |
| Week 10 DSMC cavity and shock reproduction | CPU | normally under 2 min from retained data; no TensorFlow required |
| Week 2.1 probabilistic UQ | CPU | <1 min with retained evidence; no TensorFlow required |
| Week 4.2 Stokes-to-Navier-Stokes lab and 51 x 51 addendum | CPU | <5 min reading retained evidence; full regeneration is an optional switch that needs PyTorch |
| Week 7.2 sparse-sensor state estimation | CPU | a few minutes for the validation grid; no TensorFlow required |
| Week 7.3 self-supervised pretraining and label efficiency | CPU | about 10 min on two cores (1 min pretraining, 4 min label-efficiency loop, 3 min budget control); `QUICK = True` gives about 3 min; PyTorch required |
| Week 7.4 diverse-wake representation transfer | CPU, PyTorch for regeneration | Retained evidence in seconds; full protocol with `python qa/run_week07_4_protocol.py`; regenerate LBM separately |
| Week 9 Lab 3 data-alignment audit | CPU | <2 min |
| Week 10.1 collision-map surrogate | CPU | <5 min |
| Week 11 shock/vortex, reconstruction and cavitation labs | CPU | 5 to 15 min each; PyTorch for the cavitation and reconstruction labs |
| Week 12 moment reconstruction | CPU | about 2 min including the 15 s Noise2Noise fit |
| Week 13 cavity PINNs | CPU | about 3 min; the training cell takes about a minute (PyTorch) |
| Week 14 RANS closures | CPU, separate environment | 5 to 10 min for the notebook; the optional upstream solver runs take hours |
| Week 15 geometry-aware operators | CPU, full checkout plus the 333 MB Zenodo LR evidence archive | about 5 min to verify retained fields; no PyTorch, CFD or retraining |

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
