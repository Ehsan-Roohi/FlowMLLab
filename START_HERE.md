# Start here

This page is the shortest reliable path from a fresh checkout to a meaningful scientific result.
The course has a six-week core (Weeks 1 to 6) and extension weeks (7 to 22); the
[course map](COURSE_MAP.md) shows how they fit together.

## 1. Choose your mode

Start with one of three routes. The numbered weeks are a curriculum map, not a
requirement to complete all 22 units in one semester.

| Route | Prerequisites | First outcome | Continue |
| --- | --- | --- | --- |
| Learn from the beginning | Basic fluid mechanics and algebra; Python can be new | Plot arrays and verify the cavity against an independent reference | Week 1 Python → cavity → Weeks 2–4 → P0 and a Weeks 5–6 project |
| Learn scientific ML with Python | NumPy, Matplotlib and basic derivatives | Separate numerical verification from prediction error | Week 1 cavity validation → Weeks 2–4; use the readiness rows below for extensions |
| Reproduce evidence or teach | Read the module prerequisite and data-role contract first | Rescore a retained result or plan a bounded lesson | Results guide / course map → selected module setup; full regeneration is optional and separately costed |

[Per-unit prerequisites and execution status](docs/COURSE_INVENTORY.md#unit-prerequisites-and-execution-status)
· [Actual file inventory](docs/COURSE_INVENTORY.md). Estimates are planning guidance,
not measurements of this revision or of a representative student.


| Mode | Use it when | First notebook | One-click run |
| --- | --- | --- | --- |
| Complete beginner | You know fluid mechanics but have limited Python experience | `notebooks/week01/01_python_for_cfd_ai_fluids.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| Python-ready | You can use NumPy and Matplotlib | `notebooks/week01/03_cavity_ghia.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |
| AI-assisted research-software ready | You want to qualify agent-proposed scientific code before using it in a result | `notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb) |
| Pressure-velocity coupling ready | You have completed the original cavity lab and want SIMPLE/PISO/PIMPLE at Re=100 | `notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb) |
| Scientific-ML ready | You already understand CFD validation and supervised learning | `notebooks/week05_06/P0_Project_Setup.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| Gas-dynamics ready | You know Mach number and perfect-gas relations and want exact-to-ML comparisons | `notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| Rarefied-cylinder ready | You want real DSMC fields, operator-learning anatomy, and a hard baseline test | `notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb) |
| Neural-operator ready | You want geometry-aware and shock-aligned rarefied-flow case studies | `notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) |
| DSMC article reproduction | You want complete rarefied-cavity and mono/diatomic shock data with executable checks | `notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb) |
| Week 2.1 probabilistic ML | You want posterior prediction, proper scores, and CFD uncertainty calibration | `notebooks/week02_1/Probabilistic_UQ_CFD.ipynb` | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week02_1/Probabilistic_UQ_CFD.ipynb) |
| Instructor adoption | You are planning a course or workshop | `COURSE_MAP.md`, then `lectures/` |

Project Track 6 (Weeks 5 to 6, Fokker-Planck closure, CUDA GPU) assumes case-wise splitting, scaling, offline versus closed-loop validation, and GPU troubleshooting; take it only after the earlier tracks.

The [notebook guide](notebooks/README.md) provides Colab launch links where
documented and setup links for exceptions. Most Colab bootstrap cells obtain the
repository and package; this is not a promise that every module has passed a fresh
hosted Run All. Week 14 has a separate environment, Week 15 needs the documented
evidence archive, and Weeks 17–22 use the local/ZIP setup with interactive Colab
acceptance still open.

## 2. Create an environment

### Choose what you will execute

| Operation | What it establishes | Resources and limits |
| --- | --- | --- |
| Read / rescore retained evidence | Inspect saved outputs or recompute metrics from versioned files | Usually CPU; a saved plot or checkpoint is not a new training or CFD run |
| Run a teaching exercise | Execute the stated compact numerical model, baseline or small fit | Follow module extras and input data; some labs train small CPU models |
| Regenerate a full protocol | Recreate datasets, sweeps, CFD or research-scale fits | Separate commands and budgets; P6 needs CUDA, Week 16 new CFD needs Gmsh/SU2, and research archives may require external software/resources |

Choose the retained-evidence route first when hardware is limited. Never rename
a replay as new validation, or a quick smoke protocol as release evidence.


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

The following compute times are planning estimates carried from module guides,
not timing measurements of this revision. Hardware, downloads and numerical
settings can change them. Class time is longer,
because every notebook asks for predictions and interpretation between the
cells; each week's README states it (75 to 90 minutes for Weeks 11 to 13).

<!-- course-runtime:start -->
| Unit | Mode / resources | Compute planning estimate |
| --- | --- | --- |
| [Week 1](weeks/week01/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 10–40 min per lab; classroom work longer. |
| [Week 1.1](weeks/week01_1/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 1 min for the numerical audit. |
| [Week 1.2](weeks/week01_2/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Not measured for this revision; inspect the setup before a full solve. |
| [Week 2](weeks/week02/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 20–45 min. |
| [Week 2.1](weeks/week02_1/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 1 min for retained-evidence path. |
| [Week 3](weeks/week03/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 20–40 min Maxwellian; 30–120 min mini-DSMC. |
| [Week 4](weeks/week04/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 20–90 min per lab; training settings matter. |
| [Week 4.1](weeks/week04_1/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 5 min retained-evidence path; fresh validation about 1 min. |
| [Week 4.2](weeks/week04_2/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 5 min retained-evidence path; full training not included. |
| [Week 5](weeks/week05/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | P0 about 20 min after Week 4; tracks 1–4 about 1–4 h including sweeps. |
| [Week 6](weeks/week06/README.md) | CPU/evidence paths; optional P6 requires CUDA. | Several hours for stochastic tracks; full CUDA closure study not estimated. |
| [Week 7](weeks/week07/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 2 min retained-evidence path; full three-grid CFD excluded. |
| [Week 7.1](weeks/week07_1/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 2 min teaching path. |
| [Week 7.2](weeks/week07_2/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | A few minutes for the validation grid. |
| [Week 7.3](weeks/week07_3/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | About 10 min on two cores; QUICK about 3 min. |
| [Week 7.4](weeks/week07_4/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Seconds to read retained evidence; full regeneration not estimated. |
| [Week 8](weeks/week08/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 7 min for two evidence-based labs; full model generation excluded. |
| [Week 9](weeks/week09/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | About 6–8 min for Labs 1–2; Lab 3 under 2 min. |
| [Week 10](weeks/week10/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 2 min retained-data path. |
| [Week 10.1](weeks/week10_1/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | Under 5 min CPU teaching path. |
| [Week 11](weeks/week11/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 5–15 min per lab. |
| [Week 12](weeks/week12/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | About 2 min including compact training. |
| [Week 13](weeks/week13/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | About 3 min compact CPU lab; A100 research runs excluded. |
| [Week 14](weeks/week14/README.md) | CPU teaching exercises and retained evidence; full regeneration is separate. | 5–10 min notebook; optional upstream solver runs take hours. |
| [Week 15](weeks/week15/README.md) | CPU retained-evidence verification; no model fitting in the main classroom path. | About 5 min retained-field verification after archive download. |
| [Week 16](weeks/week16/README.md) | Data-based laptop notebook; Gmsh/SU2 only for new CFD. | Not measured for this revision; CFD campaign time excluded. |
| [Week 17](weeks/week17/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
| [Week 18](weeks/week18/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
| [Week 19](weeks/week19/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
| [Week 20](weeks/week20/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
| [Week 21](weeks/week21/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
| [Week 22](weeks/week22/README.md) | CPU worked examples, exercises and evidence rescoring; fresh full-protocol fitting is separate. | Not measured for this revision. |
<!-- course-runtime:end -->

Project tracks 1–4 can include 1–4 hours of controlled sweeps. Track 5 can
require several hours for stochastic cases; Track 6 requires CUDA and a smoke
check before a substantially longer final study. These are planning estimates.
The Week 5 sparse-sensing companion uses Week 7 wakes and is taken after Week 7;
its runtime and the Week 7 modal-forecasting companion runtime are not measured
for this revision.

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


<!-- transformer-course-v3 -->
## Weeks 17-22: active Transformer laboratories

[Course guide and setup](docs/TRANSFORMER_COURSE.md). Six distinct laboratories extend Weeks 7.3, 7.4 and 15 with required language modeling, coding exercises, separate instructor solutions, matched baselines and explicit transfer audits.

## Optional self-study beyond the numbered route

The [inventory](docs/COURSE_INVENTORY.md#optional-self-study) lists optional
supplements separately from Weeks 1–22. The geometry–time evaluation bridge
requires Week 15 geometry/SDF and Week 21 rollout; Week 18 causality is useful.
It uses manufactured arrays and does not claim CFD or model-training evidence.
