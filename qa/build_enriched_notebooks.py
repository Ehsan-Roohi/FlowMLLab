#!/usr/bin/env python3
"""Build the expanded learner editions of the seven project notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
MARKER = "<!-- MIE690A enriched learner edition v2 -->"


def lines(text: str) -> list[str]:
    text = text.strip() + "\n"
    return text.splitlines(keepends=True)


def cell_id(kind: str, text: str) -> str:
    return hashlib.sha1((kind + "\n" + text).encode("utf-8")).hexdigest()[:12]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id("markdown", text),
        "metadata": {},
        "source": lines(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id("code", text),
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


REPO_BOOTSTRAP = r'''
# Repository/Colab bootstrap. Run this before the import cell below.
from pathlib import Path
import sys

def _find_course_root(start=Path.cwd()):
    candidates = [start, *start.parents]
    for base in candidates:
        if (base / "common" / "w5_common.py").exists():
            return base
    # Colab flat-upload fallback: helper files and data beside the notebook.
    if (start / "w5_common.py").exists():
        return start
    raise FileNotFoundError(
        "Course root not found. Clone the repository, or upload w4utils.py, "
        "w5_common.py, the track-specific helpers, and cavity_data.npz as listed above."
    )

COURSE_ROOT = _find_course_root()
COMMON_DIR = COURSE_ROOT / "common" if (COURSE_ROOT / "common").exists() else COURSE_ROOT
DATASET_PATH = COURSE_ROOT / "data" / "cavity_data.npz"
if not DATASET_PATH.exists():
    DATASET_PATH = COURSE_ROOT / "cavity_data.npz"
sys.path.insert(0, str(COMMON_DIR))
print("Course root:", COURSE_ROOT)
print("Common helpers:", COMMON_DIR)
print("Dataset:", DATASET_PATH)
'''


TRACK6_BOOTSTRAP = r'''
# Repository/Colab bootstrap for Track 6.
from pathlib import Path
import os, sys

def _find_track6_dir(start=Path.cwd()):
    for base in [start, *start.parents]:
        candidate = base / "advanced" / "fp_closure"
        if (candidate / "fp_cavity_reference.py").exists():
            return candidate
    if (start / "fp_cavity_reference.py").exists():
        return start
    raise FileNotFoundError(
        "Track-6 scripts not found. Run from the course repository or upload all files "
        "from advanced/fp_closure into the Colab Files pane."
    )

TRACK6_DIR = _find_track6_dir()
sys.path.insert(0, str(TRACK6_DIR))
os.chdir(TRACK6_DIR)
print("Track-6 working directory:", TRACK6_DIR)
'''


UNIVERSAL_START = f'''
{MARKER}

## How to learn from this notebook

This is a guided computational laboratory, not a script to execute without reading. For every numbered stage:

1. read the physical question and write a prediction;
2. inspect the inputs, outputs, units, and split before running code;
3. run the cell and check assertions/warnings;
4. compare with the stated baseline or physical diagnostic; and
5. write one or two sentences explaining what the result does **and does not** establish.

Use **Restart and Run All** before treating any output as final. Hidden state from out-of-order execution is a reproducibility failure.

### Evidence contract

Keep four kinds of evidence separate:

- **numerical evidence:** residuals, accepted cases, data hashes, grid/time/particle budgets;
- **statistical evidence:** losses, relative errors, variability across seeds/cases;
- **physical evidence:** centerlines, walls, divergence, vortex structure, positivity, moments;
- **computational evidence:** runtime, memory, saved configuration, and machine-readable metrics.

A claim is only as strong as the weakest relevant layer.
'''


UNIVERSAL_END = r'''
## Reproducibility record

Before closing the notebook, record:

- Python and package versions;
- dataset hash and helper versions;
- every physical case in development, validation, and blind sets;
- every seed and candidate value tried;
- the selection rule and when it was frozen;
- output filenames and units; and
- any cell that was skipped, changed, or run with a reduced budget.

Restart the kernel and run all cells in order. If the result changes materially, report the variability instead of selecting the preferred run.

## Troubleshooting without corrupting the experiment

| Symptom | Safe action | Unsafe action |
| --- | --- | --- |
| Missing helper/data file | Re-run the bootstrap and verify paths/hash | Download an unlabeled older copy |
| Training is slow | Use the documented smoke configuration, then label it “smoke” | Quietly reduce epochs/data in the final claim |
| Validation is poor | Inspect scaling, split, and baseline; revise on development data | Open the blind case to choose settings |
| Blind result fails | Report/localize failure and propose a new future experiment | Tune on the blind case while keeping its label “blind” |
| Stochastic result changes | Run multiple declared seeds and report mean/spread | Keep rerunning until one result looks good |
| A neural model loses to interpolation | Verify fairness, then recommend the simpler method | Hide the baseline |

## Final report outline

1. **Question and hypothesis** — one falsifiable sentence.
2. **Data and split** — physical cases, numerical source, and blind unit.
3. **Baseline** — simplest credible comparator using the same allowed information.
4. **Modification** — the one controlled change.
5. **Selection** — validation-only candidates and frozen rule.
6. **Blind numerical result** — aggregate errors and variability.
7. **Physical result** — at least two diagnostics tied to the flow.
8. **Failure or limitation** — where confidence ends.
9. **Cost and reproducibility** — runtime, environment, seeds, saved files.
10. **Conclusion** — helped, hurt, or revealed a tradeoff; no forced positive AI claim.
'''


TRACK_CONTENT = {
    "P0_Project_Setup.ipynb": {
        "intro": r'''
## What this setup notebook teaches

The setup is itself a scientific exercise. It verifies that every later project starts from the same numerical object rather than from a file that merely has the right name.

You will learn to distinguish:

- **identity:** the SHA-256 hash tells whether two files have identical bytes;
- **integrity:** shape, finite-value, pressure-gauge, and accepted-case checks tell whether the contents satisfy the data contract;
- **numerical quality:** residual and benchmark columns describe how the CFD labels were produced;
- **project validity:** a declared case-wise split defines what future generalization claim is allowed.

### Expected output

At the end you should have a dataset audit table, a recovered Re = 275 interpolation baseline, a plot with physical diagnostics, and `project_choice.json`. Do not begin a project if any assertion fails.
''',
        "before:## Dataset audit": r'''
## Data schema before the audit

The dataset stores one complete 65×65 field per Reynolds number. Array order is `(case, y, x)`. The primary variables are `u`, `v`, streamfunction `psi`, and vorticity `omega`; pressure is reconstructed and shifted to a zero-mean gauge. The pressure gauge means an arbitrary constant offset is not a prediction error, while a wrong pressure gradient is.

**Prediction prompt:** Which check below would catch a corrupted field full of NaNs? Which would catch a pressure offset? Which would catch convergence to a wrong but finite field? No single check catches all three.
''',
        "before:## Reproduce a non-neural baseline": r'''
## Why interpolation is the recovery baseline

For a smooth one-parameter family, linear field interpolation is cheap, transparent, and often difficult to beat. If `Re_a < Re_* < Re_b`, the baseline is

\[
q(Re_*)=(1-\alpha)q(Re_a)+\alpha q(Re_b),\qquad
\alpha=\frac{Re_*-Re_a}{Re_b-Re_a}.
\]

This operates on complete fields, not randomly selected grid points. It establishes a performance floor for every neural surrogate that uses the same neighboring development cases.

Before running, predict where interpolation error will be largest: the smooth vortex core, the moving-lid corners, or nearly uniform regions. Use the plot to check your reasoning.
''',
        "before:## Select one project variant": r'''
## From topic to falsifiable question

“Use a neural network for cavity flow” is a topic, not a research question. A usable project question names the baseline, controlled modification, test unit, metric, and possible failure.

Example: *When complete Reynolds-number cases are held out, does a wall-weighted coordinate DNN reduce corner-excluded wall RMS error relative to the same uniformly trained network without increasing global velocity error by more than 10%?*

Write the one-sentence question before choosing plots. A figure should answer the question; the question should not be reverse-engineered from the most attractive output.
''',
        "end": r'''
## Setup concept check

1. Why does a fixed hash not prove that the numerical method is accurate?
2. Why is a random grid-point split easier than a Reynolds-case split?
3. Why is pressure compared after applying a common gauge?
4. If interpolation wins, what scientific conclusion is justified?
5. Which project decision must be frozen before the blind gate?

### Reading

- Ghia, Ghia & Shin (1982) for the cavity centerline benchmark.
- Wilson et al. (2014) and Sandve et al. (2013) for reproducible-computing practice.
- `references/README.md` for the complete annotated route.
''',
    },
    "P1_Re_Generalization.ipynb": {
        "intro": r'''
## Learning objectives and prerequisites

By the end you should be able to distinguish interpolation from extrapolation geometrically, create a case-wise split, select an architecture without test leakage, retrain fairly on all permitted development cases, and localize the first tested Reynolds interval where a failure threshold is crossed.

Prerequisites: Week-4 data audit, feature/target scaling, relative L2 error, centerline interpretation, and the difference between model selection and final evaluation.
''',
        "before:## 1. Freeze": r'''
## Generalization is defined by the physical support

For Variant 1A, every blind Reynolds number lies within the convex interval spanned by development Reynolds numbers. For Variant 1B, blind cases lie above the largest development Reynolds number. The neural network sees coordinates from every part of the cavity in both variants; what changes is the operating condition.

This is why a random point-wise split is misleading: neighboring points from the same Reynolds realization leak its structure into both training and test sets.

**Prediction prompt:** Draw the development Reynolds numbers on a line. Mark every blind case. State whether each is interpolation or extrapolation before running any model.
''',
        "before:## 2. Architecture selection": r'''
## What the architecture comparison can establish

The candidate sweep tests a restricted question: with fixed data, optimizer, stopping logic, and seed, which width/depth gives the best validation evidence? It does not establish a universal best architecture.

Read the selection code in five steps: build case-wise samples; fit scalers on training data; train each candidate; evaluate one complete validation case; select and record the epoch budget. If a scaler is fit using blind coordinates/targets, the experiment is contaminated even when the network never directly trains on blind labels.
''',
        "before:## 4. Open": r'''
## Metric glossary for the blind comparison

- `relative_L2_uv`: global velocity-vector error; useful but dominated by large smooth regions.
- `relative_L2_p`: pressure-field error after a consistent gauge; does not replace pressure-gradient inspection.
- `wall_rms`: boundary fidelity with the two lid corners excluded because the ideal boundary condition is discontinuous there.
- `divergence_rms`: discrete incompressibility diagnostic; a coordinate DNN is not divergence-free by construction.
- centerlines: local profiles that reveal misplaced extrema and curvature.
- vortex evidence: structure and location that can be hidden by a small global norm.

Predeclare a failure threshold. Crossing it localizes the failure to an interval between tested cases; it does not reveal an exact critical Reynolds number.
''',
        "end": r'''
## Concept check and further reading

1. Why can direct field interpolation use fewer modeling assumptions than a coordinate DNN?
2. Why is Re = 350 only the *first tested failure* in a grid containing 300 and 350?
3. Which metric is most sensitive to a moving-lid corner error?
4. What evidence would justify the added neural complexity?
5. If the DNN wins globally but loses on divergence, how should the conclusion be written?

Read Brunton, Noack & Koumoutsakos (2020) for ML task structure in fluids and Wilson et al. (2014) for controlled computational comparisons.
''',
    },
    "P2_Physics_Guided_DNN.ipynb": {
        "intro": r'''
## Learning objectives and controlled-ablation contract

This notebook asks whether changing the loss improves a named physical diagnostic. Architecture, data, split, optimizer, and training budget remain fixed. You should be able to derive the added term, explain where it is evaluated, select a weight under a global-error constraint, and distinguish a useful tradeoff from a universal improvement claim.

Prerequisites: coordinate DNNs, automatic differentiation, boundary conditions, discrete divergence, and validation-only model selection.
''',
        "before:## 1. Baseline": r'''
## From data loss to a physics-guided objective

The baseline minimizes standardized pointwise data error. Variant 2A changes sample importance near walls; Variant 2B penalizes the divergence of the network's physical velocity output.

\[
\mathcal L=\mathcal L_{data}+\lambda_{phys}\mathcal L_{phys}.
\]

This is a *soft constraint*: a nonzero weight encourages behavior but does not guarantee it. Large weights can damage global accuracy or make optimization ill-conditioned. The scientific object is therefore a tradeoff curve, not the strongest possible weight.

The two lid corners are excluded from wall RMS because stationary side walls meet a moving lid discontinuously. Penalizing an idealized discontinuity too strongly can force the model to smooth an incompatible target.
''',
        "before:## 2A. Wall": r'''
## Variant 2A: how wall weighting changes the question

The distance `d_w` is geometric. A weight such as `1 + A exp(-d_w/delta)` increases the contribution of near-wall samples without adding new physical data. Check the minimum, maximum, and spatial distribution of weights before training.

**Prediction prompt:** As amplitude grows, which response is most plausible: monotonic improvement of every metric, wall improvement with interior degradation, or no effect? State why.
''',
        "before:## 2B. Divergence": r'''
## Variant 2B: derivatives must be in physical units

The network is trained on standardized inputs and outputs, but incompressibility is defined for physical `u`, `v`, `x`, and `y`. Chain-rule scale factors therefore matter. A divergence computed directly in standardized coordinates can be numerically small while representing the wrong physical derivative.

Automatic differentiation measures the continuous derivative of the neural approximation. The CFD labels and reported diagnostic may use a discrete finite-difference operator. Agreement is expected only up to representation and discretization differences.
''',
        "before:## 4. Blind": r'''
## How to interpret a Pareto tradeoff

A setting is scientifically useful when its intended metric improves and the cost in other relevant metrics remains within the frozen tolerance. Do not collapse wall, divergence, velocity, and pressure into one unmotivated score.

Plot each candidate in a plane such as global velocity error versus wall RMS or divergence RMS. A dominated point is worse on both axes. A nondominated point may still be unacceptable if it violates the predeclared tolerance.
''',
        "end": r'''
## Concept check and further reading

1. Why is a physics-guided loss not the same as a physics-constrained architecture?
2. Where do normalization scale factors enter the divergence derivative?
3. Why must the zero-weight baseline be retrained under the same protocol?
4. What would make a 5% wall improvement scientifically irrelevant?
5. How would a streamfunction-output architecture change the divergence question?

Read Raissi et al. (2019), Karniadakis et al. (2021), and Cai et al. (2021). For a flow-specific zonal-loss example, see Roohi & Mahdavi (2026) in the annotated references.
''',
    },
    "P3_POD_Study.ipynb": {
        "intro": r'''
## Learning objectives and notation

You will separate three error sources: finite-rank representation, coefficient prediction, and final field error. You should be able to construct POD only from development snapshots, interpret singular values and modes, compare separate and shared bases, and choose rank using compression plus physical evidence rather than cumulative energy alone.

Prerequisites: matrix shapes, centering, singular value decomposition, case-wise splitting, and regression baselines.
''',
        "before:## 1. Build POD": r'''
## POD as a change of coordinates

Arrange each flattened development field as a row of a snapshot matrix `Q`. After subtracting the development mean,

\[
Q' = U\Sigma V^T,\qquad
q(Re)\approx \bar q+\sum_{k=1}^{r}a_k(Re)\phi_k.
\]

The right singular vectors are spatial modes and the coefficients are coordinates of a case in that basis. Computing the mean or modes with blind fields leaks blind spatial information even if their Reynolds labels are withheld.

Cumulative energy measures variance captured by the development ensemble. It does not assign physical importance: a weak corner vortex or wall feature can live in a low-energy mode.
''',
        "before:## 2. Separate": r'''
## Three distinct questions

1. **Representation:** with exact projected coefficients, can rank `r` reconstruct the field?
2. **Learnability:** can the branch model predict the `r` coefficients from Reynolds number?
3. **Physical fidelity:** does the final reconstructed field preserve centerlines, walls, pressure, divergence, and vortex structure?

If exact-coefficient reconstruction is poor, a better coefficient network cannot fix the basis. If exact reconstruction is good but final prediction is poor, increasing rank may add targets that are harder to learn.

Always compare the branch MLP with direct interpolation of modal coefficients. The reduced coordinates do not automatically require a neural predictor.
''',
        "before:## 3. Freeze": r'''
## Rank is a model-selection decision

Choosing the largest tested rank guarantees the weakest compression and can increase coefficient-prediction difficulty. The supplied rule selects the smallest rank within a validation tolerance of the best global velocity error, then checks pressure and physical evidence.

Record the coefficient count. For separate bases, rank `r` may mean `3r` coefficients; for a shared basis it may mean only `r`. Equal nominal rank is not equal model size.
''',
        "before:## 5. Required low": r'''
## Read the low-rank failure spatially

Use the deliberately low-rank model as a microscope. Compare: primary-vortex center, secondary corner structure, centerline extrema, wall velocity, and pressure gradient. A visually smooth field can still have the wrong topology; a low-energy mode can carry the first scientifically important failure.
''',
        "end": r'''
## Concept check and further reading

1. Why must the mean field be computed from development cases only?
2. Can cumulative energy rise while final predictive error worsens? Explain.
3. How do exact-projection and learned-coefficient reconstructions isolate error sources?
4. Why is rank 4 in a shared basis not directly comparable with rank 4 per variable?
5. What baseline should be used for `a_k(Re)` prediction?

Read Sirovich (1987), Berkooz et al. (1993), Taira et al. (2017), Rowley et al. (2004), and Hesthaven & Ubbiali (2018).
''',
    },
    "P4_Uncertainty_Study.ipynb": {
        "intro": r'''
## Learning objectives: uncertainty is not one number

This track studies two different sources of variability. Variant 4A changes training initialization while holding data fixed. Variant 4B changes the amount of development information while holding the test and training protocol fixed. You should be able to report distributions rather than a preferred run and test whether a proposed uncertainty indicator tracks actual blind error.

Prerequisites: random seeds, case-wise splits, ensemble mean/spread, validation-only model selection, and confidence versus calibration.
''',
        "before:## 4A.": r'''
## A practical uncertainty taxonomy

- **Numerical error:** discretization, solver convergence, pressure recovery, or finite sampling window in the labels.
- **Aleatoric/statistical variability:** irreducible sampling variation in stochastic particle data.
- **Epistemic/model variability:** sensitivity to limited data, architecture, initialization, or training.
- **Distribution shift:** blind physical conditions that are not represented by development support.

An ensemble of training seeds probes only a narrow part of epistemic variability. Small ensemble spread does not prove low error; all members can be confidently wrong under extrapolation.
''',
        "before:## 4B.": r'''
## Design the comparison before running it

For a seed study, keep case lists, architecture, epoch budget, scaling rule, and metrics identical. For a data-sufficiency study, use nested development sets so that “more data” really adds cases rather than replacing them. Do not change two factors and attribute the outcome to one.

**Prediction prompt:** Will ensemble spread be largest where the true error is largest? Write a reason it might succeed and a reason it might fail.
''',
        "before:## Required": r'''
## Reporting variability honestly

Report every seed, mean, standard deviation, range, and the baseline. With only a few seeds, do not imply a precisely estimated probability distribution. For data sufficiency, show error versus number and location of development Reynolds cases; count alone can hide poor coverage.

To evaluate an uncertainty indicator, compare spread and actual error across complete blind cases. Ranking agreement is a useful first check, but calibration requires coverage analysis at declared intervals or thresholds.
''',
        "end": r'''
## Concept check and further reading

1. What uncertainty source is measured by changing only neural initialization?
2. Why can five ensemble members agree and still be wrong?
3. Why should development sets be nested in a data-density study?
4. How would label uncertainty alter the interpretation of a seed ensemble?
5. Which additional experiment would separate numerical-label error from model error?

Read the reproducibility references in `references/README.md` and the uncertainty discussion in Karniadakis et al. (2021).
''',
    },
    "P5_Rarefied_Cavity.ipynb": {
        "intro": r'''
## Learning objectives and scope warning

You will treat each `(Kn, Uwall, seed)` run as a complete stochastic physical case, quantify seed-to-seed label variability, compare single-seed and averaged labels or test unseen Knudsen number, and keep finite-window teaching evidence separate from validation-quality DSMC claims.

Prerequisites: Maxwellian sampling, DSMC move/collide/sample logic, Knudsen regimes, normalization, stochastic independence, and case-wise splitting.
''',
        "before:## 1. Generate": r'''
## What the mini solver can and cannot establish

The supplied solver preserves the pedagogical structure of particle transport, wall interaction, stochastic collisions, and macroscopic sampling. Its reduced grid, particles per cell, and averaging window make it suitable for workflow experiments, not for a new quantitative rarefied-cavity benchmark.

Knudsen number changes the ratio of molecular mean free path to cavity size. Changing `Kn` can change both the physical solution and the time required to reach a statistically stationary sampling window. A fixed number of steps is therefore not automatically a matched convergence standard across Kn.

**Budget audit:** record grid, particles per cell, time step, total steps, discarded transient, sample stride, wall model, and collision model for every case.
''',
        "before:## 2. Assemble": r'''
## Noisy labels and nondimensional targets

The model inputs use `log10(Kn)` because rarefaction spans orders of magnitude. Velocity is normalized by wall speed and temperature by wall temperature. Fit all statistical standardizers using development cases only.

Seed averaging reduces random variance approximately with the number of sufficiently independent realizations, but it does not remove systematic bias from cell size, time step, collision selection, wall model, or incomplete transient removal.
''',
        "before:## 3A.": r'''
## Variant 5A: define the independent comparison

The single-seed and seed-averaged models must be evaluated against a seed that contributed to neither training target. Compare model error with the seed-to-seed spread of the simulation labels. If model differences are smaller than label variability, the conclusion should be correspondingly cautious.
''',
        "before:## 3B.": r'''
## Variant 5B: Knudsen generalization is regime generalization

Hold out complete Kn values, not random cells. An unseen higher Kn case can combine parameter extrapolation, altered non-equilibrium structure, and different label noise. Report whether the test is within the same qualitative regime or crosses from slip toward transition behavior.
''',
        "before:## 4. Evaluate": r'''
## Physical checks for particle-derived fields

At minimum examine mass/normalization behavior, velocity boundary response, temperature positivity, centerlines, recirculation topology, and seed variability. Higher-order quantities normally converge more slowly than density and mean velocity. A smooth neural field can suppress visible noise while introducing bias; smoothness is not validation.
''',
        "end": r'''
## Concept check and further reading

1. Why is a new seed a different test from a new Knudsen number?
2. Which errors are reduced by averaging independent seeds?
3. Why can the same sampling window be less adequate at a different Kn?
4. Why should heat flux or stress be expected to converge more slowly than density?
5. What additional studies are required before making a production DSMC claim?

Read Bird (1994), Cercignani (1988), and the rarefied-flow sources in `references/README.md`.
''',
    },
    "P6_FP_Cavity_Closure.ipynb": {
        "intro": r'''
## Prerequisite gate and scope

Proceed only if you can already explain case-wise splitting, standardized multi-output loss, GPU/CPU timing synchronization, and the difference between offline and a-posteriori validation. Track 6 supplies the kinetic solver and wrapper scripts; the scientific work is the controlled closure experiment, not rewriting the research solver.

The reduced fast mode verifies software plumbing. It cannot support production speedup, convergence, or general rarefaction claims.
''',
        "before:## 1. Freeze": r'''
## Four nested levels of validation

1. **Data integrity:** finite inputs/targets, named columns, complete-condition provenance.
2. **Offline coefficient accuracy:** errors in the six stress-related `C` and three heat-flux-related `Gamma` outputs.
3. **Closed-loop macroscopic fidelity:** density, velocity, temperature, pressure, and centerlines after recursive deployment.
4. **High-order/stability fidelity:** stress, heat flux, positivity, particle health, and time histories.

Passing one level does not imply passing the next. The blind experimental design must therefore name evidence at all four levels.
''',
        "before:## 3. Audit": r'''
## Feature observability and loss weighting

The 16 inputs are low-order local features; the nine targets are closure coefficients computed by the exact local system. A coefficient can be difficult to learn because the available inputs do not uniquely observe the required high-order state. Increasing network capacity cannot repair missing information.

Q-weighting changes which standardized output errors dominate optimization. It does not add information. Inspect feature ranges by physical condition and check whether the blind condition lies outside training support.
''',
        "before:## 7. Freeze": r'''
## Why the selection rule has two clauses

Improving only the `Gamma` block can be purchased by unacceptable degradation of the `C` block. The validation rule therefore requires a targeted improvement and a bounded collateral cost. Record the rule before generating blind data. Do not select a q-weight because its blind field plot looks better.
''',
        "before:## 9. Closed": r'''
## A fair online timing experiment

Use identical physical configuration, seed policy, grid, particles, transient, sampling window, and output frequency. Synchronize GPU work before reading the clock. Report warm-up separately when relevant. Distinguish closure-kernel speedup from end-to-end solver speedup and include exact data-generation/training cost when discussing break-even.

The supplied reference path has two known savings: replacing the local exact solve and avoiding exact-path higher-moment work that is not needed by the learned path. State this explicitly when interpreting speedup.
''',
        "before:## Required interpretation": r'''
## Reading a closed-loop success or failure

Macroscopic agreement can coexist with early degradation of stress or heat flux. A stable simulation is not necessarily an accurate one. Conversely, modest local coefficient errors may be dynamically benign if the solver is insensitive in the tested regime.

Rank conclusions by strength: one reduced-budget blind case; repeated seeds at the same condition; multiple held-out speeds; unseen rarefaction; then grid/particle/time convergence. Do not generalize beyond the level actually tested.
''',
        "end": r'''
## Concept check and further reading

1. Why can low offline coefficient error fail in closed loop?
2. What does q-weighting change, and what does it not change?
3. Why are macroscopic and high-order metrics reported separately?
4. What must be synchronized for valid GPU timing?
5. What costs belong in an offline break-even analysis?
6. Which result would suggest feature non-observability rather than insufficient capacity?

Read Roohi (2026), *GPU-native neural surrogate for Fokker–Planck closure*, and the verification-oriented rarefied-ML references in `references/README.md`.
''',
    },
}


def insert_before_heading(cells: list[dict], heading: str, new_cell: dict) -> None:
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and heading in "".join(cell.get("source", [])):
            cells.insert(i, new_cell)
            return
    raise ValueError(f"heading not found: {heading}")


def enrich(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    all_text = "\n".join("".join(c.get("source", [])) for c in notebook.get("cells", []))
    if MARKER in all_text:
        print("already enriched:", path.name)
        return

    config = TRACK_CONTENT[path.name]
    cells = notebook["cells"]
    cells.insert(1, md(UNIVERSAL_START))
    cells.insert(2, md(config["intro"]))
    cells.insert(3, code(TRACK6_BOOTSTRAP if path.name.startswith("P6_") else REPO_BOOTSTRAP))

    for key, text in config.items():
        if key.startswith("before:"):
            insert_before_heading(cells, key.split(":", 1)[1], md(text))

    cells.append(md(config["end"]))
    cells.append(md(UNIVERSAL_END))

    # Make the repository layout work while keeping the flat-upload Colab path.
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace('w5_common.require_week4_files("cavity_data.npz")',
                                'w5_common.require_week4_files(str(DATASET_PATH))')
        source = source.replace("w5_common.require_week4_files()",
                                "w5_common.require_week4_files(str(DATASET_PATH))")
        cell["source"] = lines(source.rstrip())
        cell["execution_count"] = None
        cell["outputs"] = []

    notebook.setdefault("metadata", {})["mie690a_release"] = {
        "edition": "expanded-learner-v2",
        "course": "MIE 690A AI in Fluid Mechanics",
        "date": "2026-08-23",
    }
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = max(5, int(notebook.get("nbformat_minor", 5)))
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("enriched:", path.name, "cells=", len(cells))


def main() -> None:
    for name in TRACK_CONTENT:
        enrich(NOTEBOOK_DIR / name)


if __name__ == "__main__":
    main()
