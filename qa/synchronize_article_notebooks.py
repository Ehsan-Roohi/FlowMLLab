#!/usr/bin/env python3
"""Synchronize notebook entry points with the manuscript's retained evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKER = "MIE690A article-aligned validation v3"


def cell_id(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]


def markdown(label: str, source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id(label), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(label: str, source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id(label), "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


BOOTSTRAP = """from pathlib import Path
import sys

def locate_course_root(start=Path.cwd()):
    for origin in (Path(start).resolve(), Path.cwd().resolve()):
        for candidate in (origin, *origin.parents):
            if (candidate / "common" / "article_validation.py").exists():
                return candidate
    raise FileNotFoundError(
        "Run this notebook inside the complete MIE690A release; common/article_validation.py is required."
    )

COURSE_ROOT = locate_course_root()
sys.path.insert(0, str(COURSE_ROOT / "common"))
from article_validation import (
    build_dsmc_wall_pressure_validation,
    build_ghia_velocity_validation,
    build_pressure_validation,
)
print("Course root:", COURSE_ROOT)
"""


def ghia_cells() -> list[dict]:
    return [
        markdown(
            "ghia-v3-intro",
            rf"""## 0. Article-aligned validation: velocity **and** pressure

<!-- {MARKER} -->

Before deriving the teaching solver, reproduce the two continuum validation figures used in the manuscript. This is a deliberate scientific order: define the evidence first, then study how the algorithm produces it.

1. **Velocity:** the retained $Re=400$ streamfunction--vorticity solution is compared directly with the Ghia *et al.* vertical-$u$ and horizontal-$v$ centerline markers.
2. **Pressure:** pressure is not a primary variable of this formulation. We reconstruct it from the steady nondimensional momentum gradients,

$$p_x=-(u u_x+v u_y)+Re^{{-1}}\nabla^2u,\qquad
p_y=-(u v_x+v v_y)+Re^{{-1}}\nabla^2v,$$

then solve one sparse least-squares edge-gradient system with a gauge constraint. The $Re=1000$ pressure centerlines are checked against Botella--Peyret, with both $65^2$ and $129^2$ executed grids shown. The simple solver is retained because every numerical step is inspectable by new students; the grid-refinement and literature errors state its limits rather than hiding them.
""",
        ),
        code("ghia-v3-bootstrap", BOOTSTRAP),
        code(
            "ghia-v3-build",
            """from IPython.display import Image, display
import pandas as pd

velocity_validation = build_ghia_velocity_validation(COURSE_ROOT)
pressure_validation = build_pressure_validation(COURSE_ROOT)

display(Image(filename=str(velocity_validation["figure"]["png"]), width=1050))
display(pd.DataFrame([velocity_validation["metrics"]]))
display(Image(filename=str(pressure_validation["figure"]["png"]), width=1150))
display(pd.DataFrame([pressure_validation["metrics"]]))

print("Paper-ready files:")
print(" -", velocity_validation["figure"]["pdf"])
print(" -", pressure_validation["figure"]["pdf"])
""",
        ),
        code(
            "ghia-v3-rerun",
            """# Optional production recomputation. The default uses the retained, executed runs so
# the notebook opens quickly and deterministically. Set True only when you intend to
# spend several minutes regenerating the two Re=1000 grids.
RECOMPUTE_PRESSURE_RUNS = False

if RECOMPUTE_PRESSURE_RUNS:
    import subprocess
    subprocess.run(
        [sys.executable, str(COURSE_ROOT / "common" / "run_cavity_pressure_validation.py")],
        cwd=COURSE_ROOT,
        check=True,
    )
    pressure_validation = build_pressure_validation(COURSE_ROOT)
    display(Image(filename=str(pressure_validation["figure"]["png"]), width=1150))
else:
    print("Using retained Re=1000 N=65 and N=129 solver outputs; set the switch to rerun them.")
""",
        ),
    ]


def dsmc_cells() -> list[dict]:
    return [
        markdown(
            "dsmc-v3-intro",
            rf"""## 0. Article-aligned solver validation comes first

<!-- {MARKER} -->

This is a validation of **our executed HS--NTC DSMC solver**, not a pasted literature panel. The target is the hard-sphere argon cavity reported by Mohammadzadeh *et al.* at $Re=1.5$, $Kn=0.1$, and $Ma=0.09$. The observable is $p/p_0$ along A(bottom-left)--B(top-left)--C(top-right)--D(bottom-right)--A.

The black markers below were extracted from the vector paths of the published figure. The blue curve is our retained $60^2$, 50-particles-per-cell run with 6000 field samples. The ochre curve and band are the mean and run-to-run $\pm1\sigma$ spread of three independent $40^2$ runs. The second panel reports pointwise discrepancies so that visually overlapping curves are not mistaken for identical results.
""",
        ),
        code("dsmc-v3-bootstrap", BOOTSTRAP),
        code(
            "dsmc-v3-build",
            """from IPython.display import Image, display
import pandas as pd

dsmc_validation = build_dsmc_wall_pressure_validation(COURSE_ROOT)
display(Image(filename=str(dsmc_validation["figure"]["png"]), width=1150))
display(pd.DataFrame([dsmc_validation["metrics"]]))
print("Paper-ready file:", dsmc_validation["figure"]["pdf"])
""",
        ),
        code(
            "dsmc-v3-rerun",
            """# Optional full recomputation with the same solver kernel and declared budgets.
# The retained outputs above are genuine completed runs. Turn this on only when you
# want to regenerate all four realizations (CPU time depends strongly on hardware).
RUN_FULL_DSMC_VALIDATION = False

if RUN_FULL_DSMC_VALIDATION:
    import subprocess
    runner = COURSE_ROOT / "notebooks" / "week03" / "run_wall_pressure_case.py"
    output = COURSE_ROOT / "results" / "dsmc_validation" / "runs"
    specifications = [
        (40, 7, 12000, 4000, 0.05, "ref_n40_seed07"),
        (40, 19, 12000, 4000, 0.05, "ref_n40_seed19"),
        (40, 31, 12000, 4000, 0.05, "ref_n40_seed31"),
        (60, 7, 18000, 6000, 0.04, "ref_n60_seed07"),
    ]
    for grid, seed, steps, sample_start, dt_factor, case_name in specifications:
        command = [
            sys.executable, str(runner), "--grid", str(grid), "--ppc", "50",
            "--steps", str(steps), "--sample-start", str(sample_start),
            "--sample-stride", "2", "--dt-factor", str(dt_factor),
            "--seed", str(seed), "--output-dir", str(output), "--case-name", case_name,
        ]
        print("$", " ".join(command))
        subprocess.run(command, cwd=runner.parent, check=True)
    dsmc_validation = build_dsmc_wall_pressure_validation(COURSE_ROOT)
    display(Image(filename=str(dsmc_validation["figure"]["png"]), width=1150))
else:
    print("Using four retained completed runs; set the switch to rerun the production validation.")
""",
        ),
    ]


FIGURE_OWNER = {
    "week01/03_cavity_ghia.ipynb": "Figures 2 and 8: Ghia velocity centerlines and Botella--Peyret pressure validation.",
    "week03/AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb": "Figure 10a: direct wall-pressure validation of our DSMC solver.",
    "week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb": "Figure 9: CFD/MLP blind-case fields and centerlines.",
    "week04/W4_Lab3_DeepONet_Cavity_Student.ipynb": "Figure 13: POD-DeepONet blind and Ghia validation outputs.",
    "P1_Re_Generalization.ipynb": "Figure 9 and neural summary: controlled Reynolds-number generalization.",
    "P3_POD_Study.ipynb": "POD/DeepONet evidence and physical centerline validation.",
    "P5_Rarefied_Cavity.ipynb": "Figures 10--12: standard DSMC versus hybrid neural--DSMC evidence.",
    "P6_FP_Cavity_Closure.ipynb": "Fokker--Planck closure fields and profile comparison.",
}


def sync_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    managed_ids = {
        cell_id(label)
        for label in (
            "ghia-v3-intro", "ghia-v3-bootstrap", "ghia-v3-build", "ghia-v3-rerun",
            "dsmc-v3-intro", "dsmc-v3-bootstrap", "dsmc-v3-build", "dsmc-v3-rerun",
        )
    }
    cells = [
        cell for cell in cells
        if MARKER not in "".join(cell.get("source", [])) and cell.get("id") not in managed_ids
    ]

    relative = path.relative_to(ROOT / "notebooks").as_posix()
    if relative.endswith("03_cavity_ghia.ipynb"):
        cells[1:1] = ghia_cells()
    elif relative.endswith("AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb"):
        # Remove the obsolete empty Kn=0.05 digitization exercise; the direct,
        # executed Re=1.5/Kn=0.1 validation above replaces it.
        cleaned = []
        skip_next = False
        for cell in cells:
            source = "".join(cell.get("source", []))
            if "### 11.5 Validate against Mohammadzadeh" in source:
                skip_next = True
                continue
            if skip_next and cell.get("cell_type") == "code" and "FIG4_CSV" in source:
                skip_next = False
                continue
            cleaned.append(cell)
        cells = cleaned
        cells[1:1] = dsmc_cells()

    owner = FIGURE_OWNER.get(relative, "Foundational or supporting notebook; see ARTICLE_FIGURE_MAP.md for its evidence dependency.")
    map_link = "../ARTICLE_FIGURE_MAP.md" if "/" not in relative else "../../ARTICLE_FIGURE_MAP.md"
    cells.append(
        markdown(
            f"contract-{relative}",
            f"""## Article-output contract

<!-- {MARKER} -->

**Role:** {owner}

All manuscript-facing figures must be generated from retained numerical/model outputs through the documented notebook or shared helper, saved under `results/`, and accompanied by machine-readable metrics. Do not redraw curves by eye or substitute a screenshot for a solver-to-reference comparison. The complete ownership table and exact output filenames are in [`ARTICLE_FIGURE_MAP.md`]({map_link}).
""",
        )
    )
    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    for path in notebooks:
        sync_notebook(path)
    print(f"Synchronized {len(notebooks)} notebooks with {MARKER}.")


if __name__ == "__main__":
    main()
