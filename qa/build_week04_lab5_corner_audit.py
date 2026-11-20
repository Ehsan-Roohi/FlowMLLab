"""Add the physical corner audit to Lab 5 without removing its original cells."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "qa")]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/week04_review/mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nb
from PIL import Image
from flowmllab import cavity_corner_audit as audit

TAG = "week04_corner_audit_20261120"
OUTPUT = ROOT / "results/week04_corner_audit"
NOTEBOOK = ROOT / "notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb"


def cells():
    return [nb.v4.new_markdown_cell(r"""### 4. Does the field model resolve the upper-left vortex?

**Aim.** Separate a weak flow structure, a plotting omission and a genuine
prediction error. The profile networks in Table B predict only the two
centreline curves: they do **not** define a two-dimensional velocity field and
cannot predict a vortex location. This section therefore opens the **separate
frozen full-field POD-DeepONet from Lab 8**, at its whole-Re test cases 3200 and
7500. No model is retrained or selected here.

We examine the same native $513\times513$ fields in
$0\leq x/L\leq0.25$, $0.65\leq y/L\leq1$. Under
$u=\partial\psi/\partial y$, $v=-\partial\psi/\partial x$, a positive interior
streamfunction island counter-rotates relative to the main negative island.
Its native-node maximum gives a discrete centre and its maximum value gives
a circulation-strength proxy. This is not an independently validated vortex
position; a one-cell displacement is $L/512$. A small whole-field relative
error can conceal a large local error, so Table D reports both norms, the
signed peak-strength discrepancy and the centre displacement.

The first panel fills the CFD streamfunction and overlays all three methods at
**identical physical contour levels**. A symmetric-log scale exposes weak
positive islands while retaining all CFD extrema; it is linear near zero.
The two vector-error maps share a **linear** scale from zero to the largest
error across both methods in this window. Neither is clipped or independently
rescaled to look better. The fixed off-centre cuts at $x/L=0.06$ and $y/L=0.90$
show velocities and signed discrepancies; they are not Ghia centreline samples.

This is a **post-hoc physical diagnostic of already frozen tests**, not a new
blind test or a basis for further tuning on these targets. The retained CFD
mesh sensitivity and steady-branch limitations in Lab 7 remain in force.
""", metadata={TAG: True}),
        nb.v4.new_code_cell("""from flowmllab import cavity_corner_audit as corner
corner_cases, corner_provenance = corner.load_frozen_cases(ROOT)
corner_scores = pd.DataFrame(corner.diagnostic_rows(corner_cases))
print('Table D: whole-field and upper-left-window errors, fractions rather than percent')
display(corner_scores[['Re','method','field_relative_l2','corner_relative_l2',
    'corner_vector_rms_U','corner_vector_max_U','resolved_positive_island',
    'x_peak_L','y_peak_L','psi_peak_UL','peak_shift_L',
    'peak_strength_signed_relative_error']])
for re, methods in corner_cases.items():
    print(f'Re={re}: native 513 x 513 fields; frozen held-out prediction; same log-Re interpolation labels')
    corner.plot_corner(methods, re)
    plt.show()
    corner.plot_corner_profiles(methods, re)
    plt.show()
""", metadata={TAG: True}),
        nb.v4.new_markdown_cell(r"""#### Read the corner comparison and design the next experiment

1. For each test Reynolds number, compare the neural model with interpolation
   using both the whole-field and corner norms in Table D. Determine which
   method improves or deteriorates on the local velocity cuts.
2. Locate each positive streamfunction maximum, express its displacement in
   units of the native grid spacing, and report the signed relative change in
   peak strength. If no resolved positive island exists, report that rather
   than selecting a wall point and calling it a vortex centre.
3. Explain why a centreline-only MLP cannot answer this question and why a
   default streamline seeding pattern can miss a small island. Change contour
   levels or zoom limits to inspect the same unchanged field; do not train on
   test errors or change the stored predictions.
4. A proposed network improvement needs an independent development target:
   choose sampling, capacity and loss weights using training/validation cases,
   freeze the procedure, then evaluate on a new reserved test. Retain the
   current frozen result as the comparison. Stronger corner weights can improve
   a local region while worsening global errors or other corners.
5. Read the Lab 7 mesh-change results before claiming physical accuracy. These
   comparisons measure fidelity to a particular numerical steady solution,
   not three-dimensional stability or a mesh-independent exact solution.

[Full-field model and frozen protocol](../../results/cavity_field_deeponet/README.md)
 · [Machine-readable corner results](../../results/week04_corner_audit/README.md)
""", metadata={TAG: True})]


def add_cells(doc):
    doc.cells = [c for c in doc.cells if not c.metadata.get(TAG)]
    insertion = next((i for i, cell in enumerate(doc.cells)
                      if cell.cell_type == "markdown" and "### Specific student tasks" in cell.source), len(doc.cells))
    doc.cells[insertion:insertion] = cells()
    if "**Aim of Lab 5.**" not in doc.cells[0].source:
        doc.cells[0].source = doc.cells[0].source.replace("This standalone CPU lab", "**Aim of Lab 5.** Audit our CFD against the published benchmark, distinguish\nreference coverage from numerical qualification, and determine what each neural\nmodel actually predicts. Tables A-C cover CFD, centreline regression and evidence\navailability; Table D links these checks to the full-field corner diagnostic.\n\nThis standalone CPU lab", 1)
    return doc


def execute_kernel(doc):
    """Run actual Jupyter cells; keep runtime/config files on the project drive."""
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    temporary = ROOT / "tmp/week04_review"
    for key, name in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter"),
                      ("JUPYTER_CONFIG_DIR", "config")):
        folder = temporary / name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share/jupyter")
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(doc, timeout=600, km=kernel,
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    client.execute()
    doc.metadata["execution_audit"] = {"method": "nbclient with a real Jupyter kernel",
                                         "jupyter_kernel_executed": True,
                                         "code_cells": sum(c.cell_type == "code" for c in doc.cells),
                                         "hosted_colab_execution_claimed": False}
    return doc


def build():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases, provenance = audit.load_frozen_cases(ROOT)
    rows = audit.diagnostic_rows(cases)
    with (OUTPUT / "corner_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    paths = []
    for re, methods in cases.items():
        for name, plotter in (("corner", audit.plot_corner), ("profiles", audit.plot_corner_profiles)):
            fig = plotter(methods, re)
            for ext in ("png", "pdf"):
                path = OUTPUT / f"re{re}_{name}.{ext}"
                fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
                if ext == "png":
                    with Image.open(path) as original:
                        rgb = original.convert("RGB")
                    rgb.save(path)
                paths.append(path)
            plt.close(fig)
    doc = nb.read(NOTEBOOK, as_version=4)
    old_sources = [c.source for c in doc.cells if not c.metadata.get(TAG)]
    add_cells(doc)
    execute_kernel(doc)
    with NOTEBOOK.open("w", encoding="utf-8", newline="\n") as f:
        nb.write(doc, f)
    provenance.update(elapsed_seconds=time.perf_counter()-started,
                      metrics=rows, original_cells_preserved=len(old_sources),
                      notebook_sha256=audit.sha256(NOTEBOOK),
                      figure_sha256={p.relative_to(ROOT).as_posix(): audit.sha256(p) for p in paths},
                      execution=doc.metadata.get("execution_audit"))
    (OUTPUT / "verification.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": provenance["elapsed_seconds"], "metrics": rows,
                      "code_cells": sum(c.cell_type == "code" for c in doc.cells),
                      "plots": sum("image/png" in o.get("data", {}) for c in doc.cells for o in c.get("outputs", []))}, indent=2))


if __name__ == "__main__":
    build()
