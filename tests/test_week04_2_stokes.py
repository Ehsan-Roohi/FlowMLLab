from pathlib import Path
import csv

import nbformat
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def test_week04_lab4_is_executed_and_colab_link_is_current():
    path = ROOT / "notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb"
    notebook = nbformat.read(path, 4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(notebook.cells) >= 22
    assert all(cell.execution_count is not None for cell in code)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "W4_Lab4_Stokes_to_Navier_Stokes.ipynb" in source
    assert "Stokes-to-Navier-Stokes" in source
    assert "PINN" not in source


def test_week04_companion_is_ten_pages_and_not_pinn_material():
    path = ROOT / "lectures/week04_2_stokes_to_navier_stokes.pdf"
    reader = PdfReader(path)
    assert len(reader.pages) == 10
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Stokes-to-Navier-Stokes correction" in text
    assert "constant and diverse lids" in text
    assert "vortices and streamlines" in text
    assert "PINN" not in text


def test_retained_evidence_and_build_sources_exist():
    required = [
        "qa/build_week04_2_stokes_correction.py",
        "flowmllab/cavity_diversity.py",
        "flowmllab/stokes_correction.py",
        "flowmllab/stokes_refined.py",
        "results/stokes_refined/comparison.csv",
        "results/stokes_refined/vortex_comparison.csv",
        "results/stokes_refined/expanded/status.json",
        "results/stokes_refined/expanded/diverse_seed7.pt",
        "results/stokes_refined/expanded/diverse_seed17.pt",
        "results/stokes_refined/expanded/diverse_seed27.pt",
    ]
    assert all((ROOT / name).is_file() for name in required)


def test_vortex_comparison_agrees_with_retained_metrics():
    with (ROOT / "results/stokes_refined/vortex_comparison.csv").open(newline="") as stream:
        vortices = list(csv.DictReader(stream))
    with (ROOT / "results/stokes_refined/expanded/metrics.csv").open(newline="") as stream:
        metrics = list(csv.DictReader(stream))
    assert len(vortices) == 12
    retained = {(row["train_family"], row["case"]): row for row in metrics
                if row["train_family"] == row["test_family"]
                and row["seed"] == "ensemble"}
    for row in vortices:
        family = row["family"]
        assert (family, row["case"]) in retained
        assert abs(float(row["interior_vorticity_rel_l2"]) -
                   float(retained[family, row["case"]]["interior_vorticity_rel_l2"])) < 1e-10
        assert (row["primary_ref_x"], row["primary_ref_y"]) == (
            row["primary_pred_x"], row["primary_pred_y"])
        assert (row["lower_right_ref_x"], row["lower_right_ref_y"]) == (
            row["lower_right_pred_x"], row["lower_right_pred_y"])
        assert float(row["lower_right_ref_psi"]) > 0
        assert float(row["lower_right_pred_psi"]) > 0
    assert all((ROOT / f"figures/Cavity_{family}_streamlines_vorticity.png").is_file()
               for family in ("constant", "diverse"))
