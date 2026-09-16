from pathlib import Path

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


def test_week04_companion_is_eight_pages_and_not_pinn_material():
    path = ROOT / "lectures/week04_2_stokes_to_navier_stokes.pdf"
    reader = PdfReader(path)
    assert len(reader.pages) == 8
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Stokes-to-Navier-Stokes correction" in text
    assert "constant and diverse lids" in text
    assert "PINN" not in text


def test_retained_evidence_and_build_sources_exist():
    required = [
        "qa/build_week04_2_stokes_correction.py",
        "flowmllab/cavity_diversity.py",
        "flowmllab/stokes_correction.py",
        "flowmllab/stokes_refined.py",
        "results/stokes_refined/comparison.csv",
        "results/stokes_refined/expanded/status.json",
        "results/stokes_refined/expanded/diverse_seed7.pt",
        "results/stokes_refined/expanded/diverse_seed17.pt",
        "results/stokes_refined/expanded/diverse_seed27.pt",
    ]
    assert all((ROOT / name).is_file() for name in required)
