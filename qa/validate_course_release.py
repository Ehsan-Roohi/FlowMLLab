#!/usr/bin/env python3
"""Release gate for the public FlowMLLab software and tutorial tree."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATA_SHA256 = "09b96b744ee4d18126d8dcc92feb60e128774a1b4d41bb3d8c90a63ccfbabc36"


REQUIRED = [
    "README.md",
    "START_HERE.md",
    "COURSE_MAP.md",
    "requirements.txt",
    "CITATION.cff",
    "LICENSE",
    "pyproject.toml",
    "flowmllab/__init__.py",
    "flowmllab/cli.py",
    "flowmllab/core.py",
    "flowmllab/cavity_rom.py",
    "tests/test_core.py",
    "tests/test_cavity_rom.py",
    "qa/build_week04_1_rom_notebook.py",
    "qa/run_cavity_rom_validation.py",
    "data/cavity_data.npz",
    "data/case_quality.csv",
    "common/w4utils.py",
    "common/w5_common.py",
    "common/mini_dsmc.py",
    "common/run_pod_deeponet_validation.py",
    "common/article_validation.py",
    "common/run_cavity_pressure_validation.py",
    "ARTICLE_FIGURE_MAP.md",
    "notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb",
    "notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb",
    "notebooks/P0_Project_Setup.ipynb",
    "notebooks/P1_Re_Generalization.ipynb",
    "notebooks/P2_Physics_Guided_DNN.ipynb",
    "notebooks/P3_POD_Study.ipynb",
    "notebooks/P4_Uncertainty_Study.ipynb",
    "notebooks/P5_Rarefied_Cavity.ipynb",
    "notebooks/P6_FP_Cavity_Closure.ipynb",
    "lectures/week01_numerical_foundations.pdf",
    "lectures/week02_supervised_learning_rarefaction.pdf",
    "lectures/week03_kinetic_dsmc.pdf",
    "lectures/week04_cavity_surrogates_deeponet.pdf",
    "lectures/weeks05_06_project_guide.pdf",
    "references/README.md",
    "references/course_references.bib",
    "results/pod_deeponet/deeponet_selection.csv",
    "results/pod_deeponet/README.md",
    "results/pod_deeponet/deeponet_metrics.csv",
    "results/pod_deeponet/deeponet_ghia_metrics.csv",
    "results/pod_deeponet/deeponet_protocol_and_timing.json",
    "results/pod_deeponet/deeponet_predictions.csv",
    "results/pod_deeponet/pod_deeponet_ghia_validation.svg",
    "results/cavity_rom/fom_validation.csv",
    "results/cavity_rom/convergence.csv",
    "results/cavity_rom/selection.csv",
    "results/cavity_rom/blind_metrics.csv",
    "results/cavity_rom/timing.json",
    "results/cavity_rom/validation_protocol.json",
    "results/cavity_rom/validation_summary.json",
    "results/cavity_rom/cavity_rom_model.npz",
    "results/cavity_rom/cavity_rom_validation.png",
    "results/article_validation/re1000_n65.npz",
    "results/article_validation/re1000_n129.npz",
    "results/article_validation/botella_pressure_reference.csv",
    "results/article_validation/pressure_validation_protocol.json",
    "results/dsmc_validation/mohammadzadeh_fig3_dsmc_points.csv",
    "results/dsmc_validation/runs/ref_n40_seed07/wall_pressure.csv",
    "results/dsmc_validation/runs/ref_n40_seed19/wall_pressure.csv",
    "results/dsmc_validation/runs/ref_n40_seed31/wall_pressure.csv",
    "results/dsmc_validation/runs/ref_n60_seed07/wall_pressure.csv",
    "results/article_figures/fig02_cavity_benchmark.png",
    "results/article_figures/fig08_pressure_recovery.png",
    "results/article_figures/fig10a_mohammadzadeh_validation.png",
    "results/article_figures/fig02_cavity_benchmark_metrics.json",
    "results/article_figures/fig08_pressure_recovery_metrics.json",
    "results/article_figures/fig10a_mohammadzadeh_validation_metrics.json",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_notebook_code(source: str, label: str) -> None:
    kept = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    if cleaned.strip():
        ast.parse(cleaned, filename=label)


def validate_notebooks() -> tuple[int, int]:
    count = 0
    code_cells = 0
    for path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook.get("nbformat") == 4, f"unexpected nbformat: {path}"
        cells = notebook.get("cells", [])
        assert cells, f"empty notebook: {path}"
        full_source = "\n".join("".join(cell.get("source", [])) for cell in cells)
        assert "MIE690A article-aligned validation v3" in full_source, (
            f"missing article-alignment marker: {path}"
        )
        ids = [cell.get("id") for cell in cells if cell.get("id")]
        assert len(ids) == len(set(ids)), f"duplicate cell id: {path}"
        for index, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                code_cells += 1
                parse_notebook_code("".join(cell.get("source", [])), f"{path}:{index}")
        if path.parent == ROOT / "notebooks" and path.name.startswith("P"):
            words = sum(
                len("".join(cell.get("source", [])).split())
                for cell in cells
                if cell.get("cell_type") == "markdown"
            )
            minimum = 2000 if path.name.startswith("P6_") else 1000
            assert words >= minimum, f"project notebook lacks learner guidance ({words} words): {path}"
            assert any(
                "MIE690A enriched learner edition v2" in "".join(cell.get("source", []))
                for cell in cells
            ), f"missing learner-edition marker: {path}"
        count += 1
    assert count == 17, f"expected 17 notebooks, found {count}"
    return count, code_cells


def validate_article_alignment() -> dict[str, float]:
    result_dir = ROOT / "results" / "article_figures"
    ghia = json.loads((result_dir / "fig02_cavity_benchmark_metrics.json").read_text())
    pressure = json.loads((result_dir / "fig08_pressure_recovery_metrics.json").read_text())
    dsmc = json.loads((result_dir / "fig10a_mohammadzadeh_validation_metrics.json").read_text())
    assert float(ghia["u_centerline_relative_l2"]) < 0.20
    assert float(ghia["v_centerline_relative_l2"]) < 0.20
    assert float(pressure["vertical_relative_l2_n129"]) < float(pressure["vertical_relative_l2_n65"])
    assert float(pressure["horizontal_relative_l2_n129"]) < float(pressure["horizontal_relative_l2_n65"])
    assert float(pressure["vertical_relative_l2_n129"]) < 0.07
    assert float(pressure["horizontal_relative_l2_n129"]) < 0.07
    assert float(dsmc["relative_l2"]) < 0.01
    assert float(dsmc["grid_change_40_to_60"]) < 0.015

    ghia_notebook = json.loads(
        (ROOT / "notebooks" / "week01" / "03_cavity_ghia.ipynb").read_text()
    )
    dsmc_notebook = json.loads(
        (ROOT / "notebooks" / "week03" / "AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb").read_text()
    )
    ghia_opening = "\n".join("".join(cell.get("source", [])) for cell in ghia_notebook["cells"][:6])
    dsmc_opening = "\n".join("".join(cell.get("source", [])) for cell in dsmc_notebook["cells"][:6])
    assert "build_ghia_velocity_validation" in ghia_opening
    assert "build_pressure_validation" in ghia_opening
    assert "build_dsmc_wall_pressure_validation" in dsmc_opening
    assert "FIG4_CSV" not in "\n".join("".join(cell.get("source", [])) for cell in dsmc_notebook["cells"])
    return {
        "Ghia_Eu": float(ghia["u_centerline_relative_l2"]),
        "Ghia_Ev": float(ghia["v_centerline_relative_l2"]),
        "pressure_vertical_E": float(pressure["vertical_relative_l2_n129"]),
        "pressure_horizontal_E": float(pressure["horizontal_relative_l2_n129"]),
        "DSMC_wall_pressure_E": float(dsmc["relative_l2"]),
    }


def smoke_common_baseline() -> dict[str, float]:
    sys.path.insert(0, str(ROOT / "common"))
    import w5_common  # noqa: PLC0415

    data = w5_common.require_week4_files(str(ROOT / "data" / "cavity_data.npz"))
    train_re = data["Re"][data["split"].astype(str) == "train"]
    pred = w5_common.interpolate_case(data, 275, train_re)
    report = w5_common.evaluate_prediction(data, 275, pred)
    assert report
    assert all(np.isfinite(value) for value in report.values())
    assert report["relative_L2_uv"] < 0.10, report
    return {key: float(value) for key, value in report.items()}


def validate_pod_deeponet_results() -> dict[str, float]:
    result_dir = ROOT / "results" / "pod_deeponet"
    metrics = pd.read_csv(result_dir / "deeponet_metrics.csv")
    ensemble = metrics[metrics["method"] == "three-seed POD-DeepONet ensemble"].copy()
    assert sorted(ensemble["Re"].astype(int).tolist()) == [175, 275, 375]
    assert float(ensemble["relative_L2_uv"].max()) < 0.005
    assert float(ensemble["div_l2_pred"].max()) < 1.0e-12
    assert float(ensemble["wall_rms_error"].max()) == 0.0

    ghia = pd.read_csv(result_dir / "deeponet_ghia_metrics.csv")
    assert sorted(ghia["Re"].astype(int).tolist()) == [100, 400]
    ghia_delta = np.max(
        np.abs(
            ghia[["POD_DeepONet_Ghia_Eu", "POD_DeepONet_Ghia_Ev"]].to_numpy()
            - ghia[["CFD_Ghia_Eu", "CFD_Ghia_Ev"]].to_numpy()
        )
    )
    assert float(ghia_delta) < 5.0e-4

    timing = json.loads(
        (result_dir / "deeponet_protocol_and_timing.json").read_text(encoding="utf-8")
    )
    assert timing["selected_rank"] == 3
    assert timing["selected_hidden"] == [32, 32]
    assert float(timing["speedup"]) > 100.0
    assert float(timing["CFD_final_residual"]) < 1.0e-6

    predictions = pd.read_csv(result_dir / "deeponet_predictions.csv")
    field_columns = [
        "u_Re175", "v_Re175", "u_Re275", "v_Re275", "u_Re375", "v_Re375"
    ]
    assert len(predictions) == 65 * 65
    assert predictions[["iy", "ix"]].drop_duplicates().shape[0] == 65 * 65
    assert predictions[["x", "y", *field_columns]].notna().all().all()

    binary_predictions = result_dir / "deeponet_predictions.npz"
    if binary_predictions.is_file():
        with np.load(binary_predictions, allow_pickle=False) as archive:
            assert archive["u"].shape == (3, 65, 65)
            assert archive["v"].shape == (3, 65, 65)
            assert archive["seeds"].tolist() == [690, 691, 692]

    return {
        "max_blind_relative_L2_uv": float(ensemble["relative_L2_uv"].max()),
        "max_divergence_L2": float(ensemble["div_l2_pred"].max()),
        "max_Ghia_error_change": float(ghia_delta),
        "measured_speedup": float(timing["speedup"]),
    }


def validate_cavity_rom_results() -> dict[str, float]:
    result_dir = ROOT / "results" / "cavity_rom"
    summary = json.loads((result_dir / "validation_summary.json").read_text())
    assert summary["status"] == "pass"
    assert summary["selected_rank"] == 16
    assert summary["selected_deim_dimension"] == 16
    assert summary["blind_Re"] == [175, 275, 375]
    assert float(summary["maximum_blind_velocity_error"]) < 0.01
    assert float(summary["maximum_blind_final_vorticity_error"]) < 0.01
    assert float(summary["maximum_blind_divergence_l2"]) < 1.0e-12
    assert float(summary["maximum_blind_wall_rms_error"]) == 0.0

    fom = pd.read_csv(result_dir / "fom_validation.csv")
    assert fom["Re"].astype(int).tolist() == [100, 400]
    assert float(fom["archive_relative_L2_u_v_omega"].max()) < 5.0e-13
    assert float(fom[["Ghia_relative_L2_u", "Ghia_relative_L2_v"]].max().max()) < 0.20

    convergence = pd.read_csv(result_dir / "convergence.csv")
    for study in ("grid", "time_step"):
        errors = convergence.loc[
            convergence["study"] == study, "relative_L2_uv"
        ].to_numpy()
        assert len(errors) == 3 and np.all(np.diff(errors) < 0.0), (study, errors)

    selection = pd.read_csv(result_dir / "selection.csv")
    assert selection["rank"].astype(int).tolist() == [4, 8, 12, 16]
    accepted = selection[selection["passes_one_percent_gate"]]
    assert accepted["rank"].astype(int).tolist() == [16]

    blind = pd.read_csv(result_dir / "blind_metrics.csv")
    assert sorted(blind["Re"].astype(int).unique().tolist()) == [175, 275, 375]
    assert sorted(blind["method"].unique().tolist()) == ["POD-DEIM", "POD-Galerkin"]
    assert float(blind["max_time_relative_L2_uv"].max()) < 0.01
    assert float(blind["final_relative_L2_omega"].max()) < 0.01
    assert float(blind["wall_rms_error"].max()) == 0.0
    assert float(blind["divergence_l2"].max()) < 1.0e-12

    timing = json.loads((result_dir / "timing.json").read_text())
    assert float(timing["POD_Galerkin_speedup"]) < 1.0
    assert float(timing["POD_DEIM_speedup"]) > 2.0
    assert float(timing["break_even_query_count"]) > 1.0

    with np.load(result_dir / "cavity_rom_model.npz", allow_pickle=False) as archive:
        assert int(archive["n"]) == 33
        assert archive["pod_modes"].shape == (31 * 31, 16)
        assert archive["nonlinear_basis"].shape == (31 * 31, 16)
        assert archive["deim_indices"].shape == (16,)

    notebook = json.loads(
        (ROOT / "notebooks" / "week04" / "W4_1_Classical_ROM_Cavity.ipynb").read_text()
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "load_deim_model" in source and "simulate_pod_deim" in source
    assert "Stop: blind-test gate" in source
    return {
        "max_blind_relative_L2_uv": float(blind["max_time_relative_L2_uv"].max()),
        "max_blind_final_relative_L2_omega": float(
            blind["final_relative_L2_omega"].max()
        ),
        "POD_Galerkin_speedup": float(timing["POD_Galerkin_speedup"]),
        "POD_DEIM_speedup": float(timing["POD_DEIM_speedup"]),
        "break_even_query_count": float(timing["break_even_query_count"]),
    }


def validate_pdfs() -> int:
    pdfs = sorted((ROOT / "lectures").glob("*.pdf"))
    assert len(pdfs) == 5
    for path in pdfs:
        result = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        )
        assert "Pages:" in result.stdout and "Page size:" in result.stdout
    return len(pdfs)


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert not missing, "missing release files: " + ", ".join(missing)
    actual = digest(ROOT / "data" / "cavity_data.npz")
    assert actual == EXPECTED_DATA_SHA256, (actual, EXPECTED_DATA_SHA256)
    notebooks, code_cells = validate_notebooks()
    metrics = smoke_common_baseline()
    deeponet_metrics = validate_pod_deeponet_results()
    cavity_rom_metrics = validate_cavity_rom_results()
    article_metrics = validate_article_alignment()
    pdfs = validate_pdfs()
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print("FLOWMLLAB_RELEASE_QA_PASS")
    print("notebooks:", notebooks, "code cells parsed:", code_cells)
    print("lecture PDFs:", pdfs)
    print("Python files parsed:", len(python_files))
    print("dataset SHA-256:", actual)
    print("Re=275 interpolation metrics:", json.dumps(metrics, sort_keys=True))
    print("POD-DeepONet release metrics:", json.dumps(deeponet_metrics, sort_keys=True))
    print("Cavity ROM release metrics:", json.dumps(cavity_rom_metrics, sort_keys=True))
    print("Article-aligned validation metrics:", json.dumps(article_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
