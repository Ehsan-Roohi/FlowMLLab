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
    ".gitattributes",
    "README.md",
    "START_HERE.md",
    "COURSE_MAP.md",
    "THEORY_SOURCE_POLICY.md",
    "THEORY_GAP_MATRIX.md",
    "requirements.txt",
    "CITATION.cff",
    "LICENSE",
    "pyproject.toml",
    "flowmllab/__init__.py",
    "flowmllab/cli.py",
    "flowmllab/core.py",
    "flowmllab/cavity_rom.py",
    "flowmllab/cylinder_lbm.py",
    "flowmllab/cylinder_ml.py",
    "flowmllab/cylinder_cnn.py",
    "flowmllab/cylinder_phase.py",
    "flowmllab/gas_dynamics.py",
    "flowmllab/hypersonic_cylinder.py",
    "flowmllab/mahdavi_deeponet.py",
    "flowmllab/aescte_dsmc.py",
    "flowmllab/probabilistic_uq.py",
    "tests/test_core.py",
    "tests/test_cavity_rom.py",
    "tests/test_cylinder_lbm.py",
    "tests/test_cylinder_ml.py",
    "tests/test_cylinder_cnn.py",
    "tests/test_cylinder_phase.py",
    "tests/test_gas_dynamics.py",
    "tests/test_hypersonic_cylinder.py",
    "tests/test_mahdavi_deeponet.py",
    "tests/test_aescte_dsmc.py",
    "tests/test_probabilistic_uq.py",
    "qa/build_week04_1_rom_notebook.py",
    "qa/add_colab_entrypoints.py",
    "qa/run_cavity_rom_validation.py",
    "qa/run_cylinder_lbm_validation.py",
    "qa/run_cylinder_grid_independence.py",
    "qa/run_cylinder_blind_video.py",
    "qa/run_cylinder_multiscale_cnn.py",
    "qa/run_cylinder_phase_stable.py",
    "qa/build_week8_gas_dynamics_figures.py",
    "qa/run_week8_scattered_baseline.py",
    "qa/build_hypersonic_cylinder_subset.py",
    "qa/run_hypersonic_cylinder_evidence.py",
    "qa/build_week9_mahdavi_deeponet_data.py",
    "qa/build_step_independent_contours.py",
    "qa/run_nozzle_field_validation.py",
    "qa/build_week10_aescte_dsmc_data.py",
    "qa/run_week10_aescte_validation.py",
    "qa/build_probabilistic_uq_notebook.py",
    "qa/run_probabilistic_uq_validation.py",
    ".github/workflows/week8-materials.yml",
    ".github/workflows/week9-materials.yml",
    ".github/workflows/week10-materials.yml",
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
    "notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb",
    "notebooks/week07/make_week7_notebook.py",
    "notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb",
    "notebooks/week07_1/make_week7_1_notebook.py",
    "notebooks/week07_1/README.md",
    "notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb",
    "notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb",
    "notebooks/week08/make_week8_notebooks.py",
    "notebooks/week08/README.md",
    "notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb",
    "notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb",
    "notebooks/week09/make_week9_notebooks.py",
    "notebooks/week09/README.md",
    "notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb",
    "notebooks/week10/make_week10_notebook.py",
    "notebooks/week10/README.md",
    "qa/build_step_height_archive.py",
    "qa/run_step_height_teaching_validation.py",
    "notebooks/week05_06/P0_Project_Setup.ipynb",
    "notebooks/week05_06/P1_Re_Generalization.ipynb",
    "notebooks/week05_06/P2_Physics_Guided_DNN.ipynb",
    "notebooks/week05_06/P3_POD_Study.ipynb",
    "notebooks/week05_06/P4_Uncertainty_Study.ipynb",
    "notebooks/week05_06/P5_Rarefied_Cavity.ipynb",
    "notebooks/week05_06/P6_FP_Cavity_Closure.ipynb",
    "notebooks/week05_06/README.md",
    "notebooks/week02_1/README.md",
    "notebooks/week02_1/Probabilistic_UQ_CFD.ipynb",
    "lectures/week01_numerical_foundations.pdf",
    "lectures/week02_supervised_learning_rarefaction.pdf",
    "lectures/week02_1_probabilistic_uq.pdf",
    "lectures/source/week02_1_probabilistic_uq.pptx",
    "lectures/source/build_week02_1_probabilistic_uq.mjs",
    "lectures/week03_kinetic_dsmc.pdf",
    "lectures/week04_cavity_surrogates_deeponet.pdf",
    "lectures/week05_06_project_guide.pdf",
    "lectures/week07_cylinder_lbm_neural_surrogate.pdf",
    "lectures/source/week07_cylinder_lbm_neural_surrogate.tex",
    "lectures/week07_1_hypersonic_rarefied_cylinder.pdf",
    "lectures/source/build_week07_1_hypersonic_rarefied_cylinder.py",
    "lectures/week08_gas_dynamics_sciml.pdf",
    "lectures/source/week08_gas_dynamics_sciml.tex",
    "lectures/week09_rarefied_deeponet_case_studies.pdf",
    "lectures/source/week09_rarefied_deeponet_case_studies.tex",
    "lectures/week10_dsmc_data_driven_surrogates.pdf",
    "lectures/source/week10_dsmc_data_driven_surrogates.tex",
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
    "results/cylinder_lbm/regime_metrics.csv",
    "results/cylinder_lbm/README.md",
    "results/cylinder_lbm/reference_ranges.csv",
    "results/cylinder_lbm/validation_protocol.json",
    "results/cylinder_lbm/validation_summary.json",
    "results/cylinder_grid_convergence/README.md",
    "results/cylinder_grid_convergence/grid_metrics.csv",
    "results/cylinder_grid_convergence/grid_convergence.csv",
    "results/cylinder_grid_convergence/grid_protocol.json",
    "results/cylinder_grid_convergence/grid_summary.json",
    "results/cylinder_grid_convergence/cylinder_grid_independence.png",
    "results/cylinder_grid_convergence/cylinder_grid_independence.pdf",
    "results/cylinder_grid_convergence/re100_D012.npz",
    "results/cylinder_grid_convergence/re100_D018.npz",
    "results/cylinder_grid_convergence/re100_D027.npz",
    "results/cylinder_ml/README.md",
    "results/cylinder_ml/blind_re100_lbm_vs_neural.mp4",
    "results/cylinder_ml/blind_re100_lbm_vs_neural_poster.png",
    "results/cylinder_ml/blind_re100_metrics.json",
    "results/cylinder_ml/blind_re100_model_comparison.csv",
    "results/cylinder_cnn/README.md",
    "results/cylinder_cnn/multiscale_cnn.weights.h5",
    "results/cylinder_cnn/multiscale_cnn_metrics.json",
    "results/cylinder_cnn/training_history.csv",
    "results/cylinder_cnn/re100_validation_downstream.png",
    "results/cylinder_cnn/re100_validation_rollout.png",
    "results/cylinder_cnn/re105_blind_downstream.png",
    "results/cylinder_cnn/re105_retained_rollout.png",
    "results/cylinder_cnn/re105_lbm_vs_multiscale_cnn.mp4",
    "results/cylinder_cnn/re105_lbm_vs_multiscale_cnn_poster.png",
    "results/cylinder_phase/README.md",
    "results/cylinder_phase/phase_stable_metrics.json",
    "results/cylinder_phase/phase_stable_validation.png",
    "results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4",
    "results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.webp",
    "results/cylinder_phase/re095_phase_stable_poster.png",
    "results/cylinder_lbm/cylinder_lbm_regimes.png",
    "results/cylinder_lbm/cylinder_lbm_regimes.pdf",
    "results/cylinder_lbm/re5_teaching_case.npz",
    "results/cylinder_lbm/re20_teaching_case.npz",
    "results/cylinder_lbm/re40_teaching_case.npz",
    "results/cylinder_lbm/re100_teaching_case.npz",
    "results/cylinder_lbm/re180_teaching_case.npz",
    "data/hypersonic_cylinder/README.md",
    "data/hypersonic_cylinder/manifest.json",
    "data/hypersonic_cylinder/cylinder_teaching_subset.npz",
    "results/hypersonic_cylinder_week7_1/metrics.json",
    "results/hypersonic_cylinder_week7_1/mach85_baseline_audit.png",
    "results/gas_dynamics_week8/README.md",
    "results/gas_dynamics_week8/provenance.json",
    "results/gas_dynamics_week8/primary_metrics.csv",
    "results/gas_dynamics_week8/baseline_comparison.csv",
    "results/gas_dynamics_week8/range_generalization.csv",
    "results/gas_dynamics_week8/high_dimensional_scaling.csv",
    "results/gas_dynamics_week8/scattered_baseline.csv",
    "results/gas_dynamics_week8/physical_diagnostics.csv",
    "results/gas_dynamics_week8/application_audit_summary.json",
    "results/gas_dynamics_week8/week8_exact_physics.png",
    "results/gas_dynamics_week8/week8_exact_physics.pdf",
    "results/gas_dynamics_week8/week8_model_evidence.png",
    "results/gas_dynamics_week8/week8_model_evidence.pdf",
    "results/mahdavi_deeponet/README.md",
    "results/mahdavi_deeponet/DATA_LICENSE.md",
    "results/mahdavi_deeponet/provenance.json",
    "results/mahdavi_deeponet/nozzle_centerline_15cases.npz",
    "results/mahdavi_deeponet/nozzle_fields_15cases.npz",
    "results/mahdavi_deeponet/nozzle_flowmllab_selection.csv",
    "results/mahdavi_deeponet/nozzle_flowmllab_heldout_metrics.csv",
    "results/mahdavi_deeponet/nozzle_flowmllab_manifest.json",
    "results/mahdavi_deeponet/nozzle_flowmllab/nozzle_back_pressure_P16_contours.png",
    "results/mahdavi_deeponet/nozzle_flowmllab/nozzle_back_pressure_profiles.png",
    "results/mahdavi_deeponet/nozzle_flowmllab/nozzle_back_pressure_error_summary.png",
    "results/mahdavi_deeponet/nozzle_article_figures/README.md",
    "results/mahdavi_deeponet/nozzle_article_figures/nozzle_throat_X030_profiles.png",
    "results/mahdavi_deeponet/nozzle_article_figures/nozzle_throat_X030_profiles.svg",
    "results/mahdavi_deeponet/step_paper_evidence.csv",
    "results/mahdavi_deeponet/step_source_manifest.json",
    "results/mahdavi_deeponet/step_privileged_input_audit.csv",
    "results/mahdavi_deeponet/step_height_learning_7cases.npz",
    "results/mahdavi_deeponet/step_height_test_2cases.npz",
    "results/mahdavi_deeponet/step_teaching_selection.csv",
    "results/mahdavi_deeponet/step_teaching_test_metrics.csv",
    "results/mahdavi_deeponet/step_teaching_protocol.json",
    "results/mahdavi_deeponet/step_article_contour_manifest.json",
    "results/mahdavi_deeponet/step_article_contour_metrics.csv",
    "results/mahdavi_deeponet/step_article_case_coverage.csv",
    "results/mahdavi_deeponet/step_article_contours/article_figure_06_Kn0p004.png",
    "results/mahdavi_deeponet/step_article_contours/article_figure_06_Kn0p02.png",
    "results/mahdavi_deeponet/step_article_contours/article_figure_06_Kn1_DSMC_only.png",
    "results/mahdavi_deeponet/step_article_contours/article_figure_15_H44.png",
    "results/mahdavi_deeponet/step_article_contours/article_figure_15_H67.png",
    "results/mahdavi_deeponet/step_independent_contour_manifest.json",
    "results/mahdavi_deeponet/step_independent_contour_metrics.csv",
    "results/mahdavi_deeponet/step_independent_contours/held_out_H44_independent.png",
    "results/mahdavi_deeponet/step_independent_contours/held_out_H67_independent.png",
    "results/mahdavi_deeponet/nozzle_paper_field_errors.csv",
    "results/mahdavi_deeponet/nozzle_hard_case_baselines.csv",
    "results/mahdavi_deeponet/nozzle_pod_reference.csv",
    "data/aescte_dsmc/README.md",
    "data/aescte_dsmc/DATA_LICENSE.md",
    "results/aescte_dsmc/README.md",
    "results/aescte_dsmc/data_manifest.json",
    "results/aescte_dsmc/validation_summary.json",
    "results/aescte_dsmc/week10_validation_metrics.csv",
    "results/aescte_dsmc/cavity_fields_14cases.npz",
    "results/aescte_dsmc/diatomic_shock_6cases.npz",
    "results/aescte_dsmc/monatomic_shock_7cases.npz",
    "results/aescte_dsmc/week10_dsmc_reproduction_summary.png",
    "results/probabilistic_uq/README.md",
    "results/probabilistic_uq/protocol.json",
    "results/probabilistic_uq/summary.json",
    "results/probabilistic_uq/blind_metrics.csv",
    "results/probabilistic_uq/calibration.csv",
    "results/probabilistic_uq/probabilistic_uq_validation.png",
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


def validate_package_metadata() -> None:
    """Keep the editable Colab install compatible with the active runtime."""
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10,<3.14"' in metadata
    assert '"Programming Language :: Python :: 3.13"' in metadata


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
        relative = path.relative_to(ROOT).as_posix()
        colab_url = (
            "https://colab.research.google.com/github/"
            f"Ehsan-Roohi/FlowMLLab/blob/main/{relative}"
        )
        assert colab_url in full_source, f"missing direct Colab launcher: {path}"
        assert "FLOWMLLAB_COLAB_BOOTSTRAP_V1" in full_source, (
            f"missing Colab repository bootstrap: {path}"
        )
        assert any(
            marker in full_source
            for marker in (
                "MIE690A article-aligned validation v3",
                "MIE690A article-aligned validation v4",
                "MIE690A real-step-data validation v5",
            )
        ), (
            f"missing article-alignment marker: {path}"
        )
        ids = [cell.get("id") for cell in cells if cell.get("id")]
        assert len(ids) == len(set(ids)), f"duplicate cell id: {path}"
        if relative.startswith((
            "notebooks/week02_1/", "notebooks/week08/",
            "notebooks/week09/", "notebooks/week10/",
        )):
            assert len(ids) == len(cells), f"missing cell id in current teaching notebook: {path}"
        for index, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                code_cells += 1
                parse_notebook_code("".join(cell.get("source", [])), f"{path}:{index}")
        if path.parent == ROOT / "notebooks" / "week05_06" and path.name.startswith("P"):
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
    assert count == 25, f"expected 25 notebooks, found {count}"
    return count, code_cells


def validate_article_alignment() -> dict[str, float]:
    result_dir = ROOT / "results" / "article_figures"
    ghia = json.loads(
        (result_dir / "fig02_cavity_benchmark_metrics.json").read_text(encoding="utf-8")
    )
    pressure = json.loads(
        (result_dir / "fig08_pressure_recovery_metrics.json").read_text(encoding="utf-8")
    )
    dsmc = json.loads(
        (result_dir / "fig10a_mohammadzadeh_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert float(ghia["u_centerline_relative_l2"]) < 0.20
    assert float(ghia["v_centerline_relative_l2"]) < 0.20
    assert float(pressure["vertical_relative_l2_n129"]) < float(pressure["vertical_relative_l2_n65"])
    assert float(pressure["horizontal_relative_l2_n129"]) < float(pressure["horizontal_relative_l2_n65"])
    assert float(pressure["vertical_relative_l2_n129"]) < 0.07
    assert float(pressure["horizontal_relative_l2_n129"]) < 0.07
    assert float(dsmc["relative_l2"]) < 0.01
    assert float(dsmc["grid_change_40_to_60"]) < 0.015

    ghia_notebook = json.loads(
        (ROOT / "notebooks" / "week01" / "03_cavity_ghia.ipynb").read_text(
            encoding="utf-8"
        )
    )
    dsmc_notebook = json.loads(
        (ROOT / "notebooks" / "week03" / "AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb").read_text(
            encoding="utf-8"
        )
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
    assert float(ensemble["relative_L2_p"].max()) < 0.005
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
    assert timing["selected_heads"]["velocity"]["rank"] == 3
    assert timing["selected_heads"]["velocity"]["hidden"] == [32, 32]
    assert timing["selected_heads"]["velocity"]["input_transform"] == "linear"
    assert timing["selected_heads"]["pressure"]["rank"] == 3
    assert timing["selected_heads"]["pressure"]["hidden"] == [8]
    assert timing["selected_heads"]["pressure"]["input_transform"] == "log"
    assert float(timing["speedup"]) > 100.0
    assert float(timing["CFD_final_residual"]) < 1.0e-6

    predictions = pd.read_csv(result_dir / "deeponet_predictions.csv")
    field_columns = [
        "u_Re175", "v_Re175", "p_Re175",
        "u_Re275", "v_Re275", "p_Re275",
        "u_Re375", "v_Re375", "p_Re375",
    ]
    assert len(predictions) == 65 * 65
    assert predictions[["iy", "ix"]].drop_duplicates().shape[0] == 65 * 65
    assert predictions[["x", "y", *field_columns]].notna().all().all()

    binary_predictions = result_dir / "deeponet_predictions.npz"
    if binary_predictions.is_file():
        with np.load(binary_predictions, allow_pickle=False) as archive:
            assert archive["u"].shape == (3, 65, 65)
            assert archive["v"].shape == (3, 65, 65)
            assert archive["p"].shape == (3, 65, 65)
            assert np.max(np.abs(np.mean(archive["p"], axis=(1, 2)))) < 1.0e-14
            assert archive["seeds"].tolist() == [690, 691, 692]

    return {
        "max_blind_relative_L2_uv": float(ensemble["relative_L2_uv"].max()),
        "max_blind_relative_L2_p": float(ensemble["relative_L2_p"].max()),
        "max_divergence_L2": float(ensemble["div_l2_pred"].max()),
        "max_Ghia_error_change": float(ghia_delta),
        "measured_speedup": float(timing["speedup"]),
    }


def validate_cavity_rom_results() -> dict[str, float]:
    result_dir = ROOT / "results" / "cavity_rom"
    summary = json.loads(
        (result_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
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

    timing = json.loads((result_dir / "timing.json").read_text(encoding="utf-8"))
    assert float(timing["POD_Galerkin_speedup"]) < 1.0
    assert float(timing["POD_DEIM_speedup"]) > 2.0
    assert float(timing["break_even_query_count"]) > 1.0

    with np.load(result_dir / "cavity_rom_model.npz", allow_pickle=False) as archive:
        assert int(archive["n"]) == 33
        assert archive["pod_modes"].shape == (31 * 31, 16)
        assert archive["nonlinear_basis"].shape == (31 * 31, 16)
        assert archive["deim_indices"].shape == (16,)

    notebook = json.loads(
        (ROOT / "notebooks" / "week04" / "W4_1_Classical_ROM_Cavity.ipynb").read_text(
            encoding="utf-8"
        )
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


def validate_cylinder_lbm_results() -> dict[str, float]:
    """Recompute the retained Week-7 stability and regime evidence gates."""
    result_dir = ROOT / "results" / "cylinder_lbm"
    metrics = pd.read_csv(result_dir / "regime_metrics.csv")
    assert metrics["Re"].astype(int).tolist() == [5, 20, 40, 100, 180]
    assert metrics["fidelity"].eq("quick").all()
    assert metrics["stability_pass"].astype(bool).all()
    assert metrics["regime_pass"].astype(bool).all()
    assert float(metrics["Mach"].max()) < 0.1
    assert float(metrics["blockage"].max()) <= 0.15
    assert float(metrics["density_drift"].max()) < 0.01

    steady = metrics[metrics["Re"].isin([5, 20, 40])]
    shedding = metrics[metrics["Re"].isin([100, 180])]
    assert steady["observed_regime"].tolist() == [
        "attached", "steady recirculating", "steady recirculating"
    ]
    assert shedding["observed_regime"].eq("periodic shedding").all()
    assert shedding["St"].between(0.14, 0.22).all()
    assert shedding["St_valid"].astype(bool).all()
    assert (shedding["St_cycles"] >= 8.0).all()
    assert (shedding["lift_relative_rms_change"] <= 0.25).all()

    for reynolds in metrics["Re"].astype(int):
        with np.load(result_dir / f"re{reynolds}_teaching_case.npz", allow_pickle=False) as case:
            for field in ("rho", "u", "v", "p", "vorticity", "time_mean_u", "time_mean_v"):
                assert np.isfinite(case[field]).all(), (reynolds, field)
            solid = case["solid"].astype(bool)
            assert np.all(case["u"][solid] == 0.0)
            assert np.all(case["v"][solid] == 0.0)
            metadata = json.loads(str(case["metadata"]))
            assert "analytical circle" in metadata["cylinder_boundary"]
            diagnostic = json.loads(str(case["strouhal_diagnostics"]))
            assert bool(diagnostic["valid"]) == (reynolds >= 100)
            if reynolds == 100:
                assert case["restart_populations"].shape == (*solid.shape, 9)
                assert case["restart_outlet_previous"].shape == (solid.shape[0], 9)
                expected_steps = (
                    int(metadata["completed_steps_before_restart"])
                    + int(metadata["config"]["steps"])
                )
                assert int(case["restart_completed_steps"]) == expected_steps

    protocol = json.loads(
        (result_dir / "validation_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["profile"] == "quick"
    assert protocol["boundaries"]["cylinder"] == "Bouzidi interpolated circular wall"
    assert "not grid-converged" in " ".join(protocol["limitations"])
    notebook = json.loads(
        (ROOT / "notebooks" / "week07" / "W7_Lattice_Boltzmann_Cylinder_Student.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Diagnostic gate: explain the POD failure" in source
    assert "historically withheld" in source
    assert "harmonic-ridge POD" in source
    assert "blind_re100_lbm_vs_neural.mp4" in source
    assert "Four-frame multi-scale CNN" in source
    assert "re105_lbm_vs_multiscale_cnn.mp4" in source
    assert "cubic temporal extrapolation" in source
    assert "50-step recursive audit" in source
    assert "Phase-stable correction: 277 autonomous future fields" in source
    assert "fresh_test_reynolds" in source
    assert "selected_harmonics" in source
    assert "re095_phase_stable_lbm_vs_decoder.mp4" in source

    blind_metrics = json.loads(
        (ROOT / "results" / "cylinder_ml" / "blind_re100_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert blind_metrics["split"]["blind_reynolds"] == 100
    assert blind_metrics["split"]["training_reynolds"] == [60, 80, 90, 110, 120, 140]
    assert "not autonomous rollout" in blind_metrics["claim_scope"]
    neural = blind_metrics["neural_blind_metrics"]
    baseline = blind_metrics["harmonic_pod_baseline_metrics"]
    assert neural["combined_relative_l2"] < 0.10
    assert neural["vorticity_relative_l2"] < baseline["vorticity_relative_l2"]
    assert neural["solid_speed_rms_normalized"] == 0.0
    assert (ROOT / "results" / "cylinder_ml" / "blind_re100_lbm_vs_neural.mp4").stat().st_size > 1_000_000

    cnn_metrics = json.loads(
        (ROOT / "results" / "cylinder_cnn" / "multiscale_cnn_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = cnn_metrics["protocol"]
    assert protocol["development_reynolds"] == [60, 80, 90, 110, 120, 140]
    assert protocol["validation_reynolds"] == 100
    assert protocol["blind_reynolds"] == 105
    assert "not a fresh test" in protocol["blind_status"]
    assert "not autonomous rollout" in protocol["claim_scope"]
    assert protocol["history_frames"] == 4
    assert abs(protocol["dimensionless_snapshot_spacing"] - 0.1041666667) < 1.0e-8
    assert protocol["teacher_forced_one_step"]
    assert cnn_metrics["validation_pass"]
    assert all(cnn_metrics["validation_gates"].values())
    for split_name in ("validation", "blind"):
        split = cnn_metrics[split_name]
        assert split["best_non_neural_baseline"] == "cubic"
        split_metrics = split["models"]
        cnn = split_metrics["prediction"]
        persistence = split_metrics["persistence"]
        cubic = split_metrics["cubic"]
        assert cnn["vorticity_relative_l2"] < cubic["vorticity_relative_l2"] < persistence["vorticity_relative_l2"]
        assert cnn["mean_station_profile_relative_l2"] < cubic["mean_station_profile_relative_l2"] < persistence["mean_station_profile_relative_l2"]
        assert cnn["mean_station_complex_spectral_incoherence"] < cubic["mean_station_complex_spectral_incoherence"]
        assert cnn["solid_speed_max"] == 0.0
        assert len(cnn["stationwise"]) == 4
        assert [row["x_over_d"] for row in cnn["stationwise"]] == [2.0, 4.0, 6.0, 8.0]
        assert all(0.75 <= row["enstrophy_ratio"] <= 1.25 for row in cnn["stationwise"])
    for rollout_name in ("validation_rollout", "blind_rollout"):
        rollout = cnn_metrics[rollout_name]
        assert rollout["status"] == "fails_declared_rollout_gate"
        cnn_rows = {
            int(row["horizon_frames"]): row
            for row in rollout["rows"]
            if row["model"] == "cnn_autoregressive"
        }
        assert sorted(cnn_rows) == [1, 5, 10, 20, 30, 40, 50]
        assert cnn_rows[10]["mean_vorticity_relative_l2"] < 0.15
        assert cnn_rows[20]["mean_vorticity_relative_l2"] > 0.15
    assert cnn_metrics["blind"]["reynolds"] == 105
    assert cnn_metrics["blind"]["models"]["prediction"]["vorticity_relative_l2"] < 0.02
    assert (ROOT / "results" / "cylinder_cnn" / "re105_lbm_vs_multiscale_cnn.mp4").stat().st_size > 1_000_000
    phase_metrics = json.loads(
        (ROOT / "results" / "cylinder_phase" / "phase_stable_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    phase_protocol = phase_metrics["protocol"]
    assert phase_protocol["development_reynolds"] == [90, 110, 120, 140]
    assert phase_protocol["validation_reynolds"] == 100
    assert phase_protocol["fresh_test_reynolds"] == 95
    assert phase_protocol["retained_test_reynolds"] == 105
    assert phase_protocol["initial_true_frames"] == 4
    assert phase_protocol["future_cfd_inputs"] == 0
    assert phase_metrics["selected_harmonics"] == 6
    assert phase_metrics["all_gates_pass"]
    for split_name in ("validation", "fresh_test", "retained_test"):
        split = phase_metrics[split_name]
        assert split["future_frames"] == 277
        assert split["vorticity_global_relative_l2"] < 0.15
        assert split["vorticity_max_frame_relative_l2"] < 0.15
        assert split["strouhal_relative_error"] < 0.02
        assert split["passes"]
    assert (ROOT / "results" / "cylinder_phase" / "re095_phase_stable_lbm_vs_decoder.mp4").stat().st_size > 1_000_000
    return {
        "max_density_drift": float(metrics["density_drift"].max()),
        "Re100_St": float(metrics.loc[metrics["Re"] == 100, "St"].iloc[0]),
        "Re180_St": float(metrics.loc[metrics["Re"] == 180, "St"].iloc[0]),
        "blind_Re100_neural_uvp_error": float(neural["combined_relative_l2"]),
        "blind_Re100_neural_vorticity_error": float(neural["vorticity_relative_l2"]),
        "retained_Re105_CNN_vorticity_error": float(
            cnn_metrics["blind"]["models"]["prediction"]["vorticity_relative_l2"]
        ),
        "fresh_Re95_phase_vorticity_error": float(
            phase_metrics["fresh_test"]["vorticity_global_relative_l2"]
        ),
        "fresh_Re95_phase_strouhal_error": float(
            phase_metrics["fresh_test"]["strouhal_relative_error"]
        ),
    }


def validate_cylinder_grid_results() -> dict[str, float | bool]:
    """Verify the retained, deliberately failed three-grid formal gate."""
    result_dir = ROOT / "results" / "cylinder_grid_convergence"
    metrics = pd.read_csv(result_dir / "grid_metrics.csv")
    convergence = pd.read_csv(result_dir / "grid_convergence.csv")
    summary = json.loads((result_dir / "grid_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads(
        (result_dir / "grid_protocol.json").read_text(encoding="utf-8")
    )

    assert metrics["nodes_per_diameter"].astype(int).tolist() == [12, 18, 27]
    assert metrics["Re"].astype(int).eq(100).all()
    assert np.allclose(metrics["Mach"], metrics["Mach"].iloc[0])
    assert np.allclose(metrics["blockage"], 0.125)
    assert metrics["observation_time_D_over_U"].eq(100).all()
    assert metrics["statistics_start_D_over_U"].eq(45).all()
    assert metrics["statistical_convergence_pass"].astype(bool).all()
    assert convergence["quantity"].tolist() == ["Cd_mean", "St", "Lr_over_D"]
    assert not convergence["valid_asymptotic_sequence"].astype(bool).any()
    assert not convergence["pass"].astype(bool).any()
    assert not summary["grid_independent"] and summary["status"] == "fail"
    for _, row in convergence.iterrows():
        assert row["fine_pair_relative_change_percent"] <= row["fine_pair_gate_percent"]
    assert protocol["changed_parameter"] == "nodes per cylinder diameter"
    assert protocol["nodes_per_diameter"] == [12, 18, 27]
    assert protocol["refinement_ratio"] == 1.5
    assert protocol["next_required_resolution"] == 40
    assert "Gates were not relaxed" in protocol["continuation_rule"]

    for resolution in (12, 18, 27):
        path = result_dir / f"re100_D{resolution:03d}.npz"
        with np.load(path, allow_pickle=False) as case:
            solid = case["solid"].astype(bool)
            for field in ("rho", "u", "v", "p", "vorticity", "time_mean_u"):
                assert np.isfinite(case[field]).all(), (resolution, field)
            assert np.all(case["u"][solid] == 0.0)
            assert np.all(case["v"][solid] == 0.0)
            metadata = json.loads(str(case["metadata"]))
            assert int(metadata["config"]["diameter"]) == resolution
            assert metadata["config"]["nx"] == 20 * resolution
            assert metadata["config"]["ny"] == 8 * resolution

    notebook = json.loads(
        (ROOT / "notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "LBM algorithm: one time step in seven operations" in source
    assert "Executed grid study: retain failure, then refine" in source
    assert "richardson" in source.lower() and "GCI" in source
    assert "ML-versus-LBM error cannot repair" in source

    return {
        "finest_nodes_per_diameter": float(metrics["nodes_per_diameter"].iloc[-1]),
        "finest_Cd": float(metrics["Cd_mean"].iloc[-1]),
        "finest_St": float(metrics["St"].iloc[-1]),
        "max_fine_pair_change_percent": float(
            convergence["fine_pair_relative_change_percent"].max()
        ),
        "formal_grid_independent": False,
    }


def validate_week8_gas_dynamics_results() -> dict[str, object]:
    """Verify the pinned evidence, notebook pedagogy, and source boundaries."""
    sys.path.insert(0, str(ROOT))
    from flowmllab.gas_dynamics import validate_week8_evidence  # noqa: PLC0415

    report = validate_week8_evidence(ROOT)
    result_dir = ROOT / "results" / "gas_dynamics_week8"
    provenance = json.loads(
        (result_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert "not_permitted" in provenance["claim_boundary"]
    assert "other eight distributed cases remain unverified" in provenance[
        "claim_boundary"
    ]["su2_status"]

    lab1 = json.loads(
        (ROOT / "notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb").read_text(
            encoding="utf-8"
        )
    )
    lab2 = json.loads(
        (ROOT / "notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb").read_text(
            encoding="utf-8"
        )
    )
    lab1_source = "\n".join("".join(cell.get("source", [])) for cell in lab1["cells"])
    lab2_source = "\n".join("".join(cell.get("source", [])) for cell in lab2["cells"])
    assert "Exact gas dynamics before machine learning" in lab1_source
    assert "nine classical chapter notebooks" in lab1_source
    assert "other eight distributed cases remain" in lab1_source
    assert "deliberately ill-posed inverse" in lab2_source
    assert "The baseline can win" in lab2_source
    assert "Ordinary blind accuracy is not edge generalization" in lab2_source
    assert "Dimensional scaling under a matched offline budget" in lab2_source
    return report


def validate_week9_mahdavi_deeponet_results() -> dict[str, object]:
    """Verify the public DSMC derivative, paper tables, and claim boundaries."""
    sys.path.insert(0, str(ROOT))
    from flowmllab.mahdavi_deeponet import validate_week9_evidence  # noqa: PLC0415

    report = validate_week9_evidence(ROOT)
    result_dir = ROOT / "results" / "mahdavi_deeponet"

    step = pd.read_csv(result_dir / "step_paper_evidence.csv")
    step_values = step.pivot(
        index="objective", columns="scope", values="reported_error_percent"
    )
    assert abs(float(step_values.loc["MSE", "full_domain"]) - 2.1739) < 1e-10
    assert abs(float(step_values.loc["MSE", "recirculation_zone"]) - 14.6135) < 1e-10
    assert abs(float(step_values.loc["Zonal", "full_domain"]) - 2.2254) < 1e-10
    assert abs(float(step_values.loc["Zonal", "recirculation_zone"]) - 11.9413) < 1e-10

    field_errors = pd.read_csv(result_dir / "nozzle_paper_field_errors.csv")
    assert sorted(field_errors["held_out_pressure_kpa"].unique().tolist()) == [16, 25, 30]
    assert sorted(field_errors["field"].unique().tolist()) == [
        "Mach", "U", "V", "density", "pressure", "temperature"
    ]
    assert len(field_errors) == 18

    baselines = pd.read_csv(result_dir / "nozzle_hard_case_baselines.csv")
    aligned = baselines[baselines["model"] == "Shock-aligned reduced d"].iloc[0]
    cartesian = baselines[
        baselines["model"] == "Cartesian Hadamard branch/trunk"
    ].iloc[0]
    assert float(aligned["shock_window_mean_percent"]) == 9.12
    assert float(aligned["shock_window_std_percent"]) == 1.01
    assert float(cartesian["shock_window_mean_percent"]) == 34.95
    assert baselines["evidence_status"].eq(
        "retained final-article Table XIII evidence"
    ).all()

    provenance = json.loads(
        (result_dir / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_commit"] == "e1b234ba499408d3b6224633972f939f3b2301d6"
    assert len(provenance["source_files_sha256"]) == 15
    assert "real DSMC" in provenance["claim_boundary"]["microstep_data"]
    assert "does not receive" in provenance["claim_boundary"]["microstep_teaching_model"]
    assert "privileged-input" in provenance["claim_boundary"]["microstep_published_model"]
    assert "not the paper's trained six-output DeepONet" in provenance["claim_boundary"]["micro_nozzle_teaching_model"]

    step_manifest = json.loads(
        (result_dir / "step_source_manifest.json").read_text(encoding="utf-8")
    )
    assert step_manifest["source_commit"] == "c3f211376b42b8dc30daad380eaef5e0ab800b5c"
    assert step_manifest["height_percent"] == [16, 21, 25, 33, 44, 50, 58, 67, 75]
    assert step_manifest["split"]["held_out_test_percent"] == [44, 67]
    assert len(step_manifest["smoothed_files"]) == 9
    assert step_manifest["study_scope"]["joint_generalization"] == "not demonstrated"
    assert step_manifest["derived_archives_sha256"] == report["step_dataset"]["archive_sha256"]
    assert report["step_dataset"]["file_level_test_isolation"] is True
    assert report["step_teaching_validation"]["test_used_for_selection"] is False
    assert report["step_teaching_validation"]["selected_alpha"] == 0.6

    patch_audit = pd.read_csv(result_dir / "step_privileged_input_audit.csv")
    assert patch_audit["height_percent"].tolist() == [44, 67]
    assert (
        patch_audit["target_patch_nearest_sample_relative_l2_percent"]
        < patch_audit["upstream_stored_prediction_relative_l2_percent"]
    ).all()

    lab1 = json.loads(
        (
            ROOT / "notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb"
        ).read_text(encoding="utf-8")
    )
    lab2 = json.loads(
        (
            ROOT / "notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb"
        ).read_text(encoding="utf-8")
    )
    lab1_source = "\n".join("".join(cell.get("source", [])) for cell in lab1["cells"])
    lab2_source = "\n".join("".join(cell.get("source", [])) for cell in lab2["cells"])
    assert "nine real DSMC height fields" in lab1_source
    assert "model uses only $h/H$" in lab1_source
    assert "Freeze the case-wise split and selection rule" in lab1_source
    assert "Stop: held-out geometry gate" in lab1_source
    assert "step_height_test_2cases.npz" in lab1_source
    assert "independent coordinate surrogate" in lab1_source
    assert "Joint" in lab1_source and "has not been demonstrated" in lab1_source
    assert "all 15 real" in lab2_source and "public DSMC snapshots" in lab2_source
    assert "leave-one-case-out" in lab2_source
    assert "Stop: held-out pressure gate" in lab2_source
    assert "run_nozzle_field_validation.py" in lab2_source
    assert "physical-coordinate and shock-aligned interpolation baselines" in lab2_source
    nozzle_rows = report["nozzle_flowmllab_validation"]["held_out_metrics"]
    assert len(nozzle_rows) == 18
    assert max(float(row["selected_global_relative_l2_percent"]) for row in nozzle_rows) < 15.0
    return report


def validate_probabilistic_uq_results() -> dict[str, object]:
    """Verify the Week-2.1 UQ evidence, blind split, and originality boundary."""
    sys.path.insert(0, str(ROOT))
    from flowmllab.probabilistic_uq import (  # noqa: PLC0415
        validate_probabilistic_uq_evidence,
    )

    report = validate_probabilistic_uq_evidence(ROOT)
    notebook = json.loads(
        (
            ROOT / "notebooks/week02_1/Probabilistic_UQ_CFD.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Observation model before loss function" in source
    assert "Blind gate" in source
    assert "coverage over correlated CFD nodes is descriptive" in source
    assert "No restricted course handout" in source
    policy = (ROOT / "THEORY_SOURCE_POLICY.md").read_text(encoding="utf-8")
    assert "Homework, solution sets" in policy
    assert "Independent-authoring workflow" in policy
    return report


def validate_week10_aescte_results() -> dict[str, object]:
    """Verify the article-backed DSMC data and retained numerical gates."""
    result_dir = ROOT / "results" / "aescte_dsmc"
    manifest = json.loads((result_dir / "data_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "validation_summary.json").read_text(encoding="utf-8"))
    assert manifest["article"]["doi"] == "10.1016/j.ast.2025.110785"
    assert len(manifest["source_file_sha256"]) == 85
    for relative, expected in manifest["source_file_sha256"].items():
        assert digest(ROOT / relative) == expected, f"Week-10 source hash mismatch: {relative}"
    for relative, expected in manifest["derived_file_sha256"].items():
        assert digest(ROOT / relative) == expected, f"Week-10 archive hash mismatch: {relative}"
    metrics = pd.read_csv(result_dir / "week10_validation_metrics.csv")
    assert len(metrics) == 34
    assert summary["status"] == "pass"
    assert summary["cavity_primary_max_nrmse_percent"] < 2.0
    assert summary["shock_max_relative_l2_percent"] < 1.5
    assert summary["cavity_100_ms_target_available"] is False
    assert summary["diatomic_mach_2_target_available"] is False
    for filename, expected in summary["figures_sha256"].items():
        assert digest(result_dir / filename) == expected
    return summary


def validate_hypersonic_cylinder_results() -> dict[str, object]:
    """Verify the Week-7.1 provenance, split, baseline, and claim boundary."""
    from flowmllab.hypersonic_cylinder import (  # noqa: PLC0415
        validate_hypersonic_cylinder_evidence,
    )

    report = validate_hypersonic_cylinder_evidence(ROOT)
    assert report["cases"] == 20
    assert report["points"] == 44_500
    assert report["paper_doi"] == "10.1063/5.0334590"
    retained = json.loads(
        (ROOT / "results/hypersonic_cylinder_week7_1/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    interpolation = retained["splits"]["interpolation"]
    baseline = interpolation["linear_case_baseline_relative_l2"]
    teaching = interpolation["teaching_operator_relative_l2"]
    assert all(value < 0.01 for value in baseline.values())
    assert all(teaching[name] > baseline[name] for name in baseline)
    notebook = json.loads(
        (
            ROOT
            / "notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "Mandatory strong baseline" in source
    assert "does **not** report the published full-resolution accuracy" in source
    assert "RUN_FULL_NEURAL = False" in source
    return report


def validate_pdfs() -> int:
    pdfs = sorted((ROOT / "lectures").glob("*.pdf"))
    assert len(pdfs) == 11
    for path in pdfs:
        result = subprocess.run(
            ["pdfinfo", str(path)], check=True, capture_output=True, text=True
        )
        assert "Pages:" in result.stdout and "Page size:" in result.stdout
    return len(pdfs)


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert not missing, "missing release files: " + ", ".join(missing)
    validate_package_metadata()
    actual = digest(ROOT / "data" / "cavity_data.npz")
    assert actual == EXPECTED_DATA_SHA256, (actual, EXPECTED_DATA_SHA256)
    notebooks, code_cells = validate_notebooks()
    metrics = smoke_common_baseline()
    deeponet_metrics = validate_pod_deeponet_results()
    cavity_rom_metrics = validate_cavity_rom_results()
    cylinder_metrics = validate_cylinder_lbm_results()
    cylinder_grid_metrics = validate_cylinder_grid_results()
    week8_metrics = validate_week8_gas_dynamics_results()
    week9_metrics = validate_week9_mahdavi_deeponet_results()
    week10_metrics = validate_week10_aescte_results()
    uq_metrics = validate_probabilistic_uq_results()
    hypersonic_cylinder_metrics = validate_hypersonic_cylinder_results()
    article_metrics = validate_article_alignment()
    pdfs = validate_pdfs()
    excluded_roots = {".external", ".git", ".venv", "tmp", "venv"}
    python_files = sorted(
        path
        for path in ROOT.rglob("*.py")
        if excluded_roots.isdisjoint(path.relative_to(ROOT).parts)
    )
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
    print("Cylinder LBM release metrics:", json.dumps(cylinder_metrics, sort_keys=True))
    print("Cylinder grid-verification metrics:", json.dumps(cylinder_grid_metrics, sort_keys=True))
    print(
        "Hypersonic-cylinder Week-7.1 metrics:",
        json.dumps(hypersonic_cylinder_metrics, sort_keys=True),
    )
    print("Week-8 gas-dynamics metrics:", json.dumps(week8_metrics, sort_keys=True))
    print("Week-9 Roohi--Mahdavi metrics:", json.dumps(week9_metrics, sort_keys=True))
    print("Week-10 DSMC reproduction metrics:", json.dumps(week10_metrics, sort_keys=True))
    print("Probabilistic-UQ metrics:", json.dumps(uq_metrics, sort_keys=True))
    print("Article-aligned validation metrics:", json.dumps(article_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
