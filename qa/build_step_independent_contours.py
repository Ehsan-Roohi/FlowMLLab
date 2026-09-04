#!/usr/bin/env python3
"""Build held-out H44/H67 contours for the frozen independent teaching model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "qa"))

from build_step_article_contours import (  # noqa: E402
    geometry_contract,
    plot_case,
    reattachment_length_over_l,
    safe_relative_l2,
)
from flowmllab.mahdavi_deeponet import (  # noqa: E402
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
    fit_step_coordinate_surrogate,
    load_step_height_archive,
    predict_step_coordinate_surrogate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_result(
    height: int,
    case: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, object]:
    x = np.asarray(case["x"], dtype=float)
    y = np.asarray(case["y"], dtype=float)
    truth = np.column_stack((case["u"], case["v"]))
    delta = prediction - truth
    vortex = truth[:, 0] < 0.0
    predicted_vortex = prediction[:, 0] < 0.0
    union = np.count_nonzero(vortex | predicted_vortex)
    geometry = geometry_contract(x, y)
    metrics: dict[str, object] = {
        "case_id": f"H{height}",
        "nominal_parameter": f"h/H={height / 100:.2f}",
        "height_percent": height,
        "point_count": len(x),
        "combined_relative_l2_percent": 100.0 * safe_relative_l2(delta, truth),
        "u_relative_l2_percent": 100.0 * safe_relative_l2(delta[:, 0], truth[:, 0]),
        "v_relative_l2_percent": 100.0 * safe_relative_l2(delta[:, 1], truth[:, 1]),
        "vortex_relative_l2_percent": 100.0
        * safe_relative_l2(delta[vortex], truth[vortex]),
        "negative_u_iou_percent": 100.0
        * (np.count_nonzero(vortex & predicted_vortex) / union if union else 1.0),
        "dsmc_reattachment_length_over_L": reattachment_length_over_l(
            x, y, truth[:, 0], geometry
        ),
        "independent_reattachment_length_over_L": reattachment_length_over_l(
            x, y, prediction[:, 0], geometry
        ),
        "abs_u_error_p99": float(np.percentile(np.abs(delta[:, 0]), 99)),
        "abs_u_error_max": float(np.max(np.abs(delta[:, 0]))),
        "abs_v_error_p99": float(np.percentile(np.abs(delta[:, 1]), 99)),
        "abs_v_error_max": float(np.max(np.abs(delta[:, 1]))),
        "L_over_H_from_grid": geometry["L_over_H"],
        "step_x_over_H_from_grid": geometry["step_x_over_H"],
        "step_y_over_H_from_grid": geometry["step_y_over_H"],
    }
    return {
        "x": x,
        "y": y,
        "truth": truth,
        "prediction": prediction,
        "geometry": geometry,
        "metrics": metrics,
    }


def load_recorded_metrics(path: Path, selected_alpha: float) -> dict[int, dict[str, str]]:
    label = f"zonal_alpha_{selected_alpha:.1f}"
    with path.open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["model"] == label]
    return {int(row["height_percent"]): row for row in rows}


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = root / "results" / "mahdavi_deeponet"
    protocol_path = result_dir / "step_teaching_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected_alpha = float(protocol["selected_alpha"])
    if protocol["test_used_for_selection"] is not False:
        raise ValueError("the frozen protocol does not prohibit test-based selection")

    # Only the physically separate learning archive is opened before final fit.
    learning_cases = load_step_height_archive(root, split="learning")
    first = learning_cases[int(STEP_HEIGHT_DEVELOPMENT_PERCENT[0])]
    bounds_m = (
        float(first["x"].min()),
        float(first["x"].max()),
        float(first["y"].min()),
        float(first["y"].max()),
    )
    final_training_heights = np.concatenate(
        (STEP_HEIGHT_DEVELOPMENT_PERCENT, STEP_HEIGHT_VALIDATION_PERCENT)
    )
    fitted = fit_step_coordinate_surrogate(
        learning_cases,
        final_training_heights,
        selected_alpha,
        bounds_m=bounds_m,
        seed=int(protocol["model"]["seed"]),
        sample_size=int(protocol["model"]["optimizer_sample_size"]),
        max_iter=int(protocol["model"]["max_iter"]),
        hidden_layer_sizes=tuple(protocol["model"]["hidden_layers"]),
    )

    # First test access occurs after every model choice is frozen and fit completes.
    test_cases = load_step_height_archive(root, split="test")
    recorded = load_recorded_metrics(
        result_dir / "step_teaching_test_metrics.csv", selected_alpha
    )
    figure_dir = result_dir / "step_independent_contours"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    figure_hashes: dict[str, str] = {}
    for height in STEP_HEIGHT_HELD_OUT_PERCENT:
        integer_height = int(height)
        case = test_cases[integer_height]
        prediction = predict_step_coordinate_surrogate(fitted, integer_height, case)
        result = comparison_result(integer_height, case, prediction)
        metrics = dict(result["metrics"])
        expected = recorded[integer_height]
        for current_name, recorded_name in (
            ("combined_relative_l2_percent", "global_relative_l2_percent"),
            ("vortex_relative_l2_percent", "vortex_relative_l2_percent"),
        ):
            if abs(float(metrics[current_name]) - float(expected[recorded_name])) > 5.0e-7:
                raise ValueError(
                    f"H{integer_height}: contour prediction does not reproduce {recorded_name}"
                )
        output = figure_dir / f"held_out_H{integer_height}_independent.png"
        plot_case(
            result,
            output,
            prediction_label="FlowMLLab independent MLP",
            title_prefix="Independent held-out test",
            metric_label="independent-model vector relative L2",
            footnote=(
                "Model frozen on H16,H21,H25,H33,H50,H58,H75; the separate "
                "H44/H67 held-out archive was evaluated only after final fitting."
            ),
            footnote_color="#174f2a",
        )
        rows.append(metrics)
        figure_hashes[output.relative_to(root).as_posix()] = sha256(output)

    metrics_path = result_dir / "step_independent_contour_metrics.csv"
    write_metrics(metrics_path, rows)
    manifest = {
        "schema_version": 1,
        "selected_alpha": selected_alpha,
        "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
        "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
        "held_out_test_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
        "training_for_final_fit_percent": final_training_heights.tolist(),
        "test_archive_opened_after_final_fit": True,
        "test_used_for_selection": False,
        "allowed_inputs": protocol["features"],
        "forbidden_inputs": protocol["forbidden_inputs"],
        "protocol_sha256": sha256(protocol_path),
        "metrics_file": metrics_path.relative_to(root).as_posix(),
        "metrics_sha256": sha256(metrics_path),
        "figure_sha256": figure_hashes,
        "cases": rows,
    }
    manifest_path = result_dir / "step_independent_contour_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "metrics": metrics_path.relative_to(root).as_posix(),
                "manifest": manifest_path.relative_to(root).as_posix(),
                "figures": figure_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
