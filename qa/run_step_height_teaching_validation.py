#!/usr/bin/env python3
"""Run and record the leakage-free Week-9 micro-step teaching validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (  # noqa: E402
    STEP_HEIGHT_ARCHIVE_SHA256,
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
    STEP_SOURCE_COMMIT,
    evaluate_step_coordinate_surrogate,
    fit_step_coordinate_surrogate,
    load_step_height_archive,
    validate_step_height_archives,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mean_percent(rows: list[dict[str, float | int]], key: str) -> float:
    return 100.0 * float(np.mean([float(row[key]) for row in rows]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = root / "results" / "mahdavi_deeponet"
    # The sealed test file is deliberately not opened in this selection phase.
    learning_cases = load_step_height_archive(root, split="learning")
    learning_path = result_dir / "step_height_learning_7cases.npz"
    if digest(learning_path) != STEP_HEIGHT_ARCHIVE_SHA256[learning_path.name]:
        raise ValueError("learning archive SHA-256 mismatch")
    first = learning_cases[int(STEP_HEIGHT_DEVELOPMENT_PERCENT[0])]
    bounds_m = (
        float(first["x"].min()),
        float(first["x"].max()),
        float(first["y"].min()),
        float(first["y"].max()),
    )
    fit_options = {"bounds_m": bounds_m, "seed": 690, "sample_size": 60_000}

    baseline = fit_step_coordinate_surrogate(
        learning_cases, STEP_HEIGHT_DEVELOPMENT_PERCENT, None, **fit_options
    )
    baseline_rows = evaluate_step_coordinate_surrogate(
        baseline, learning_cases, STEP_HEIGHT_VALIDATION_PERCENT
    )
    baseline_global = mean_percent(baseline_rows, "full_relative_l2")
    baseline_vortex = mean_percent(baseline_rows, "vortex_relative_l2")
    selection_rows: list[dict[str, object]] = [
        {
            "model": "unweighted",
            "alpha": "",
            "validation_global_relative_l2_percent": f"{baseline_global:.9f}",
            "validation_vortex_relative_l2_percent": f"{baseline_vortex:.9f}",
            "eligible_global_guard": True,
            "selected": False,
        }
    ]
    candidates = {}
    candidate_summary = []
    for alpha in (0.5, 0.6, 0.7, 0.8):
        fitted = fit_step_coordinate_surrogate(
            learning_cases, STEP_HEIGHT_DEVELOPMENT_PERCENT, alpha, **fit_options
        )
        candidates[alpha] = fitted
        rows = evaluate_step_coordinate_surrogate(
            fitted, learning_cases, STEP_HEIGHT_VALIDATION_PERCENT
        )
        global_error = mean_percent(rows, "full_relative_l2")
        vortex_error = mean_percent(rows, "vortex_relative_l2")
        candidate_summary.append(
            {
                "alpha": alpha,
                "global": global_error,
                "vortex": vortex_error,
                "eligible": global_error <= baseline_global + 2.0,
            }
        )
    eligible = [row for row in candidate_summary if row["eligible"]]
    if not eligible:
        raise RuntimeError("no zonal candidate satisfies the predeclared global-error guard")
    selected_alpha = float(min(eligible, key=lambda row: row["vortex"])["alpha"])
    for row in candidate_summary:
        selection_rows.append(
            {
                "model": "zonal",
                "alpha": f"{row['alpha']:.1f}",
                "validation_global_relative_l2_percent": f"{row['global']:.9f}",
                "validation_vortex_relative_l2_percent": f"{row['vortex']:.9f}",
                "eligible_global_guard": bool(row["eligible"]),
                "selected": float(row["alpha"]) == selected_alpha,
            }
        )

    # Freeze every choice and fit on development+validation before opening test data.
    learning_heights = np.concatenate(
        (STEP_HEIGHT_DEVELOPMENT_PERCENT, STEP_HEIGHT_VALIDATION_PERCENT)
    )
    final_unweighted = fit_step_coordinate_surrogate(
        learning_cases, learning_heights, None, **fit_options
    )
    final_zonal = fit_step_coordinate_surrogate(
        learning_cases, learning_heights, selected_alpha, **fit_options
    )
    frozen_protocol = {
        "schema_version": 1,
        "source_commit": STEP_SOURCE_COMMIT,
        "archive_sha256": STEP_HEIGHT_ARCHIVE_SHA256,
        "split": {
            "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
            "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
            "held_out_test_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
        },
        "file_level_test_isolation": True,
        "test_archive_opened_only_after_model_selection_and_final_fit": True,
        "test_used_for_selection": False,
        "features": [
            "h/H", "normalized x", "normalized y", "step-relative x",
            "geometry-derived wall coordinate", "h*x", "h*y", "x*y",
        ],
        "forbidden_inputs": [
            "held-out U", "held-out V", "target-derived mask", "target-field patch",
        ],
        "model": {
            "kind": "coordinate MLP teaching baseline",
            "hidden_layers": [48, 48],
            "activation": "tanh",
            "seed": 690,
            "optimizer_sample_size": 60000,
            "max_iter": 90,
        },
        "candidate_alpha": [0.5, 0.6, 0.7, 0.8],
        "selection_rule": (
            "Minimize mean validation vortex relative L2 among candidates whose mean "
            "validation global relative L2 is no more than 2 percentage points above "
            "the unweighted baseline."
        ),
        "selected_alpha": selected_alpha,
        "software": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }

    # First access to the physically separate test archive occurs here.
    test_cases = load_step_height_archive(root, split="test")
    validate_step_height_archives(root)
    test_rows: list[dict[str, object]] = []
    for label, fitted in (
        ("unweighted", final_unweighted),
        (f"zonal_alpha_{selected_alpha:.1f}", final_zonal),
    ):
        for row in evaluate_step_coordinate_surrogate(
            fitted, test_cases, STEP_HEIGHT_HELD_OUT_PERCENT
        ):
            test_rows.append(
                {
                    "model": label,
                    "height_percent": int(row["height_percent"]),
                    "global_relative_l2_percent": f"{100 * float(row['full_relative_l2']):.9f}",
                    "vortex_relative_l2_percent": f"{100 * float(row['vortex_relative_l2']):.9f}",
                    "full_mse": f"{float(row['full_mse']):.12e}",
                    "vortex_mse": f"{float(row['vortex_mse']):.12e}",
                    "main_mse": f"{float(row['main_mse']):.12e}",
                    "vortex_points": int(row["vortex_points"]),
                    "main_points": int(row["main_points"]),
                }
            )

    selection_path = result_dir / "step_teaching_selection.csv"
    test_path = result_dir / "step_teaching_test_metrics.csv"
    protocol_path = result_dir / "step_teaching_protocol.json"
    write_csv(selection_path, selection_rows)
    write_csv(test_path, test_rows)
    protocol_path.write_text(
        json.dumps(frozen_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "selection_sha256": digest(selection_path),
                "test_metrics_sha256": digest(test_path),
                "protocol_sha256": digest(protocol_path),
                "test_metrics": test_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
