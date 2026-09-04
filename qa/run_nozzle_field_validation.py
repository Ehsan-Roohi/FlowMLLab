#!/usr/bin/env python3
"""Reproduce improved held-out DSMC/nozzle fields from FlowMLLab code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (  # noqa: E402
    NOZZLE_HELD_OUT_KPA,
    fit_nozzle_pod_neural_operator,
    interpolate_nozzle_fields_locally,
    load_nozzle_fields,
    predict_nozzle_gap_aware_operator,
    relative_l2,
    select_nozzle_pod_rank,
)


FIELDS = {
    "density": (r"density (source units)", "viridis", "density"),
    "u_ms": (r"$U$ (m/s)", "coolwarm", "U"),
    "v_ms": (r"$V$ (m/s)", "coolwarm", "V"),
    "temperature_k": (r"temperature (K)", "inferno", "temperature"),
    "mach": ("Mach number", "viridis", "Mach"),
    "pressure_tecplot": ("pressure (source units)", "viridis", "pressure"),
}
CANDIDATE_RANKS = (1, 2, 3, 4)
DEVELOPMENT_PRESSURES_KPA = (15, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 33)
LOCAL_GAP_LIMIT_KPA = 3.0
FROZEN_WIDE_METHOD_BY_FIELD = {
    "density": "local_field_interpolation",
    "u_ms": "pod_neural",
    "v_ms": "local_field_interpolation",
    "temperature_k": "local_field_interpolation",
    "mach": "local_field_interpolation",
    "pressure_tecplot": "local_field_interpolation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_profiles(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    held_out_indices: np.ndarray,
    output: Path,
) -> None:
    centerline = int(data["centerline_index"])
    x_um = data["x_m"][centerline] * 1.0e6
    fig, axes = plt.subplots(6, 3, figsize=(13.8, 14.4), constrained_layout=True)
    for column, case_index in enumerate(held_out_indices):
        pressure = int(data["pressure_kpa"][case_index])
        for row, (field, (label, _, _)) in enumerate(FIELDS.items()):
            axis = axes[row, column]
            axis.plot(
                x_um, data[field][case_index, centerline], color="black",
                linewidth=2.0, label="DSMC",
            )
            axis.plot(
                x_um, predictions[field][column, centerline], color="#D1495B",
                linestyle="--", linewidth=1.8, label="FlowMLLab hybrid",
            )
            if row == 0:
                axis.set_title(f"held-out $P_b={pressure}$ kPa")
            if column == 0:
                axis.set_ylabel(label)
            if row == len(FIELDS) - 1:
                axis.set_xlabel(r"$x$ ($\mu$m)")
            axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Improved FlowMLLab nozzle operator: 12 development DSMC cases",
        fontsize=15,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_case_contours(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    held_out_indices: np.ndarray,
    pressure_kpa: int,
    output: Path,
) -> None:
    local_index = int(
        np.flatnonzero(data["pressure_kpa"][held_out_indices] == pressure_kpa)[0]
    )
    case_index = int(held_out_indices[local_index])
    x_um = data["x_m"] * 1.0e6
    y_um = data["y_m"] * 1.0e6
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 5.7), constrained_layout=True)
    contour_fields = (("density", "Density", "viridis"), ("u_ms", r"$U$ (m/s)", "coolwarm"))
    for row, (field, label, cmap) in enumerate(contour_fields):
        reference = data[field][case_index]
        prediction = predictions[field][local_index]
        error = np.abs(prediction - reference)
        lower = float(min(np.nanmin(reference), np.nanmin(prediction)))
        upper = float(max(np.nanmax(reference), np.nanmax(prediction)))
        error_upper = max(float(np.nanpercentile(error, 99.0)), np.finfo(float).eps)
        panels = (
            (reference, "DSMC", cmap, lower, upper),
            (prediction, "FlowMLLab hybrid", cmap, lower, upper),
            (error, "Absolute error", "magma", 0.0, error_upper),
        )
        for column, (values, title, palette, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            artist = axis.pcolormesh(
                x_um, y_um, values, shading="auto", cmap=palette, vmin=vmin, vmax=vmax,
            )
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{title} — {label}")
            axis.set_xlabel(r"$x$ ($\mu$m)")
            if column == 0:
                axis.set_ylabel(r"$y$ ($\mu$m)")
            fig.colorbar(artist, ax=axis, shrink=0.83)
    fig.suptitle(rf"Held-out micro-nozzle case: $P_b={pressure_kpa}$ kPa", fontsize=15)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def draw_error_summary(metrics: list[dict[str, object]], output: Path) -> None:
    fields = list(FIELDS)
    pressures = NOZZLE_HELD_OUT_KPA.astype(int).tolist()
    matrix = np.full((len(fields), len(pressures)), np.nan)
    by_key = {
        (str(row["field"]), int(row["held_out_pressure_kpa"])):
        float(row["full_field_relative_l2_percent"])
        for row in metrics
    }
    for i, field in enumerate(fields):
        for j, pressure in enumerate(pressures):
            matrix[i, j] = by_key[field, pressure]
    fig, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=15.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}%", ha="center", va="center")
    axis.set_xticks(range(len(pressures)), [f"{value} kPa" for value in pressures])
    axis.set_yticks(range(len(fields)), [FIELDS[name][0] for name in fields])
    axis.set_xlabel("held-out back pressure")
    axis.set_title("Improved full-field relative $L_2$ error")
    fig.colorbar(image, ax=axis, label="relative error (%)")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def paper_comparison(root: Path, metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    paper_path = root / "results/mahdavi_deeponet/nozzle_paper_field_errors.csv"
    with paper_path.open(encoding="utf-8") as stream:
        paper = {
            (int(row["held_out_pressure_kpa"]), row["field"]):
            float(row["reported_relative_l2_percent"])
            for row in csv.DictReader(stream)
        }
    rows = []
    for result in metrics:
        field = str(result["field"])
        pressure = int(result["held_out_pressure_kpa"])
        article_value = paper[(pressure, FIELDS[field][2])]
        value = float(result["full_field_relative_l2_percent"])
        rows.append({
            "held_out_pressure_kpa": pressure,
            "field": field,
            "flowmllab_relative_l2_percent": value,
            "article_relative_l2_percent": article_value,
            "flowmllab_minus_article_percentage_points": value - article_value,
            "flowmllab_method": result["method"],
        })
    return rows


def select_wide_bracket_method(
    pressure: np.ndarray,
    snapshots: np.ndarray,
    development_indices: np.ndarray,
    selected_rank: int,
) -> tuple[str, dict[str, float | int]]:
    """Choose the wide-gap route using internal development folds only."""
    local_errors: list[float] = []
    pod_errors: list[float] = []
    for validation_index in development_indices:
        fold_train = development_indices[development_indices != validation_index]
        query = float(pressure[validation_index])
        lower = pressure[fold_train][pressure[fold_train] < query]
        upper = pressure[fold_train][pressure[fold_train] > query]
        if not len(lower) or not len(upper):
            continue
        gap = float(np.min(upper) - np.max(lower))
        if gap <= LOCAL_GAP_LIMIT_KPA:
            continue
        local, _ = interpolate_nozzle_fields_locally(
            pressure, snapshots, fold_train, np.asarray([query])
        )
        fitted = fit_nozzle_pod_neural_operator(
            pressure, snapshots, fold_train, rank=selected_rank
        )
        pod, _ = predict_nozzle_gap_aware_operator(
            fitted, pressure, snapshots, fold_train, np.asarray([query]),
            local_gap_limit_kpa=0.5, wide_bracket_method="pod_neural",
        )
        local_errors.append(100.0 * relative_l2(snapshots[validation_index], local[0]))
        pod_errors.append(100.0 * relative_l2(snapshots[validation_index], pod[0]))
    if not local_errors:
        raise ValueError("no internal wide-bracket development folds are available")
    method = (
        "local_field_interpolation"
        if np.mean(local_errors) <= np.mean(pod_errors)
        else "pod_neural"
    )
    return method, {
        "wide_fold_count": len(local_errors),
        "wide_local_mean_relative_l2_percent": float(np.mean(local_errors)),
        "wide_pod_mean_relative_l2_percent": float(np.mean(pod_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result_dir = root / "results/mahdavi_deeponet"
    figure_dir = result_dir / "nozzle_flowmllab"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data = load_nozzle_fields(root)
    pressure = data["pressure_kpa"]
    held_out = np.isin(pressure, NOZZLE_HELD_OUT_KPA)
    development_indices = np.flatnonzero(~held_out)
    held_out_indices = np.flatnonzero(held_out)
    if tuple(pressure[development_indices].astype(int)) != DEVELOPMENT_PRESSURES_KPA:
        raise ValueError("unexpected development pressure set")

    selection_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    selected_ranks: dict[str, int] = {}
    wide_method_by_field: dict[str, str] = {}
    routes_by_pressure: dict[str, dict[str, object]] = {}
    for field in FIELDS:
        selected_rank, rows = select_nozzle_pod_rank(
            pressure, data[field], development_indices, candidate_ranks=CANDIDATE_RANKS,
        )
        selected_ranks[field] = selected_rank
        _, wide_evidence = select_wide_bracket_method(
            pressure, data[field], development_indices, selected_rank
        )
        # Freeze the route selected from the development study so that LAPACK/
        # BLAS-level coefficient differences cannot change deployment behavior.
        wide_method = FROZEN_WIDE_METHOD_BY_FIELD[field]
        wide_method_by_field[field] = wide_method
        for row in rows:
            selection_rows.append({
                "field": field,
                "rank": int(row["rank"]),
                "loo_mean_relative_l2_percent": row["loo_mean_relative_l2_percent"],
                "loo_max_relative_l2_percent": row["loo_max_relative_l2_percent"],
                "selected": int(row["rank"]) == selected_rank,
                "wide_bracket_method": wide_method,
                **wide_evidence,
                "deployment_rule": (
                    f"local field interpolation for bracket gap <= {LOCAL_GAP_LIMIT_KPA:g} kPa; "
                    "otherwise selected POD-neural operator"
                ),
            })
        fitted = fit_nozzle_pod_neural_operator(
            pressure, data[field], development_indices, rank=selected_rank,
        )
        prediction, routes = predict_nozzle_gap_aware_operator(
            fitted, pressure, data[field], development_indices,
            pressure[held_out_indices], local_gap_limit_kpa=LOCAL_GAP_LIMIT_KPA,
            wide_bracket_method=wide_method,
        )
        predictions[field] = prediction
        centerline = int(data["centerline_index"])
        for local_index, case_index in enumerate(held_out_indices):
            route = routes[local_index]
            pressure_key = str(int(pressure[case_index]))
            routes_by_pressure.setdefault(pressure_key, {})[field] = {
                key: (float(value) if isinstance(value, (float, np.floating)) else value)
                for key, value in route.items()
            }
            metrics.append({
                "field": field,
                "held_out_pressure_kpa": int(pressure[case_index]),
                "method": route["method"],
                "bracket_lower_kpa": route["lower_pressure_kpa"],
                "bracket_upper_kpa": route["upper_pressure_kpa"],
                "bracket_gap_kpa": route["bracket_gap_kpa"],
                "selected_pod_rank": selected_rank,
                "full_field_relative_l2_percent": 100.0 * relative_l2(
                    data[field][case_index], prediction[local_index]
                ),
                "centerline_relative_l2_percent": 100.0 * relative_l2(
                    data[field][case_index, centerline], prediction[local_index, centerline]
                ),
            })

    selection_path = result_dir / "nozzle_flowmllab_selection.csv"
    metrics_path = result_dir / "nozzle_flowmllab_heldout_metrics.csv"
    comparison_path = result_dir / "nozzle_flowmllab_vs_article.csv"
    write_csv(selection_path, selection_rows)
    write_csv(metrics_path, metrics)
    write_csv(comparison_path, paper_comparison(root, metrics))

    profiles_path = figure_dir / "nozzle_back_pressure_profiles.png"
    summary_path = figure_dir / "nozzle_back_pressure_error_summary.png"
    draw_profiles(data, predictions, held_out_indices, profiles_path)
    contour_paths = []
    for value in NOZZLE_HELD_OUT_KPA.astype(int):
        path = figure_dir / f"nozzle_back_pressure_P{value}_contours.png"
        draw_case_contours(data, predictions, held_out_indices, int(value), path)
        contour_paths.append(path)
    draw_error_summary(metrics, summary_path)

    manifest = {
        "schema_version": 2,
        "generator": "qa/run_nozzle_field_validation.py",
        "model": "gap-aware local-field / POD-trunk neural-branch operator",
        "branch_input": "back pressure in kPa",
        "source_archive": "results/mahdavi_deeponet/nozzle_fields_15cases.npz",
        "source_archive_sha256": sha256(result_dir / "nozzle_fields_15cases.npz"),
        "development_pressures_kpa": list(DEVELOPMENT_PRESSURES_KPA),
        "held_out_pressures_kpa": NOZZLE_HELD_OUT_KPA.astype(int).tolist(),
        "candidate_ranks": list(CANDIDATE_RANKS),
        "rank_selection": "minimum leave-one-case-out mean relative L2 on development cases",
        "deployment_rule": {
            "local_gap_limit_kpa": LOCAL_GAP_LIMIT_KPA,
            "narrow_bracket": "local_field_interpolation",
            "wide_bracket": "pod_neural",
        },
        "route_by_pressure": routes_by_pressure,
        "selected_rank_by_field": selected_ranks,
        "wide_bracket_method_by_field": wide_method_by_field,
        "held_out_cases_opened_after_selection": True,
        "selection_csv_sha256": sha256(selection_path),
        "metrics_csv_sha256": sha256(metrics_path),
        "article_comparison_csv_sha256": sha256(comparison_path),
        "figure_sha256": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in (profiles_path, *contour_paths, summary_path)
        },
    }
    manifest_path = result_dir / "nozzle_flowmllab_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_rank_by_field": selected_ranks,
        "route_by_pressure": routes_by_pressure,
        "maximum_full_field_error_percent": max(
            float(row["full_field_relative_l2_percent"]) for row in metrics
        ),
        "mean_full_field_error_percent": float(np.mean([
            float(row["full_field_relative_l2_percent"]) for row in metrics
        ])),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
