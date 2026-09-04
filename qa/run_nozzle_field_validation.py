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
    interpolate_nozzle_fields_shock_aligned,
    interpolate_nozzle_fields_locally,
    load_nozzle_fields,
    nozzle_field_error_metrics,
    relative_l2,
)


FIELDS = {
    "density": (r"density (source units)", "viridis", "density"),
    "u_ms": (r"$U$ (m/s)", "coolwarm", "U"),
    "v_ms": (r"$V$ (m/s)", "coolwarm", "V"),
    "temperature_k": (r"temperature (K)", "inferno", "temperature"),
    "mach": ("Mach number", "viridis", "Mach"),
    "pressure_tecplot": ("pressure (source units)", "viridis", "pressure"),
}
DEVELOPMENT_PRESSURES_KPA = (15, 18, 19, 20, 22, 23, 24, 26, 27, 28, 29, 33)


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
    physical_predictions: dict[str, np.ndarray],
    aligned_predictions: dict[str, np.ndarray],
    selected_method: dict[str, str],
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
                x_um, physical_predictions[field][column, centerline], color="#D1495B",
                linestyle="--", linewidth=1.5, label="physical-coordinate interpolation",
            )
            axis.plot(
                x_um, aligned_predictions[field][column, centerline], color="#2166AC",
                linestyle="-.", linewidth=1.7, label="shock-aligned interpolation",
            )
            if row == 0:
                axis.set_title(f"held-out $P_b={pressure}$ kPa")
            if column == 0:
                axis.set_ylabel(label)
            if row == len(FIELDS) - 1:
                axis.set_xlabel(r"$x$ ($\mu$m)")
            axis.text(
                0.98, 0.06,
                "selected: aligned" if selected_method[field] == "shock_aligned" else "selected: physical",
                transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5,
                color="#2166AC" if selected_method[field] == "shock_aligned" else "#D1495B",
            )
            axis.grid(alpha=0.22)
            if field == "v_ms":
                axis.axhline(0, color="#B42318", linewidth=1)
                axis.set_title("Exported boundary V violates symmetry", fontsize=9, color="#B42318")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Held-out nozzle profiles: physical versus shock-aligned baselines",
        fontsize=15,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_case_contours(
    data: dict[str, np.ndarray],
    physical_predictions: dict[str, np.ndarray],
    aligned_predictions: dict[str, np.ndarray],
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
    fig, axes = plt.subplots(2, 5, figsize=(18.6, 5.5), constrained_layout=True)
    contour_fields = (("density", "Density", "viridis"), ("u_ms", r"$U$ (m/s)", "coolwarm"))
    for row, (field, label, cmap) in enumerate(contour_fields):
        reference = data[field][case_index]
        physical = physical_predictions[field][local_index]
        aligned = aligned_predictions[field][local_index]
        physical_error = np.abs(physical - reference)
        aligned_error = np.abs(aligned - reference)
        lower = float(min(np.nanmin(reference), np.nanmin(physical), np.nanmin(aligned)))
        upper = float(max(np.nanmax(reference), np.nanmax(physical), np.nanmax(aligned)))
        error_upper = max(
            float(np.nanpercentile(np.concatenate((physical_error.ravel(), aligned_error.ravel())), 99.0)),
            np.finfo(float).eps,
        )
        panels = (
            (reference, "DSMC", cmap, lower, upper),
            (physical, "physical", cmap, lower, upper),
            (aligned, "shock-aligned", cmap, lower, upper),
            (physical_error, "physical |error|", "magma", 0.0, error_upper),
            (aligned_error, "aligned |error|", "magma", 0.0, error_upper),
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
        float(row["selected_global_relative_l2_percent"])
        for row in metrics
    }
    for i, field in enumerate(fields):
        for j, pressure in enumerate(pressures):
            matrix[i, j] = by_key[field, pressure]
    fig, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=15.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row = next(
                item for item in metrics
                if item["field"] == fields[i]
                and int(item["held_out_pressure_kpa"]) == pressures[j]
            )
            suffix = "A" if row["selected_method"] == "shock_aligned" else "P"
            axis.text(j, i, f"{matrix[i, j]:.2f}%\n({suffix})", ha="center", va="center")
    axis.set_xticks(range(len(pressures)), [f"{value} kPa" for value in pressures])
    axis.set_yticks(range(len(fields)), [FIELDS[name][0] for name in fields])
    axis.set_xlabel("held-out back pressure")
    axis.set_title("Development-selected full-field relative $L_2$ error (P/A)")
    fig.colorbar(image, ax=axis, label="relative error (%)")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_interpolation_method(
    pressure: np.ndarray,
    snapshots: np.ndarray,
    density_snapshots: np.ndarray,
    x_m: np.ndarray,
    development_indices: np.ndarray,
) -> tuple[str, list[dict[str, float | int | bool | str]]]:
    """Choose alignment by shock error with a two-point global guardrail."""
    metric_names = (
        "global_relative_l2_percent",
        "shock_window_relative_l2_percent",
        "gradient_weighted_relative_l2_percent",
    )
    scores = {
        name: {metric: [] for metric in metric_names}
        for name in ("physical_coordinate", "shock_aligned")
    }
    for validation_index in development_indices:
        fold_train = development_indices[development_indices != validation_index]
        query = float(pressure[validation_index])
        lower = pressure[fold_train][pressure[fold_train] < query]
        upper = pressure[fold_train][pressure[fold_train] > query]
        if not len(lower) or not len(upper):
            continue
        physical, _ = interpolate_nozzle_fields_locally(
            pressure, snapshots, fold_train, np.asarray([query])
        )
        aligned, _ = interpolate_nozzle_fields_shock_aligned(
            pressure, snapshots, density_snapshots, x_m,
            fold_train, np.asarray([query]),
        )
        for name, prediction in (
            ("physical_coordinate", physical[0]),
            ("shock_aligned", aligned[0]),
        ):
            errors = nozzle_field_error_metrics(
                snapshots[validation_index], prediction,
                density_snapshots[validation_index], x_m,
            )
            for metric in metric_names:
                scores[name][metric].append(errors[metric])
    if not scores["physical_coordinate"][metric_names[0]]:
        raise ValueError("no bracketed leave-one-case-out development folds are available")
    means = {
        name: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for name, metrics in scores.items()
    }
    aligned_is_eligible = (
        means["shock_aligned"]["global_relative_l2_percent"]
        <= means["physical_coordinate"]["global_relative_l2_percent"] + 2.0
    )
    aligned_improves_shock = (
        means["shock_aligned"]["shock_window_relative_l2_percent"]
        < means["physical_coordinate"]["shock_window_relative_l2_percent"]
    )
    method = "shock_aligned" if aligned_is_eligible and aligned_improves_shock else "physical_coordinate"
    rows = []
    for name in ("physical_coordinate", "shock_aligned"):
        global_errors = scores[name]["global_relative_l2_percent"]
        rows.append({
            "method": name,
            "loo_fold_count": len(global_errors),
            "loo_mean_global_relative_l2_percent": means[name]["global_relative_l2_percent"],
            "loo_max_global_relative_l2_percent": float(np.max(global_errors)),
            "loo_mean_shock_window_relative_l2_percent": means[name]["shock_window_relative_l2_percent"],
            "loo_mean_gradient_weighted_relative_l2_percent": means[name]["gradient_weighted_relative_l2_percent"],
            "global_guardrail_eligible": (
                True if name == "physical_coordinate" else aligned_is_eligible
            ),
            "selected": name == method,
            "selection_rule": (
                "select shock alignment when it lowers mean shock-window error and "
                "raises mean global error by no more than 2 percentage points"
            ),
        })
    return method, rows


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
    physical_predictions: dict[str, np.ndarray] = {}
    aligned_predictions: dict[str, np.ndarray] = {}
    selected_method_by_field: dict[str, str] = {}
    routes_by_pressure: dict[str, dict[str, object]] = {}
    for field in FIELDS:
        selected_method, rows = select_interpolation_method(
            pressure, data[field], data["density"], data["x_m"], development_indices,
        )
        selected_method_by_field[field] = selected_method
        selection_rows.extend({"field": field, **row} for row in rows)
        physical, physical_routes = interpolate_nozzle_fields_locally(
            pressure, data[field], development_indices, pressure[held_out_indices],
        )
        aligned, aligned_routes = interpolate_nozzle_fields_shock_aligned(
            pressure, data[field], data["density"], data["x_m"],
            development_indices, pressure[held_out_indices],
        )
        physical_predictions[field] = physical
        aligned_predictions[field] = aligned
        centerline = int(data["centerline_index"])
        for local_index, case_index in enumerate(held_out_indices):
            physical_route = physical_routes[local_index]
            aligned_route = aligned_routes[local_index]
            pressure_key = str(int(pressure[case_index]))
            routes_by_pressure.setdefault(pressure_key, {})[field] = {
                "selected_method": selected_method,
                "physical_coordinate": physical_route,
                "shock_aligned": aligned_route,
            }
            physical_error = nozzle_field_error_metrics(
                data[field][case_index], physical[local_index],
                data["density"][case_index], data["x_m"],
            )
            aligned_error = nozzle_field_error_metrics(
                data[field][case_index], aligned[local_index],
                data["density"][case_index], data["x_m"],
            )
            selected_error = (
                aligned_error if selected_method == "shock_aligned" else physical_error
            )
            metrics.append({
                "field": field,
                "held_out_pressure_kpa": int(pressure[case_index]),
                "bracket_lower_kpa": physical_route["lower_pressure_kpa"],
                "bracket_upper_kpa": physical_route["upper_pressure_kpa"],
                "bracket_gap_kpa": physical_route["bracket_gap_kpa"],
                "selected_method": selected_method,
                **{f"physical_{key}": value for key, value in physical_error.items()},
                **{f"aligned_{key}": value for key, value in aligned_error.items()},
                **{f"selected_{key}": value for key, value in selected_error.items()},
                "physical_centerline_relative_l2_percent": 100.0 * relative_l2(
                    data[field][case_index, centerline], physical[local_index, centerline]
                ),
                "aligned_centerline_relative_l2_percent": 100.0 * relative_l2(
                    data[field][case_index, centerline], aligned[local_index, centerline]
                ),
            })

    selection_path = result_dir / "nozzle_flowmllab_selection.csv"
    metrics_path = result_dir / "nozzle_flowmllab_heldout_metrics.csv"
    write_csv(selection_path, selection_rows)
    write_csv(metrics_path, metrics)

    profiles_path = figure_dir / "nozzle_back_pressure_profiles.png"
    summary_path = figure_dir / "nozzle_back_pressure_error_summary.png"
    draw_profiles(
        data, physical_predictions, aligned_predictions,
        selected_method_by_field, held_out_indices, profiles_path,
    )
    contour_paths = []
    for value in NOZZLE_HELD_OUT_KPA.astype(int):
        path = figure_dir / f"nozzle_back_pressure_P{value}_contours.png"
        draw_case_contours(
            data, physical_predictions, aligned_predictions,
            held_out_indices, int(value), path,
        )
        contour_paths.append(path)
    draw_error_summary(metrics, summary_path)

    manifest = {
        "schema_version": 3,
        "generator": "qa/run_nozzle_field_validation.py",
        "model": "development-selected physical-coordinate or shock-aligned interpolation baseline",
        "branch_input": "back pressure in kPa",
        "source_archive": "results/mahdavi_deeponet/nozzle_fields_15cases.npz",
        "source_archive_sha256": sha256(result_dir / "nozzle_fields_15cases.npz"),
        "development_pressures_kpa": list(DEVELOPMENT_PRESSURES_KPA),
        "held_out_pressures_kpa": NOZZLE_HELD_OUT_KPA.astype(int).tolist(),
        "selection_rule": (
            "select shock alignment when it lowers mean shock-window error and raises "
            "mean global error by no more than 2 percentage points on bracketed development LOO folds"
        ),
        "candidate_methods": ["physical_coordinate", "shock_aligned"],
        "shock_alignment": {
            "sensor": "strongest density gradient in downstream 58% of each source-field row",
            "query_surface": "linear pressure interpolation of source shock surfaces",
            "target_field_used_for_prediction": False,
        },
        "error_metrics": {
            "global": "relative L2 over the complete 101-by-31 field",
            "shock_window": "+/- 3 detected reference jump widths; evaluation only",
            "gradient_weighted": "relative L2 weighted by 1 + 4 normalized reference density-gradient magnitude",
        },
        "route_by_pressure": routes_by_pressure,
        "selected_method_by_field": selected_method_by_field,
        "held_out_cases_opened_after_selection": True,
        "selection_csv_sha256": sha256(selection_path),
        "metrics_csv_sha256": sha256(metrics_path),
        "retained_article_evidence_csv": "results/mahdavi_deeponet/nozzle_paper_field_errors.csv",
        "retained_article_evidence_csv_sha256": sha256(
            result_dir / "nozzle_paper_field_errors.csv"
        ),
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
        "selected_method_by_field": selected_method_by_field,
        "route_by_pressure": routes_by_pressure,
        "maximum_full_field_error_percent": max(
            float(row["selected_global_relative_l2_percent"]) for row in metrics
        ),
        "mean_full_field_error_percent": float(np.mean([
            float(row["selected_global_relative_l2_percent"]) for row in metrics
        ])),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
