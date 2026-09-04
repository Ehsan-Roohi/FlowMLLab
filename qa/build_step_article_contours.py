#!/usr/bin/env python3
"""Rebuild article-case micro-step contour comparisons from pinned Tecplot data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c3f211376b42b8dc30daad380eaef5e0ab800b5c"
ARTICLE_URL = "https://doi.org/10.1007/s10404-026-02899-8"
ARXIV_URL = "https://arxiv.org/abs/2509.17254"
FINAL_PAPER_SHA256 = "47d130cb4fae08608cef2578460665bebdf20a938b489eb5efc914302afcb980"
FINAL_SUPPLEMENT_SHA256 = "5fda3881c60bb77c5c6a7572e269822628e6e844cc438bf165c396f218ab4133"
KN1_REFERENCE = "data/kn/smoothed/Kn=1_smoothed.dat"


@dataclass(frozen=True)
class ArticleCase:
    case_id: str
    study: str
    nominal_parameter: str
    article_figure: int
    reference: str
    prediction: str


CASES = (
    ArticleCase(
        "Kn0p004",
        "Knudsen-number",
        "Kn=0.004",
        6,
        "data/kn/smoothed/Kn=0-004_smoothed.dat",
        "results/kn/best_run_seed_2035/tecplot_pred/pred_Kn=0-004.dat",
    ),
    ArticleCase(
        "Kn0p02",
        "Knudsen-number",
        "Kn=0.02",
        6,
        "data/kn/smoothed/Kn=0-02_smoothed.dat",
        "results/kn/best_run_seed_2035/tecplot_pred/pred_Kn=0-02.dat",
    ),
    ArticleCase(
        "H44",
        "step-height",
        "h/H=0.44",
        15,
        "data/height/smoothed/H44_smoothed.dat",
        "results/height/full_data/tecplot_pred/pred_H44_smoothed.dat",
    ),
    ArticleCase(
        "H67",
        "step-height",
        "h/H=0.67",
        15,
        "data/height/smoothed/H67_smoothed.dat",
        "results/height/full_data/tecplot_pred/pred_H67_smoothed.dat",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tecplot(path: Path) -> tuple[list[str], np.ndarray]:
    """Read quoted variable names and numeric point rows from Tecplot ASCII."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    variables: list[str] = []
    collecting = False
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("VARIABLES"):
            variables.extend(re.findall(r'"([^"]+)"', stripped))
            collecting = True
            continue
        if collecting and stripped.startswith('"'):
            variables.extend(re.findall(r'"([^"]+)"', stripped))
            continue
        collecting = False
        if not variables or stripped.upper().startswith(
            ("TITLE", "ZONE", "AUXDATA", "STRANDID", "I=", "DATAPACKING", "DT=")
        ):
            continue
        try:
            values = [float(token) for token in stripped.split()]
        except ValueError:
            continue
        if len(values) >= len(variables):
            rows.append(values[: len(variables)])
    if not variables or not rows:
        raise ValueError(f"no Tecplot point data found in {path}")
    return [name.upper() for name in variables], np.asarray(rows, dtype=np.float64)


def column(names: list[str], data: np.ndarray, name: str) -> np.ndarray:
    try:
        return data[:, names.index(name.upper())]
    except ValueError as exc:
        raise ValueError(f"{name!r} is missing from Tecplot variables") from exc


def parent_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """Map point data to its parent Cartesian grid without interpolating solids."""
    points = np.column_stack((x, y))
    unique_points, first = np.unique(points, axis=0, return_index=True)
    unique_values = values[first]
    xs = np.unique(unique_points[:, 0])
    ys = np.unique(unique_points[:, 1])
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float64)
    ix = np.searchsorted(xs, unique_points[:, 0])
    iy = np.searchsorted(ys, unique_points[:, 1])
    grid[iy, ix] = unique_values
    return xs, ys, np.ma.masked_invalid(grid)


def geometry_contract(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xs, ys = np.unique(x), np.unique(y)
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    x0, x1 = float(xs.min() - dx / 2), float(xs.max() + dx / 2)
    y0, y1 = float(ys.min() - dy / 2), float(ys.max() + dy / 2)
    length, height = x1 - x0, y1 - y0
    bottom = np.isclose(y, ys.min(), rtol=0.0, atol=0.0)
    step_x = float(np.min(x[bottom]) - dx / 2)
    upstream = x < step_x - dx / 2
    step_y = float(np.min(y[upstream]) - dy / 2)
    aspect = length / height
    if not 4.99 < aspect < 5.01:
        raise ValueError(f"unexpected data-domain aspect ratio L/H={aspect:.8f}")
    return {
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "L": length,
        "H": height,
        "L_over_H": aspect,
        "step_x_over_H": (step_x - x0) / height,
        "step_y_over_H": (step_y - y0) / height,
    }


def add_geometry(axis: plt.Axes, geometry: dict[str, float]) -> None:
    sx, sy = geometry["step_x_over_H"], geometry["step_y_over_H"]
    xmax = geometry["L_over_H"]
    axis.fill([0, sx, sx, 0], [0, 0, sy, sy], color="#dedede", zorder=8)
    axis.plot([0, sx, sx, xmax], [sy, sy, 0, 0], color="black", lw=0.8, zorder=9)
    axis.plot([0, xmax], [1, 1], color="black", lw=0.8, zorder=9)
    axis.set_xlim(0, xmax)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/H$")
    axis.set_ylabel(r"$y/H$")
    axis.tick_params(labelsize=8)


def last_negative_zero_crossing(x: np.ndarray, u: np.ndarray) -> float:
    negative = np.flatnonzero(u < 0.0)
    if len(negative) == 0 or negative[-1] + 1 >= len(u):
        return float("nan")
    left = int(negative[-1])
    right = left + 1
    denominator = u[right] - u[left]
    if denominator == 0.0:
        return float(x[left])
    return float(x[left] - u[left] * (x[right] - x[left]) / denominator)


def reattachment_length_over_l(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    geometry: dict[str, float],
) -> float:
    bottom = np.isclose(y, np.min(y), rtol=0.0, atol=0.0)
    order = np.argsort(x[bottom])
    crossing = last_negative_zero_crossing(x[bottom][order], u[bottom][order])
    step_x = geometry["x0"] + geometry["step_x_over_H"] * geometry["H"]
    return (crossing - step_x) / geometry["L"]


def safe_relative_l2(delta: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(delta) / denominator) if denominator else float("nan")


def compare_case(source: Path, case: ArticleCase) -> dict[str, object]:
    reference_path = source / case.reference
    prediction_path = source / case.prediction
    ref_names, ref = read_tecplot(reference_path)
    pred_names, pred = read_tecplot(prediction_path)
    if len(ref) != len(pred):
        raise ValueError(f"{case.case_id}: reference/prediction row-count mismatch")
    x, y = column(ref_names, ref, "X"), column(ref_names, ref, "Y")
    px, py = column(pred_names, pred, "X"), column(pred_names, pred, "Y")
    coordinate_delta = float(np.max(np.abs(np.column_stack((x - px, y - py)))))
    if coordinate_delta > 1.0e-15:
        raise ValueError(f"{case.case_id}: coordinates differ by {coordinate_delta:.3e}")
    truth = np.column_stack((column(ref_names, ref, "U"), column(ref_names, ref, "V")))
    prediction = np.column_stack(
        (column(pred_names, pred, "U"), column(pred_names, pred, "V"))
    )
    delta = prediction - truth
    vortex = truth[:, 0] < 0.0
    predicted_vortex = prediction[:, 0] < 0.0
    union = np.count_nonzero(vortex | predicted_vortex)
    vortex_iou = np.count_nonzero(vortex & predicted_vortex) / union if union else 1.0
    geometry = geometry_contract(x, y)
    unique_points = len(np.unique(np.column_stack((x, y)), axis=0))
    metrics: dict[str, object] = {
        "case_id": case.case_id,
        "study": case.study,
        "nominal_parameter": case.nominal_parameter,
        "article_figure": case.article_figure,
        "point_count": len(ref),
        "unique_point_count": unique_points,
        "max_coordinate_delta": coordinate_delta,
        "combined_relative_l2_percent": 100.0 * safe_relative_l2(delta, truth),
        "u_relative_l2_percent": 100.0 * safe_relative_l2(delta[:, 0], truth[:, 0]),
        "v_relative_l2_percent": 100.0 * safe_relative_l2(delta[:, 1], truth[:, 1]),
        "vortex_relative_l2_percent": 100.0 * safe_relative_l2(delta[vortex], truth[vortex]),
        "negative_u_iou_percent": 100.0 * vortex_iou,
        "dsmc_reattachment_length_over_L": reattachment_length_over_l(
            x, y, truth[:, 0], geometry
        ),
        "stored_nn_reattachment_length_over_L": reattachment_length_over_l(
            x, y, prediction[:, 0], geometry
        ),
        "abs_u_error_p99": float(np.percentile(np.abs(delta[:, 0]), 99)),
        "abs_u_error_max": float(np.max(np.abs(delta[:, 0]))),
        "abs_v_error_p99": float(np.percentile(np.abs(delta[:, 1]), 99)),
        "abs_v_error_max": float(np.max(np.abs(delta[:, 1]))),
        "L_over_H_from_grid": geometry["L_over_H"],
        "step_x_over_H_from_grid": geometry["step_x_over_H"],
        "step_y_over_H_from_grid": geometry["step_y_over_H"],
        "reference_sha256": sha256(reference_path),
        "prediction_sha256": sha256(prediction_path),
        "reference_path": case.reference,
        "prediction_path": case.prediction,
    }
    return {
        "x": x,
        "y": y,
        "truth": truth,
        "prediction": prediction,
        "geometry": geometry,
        "metrics": metrics,
    }


def plot_case(
    result: dict[str, object],
    output: Path,
    *,
    prediction_label: str = "DeepONet",
    title_prefix: str | None = None,
    metric_label: str = "stored-field vector relative L2",
    footnote: str | None = None,
    footnote_color: str = "#7a1f1f",
) -> None:
    x = np.asarray(result["x"])
    y = np.asarray(result["y"])
    truth = np.asarray(result["truth"])
    prediction = np.asarray(result["prediction"])
    geometry = dict(result["geometry"])
    metrics = dict(result["metrics"])
    xn = (x - geometry["x0"]) / geometry["H"]
    yn = (y - geometry["y0"]) / geometry["H"]

    xs = np.unique(xn)
    ys = np.unique(yn)
    # Tecplot coordinates are rounded in ASCII, so nominally uniform spacings
    # differ at about 4e-5 H. Matplotlib streamplot requires exact uniformity;
    # regularize coordinates only for streamline tracing, never field values.
    stream_x = np.linspace(xs[0], xs[-1], len(xs))
    stream_y = np.linspace(ys[0], ys[-1], len(ys))
    truth_velocity_grids = (
        parent_grid(xn, yn, truth[:, 0])[2],
        parent_grid(xn, yn, truth[:, 1])[2],
    )
    prediction_velocity_grids = (
        parent_grid(xn, yn, prediction[:, 0])[2],
        parent_grid(xn, yn, prediction[:, 1])[2],
    )

    fig = plt.figure(figsize=(15.6, 7.0), facecolor="white")
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(1.0, 0.12, 1.0, 0.12),
        left=0.055,
        right=0.985,
        bottom=0.105,
        top=0.89,
        hspace=0.60,
        wspace=0.14,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, col]) for col in range(3)],
            [fig.add_subplot(grid[2, col]) for col in range(3)],
        ]
    )
    color_axes = (
        (fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2])),
        (fig.add_subplot(grid[3, :2]), fig.add_subplot(grid[3, 2])),
    )
    labels = (r"$U$", r"$V$")
    for row, label in enumerate(labels):
        truth_component = truth[:, row]
        pred_component = prediction[:, row]
        error = np.abs(pred_component - truth_component)
        vmin = float(min(truth_component.min(), pred_component.min()))
        vmax = float(max(truth_component.max(), pred_component.max()))
        levels = np.linspace(vmin, vmax, 45)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        truth_grid = parent_grid(xn, yn, truth_component)[2]
        pred_grid = parent_grid(xn, yn, pred_component)[2]
        err_grid = parent_grid(xn, yn, error)[2]
        field_artist = None
        for col, (field, title) in enumerate(
            ((truth_grid, f"DSMC {label}"), (pred_grid, f"{prediction_label} {label}"))
        ):
            field_artist = axes[row, col].contourf(
                xs, ys, field, levels=levels, cmap="RdBu_r", norm=norm, extend="both"
            )
            if row == 0 and np.nanmin(field) < 0.0 < np.nanmax(field):
                axes[row, col].contour(
                    xs, ys, field, levels=[0.0], colors="black", linewidths=0.65
                )
            velocity_grids = (
                truth_velocity_grids if col == 0 else prediction_velocity_grids
            )
            axes[row, col].streamplot(
                stream_x,
                stream_y,
                velocity_grids[0],
                velocity_grids[1],
                color="#242424",
                density=0.55,
                linewidth=0.35,
                arrowsize=0.55,
            )
            bubble_seeds = np.column_stack(
                (
                    np.full(7, geometry["step_x_over_H"] + 0.12),
                    np.linspace(0.045, max(0.055, 0.90 * geometry["step_y_over_H"]), 7),
                )
            )
            axes[row, col].streamplot(
                stream_x,
                stream_y,
                velocity_grids[0],
                velocity_grids[1],
                start_points=bubble_seeds,
                integration_direction="both",
                color="#111111",
                density=1.0,
                linewidth=0.42,
                arrowsize=0.55,
            )
            axes[row, col].set_title(title, fontsize=10)
            add_geometry(axes[row, col], geometry)
        assert field_artist is not None
        fig.colorbar(
            field_artist,
            cax=color_axes[row][0],
            orientation="horizontal",
            label=f"{label} (source velocity units; shared DSMC/NN scale)",
        )

        clip = float(np.percentile(error, 99))
        error_artist = axes[row, 2].contourf(
            xs,
            ys,
            err_grid,
            levels=np.linspace(0.0, clip, 45),
            cmap="magma",
            extend="max",
        )
        axes[row, 2].set_title(rf"$|\Delta {label.strip('$')}|$ (clipped at p99)", fontsize=10)
        add_geometry(axes[row, 2], geometry)
        fig.colorbar(
            error_artist,
            cax=color_axes[row][1],
            orientation="horizontal",
            label=f"absolute error (source velocity units)",
        )

    prefix = title_prefix or f"Article Fig. {metrics['article_figure']} reproduction"
    title = (
        f"{prefix} — "
        f"{metrics['nominal_parameter']} | stored-field vector relative L2 = "
        f"{metrics['combined_relative_l2_percent']:.3f}% | "
        f"reverse-flow IoU = {metrics['negative_u_iou_percent']:.2f}%"
    )
    title = title.replace("stored-field vector relative L2", metric_label)
    fig.suptitle(title, fontsize=13)
    if footnote is None:
        footnote = (
            "DSMC and DeepONet fields are evaluated on the same source grid; "
            "axes preserve the measured L/H=5 domain."
        )
    fig.text(
        0.5,
        0.018,
        footnote,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=footnote_color,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


def plot_reference_only_kn1(source: Path, output: Path) -> dict[str, object]:
    """Plot the exact Figure-6 Kn=1 DSMC case without inventing an NN field."""
    reference_path = source / KN1_REFERENCE
    names, data = read_tecplot(reference_path)
    x, y = column(names, data, "X"), column(names, data, "Y")
    velocity = np.column_stack((column(names, data, "U"), column(names, data, "V")))
    geometry = geometry_contract(x, y)
    xn = (x - geometry["x0"]) / geometry["H"]
    yn = (y - geometry["y0"]) / geometry["H"]
    xs, ys = np.unique(xn), np.unique(yn)
    stream_x = np.linspace(xs[0], xs[-1], len(xs))
    stream_y = np.linspace(ys[0], ys[-1], len(ys))
    velocity_grids = (
        parent_grid(xn, yn, velocity[:, 0])[2],
        parent_grid(xn, yn, velocity[:, 1])[2],
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 3.8), facecolor="white")
    for index, label in enumerate((r"$U$", r"$V$")):
        component = velocity[:, index]
        vmin, vmax = float(component.min()), float(component.max())
        norm = (
            TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            if vmin < 0.0 < vmax
            else Normalize(vmin=vmin, vmax=vmax)
        )
        artist = axes[index].contourf(
            xs,
            ys,
            parent_grid(xn, yn, component)[2],
            levels=np.linspace(vmin, vmax, 45),
            cmap="RdBu_r",
            norm=norm,
            extend="both",
        )
        axes[index].streamplot(
            stream_x,
            stream_y,
            velocity_grids[0],
            velocity_grids[1],
            color="#242424",
            density=0.58,
            linewidth=0.35,
            arrowsize=0.55,
        )
        axes[index].set_title(f"DSMC {label} — exact Kn=1 source field", fontsize=10)
        add_geometry(axes[index], geometry)
        fig.colorbar(
            artist,
            ax=axes[index],
            orientation="horizontal",
            pad=0.24,
            fraction=0.07,
            ticks=np.linspace(vmin, vmax, 6),
            format="%.1f",
            label=f"{label} (source velocity units)",
        )
    fig.suptitle(
        "Article Fig. 6 case coverage — Kn=1 DSMC reference only",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "The pinned article repository has no stored Kn=1 neural prediction; the missing panel is not inferred from the paper image.",
        ha="center",
        fontsize=8.7,
        color="#7a1f1f",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.23, top=0.85, wspace=0.13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "reference_path": KN1_REFERENCE,
        "reference_sha256": sha256(reference_path),
        "point_count": len(data),
        "L_over_H_from_grid": geometry["L_over_H"],
        "step_x_over_H_from_grid": geometry["step_x_over_H"],
        "step_y_over_H_from_grid": geometry["step_y_over_H"],
    }


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    numeric_order = (
        "case_id",
        "study",
        "nominal_parameter",
        "article_figure",
        "point_count",
        "unique_point_count",
        "max_coordinate_delta",
        "combined_relative_l2_percent",
        "u_relative_l2_percent",
        "v_relative_l2_percent",
        "vortex_relative_l2_percent",
        "negative_u_iou_percent",
        "dsmc_reattachment_length_over_L",
        "stored_nn_reattachment_length_over_L",
        "abs_u_error_p99",
        "abs_u_error_max",
        "abs_v_error_p99",
        "abs_v_error_max",
        "L_over_H_from_grid",
        "step_x_over_H_from_grid",
        "step_y_over_H_from_grid",
        "reference_sha256",
        "prediction_sha256",
        "reference_path",
        "prediction_path",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=numeric_order, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_case_coverage(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_head(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot identify source checkout commit: {source}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Pinned checkout of Ehsan-Roohi/roohi-step-dnn-mahdavi",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.root.resolve()
    head = source_head(source)
    if head != SOURCE_COMMIT:
        raise ValueError(f"expected source commit {SOURCE_COMMIT}, found {head}")

    result_dir = root / "results" / "mahdavi_deeponet"
    figure_dir = result_dir / "step_article_contours"
    rows: list[dict[str, object]] = []
    outputs: dict[str, str] = {}
    for case in CASES:
        result = compare_case(source, case)
        output = figure_dir / f"article_figure_{case.article_figure:02d}_{case.case_id}.png"
        plot_case(result, output)
        metrics = dict(result["metrics"])
        rows.append(metrics)
        outputs[output.relative_to(root).as_posix()] = sha256(output)

    kn1_output = figure_dir / "article_figure_06_Kn1_DSMC_only.png"
    kn1_metadata = plot_reference_only_kn1(source, kn1_output)
    outputs[kn1_output.relative_to(root).as_posix()] = sha256(kn1_output)

    coverage_rows = [
        {
            "article_figure": 6,
            "case_id": "Kn0p004",
            "nominal_parameter": "Kn=0.004",
            "reference_field": "available",
            "stored_nn_field": "available",
            "reproduction_status": "DSMC-NN-error comparison rebuilt",
        },
        {
            "article_figure": 6,
            "case_id": "Kn0p02",
            "nominal_parameter": "Kn=0.02",
            "reference_field": "available",
            "stored_nn_field": "available",
            "reproduction_status": "DSMC-NN-error comparison rebuilt",
        },
        {
            "article_figure": 6,
            "case_id": "Kn1",
            "nominal_parameter": "Kn=1",
            "reference_field": "available",
            "stored_nn_field": "missing from pinned repository",
            "reproduction_status": "DSMC-only contour rebuilt; NN comparison not fabricated",
        },
        {
            "article_figure": 15,
            "case_id": "H44",
            "nominal_parameter": "h/H=0.44",
            "reference_field": "available",
            "stored_nn_field": "available",
            "reproduction_status": "DSMC-NN-error comparison rebuilt",
        },
        {
            "article_figure": 15,
            "case_id": "H67",
            "nominal_parameter": "h/H=0.67",
            "reference_field": "available",
            "stored_nn_field": "available",
            "reproduction_status": "DSMC-NN-error comparison rebuilt",
        },
    ]
    coverage_path = result_dir / "step_article_case_coverage.csv"
    write_case_coverage(coverage_path, coverage_rows)

    metrics_path = result_dir / "step_article_contour_metrics.csv"
    write_metrics(metrics_path, rows)
    manifest = {
        "schema_version": 2,
        "article": {
            "doi": ARTICLE_URL,
            "arxiv": ARXIV_URL,
            "version_used_for_figure_numbering": "final Springer PDF supplied by the authors",
            "final_paper_sha256": FINAL_PAPER_SHA256,
            "final_supplement_sha256": FINAL_SUPPLEMENT_SHA256,
        },
        "source_repository": "https://github.com/Ehsan-Roohi/roohi-step-dnn-mahdavi",
        "source_commit": SOURCE_COMMIT,
        "comparison_contract": {
            "coordinates": "the prediction and DSMC point coordinates must match",
            "geometry": "solid cells remain masked; no interpolation fills the step",
            "aspect_ratio": "axes use x/H and y/H with equal data aspect; grid L/H must be 5",
            "color_limits": "DSMC and stored NN share one full-range linear scale per velocity component",
            "error_limits": "absolute-error panels use the 99th percentile with over-range values marked by colorbar extension",
            "streamlines": "velocity samples are unchanged; rounded Tecplot coordinates are regularized by less than 4e-5 H solely because Matplotlib streamplot requires exact uniform spacing",
        },
        "evidence_boundary": (
            "Stored upstream NN contours are privileged-input reconstruction evidence because "
            "the upstream test path constructs local patches from the target DSMC U,V field."
        ),
        "article_repository_inconsistencies": [
            (
                "The article problem-definition text lists Kn=0.2 as held out, while Figure 6 "
                "and its discussion label Kn=0.02; the pinned source code and stored prediction "
                "also use Kn=0.02. This reproduction follows the figure caption and repository artifact."
            ),
            (
                "Final article Figure 6 shows Kn=1, but the pinned repository contains no stored "
                "Kn=1 prediction. The exact DSMC source contour is included, while the neural "
                "comparison is explicitly unavailable rather than fabricated."
            ),
            (
                "The H44 and H67 titles retain the nominal article case labels. The solid mask "
                "in the supplied parent grid implies step-top locations of about 0.408H and "
                "0.658H, respectively; both values are recorded per case rather than silently "
                "redrawing the geometry."
            ),
        ],
        "metrics_file": metrics_path.relative_to(root).as_posix(),
        "metrics_sha256": sha256(metrics_path),
        "case_coverage_file": coverage_path.relative_to(root).as_posix(),
        "case_coverage_sha256": sha256(coverage_path),
        "kn1_reference_only": kn1_metadata,
        "figure_sha256": outputs,
        "cases": rows,
    }
    manifest_path = result_dir / "step_article_contour_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_commit": head,
                "metrics": metrics_path.relative_to(root).as_posix(),
                "manifest": manifest_path.relative_to(root).as_posix(),
                "figures": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
