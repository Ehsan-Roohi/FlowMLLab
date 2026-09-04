#!/usr/bin/env python3
"""Regenerate Week-10 article-backed DSMC surrogate results and figures."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.aescte_dsmc import (  # noqa: E402
    CAVITY_LID_SPEEDS_MS,
    CAVITY_TEST_KNUDSEN,
    cavity_case,
    fit_pod_polynomial_operator,
    load_cavity_archive,
    load_shock_archive,
    logarithmic_kn_prediction,
    maxwell_speed_pdf,
    normalized_rmse,
    predict_pod_polynomial_operator,
    relative_l2,
    sha256,
)


RESULTS = ROOT / "results" / "aescte_dsmc"
PALETTE = {"dsmc": "#14213D", "model": "#D1495B", "accent": "#1B998B"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def shock_prediction(
    data: dict[str, np.ndarray],
    *,
    test_mach: float,
    training_mach: list[float],
    fields: list[str],
    rank: int,
    degree: int,
) -> dict[str, np.ndarray]:
    mach = data["mach"]
    train = np.flatnonzero(np.isin(np.round(mach, 8), np.round(training_mach, 8)))
    result: dict[str, np.ndarray] = {}
    for field in fields:
        model = fit_pod_polynomial_operator(
            mach, data[field], train, rank=rank, degree=degree
        )
        result[field] = predict_pod_polynomial_operator(model, [test_mach])[0]
    return result


def cavity_metrics_and_predictions(data: dict[str, np.ndarray]):
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    for lid in CAVITY_LID_SPEEDS_MS:
        for kn in CAVITY_TEST_KNUDSEN:
            prediction = logarithmic_kn_prediction(
                data,
                lid_speed_ms=float(lid),
                test_knudsen=float(kn),
                fields=("u_ms", "v_ms", "temperature_k", "qx", "qy", "txy"),
            )
            predictions[(int(lid), float(kn))] = prediction
            reference_index = cavity_case(data, lid, kn)
            scales = {
                "u_ms": float(lid),
                "v_ms": float(lid),
                "temperature_k": 50.0,
                "qx": float(np.max(np.abs(data["qx"][reference_index]))),
                "qy": float(np.max(np.abs(data["qy"][reference_index]))),
                "txy": float(np.max(np.abs(data["txy"][reference_index]))),
            }
            for field, scale in scales.items():
                rows.append({
                    "case_family": "cavity",
                    "lid_speed_ms": int(lid),
                    "test_parameter": f"Kn={kn:g}",
                    "field": field,
                    "metric": "NRMSE_percent",
                    "value": normalized_rmse(
                        data[field][reference_index], np.asarray(prediction[field]), scale
                    ),
                    "normalization": (
                        "lid_speed" if field in {"u_ms", "v_ms"}
                        else "wall_delta_T" if field == "temperature_k"
                        else "target_max_abs"
                    ),
                })
    return rows, predictions


def shock_metrics(data, predictions, test_mach, family, fields):
    target_index = int(np.flatnonzero(np.isclose(data["mach"], test_mach))[0])
    rows = []
    for field in fields:
        rows.append({
            "case_family": family,
            "lid_speed_ms": "",
            "test_parameter": f"Mach={test_mach:g}",
            "field": field,
            "metric": "relative_L2_percent",
            "value": relative_l2(data[field][target_index], predictions[field]),
            "normalization": "target_L2",
        })
    return rows


def draw_cavity_fields(data, predictions, output):
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 7.3), constrained_layout=True)
    for row, lid in enumerate(CAVITY_LID_SPEEDS_MS.astype(int)):
        kn = 0.05
        index = cavity_case(data, lid, kn)
        pred = predictions[(lid, kn)]
        speed_ref = np.hypot(data["u_ms"][index], data["v_ms"][index])
        speed_pred = np.hypot(pred["u_ms"], pred["v_ms"])
        speed_limits = (float(min(speed_ref.min(), speed_pred.min())), float(max(speed_ref.max(), speed_pred.max())))
        temperature_limits = (
            float(min(data["temperature_k"][index].min(), np.min(pred["temperature_k"]))),
            float(max(data["temperature_k"][index].max(), np.max(pred["temperature_k"]))),
        )
        panels = (
            (speed_ref, "DSMC speed", "viridis", speed_limits),
            (speed_pred, "log-Kn surrogate speed", "viridis", speed_limits),
            (data["temperature_k"][index], "DSMC temperature", "coolwarm", temperature_limits),
            (pred["temperature_k"], "log-Kn surrogate temperature", "coolwarm", temperature_limits),
        )
        for column, (values, title, cmap, limits) in enumerate(panels):
            ax = axes[row, column]
            image = ax.imshow(
                values, origin="lower", extent=(0, 1, 0, 1), cmap=cmap,
                aspect="equal", vmin=limits[0], vmax=limits[1],
            )
            ax.set_title(title)
            ax.set_xlabel("x/L")
            if column == 0:
                ax.set_ylabel(f"U_lid={lid} m/s\ny/L")
            fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle("Article case reproduced from complete DSMC fields: held-out Kn=0.05", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_cavity_profiles(data, predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    y = data["y"][0, :, 25]
    for row, lid in enumerate(CAVITY_LID_SPEEDS_MS.astype(int)):
        for kn, linestyle in ((0.05, "-"), (0.5, "--")):
            index = cavity_case(data, lid, kn)
            pred = predictions[(lid, kn)]
            axes[row, 0].plot(data["u_ms"][index, :, 25] / lid, y, color=PALETTE["dsmc"], ls=linestyle, lw=2)
            axes[row, 0].plot(pred["u_ms"][:, 25] / lid, y, color=PALETTE["model"], ls=linestyle, lw=1.6)
            axes[row, 1].plot(data["temperature_k"][index, :, 25], y, color=PALETTE["dsmc"], ls=linestyle, lw=2)
            axes[row, 1].plot(pred["temperature_k"][:, 25], y, color=PALETTE["model"], ls=linestyle, lw=1.6)
        axes[row, 0].set(ylabel=f"U_lid={lid} m/s\ny/L", xlabel="U/U_lid", title="vertical centerline velocity")
        axes[row, 1].set(xlabel="temperature (K)", title="vertical centerline temperature")
        for ax in axes[row]:
            ax.grid(alpha=0.25)
    axes[0, 0].plot([], [], color=PALETTE["dsmc"], lw=2, label="DSMC")
    axes[0, 0].plot([], [], color=PALETTE["model"], lw=1.6, label="surrogate")
    axes[0, 0].plot([], [], color="gray", ls="-", label="Kn=0.05")
    axes[0, 0].plot([], [], color="gray", ls="--", label="Kn=0.5")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Independent cavity profiles at both supplied held-out Kn values", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_diatomic(data, prediction_17, prediction_14, output):
    fields = ["rotational_temperature", "translational_temperature", "normalized_velocity"]
    labels = [r"$T_{rot}^*$", r"$T_{tr}^*$", r"$U^*$"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), constrained_layout=True, sharex=True)
    for row, (mach, prediction, subtitle) in enumerate(((1.7, prediction_17, "interpolation"), (1.4, prediction_14, "extrapolation"))):
        index = int(np.flatnonzero(np.isclose(data["mach"], mach))[0])
        x = data["x_over_lambda"][index]
        for column, (field, label) in enumerate(zip(fields, labels)):
            axes[row, column].plot(x, data[field][index], color=PALETTE["dsmc"], lw=2.3, label="DSMC")
            axes[row, column].plot(x, prediction[field], color=PALETTE["model"], lw=1.8, ls="--", label="POD branch-trunk")
            axes[row, column].set(title=f"Mach {mach:g} {subtitle}: {label}", xlabel=r"$x/\lambda$")
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("normalized value")
    axes[1, 0].set_ylabel("normalized value")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Diatomic nitrogen shock: overshoot and delayed rotational relaxation", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_monatomic_relaxation(data, prediction, output):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True)
    target = int(np.flatnonzero(np.isclose(data["mach"], 1.7))[0])
    x = data["x_over_lambda"][target]
    for ax, field, label in zip(axes[:3], ["density", "velocity", "temperature"], ["density", "velocity", "temperature"]):
        ax.plot(x, data[field][target], color=PALETTE["dsmc"], lw=2.2, label="DSMC")
        ax.plot(x, prediction[field], color=PALETTE["model"], lw=1.7, ls="--", label="operator")
        ax.set(title=f"held-out Mach 1.7: {label}", xlabel=r"$x/\lambda$")
        ax.grid(alpha=0.25)
    speed = np.linspace(0, 1250, 500)
    argon_mass = 39.948e-3 / 6.02214076e23
    for temperature, color in ((325.0, PALETTE["accent"]), (500.0, PALETTE["model"])):
        axes[3].plot(speed, maxwell_speed_pdf(speed, temperature, argon_mass), color=color, lw=2, label=f"{temperature:g} K")
    axes[3].set(title="Maxwell speed PDF", xlabel="speed (m/s)", ylabel="probability density")
    axes[3].legend(frameon=False)
    axes[3].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Monatomic shock interpolation and equilibrium-distribution check", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_summary(cavity_figure, diatomic_figure, metrics, output):
    from PIL import Image
    left = Image.open(cavity_figure).convert("RGB")
    right = Image.open(diatomic_figure).convert("RGB")
    left.thumbnail((1300, 620)); right.thumbnail((1300, 620))
    canvas = Image.new("RGB", (1320, 1240), "white")
    canvas.paste(left, ((1320-left.width)//2, 0))
    canvas.paste(right, ((1320-right.width)//2, 620))
    canvas.save(output, quality=94)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cavity_path = RESULTS / "cavity_fields_14cases.npz"
    diatomic_path = RESULTS / "diatomic_shock_6cases.npz"
    monatomic_path = RESULTS / "monatomic_shock_7cases.npz"
    for path in (cavity_path, diatomic_path, monatomic_path):
        if not path.is_file():
            raise FileNotFoundError(f"run qa/build_week10_aescte_dsmc_data.py first: {path}")
    cavity = load_cavity_archive(cavity_path)
    diatomic = load_shock_archive(diatomic_path)
    monatomic = load_shock_archive(monatomic_path)

    rows, cavity_predictions = cavity_metrics_and_predictions(cavity)
    diatomic_fields = ["rotational_temperature", "translational_temperature", "normalized_velocity"]
    prediction_17 = shock_prediction(diatomic, test_mach=1.7, training_mach=[1.4, 1.5, 1.6, 1.8, 1.9], fields=diatomic_fields, rank=4, degree=3)
    prediction_14 = shock_prediction(diatomic, test_mach=1.4, training_mach=[1.5, 1.6, 1.7, 1.8, 1.9], fields=diatomic_fields, rank=4, degree=2)
    rows += shock_metrics(diatomic, prediction_17, 1.7, "diatomic_interpolation", diatomic_fields)
    rows += shock_metrics(diatomic, prediction_14, 1.4, "diatomic_extrapolation", diatomic_fields)

    monatomic_fields = ["density", "velocity", "temperature"]
    monatomic_prediction = shock_prediction(monatomic, test_mach=1.7, training_mach=[1.4, 1.5, 1.6, 1.8, 1.9, 2.0], fields=monatomic_fields, rank=5, degree=3)
    rows += shock_metrics(monatomic, monatomic_prediction, 1.7, "monatomic_interpolation", monatomic_fields)

    speed = np.linspace(0.0, 3000.0, 10000)
    argon_mass = 39.948e-3 / 6.02214076e23
    mb_errors = []
    for temperature in (200, 275, 325, 350, 425, 500, 575, 650):
        integral = float(np.trapezoid(maxwell_speed_pdf(speed, temperature, argon_mass), speed))
        mb_errors.append(abs(integral - 1.0))
    rows.append({
        "case_family": "argon_relaxation",
        "lid_speed_ms": "",
        "test_parameter": "T=325 K",
        "field": "Maxwell_speed_PDF",
        "metric": "normalization_absolute_error",
        "value": max(mb_errors),
        "normalization": "unit_integral",
    })
    metrics_path = RESULTS / "week10_validation_metrics.csv"
    write_csv(metrics_path, rows)

    cavity_figure = RESULTS / "cavity_kn005_reproduction.png"
    cavity_profiles = RESULTS / "cavity_validation_profiles.png"
    diatomic_figure = RESULTS / "diatomic_shock_reproduction.png"
    mono_figure = RESULTS / "monatomic_relaxation_reproduction.png"
    summary_figure = RESULTS / "week10_dsmc_reproduction_summary.png"
    draw_cavity_fields(cavity, cavity_predictions, cavity_figure)
    draw_cavity_profiles(cavity, cavity_predictions, cavity_profiles)
    draw_diatomic(diatomic, prediction_17, prediction_14, diatomic_figure)
    draw_monatomic_relaxation(monatomic, monatomic_prediction, mono_figure)
    draw_summary(cavity_figure, diatomic_figure, rows, summary_figure)

    primary_cavity = [row["value"] for row in rows if row["case_family"] == "cavity" and row["field"] in {"u_ms", "v_ms", "temperature_k"}]
    shock_values = [row["value"] for row in rows if str(row["case_family"]).startswith(("diatomic", "monatomic"))]
    summary = {
        "schema_version": 1,
        "status": "pass",
        "cavity_primary_max_nrmse_percent": max(primary_cavity),
        "cavity_primary_mean_nrmse_percent": float(np.mean(primary_cavity)),
        "shock_max_relative_l2_percent": max(shock_values),
        "shock_mean_relative_l2_percent": float(np.mean(shock_values)),
        "diatomic_mach_2_target_available": False,
        "cavity_100_ms_target_available": False,
        "data_manifest_sha256": sha256(RESULTS / "data_manifest.json"),
        "metrics_sha256": sha256(metrics_path),
        "figures_sha256": {
            path.name: sha256(path)
            for path in (cavity_figure, cavity_profiles, diatomic_figure, mono_figure, summary_figure)
        },
    }
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if summary["cavity_primary_max_nrmse_percent"] > 2.0:
        raise SystemExit("cavity primary-variable gate failed")
    if summary["shock_max_relative_l2_percent"] > 1.5:
        raise SystemExit("shock profile gate failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
