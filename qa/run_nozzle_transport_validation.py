#!/usr/bin/env python3
"""Select registered nozzle models on development cases, then assess regressions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (  # noqa: E402
    NOZZLE_HELD_OUT_KPA, detect_density_shock_surface, interpolate_nozzle_fields_locally,
    interpolate_nozzle_fields_shock_aligned, load_nozzle_fields, nozzle_field_error_metrics,
)
from flowmllab.nozzle_transport import (  # noqa: E402
    FIELDS, compression_surface, fit_transport_pod, predict_transport_pod,
    predict_with_symmetry, save_transport_model,
)

LABELS = ("Density (source units)", r"$U$ (m/s)", r"$V$ (m/s)",
          "Temperature (K)", "Mach number", "Pressure (source units)")
SEEDS = (690, 691, 692)
RANKS = (2, 4, 6)
WIDTHS = (8, 16)


def sha256(path: Path) -> str:
    """Hash a retained file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write portable machine-readable evidence."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def field_errors(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """One unweighted full-grid relative L2 percentage per physical field."""
    return 100 * np.sqrt(np.sum((prediction - reference)**2, axis=(0, 1))
                         / np.sum(reference**2, axis=(0, 1)))


def select_models(data: dict, output: Path) -> dict:
    """Refit registration, scaling, POD, and branch inside every development fold."""
    indices = np.flatnonzero(~np.isin(data["pressure_kpa"], NOZZLE_HELD_OUT_KPA))
    pressure = data["pressure_kpa"][indices].copy()
    fields = np.stack([data[name][indices] for name in FIELDS], axis=-1)
    x, y = data["x_m"], data["y_m"]
    rows, folds, candidates = [], [], []
    for rank in RANKS:
        candidates.append((rank, "polynomial", 0))
        candidates.extend((rank, "neural", width) for width in WIDTHS)
    start = time.perf_counter()
    for rank, branch, width in candidates:
        errors, converged = [], []
        for seed in SEEDS if branch == "neural" else (SEEDS[0],):
            for validation in range(1, len(pressure) - 1):
                train = np.delete(np.arange(len(pressure)), validation)
                fitted = fit_transport_pod(
                    pressure[train], fields[train], x, y,
                    rank=rank, branch=branch, width=width or 16, seed=seed,
                )
                predicted = predict_transport_pod(fitted, pressure[validation:validation + 1])[0]
                error = field_errors(fields[validation], predicted)
                errors.append(error)
                converged.append(bool(fitted["fit_converged"]))
                folds.append({
                    "branch": branch, "rank": rank, "width": width, "seed": seed,
                    "validation_pressure_kpa": float(pressure[validation]),
                    "fit_converged": bool(fitted["fit_converged"]),
                    **{f"{name}_global_percent": float(value) for name, value in zip(FIELDS, error, strict=True)},
                })
        row = {
            "branch": branch, "rank": rank, "width": width,
            "seed_count": 3 if branch == "neural" else 1,
            "mean_global_percent": float(np.mean(errors)),
            "nonconverged_fits": len(converged) - sum(converged),
            **{f"{name}_global_percent": float(value) for name, value in zip(FIELDS, np.mean(errors, axis=0), strict=True)},
        }
        rows.append(row)
        print("DEVELOPMENT_CANDIDATE=" + json.dumps(row), flush=True)
    # One scalar objective for all six fields; no selection on test or paper errors.
    selected = min(rows, key=lambda row: (row["mean_global_percent"], row["rank"], row["width"]))
    neural = min((row for row in rows if row["branch"] == "neural"),
                 key=lambda row: (row["mean_global_percent"], row["rank"], row["width"]))
    report = {
        "schema_version": 1,
        "development_pressures_kpa": pressure.tolist(),
        "validation_pressures_kpa": pressure[1:-1].tolist(),
        "endpoint_policy": "15 and 33 kPa are training anchors; extrapolation is outside this experiment",
        "regression_pressures_kpa": NOZZLE_HELD_OUT_KPA.tolist(),
        "selection_rule": "minimize mean full-grid relative L2 percentage over six fields, ten complete-case folds and, for neural candidates, three seeds",
        "candidate_ranks": list(RANKS), "candidate_neural_widths": list(WIDTHS), "seeds": list(SEEDS),
        "selected": selected, "selected_neural": neural, "candidates": rows,
        "test_fields_used_for_selection": False,
        "test_cases_previously_inspected": True,
        "assessment_scope": "regression assessment on previously inspected holdouts; fresh unseen pressures/geometries are still required",
        "source_archive_sha256": sha256(ROOT / "results/mahdavi_deeponet/nozzle_fields_15cases.npz"),
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_csv(output / "development_candidates.csv", rows)
    write_csv(output / "development_folds.csv", folds)
    (output / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def plot_results(data: dict, indices: np.ndarray, old: np.ndarray,
                 selected: np.ndarray, neural: np.ndarray, output: Path,
                 *, constrained: np.ndarray, profiles_only: bool = False) -> list[Path]:
    """Draw actual profiles and all six fields, with shared scales and full errors."""
    paths = []
    x, y = data["x_m"] * 1e6, data["y_m"] * 1e6
    centerline = int(data["centerline_index"])
    if not np.allclose(data["y_m"][centerline], 92e-6, rtol=0, atol=1e-11):
        raise ValueError("profile row does not match the documented symmetry plane")
    if np.any(constrained[:, centerline, :, 2] != 0):
        raise ValueError("profile prediction violates the supplied symmetry boundary")
    for local, index in enumerate(indices):
        pressure = int(data["pressure_kpa"][index])
        reference = np.stack([data[name][index] for name in FIELDS], axis=-1)
        errors = field_errors(reference, selected[local])
        fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), layout="constrained")
        for channel, axis in enumerate(axes.flat):
            if channel == 2:
                axis.plot(x[centerline], constrained[local, centerline, :, channel],
                          color="#008679", lw=2, label=r"Prescribed symmetry: $V=0$")
                axis.set(xlabel=r"$x$ ($\mu$m)", ylabel=LABELS[channel],
                         title="Symmetry boundary", ylim=(-0.5, 0.5), yticks=[-0.5, 0, 0.5])
                axis.legend(loc="lower center", frameon=False, fontsize=9)
                axis.grid(alpha=0.16)
                axis.spines[["top", "right"]].set_visible(False)
                continue
            axis.plot(x[centerline], reference[centerline, :, channel], color="#172B3A",
                      lw=1.6, marker="o", ms=2.4, markevery=4, label="DSMC reference")
            axis.plot(x[centerline], old[local, centerline, :, channel], color="#9B9FA5",
                      lw=1.2, ls=":", label="previous selected baseline")
            axis.plot(x[centerline], neural[local, centerline, :, channel], color="#756BB1",
                      lw=1.4, ls="--", label="registered POD neural ensemble")
            axis.plot(x[centerline], selected[local, centerline, :, channel], color="#008679",
                      lw=1.7, label="development-selected transport model")
            axis.set(xlabel=r"$x$ ($\mu$m)", ylabel=LABELS[channel])
            axis.set_title(f"Full-field error: {errors[channel]:.2f}%", fontsize=10)
            axis.grid(alpha=0.16)
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(f"Micro-nozzle at {pressure} kPa | centerline y = 92 um", fontsize=16)
        for suffix in ("png", "pdf"):
            path = output / f"nozzle_P{pressure}_profiles.{suffix}"
            fig.savefig(path, dpi=220)
            paths.append(path)
        plt.close(fig)
        if profiles_only:
            continue
        fig, axes = plt.subplots(6, 3, figsize=(12.8, 15.0), layout="constrained")
        for channel in range(6):
            ref, prediction = reference[:, :, channel], selected[local, :, :, channel]
            error = np.abs(prediction - ref)
            lo, hi = min(ref.min(), prediction.min()), max(ref.max(), prediction.max())
            palette = "RdBu_r" if channel == 2 else ("inferno" if channel == 3 else "viridis")
            for column, values in enumerate((ref, prediction, error)):
                axis = axes[channel, column]
                artist = axis.pcolormesh(x, y, values, shading="auto",
                                        cmap="magma" if column == 2 else palette,
                                        vmin=0 if column == 2 else lo,
                                        vmax=max(error.max(), 1e-12) if column == 2 else hi,
                                        rasterized=True)
                axis.set_aspect("equal", adjustable="box")
                if channel == 0:
                    axis.set_title(("DSMC reference", "Selected transport model", "Absolute error")[column])
                axis.set_xlabel(r"$x$ ($\mu$m)")
                if column == 0:
                    axis.set_ylabel(LABELS[channel] + "\n" + r"$y$ ($\mu$m)")
                fig.colorbar(artist, ax=axis, shrink=0.78, pad=0.025)
        fig.suptitle(f"Micro-nozzle at {pressure} kPa | all six physical fields", fontsize=15)
        for suffix in ("png", "pdf"):
            path = output / f"nozzle_P{pressure}_fields.{suffix}"
            fig.savefig(path, dpi=220)
            paths.append(path)
        plt.close(fig)
    return paths


def evaluate(data: dict, selection: dict, output: Path) -> dict:
    """Refit selected configurations and assess the three historical holdouts."""
    train = np.flatnonzero(~np.isin(data["pressure_kpa"], NOZZLE_HELD_OUT_KPA))
    test = np.flatnonzero(np.isin(data["pressure_kpa"], NOZZLE_HELD_OUT_KPA))
    values = np.stack([data[name][train] for name in FIELDS], axis=-1)
    query = data["pressure_kpa"][test]
    cfg = selection["selected"]
    start = time.perf_counter()
    fitted = fit_transport_pod(data["pressure_kpa"][train], values, data["x_m"], data["y_m"],
                               rank=cfg["rank"], branch=cfg["branch"], width=cfg["width"] or 16)
    fit_seconds = time.perf_counter() - start
    selected = predict_transport_pod(fitted, query)
    constrained = predict_with_symmetry(fitted, query, symmetry_y_m=92e-6)
    save_transport_model(fitted, output / "selected_model.npz")
    neural_predictions, neural_training = [], []
    nc = selection["selected_neural"]
    for seed in SEEDS:
        model = fit_transport_pod(data["pressure_kpa"][train], values, data["x_m"], data["y_m"],
                                  rank=nc["rank"], branch="neural", width=nc["width"], seed=seed)
        neural_predictions.append(predict_transport_pod(model, query))
        neural_training.append({"seed": seed, "fit_converged": bool(model["fit_converged"]),
                                "iterations": int(model["training_iterations"])})
        save_transport_model(model, output / f"neural_seed{seed}.npz")
    neural_predictions = np.array(neural_predictions)
    neural = neural_predictions.mean(axis=0)
    previous_manifest = json.loads((ROOT / "results/mahdavi_deeponet/nozzle_flowmllab_manifest.json").read_text())
    old = np.empty_like(selected)
    for channel, field in enumerate(FIELDS):
        if previous_manifest["selected_method_by_field"][field] == "shock_aligned":
            prediction, _ = interpolate_nozzle_fields_shock_aligned(
                data["pressure_kpa"], data[field], data["density"], data["x_m"], train, query,
            )
        else:
            prediction, _ = interpolate_nozzle_fields_locally(data["pressure_kpa"], data[field], train, query)
        old[:, :, :, channel] = prediction
    metrics, diagnostics = [], []
    for local, index in enumerate(test):
        ref = np.stack([data[name][index] for name in FIELDS], axis=-1)
        for channel, field in enumerate(FIELDS):
            row = {"pressure_kpa": int(query[local]), "field": field}
            for prefix, prediction in (("previous", old), ("selected", selected), ("neural", neural)):
                result = nozzle_field_error_metrics(ref[:, :, channel], prediction[local, :, :, channel],
                                                    ref[:, :, 0], data["x_m"])
                row.update({f"{prefix}_{key}": value for key, value in result.items()})
            seed_errors = [field_errors(ref, pred[local])[channel] for pred in neural_predictions]
            row["neural_seed_global_mean_percent"] = float(np.mean(seed_errors))
            row["neural_seed_global_std_percent"] = float(np.std(seed_errors, ddof=1))
            metrics.append(row)
        # Diagnostics use target fields only after predictions have been frozen.
        reference_surface = compression_surface(fitted["x_m"], ref[:, :, 0], float(fitted["throat_m"]))
        p = (query[local] - fitted["parameter_center"]) / fitted["parameter_half_range"]
        predicted_surface = fitted["station_coefficients"] @ np.array([1, p, p*p])
        row = {"pressure_kpa": int(query[local]),
               "shock_station_mean_absolute_error_um": float(np.mean(abs(predicted_surface-reference_surface))*1e6),
               "shock_station_max_absolute_error_um": float(np.max(abs(predicted_surface-reference_surface))*1e6)}
        for prefix, field in (("reference", ref), ("selected", selected[local]), ("neural", neural[local])):
            boundary_v = field[int(data["centerline_index"]), :, 2]
            row[f"{prefix}_symmetry_v_rms_ms"] = float(np.sqrt(np.mean(boundary_v**2)))
            row[f"{prefix}_symmetry_v_max_abs_ms"] = float(np.max(abs(boundary_v)))
            # Density source-unit factor cancels in the relative diagnostics.
            mass = np.trapezoid(field[:, :, 0] * field[:, :, 1], data["y_m"], axis=0)
            row[f"{prefix}_mass_flow_spread_percent"] = float(100*np.ptp(mass)/np.mean(abs(mass)))
            row[f"{prefix}_positive_density_temperature_pressure"] = bool(np.all(field[:, :, [0, 3, 5]] > 0))
            if prefix != "reference":
                reference_mass = np.trapezoid(ref[:, :, 0]*ref[:, :, 1], data["y_m"], axis=0)
                row[f"{prefix}_mass_flow_profile_relative_l2_percent"] = float(100*np.linalg.norm(mass-reference_mass)/np.linalg.norm(reference_mass))
        diagnostics.append(row)
    write_csv(output / "regression_metrics.csv", metrics)
    write_csv(output / "physical_diagnostics.csv", diagnostics)
    np.savez_compressed(output / "predictions.npz", pressure_kpa=query, field_names=np.array(FIELDS),
                        selected=selected, symmetry_constrained=constrained, neural=neural,
                        neural_by_seed=neural_predictions, previous=old)
    # Separate boundary audit; never overwrite raw labels or silently rescore them.
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), layout="constrained")
    for local, index in enumerate(test):
        axis = axes[local]
        axis.plot(data["x_m"][-1]*1e6, data["v_ms"][index, -1], color="#B42318",
                  label="raw DSMC export at symmetry node")
        axis.plot(data["x_m"][-1]*1e6, selected[local, -1, :, 2], "--", color="#756BB1",
                  label="unconstrained fit to raw export")
        axis.plot(data["x_m"][-1]*1e6, constrained[local, -1, :, 2], color="#008679", lw=2,
                  label="prediction with symmetry boundary")
        axis.set(title=f"{int(query[local])} kPa | y = 92 um", xlabel="x (um)", ylabel="V (m/s)")
        axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=1, frameon=False)
    fig.suptitle("Symmetry boundary audit: V = 0 is required, not a learned target")
    fig.savefig(output / "symmetry_boundary_audit.png", dpi=200)
    plt.close(fig)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        predict_transport_pod(fitted, query)
        times.append(time.perf_counter() - start)
    figures = plot_results(data, test, old, selected, neural, output, constrained=constrained)
    figures.append(output / "symmetry_boundary_audit.png")
    report = {
        "schema_version": 1, "selected_branch": cfg["branch"], "selected_rank": cfg["rank"],
        "neural_rank": nc["rank"], "neural_width": nc["width"],
        "neural_training": neural_training,
        "selection_sha256": sha256(output / "selection.json"),
        "regression_metrics_sha256": sha256(output / "regression_metrics.csv"),
        "diagnostics_sha256": sha256(output / "physical_diagnostics.csv"),
        "source_archive_sha256": selection["source_archive_sha256"],
        "test_cases_previously_inspected": True,
        "target_field_or_target_shock_used_for_prediction": False,
        "article_checkpoint_reproduced": False,
        "physical_validation_passed": False,
        "symmetry_boundary_y_m": 92e-6,
        "source_boundary_condition_defect": "exported nodal V is nonzero on the stated symmetry plane; raw fitting metrics are not physical validation",
        "symmetry_projection_scope": "separately stored prediction applies V=0 at known boundary, leaves interior unchanged, and is not used for the reported raw regression improvement",
        "profile_v_semantics": "boundary-constrained prediction V=0; raw nonzero V curves appear only in the separate symmetry audit; other profile panels and all raw-label scores are unchanged",
        "metric_contract": "same historical FlowMLLab global, shock-window and gradient-weighted definitions for every model; article-local metrics are not assumed identical",
        "selected_fit_seconds": fit_seconds,
        "selected_prediction_seconds_per_case_median": float(np.median(times)/len(query)),
        "timing_scope": "CPU, excludes file I/O; no measured DSMC speedup is claimed",
        "python": platform.python_version(), "numpy": np.__version__,
        "model_and_predictions_sha256": {p.name: sha256(p) for p in output.glob("*.npz")},
        "physical_diagnostics": diagnostics,
        "figures_sha256": {p.name: sha256(p) for p in figures},
    }
    for prefix in ("previous", "selected", "neural"):
        vals = [row[f"{prefix}_global_relative_l2_percent"] for row in metrics]
        report[f"{prefix}_mean_global_percent"] = float(np.mean(vals))
        report[f"{prefix}_maximum_global_percent"] = float(np.max(vals))
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("NOZZLE_TRANSPORT_REPORT=" + json.dumps(report), flush=True)
    return report


def main() -> None:
    """Run selection and evaluation in separate, reviewable phases if requested."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/nozzle_transport")
    parser.add_argument("--stage", choices=("select", "evaluate", "profiles", "all"), default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_nozzle_fields(ROOT)
    if args.stage == "profiles":
        report = json.loads((output / "report.json").read_text())
        if report["source_archive_sha256"] != sha256(ROOT / "results/mahdavi_deeponet/nozzle_fields_15cases.npz"):
            raise ValueError("retained predictions and source archive differ")
        path = output / "predictions.npz"
        if report["model_and_predictions_sha256"][path.name] != sha256(path):
            raise ValueError("retained prediction hash mismatch")
        with np.load(path, allow_pickle=False) as stored:
            indices = np.array([int(np.flatnonzero(data["pressure_kpa"] == p)[0])
                                for p in stored["pressure_kpa"]])
            figures = plot_results(data, indices, stored["previous"], stored["selected"],
                                   stored["neural"], output,
                                   constrained=stored["symmetry_constrained"], profiles_only=True)
        report["figures_sha256"].update({p.name: sha256(p) for p in figures})
        report["profile_v_semantics"] = "boundary-constrained prediction V=0; raw nonzero V curves appear only in the separate symmetry audit; other profile panels and all raw-label scores are unchanged"
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("Regenerated three profile figures from unchanged retained predictions.")
        return
    if args.stage in {"select", "all"}:
        select_models(data, output)
    if args.stage in {"evaluate", "all"}:
        selection = json.loads((output / "selection.json").read_text())
        if selection["source_archive_sha256"] != sha256(ROOT / "results/mahdavi_deeponet/nozzle_fields_15cases.npz"):
            raise ValueError("selection and evaluation source archives differ")
        evaluate(data, selection, output)


if __name__ == "__main__":
    main()
