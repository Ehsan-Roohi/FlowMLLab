#!/usr/bin/env python3
"""Generate the Week-5 complete-Re blind cylinder-wake comparison video.

The full Re=100 trajectory is excluded from training.  A phase-conditioned
POD--MLP is fit only to Re=60, 80, 90, 110, 120, and 140 LBM snapshots, then
evaluated once on Re=100.  The resulting animation compares dimensionless vorticity from the
blind LBM trajectory with vorticity derived from the predicted velocity field.

This is an educational interpolation test, not an autonomous time rollout and
not a claim that the quick LBM grid is a grid-converged cylinder DNS.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab import cylinder_lbm, cylinder_ml  # noqa: E402


TRAIN_RE = (60, 80, 90, 110, 120, 140)
BLIND_RE = 100
ALL_RE = tuple(sorted(TRAIN_RE + (BLIND_RE,)))
SEED = 690
POD_RANK = 16
GRID = {
    "nx": 240,
    "ny": 96,
    "diameter": 12.0,
    "center": (60.0, 47.5),
    "inflow_velocity": 0.05,
    "steps": 22000,
    "history_stride": 4,
    "snapshot_start": 15000,
    "snapshot_stride": 125,
    "perturbation": 1.0e-2,
    "seed": SEED,
    "collision_model": "trt",
    "cylinder_boundary": "bouzidi",
}


def _simulate(reynolds: int) -> dict[str, Any]:
    started = time.perf_counter()
    result = cylinder_lbm.simulate_cylinder(reynolds, **GRID)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _phase(case: dict[str, Any]) -> np.ndarray:
    lift = np.asarray(case["lift_coefficient"], dtype=float)
    analytic = hilbert(lift - lift.mean())
    history_phase = np.unwrap(np.angle(analytic))
    phase = np.interp(case["snapshot_time"], case["time"], history_phase)
    return np.mod(phase, 2.0 * np.pi)


def _normalized_fields(case: dict[str, Any]) -> dict[str, np.ndarray]:
    speed = float(case["metadata"]["config"]["inflow_velocity"])
    return {
        "u": np.asarray(case["snapshots"]["u"], dtype=float) / speed,
        "v": np.asarray(case["snapshots"]["v"], dtype=float) / speed,
        "p": np.asarray(case["snapshots"]["p"], dtype=float) / speed**2,
    }


def _branch_features(reynolds: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return np.column_stack((reynolds, np.sin(phase), np.cos(phase)))


def _fit_and_predict(cases: dict[int, dict[str, Any]]) -> dict[str, Any]:
    parts = {name: [] for name in ("u", "v", "p")}
    re_labels: list[float] = []
    phase_labels: list[float] = []
    for reynolds in ALL_RE:
        fields = _normalized_fields(cases[reynolds])
        count = fields["u"].shape[0]
        for name in parts:
            parts[name].append(fields[name])
        re_labels.extend([float(reynolds)] * count)
        phase_labels.extend(_phase(cases[reynolds]).tolist())

    fields = {name: np.concatenate(values) for name, values in parts.items()}
    re_array = np.asarray(re_labels)
    phase_array = np.asarray(phase_labels)
    split = cylinder_ml.casewise_reynolds_split(
        re_array, test_reynolds=[BLIND_RE]
    )
    if tuple(split.train_reynolds.astype(int)) != TRAIN_RE:
        raise AssertionError("the complete-Re split changed unexpectedly")

    train_fields = {
        name: values[split.train_indices] for name, values in fields.items()
    }
    truth = {name: values[split.test_indices] for name, values in fields.items()}
    baseline = cylinder_ml.fit_pod_regressor(
        train_fields,
        re_array[split.train_indices],
        phase_array[split.train_indices],
        field_names=("u", "v", "p"),
        rank=POD_RANK,
        reynolds_degree=2,
        phase_harmonics=2,
        ridge=1.0e-8,
    )
    baseline_prediction = baseline.predict_fields(
        re_array[split.test_indices], phase_array[split.test_indices]
    )

    train_vectors, _ = cylinder_ml.pack_fields(
        train_fields, ("u", "v", "p")
    )
    train_coefficients = cylinder_ml.project_pod(baseline.pod, train_vectors)
    x_scaler = StandardScaler().fit(
        _branch_features(
            re_array[split.train_indices], phase_array[split.train_indices]
        )
    )
    y_scaler = StandardScaler().fit(train_coefficients)
    network = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        solver="lbfgs",
        alpha=1.0e-2,
        max_iter=10000,
        random_state=SEED,
    )
    network.fit(
        x_scaler.transform(
            _branch_features(
                re_array[split.train_indices], phase_array[split.train_indices]
            )
        ),
        y_scaler.transform(train_coefficients),
    )
    blind_coefficients = y_scaler.inverse_transform(
        network.predict(
            x_scaler.transform(
                _branch_features(
                    re_array[split.test_indices], phase_array[split.test_indices]
                )
            )
        )
    )
    blind_vectors = cylinder_ml.reconstruct_pod(
        baseline.pod, blind_coefficients
    )
    prediction = cylinder_ml.unpack_fields(blind_vectors, baseline.layout)
    return {
        "truth": truth,
        "prediction": prediction,
        "baseline_prediction": baseline_prediction,
        "blind_phase": phase_array[split.test_indices],
        "pod_energy": float(baseline.pod.cumulative_energy[POD_RANK - 1]),
        "network_iterations": int(network.n_iter_),
    }


def _vorticity(fields: dict[str, np.ndarray], solid: np.ndarray) -> np.ndarray:
    values = []
    for u, v in zip(fields["u"], fields["v"]):
        omega = cylinder_lbm.compute_vorticity(u, v, solid)
        values.append(omega)
    return np.stack(values)


def _metrics(
    cases: dict[int, dict[str, Any]], products: dict[str, Any]
) -> dict[str, Any]:
    truth = products["truth"]
    prediction = products["prediction"]
    baseline_prediction = products["baseline_prediction"]
    solid = np.asarray(cases[BLIND_RE]["solid"], dtype=bool)
    omega_truth = _vorticity(truth, solid)
    omega_prediction = _vorticity(prediction, solid)
    omega_baseline = _vorticity(baseline_prediction, solid)
    fluid = ~solid

    def omega_error(estimate: np.ndarray) -> tuple[float, np.ndarray]:
        residual = (estimate - omega_truth)[:, fluid]
        exact = omega_truth[:, fluid]
        framewise = np.linalg.norm(residual, axis=1) / np.maximum(
            np.linalg.norm(exact, axis=1), np.finfo(float).eps
        )
        total = float(np.linalg.norm(residual) / np.linalg.norm(exact))
        return total, framewise

    neural_omega, framewise = omega_error(omega_prediction)
    baseline_omega, _ = omega_error(omega_baseline)
    field_metrics = cylinder_ml.reconstruction_diagnostics(truth, prediction)
    baseline_metrics = cylinder_ml.reconstruction_diagnostics(
        truth, baseline_prediction
    )
    physics = cylinder_ml.flow_diagnostics(
        {"u": prediction["u"], "v": prediction["v"]},
        dx=1.0,
        dy=1.0,
        solid_mask=solid,
    )
    blind = cases[BLIND_RE]
    tail = slice(len(blind["time"]) // 2, None)
    return {
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"training_reynolds": list(TRAIN_RE), "blind_reynolds": BLIND_RE},
        "conditioning": "Re, sin(phase), cos(phase); phase is measured from LBM lift",
        "claim_scope": (
            "educational phase-conditioned interpolation; not autonomous rollout; "
            "quick LBM target is not grid-converged DNS"
        ),
        "model": {
            "fields": ["u/U", "v/U", "p/U^2"],
            "pod_rank": POD_RANK,
            "pod_training_energy": products["pod_energy"],
            "hidden_layers": [64, 64],
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 1.0e-2,
            "seed": SEED,
            "iterations": products["network_iterations"],
        },
        "neural_blind_metrics": {
            **field_metrics,
            "vorticity_relative_l2": neural_omega,
            "vorticity_frame_median_relative_l2": float(np.median(framewise)),
            "vorticity_frame_max_relative_l2": float(np.max(framewise)),
            "divergence_rms_normalized": float(physics["divergence_rms"]),
            "solid_speed_rms_normalized": float(physics["solid_speed_rms"]),
            "mean_pressure_gauge_normalized": float(prediction["p"].mean()),
        },
        "harmonic_pod_baseline_metrics": {
            **baseline_metrics,
            "vorticity_relative_l2": baseline_omega,
        },
        "blind_lbm_diagnostics": {
            "cylinder_boundary": str(blind["metadata"]["cylinder_boundary"]),
            "strouhal": float(blind["strouhal"]),
            "mean_drag_coefficient": float(
                np.mean(blind["drag_coefficient"][tail])
            ),
            "lift_rms": float(np.std(blind["lift_coefficient"][tail])),
            "max_mean_density_drift": float(
                np.max(np.abs(blind["mean_density_ratio"] - 1.0))
            ),
            "lattice_mach": float(blind["metadata"]["lattice_mach"]),
            "relaxation_time": float(blind["metadata"]["relaxation_time"]),
            "blockage_ratio": float(blind["metadata"]["blockage_ratio"]),
            "snapshots": int(omega_truth.shape[0]),
        },
        "runtime": {
            str(reynolds): float(cases[reynolds]["elapsed_seconds"])
            for reynolds in ALL_RE
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def _make_video(
    cases: dict[int, dict[str, Any]],
    products: dict[str, Any],
    metrics: dict[str, Any],
    output: Path,
) -> None:
    blind = cases[BLIND_RE]
    solid = np.asarray(blind["solid"], dtype=bool)
    # Hilbert phase is least reliable at the two ends of a finite record.  The
    # retained metrics use every blind snapshot, while the presentation video
    # omits five leading and three trailing edge-affected frames.
    display_window = slice(5, -3)
    truth = _vorticity(products["truth"], solid)[display_window]
    prediction = _vorticity(products["prediction"], solid)[display_window]
    error = prediction - truth
    diameter = float(blind["metadata"]["config"]["diameter"])
    center_x, center_y = blind["metadata"]["cylinder_center"]
    x_d = (np.asarray(blind["x"]) - center_x) / diameter
    y_d = (np.asarray(blind["y"]) - center_y) / diameter
    extent = (x_d.min(), x_d.max(), y_d.min(), y_d.max())
    time_star = (
        np.asarray(blind["snapshot_time"])[display_window]
        * GRID["inflow_velocity"]
        / diameter
    )
    display_phase = products["blind_phase"][display_window]
    history_star = np.asarray(blind["time"]) * GRID["inflow_velocity"] / diameter
    dimensionless_truth = truth * diameter
    dimensionless_prediction = prediction * diameter
    dimensionless_error = error * diameter
    limit = float(np.percentile(np.abs(dimensionless_truth[:, ~solid]), 99.5))
    error_limit = float(np.percentile(np.abs(dimensionless_error[:, ~solid]), 99.0))

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(3.25, 0.16, 1.10),
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.855,
        hspace=0.48,
        wspace=0.20,
    )
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    shared_color_axis = fig.add_subplot(grid[1, :2])
    error_color_axis = fig.add_subplot(grid[1, 2])
    lift_axis = fig.add_subplot(grid[2, :2])
    note_axis = fig.add_subplot(grid[2, 2])
    images = [
        axes[0].imshow(
            dimensionless_truth[0], origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="bilinear"
        ),
        axes[1].imshow(
            dimensionless_prediction[0], origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="bilinear"
        ),
        axes[2].imshow(
            dimensionless_error[0], origin="lower", extent=extent, aspect="auto",
            cmap="coolwarm", vmin=-error_limit, vmax=error_limit,
            interpolation="bilinear"
        ),
    ]
    titles = (
        "LBM target — withheld from training",
        "Neural POD prediction",
        "Signed error: Neural − LBM",
    )
    for axis, title in zip(axes, titles):
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                0.5,
                facecolor="#F7F7F7",
                edgecolor="black",
                linewidth=1.5,
                zorder=5,
            )
        )
        axis.set_xlim(-1.5, 13.0)
        axis.set_ylim(-3.7, 3.7)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, pad=10)
        axis.set_xlabel(r"$(x-x_c)/D$")
        axis.set_ylabel(r"$(y-y_c)/D$")
    fig.colorbar(
        images[1], cax=shared_color_axis, orientation="horizontal",
        label=r"dimensionless vorticity  $\omega_z D/U_\infty$"
    )
    fig.colorbar(
        images[2], cax=error_color_axis, orientation="horizontal",
        label=r"vorticity error  $\Delta\omega_z D/U_\infty$"
    )

    lift_axis.plot(
        history_star, blind["lift_coefficient"], color="#243B53", lw=1.5
    )
    marker = lift_axis.axvline(time_star[0], color="#D1495B", lw=2.0)
    lift_axis.axhline(0.0, color="0.55", lw=0.8)
    lift_axis.set(
        xlabel=r"convective time $tU_\infty/D$",
        ylabel=r"$C_L$",
        title="Blind LBM lift history and displayed snapshot",
    )
    lift_axis.grid(alpha=0.22)

    neural = metrics["neural_blind_metrics"]
    lbm = metrics["blind_lbm_diagnostics"]
    note_axis.axis("off")
    note_axis.text(
        0.02,
        0.98,
        "Complete-Re blind test\n"
        "Train Re: 60, 80, 90,\n"
        "          110, 120, 140\n"
        "Test: Re = 100\n\n"
        f"Vorticity relative L2: {100*neural['vorticity_relative_l2']:.2f}%\n"
        f"Combined u,v,p L2: {100*neural['combined_relative_l2']:.2f}%\n"
        f"LBM St: {lbm['strouhal']:.3f}\n"
        f"LBM density drift: {100*lbm['max_mean_density_drift']:.3f}%\n\n"
        "Curved Bouzidi wall; phase-conditioned",
        va="top",
        ha="left",
        fontsize=10.5,
        linespacing=1.24,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F4F7FA", "edgecolor": "#9FB3C8"},
    )
    clock = fig.text(0.5, 0.907, "", ha="center", va="top", fontsize=13)
    fig.suptitle(
        "FlowMLLab Week 5 — Vortex shedding in an unseen Reynolds case",
        fontsize=20,
        y=0.975,
    )

    def update(frame: int):
        images[0].set_data(dimensionless_truth[frame])
        images[1].set_data(dimensionless_prediction[frame])
        images[2].set_data(dimensionless_error[frame])
        marker.set_xdata([time_star[frame], time_star[frame]])
        clock.set_text(
            rf"Re = {BLIND_RE}   |   $tU_\infty/D={time_star[frame]:.2f}$   |   "
            rf"phase $={display_phase[frame]:.2f}$ rad"
        )
        return (*images, marker, clock)

    animation = FuncAnimation(
        fig, update, frames=len(time_star), interval=100, blit=False
    )
    writer = FFMpegWriter(
        fps=10,
        codec="libx264",
        bitrate=6000,
        metadata={
            "title": "FlowMLLab blind Re=100 LBM versus neural POD",
            "artist": "FlowMLLab",
            "comment": "Training Reynolds numbers: 60, 80, 90, 110, 120, 140; blind Re=100",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(output / "blind_re100_lbm_vs_neural.mp4", writer=writer, dpi=120)
    update(len(time_star) // 2)
    fig.savefig(output / "blind_re100_lbm_vs_neural_poster.png", dpi=220)
    plt.close(fig)


def regenerate(output: Path, workers: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        results = [_simulate(reynolds) for reynolds in ALL_RE]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_simulate, ALL_RE))
    cases = {int(case["metadata"]["reynolds"]): case for case in results}
    products = _fit_and_predict(cases)
    metrics = _metrics(cases, products)
    _make_video(cases, products, metrics, output)
    (output / "blind_re100_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for name, values in (
        ("harmonic POD baseline", metrics["harmonic_pod_baseline_metrics"]),
        ("neural POD", metrics["neural_blind_metrics"]),
    ):
        rows.append({"model": name, **values})
    pd.DataFrame(rows).to_csv(output / "blind_re100_model_comparison.csv", index=False)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    output = args.root.resolve() / "results" / "cylinder_ml"
    report = regenerate(output, max(1, args.workers))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
