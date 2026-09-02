#!/usr/bin/env python3
"""Train and validate the Week-5 four-frame multi-scale cylinder CNN.

Protocol
--------
* development Reynolds cases: 60, 80, 90, 110, 120, 140;
* model selection and stopping: complete Re=100 validation trajectory;
* final blind case: Re=105, opened only after the validation gates pass;
* temporal spacing: dt* = snapshot_stride * U / D ~= 0.1;
* matched next-frame baseline: persistence of the latest input frame.

The older phase-conditioned POD--MLP remains a documented failure baseline.  It
is not silently relabelled as the new CNN result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab import cylinder_lbm  # noqa: E402
from flowmllab import cylinder_cnn  # noqa: E402


DEVELOPMENT_RE = (60, 80, 90, 110, 120, 140)
VALIDATION_RE = 100
BLIND_RE = 105
SEED = 690
HISTORY = 4
SNAPSHOT_STRIDE = 25
GRID = {
    "nx": 240,
    "ny": 96,
    "diameter": 12.0,
    "center": (60.0, 47.5),
    "inflow_velocity": 0.05,
    "steps": 22000,
    "history_stride": 4,
    "snapshot_start": 15000,
    "snapshot_stride": SNAPSHOT_STRIDE,
    "perturbation": 1.0e-2,
    "seed": SEED,
    "collision_model": "trt",
    "cylinder_boundary": "bouzidi",
}
FIELD_NAMES = ("u", "v", "p")
LOSS_WEIGHTS = {
    "gradient": 0.20,
    "vorticity": 0.20,
    "divergence": 0.05,
}


def _cache_name(reynolds: int) -> str:
    return (
        f"re{reynolds:03d}_nx{GRID['nx']}_ny{GRID['ny']}_d{int(GRID['diameter'])}"
        f"_s{GRID['steps']}_start{GRID['snapshot_start']}_stride{SNAPSHOT_STRIDE}.npz"
    )


def _run_case(reynolds: int, cache_dir: str) -> str:
    """Run one dense LBM case and cache only the arrays required by the CNN."""
    destination = Path(cache_dir) / _cache_name(reynolds)
    if destination.exists():
        return str(destination)
    started = time.perf_counter()
    result = cylinder_lbm.simulate_cylinder(reynolds, **GRID)
    speed = float(GRID["inflow_velocity"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        reynolds=np.asarray(float(reynolds)),
        u=np.asarray(result["snapshots"]["u"] / speed, dtype=np.float32),
        v=np.asarray(result["snapshots"]["v"] / speed, dtype=np.float32),
        p=np.asarray(result["snapshots"]["p"] / speed**2, dtype=np.float32),
        solid=np.asarray(result["solid"], dtype=np.uint8),
        x=np.asarray(result["x"], dtype=np.float32),
        y=np.asarray(result["y"], dtype=np.float32),
        snapshot_time=np.asarray(result["snapshot_time"], dtype=np.float32),
        time=np.asarray(result["time"], dtype=np.float32),
        drag_coefficient=np.asarray(result["drag_coefficient"], dtype=np.float32),
        lift_coefficient=np.asarray(result["lift_coefficient"], dtype=np.float32),
        mean_density_ratio=np.asarray(result["mean_density_ratio"], dtype=np.float32),
        strouhal=np.asarray(float(result["strouhal"])),
        elapsed_seconds=np.asarray(time.perf_counter() - started),
        metadata_json=np.asarray(json.dumps(result["metadata"], sort_keys=True)),
    )
    return str(destination)


def _ensure_cases(
    reynolds_numbers: Iterable[int], cache_dir: Path, workers: int
) -> dict[int, dict[str, Any]]:
    values = tuple(int(value) for value in reynolds_numbers)
    missing = [value for value in values if not (cache_dir / _cache_name(value)).exists()]
    if missing:
        print("Dense LBM cases to run:", missing, flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            futures = {
                pool.submit(_run_case, value, str(cache_dir)): value for value in missing
            }
            for future in as_completed(futures):
                value = futures[future]
                path = future.result()
                print(f"  Re={value}: cached at {path}", flush=True)
    cases: dict[int, dict[str, Any]] = {}
    for value in values:
        path = cache_dir / _cache_name(value)
        with np.load(path, allow_pickle=False) as archive:
            cases[value] = {key: archive[key] for key in archive.files}
        cases[value]["solid"] = np.asarray(cases[value]["solid"], dtype=bool)
    return cases


def _training_scales(cases: dict[int, dict[str, Any]]) -> tuple[np.ndarray, float]:
    sums = np.zeros(3)
    sums_squared = np.zeros(3)
    count = 0
    omega_samples = []
    for case in cases.values():
        arrays = [np.asarray(case[name], dtype=np.float64) for name in FIELD_NAMES]
        local_count = arrays[0].size
        sums += [array.sum() for array in arrays]
        sums_squared += [(array**2).sum() for array in arrays]
        count += local_count
        omega_samples.append(
            cylinder_cnn.vorticity(
                arrays[0][::10], arrays[1][::10], diameter=float(GRID["diameter"])
            ).reshape(-1)
        )
    means = sums / count
    scales = np.sqrt(np.maximum(sums_squared / count - means**2, 1.0e-12))
    omega = np.concatenate(omega_samples)
    omega_scale = max(float(np.sqrt(np.mean(omega**2))), 1.0e-6)
    return scales.astype(np.float32), omega_scale


def _sample_generator(
    cases: dict[int, dict[str, Any]],
    *,
    batch_size: int,
    patch_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    keys = np.asarray(sorted(cases))
    re_center = 0.5 * (min(DEVELOPMENT_RE) + max(DEVELOPMENT_RE))
    re_scale = 0.5 * (max(DEVELOPMENT_RE) - min(DEVELOPMENT_RE))
    diameter = float(GRID["diameter"])
    center_x = float(GRID["center"][0])
    while True:
        inputs = np.empty(
            (batch_size, patch_size, patch_size, HISTORY * 3 + 2), dtype=np.float32
        )
        targets = np.empty((batch_size, patch_size, patch_size, 3), dtype=np.float32)
        for batch_index in range(batch_size):
            reynolds = int(rng.choice(keys))
            case = cases[reynolds]
            windows = cylinder_cnn.temporal_windows(case["u"].shape[0], history=HISTORY)
            window = windows[int(rng.integers(0, len(windows)))]
            ny, nx = case["solid"].shape
            y0 = int(rng.integers(0, ny - patch_size + 1))
            if rng.random() < 0.85:
                low = max(0, int(center_x - 1.5 * diameter))
                high = max(low + 1, nx - patch_size + 1)
                x0 = int(rng.integers(low, high))
            else:
                x0 = int(rng.integers(0, nx - patch_size + 1))
            ys = slice(y0, y0 + patch_size)
            xs = slice(x0, x0 + patch_size)
            history_fields = {name: case[name][:, ys, xs] for name in FIELD_NAMES}
            flow = cylinder_cnn.stack_history(history_fields, window.history)
            target = np.stack([case[name][window.target, ys, xs] for name in FIELD_NAMES], axis=-1)
            fluid = (~case["solid"][ys, xs]).astype(np.float32)[..., None]
            re_channel = np.full_like(
                fluid, (float(reynolds) - re_center) / re_scale, dtype=np.float32
            )
            if rng.random() < 0.5:
                flow = flow[::-1].copy()
                target = target[::-1].copy()
                fluid = fluid[::-1].copy()
                re_channel = re_channel[::-1].copy()
                flow[..., 1::3] *= -1.0
                target[..., 1] *= -1.0
            inputs[batch_index] = np.concatenate((flow, re_channel, fluid), axis=-1)
            targets[batch_index] = target
        yield inputs, targets


def _dataset(
    tf: Any,
    cases: dict[int, dict[str, Any]],
    *,
    batch_size: int,
    patch_size: int,
    seed: int,
) -> Any:
    signature = (
        tf.TensorSpec(
            shape=(batch_size, patch_size, patch_size, HISTORY * 3 + 2),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(batch_size, patch_size, patch_size, 3), dtype=tf.float32
        ),
    )
    return tf.data.Dataset.from_generator(
        lambda: _sample_generator(
            cases, batch_size=batch_size, patch_size=patch_size, seed=seed
        ),
        output_signature=signature,
    ).prefetch(1)


def _full_inputs(
    case: dict[str, Any], windows: list[cylinder_cnn.TemporalWindow], reynolds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = case["solid"].shape
    flow_inputs = np.empty((len(windows), ny, nx, HISTORY * 3 + 2), dtype=np.float32)
    truth = np.empty((len(windows), ny, nx, 3), dtype=np.float32)
    persistence = np.empty_like(truth)
    re_center = 0.5 * (min(DEVELOPMENT_RE) + max(DEVELOPMENT_RE))
    re_scale = 0.5 * (max(DEVELOPMENT_RE) - min(DEVELOPMENT_RE))
    re_channel = np.full(
        (ny, nx, 1), (float(reynolds) - re_center) / re_scale, dtype=np.float32
    )
    fluid = (~case["solid"]).astype(np.float32)[..., None]
    for index, window in enumerate(windows):
        flow = cylinder_cnn.stack_history(case, window.history)
        flow_inputs[index] = np.concatenate((flow, re_channel, fluid), axis=-1)
        truth[index] = np.stack(
            [case[name][window.target] for name in FIELD_NAMES], axis=-1
        )
        persistence[index] = flow[..., -3:]
    return flow_inputs, truth, persistence


def _predict_case(model: Any, case: dict[str, Any], reynolds: int) -> dict[str, Any]:
    windows = cylinder_cnn.temporal_windows(case["u"].shape[0], history=HISTORY)
    inputs, truth, persistence = _full_inputs(case, windows, reynolds)
    prediction_parts = []
    for start in range(0, len(windows), 4):
        prediction_parts.append(
            np.asarray(model.predict_on_batch(inputs[start : start + 4]), dtype=np.float32)
        )
    prediction = np.concatenate(prediction_parts)
    return {
        "truth": truth,
        "prediction": prediction,
        "persistence": persistence,
        "times": np.asarray([case["snapshot_time"][window.target] for window in windows]),
    }


def _case_metrics(
    products: dict[str, Any], case: dict[str, Any], reynolds: int
) -> dict[str, Any]:
    truth = products["truth"]
    solid = np.asarray(case["solid"], dtype=bool)
    fluid = ~solid
    truth_omega = cylinder_cnn.vorticity(
        truth[..., 0], truth[..., 1], diameter=float(GRID["diameter"])
    )
    output: dict[str, Any] = {
        "reynolds": int(reynolds),
        "snapshot_count": int(truth.shape[0]),
        "lbm_strouhal": float(case["strouhal"]),
        "models": {},
    }
    for name in ("persistence", "prediction"):
        estimate = products[name]
        omega = cylinder_cnn.vorticity(
            estimate[..., 0], estimate[..., 1], diameter=float(GRID["diameter"])
        )
        div = cylinder_cnn.divergence(
            estimate[..., 0], estimate[..., 1], diameter=float(GRID["diameter"])
        )
        field_errors = {
            field: cylinder_cnn.relative_l2(
                truth[..., field_index][:, fluid], estimate[..., field_index][:, fluid]
            )
            for field_index, field in enumerate(FIELD_NAMES)
        }
        stations = cylinder_cnn.stationwise_wake_metrics(
            truth_omega,
            omega,
            center_x=float(GRID["center"][0]),
            diameter=float(GRID["diameter"]),
        )
        output["models"][name] = {
            **{f"{field}_relative_l2": value for field, value in field_errors.items()},
            "vorticity_relative_l2": cylinder_cnn.relative_l2(
                truth_omega[:, fluid], omega[:, fluid]
            ),
            "divergence_rms": float(np.sqrt(np.mean(div[:, fluid] ** 2))),
            "solid_speed_max": float(
                np.sqrt(estimate[..., 0][:, solid] ** 2 + estimate[..., 1][:, solid] ** 2).max()
            ),
            "stationwise": stations,
            "mean_station_enstrophy_relative_error": float(
                np.mean([row["enstrophy_relative_error"] for row in stations])
            ),
            "mean_station_profile_relative_l2": float(
                np.mean([row["vorticity_profile_relative_l2"] for row in stations])
            ),
            "mean_station_psd_relative_l2": float(
                np.mean([row["normalized_psd_relative_l2"] for row in stations])
            ),
        }
    return output


def _validation_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    cnn = metrics["models"]["prediction"]
    persistence = metrics["models"]["persistence"]
    ratios = [row["enstrophy_ratio"] for row in cnn["stationwise"]]
    return {
        "cnn_vorticity_beats_persistence": (
            cnn["vorticity_relative_l2"] < persistence["vorticity_relative_l2"]
        ),
        "cnn_downstream_profile_beats_persistence": (
            cnn["mean_station_profile_relative_l2"]
            < persistence["mean_station_profile_relative_l2"]
        ),
        "mean_downstream_enstrophy_error_below_15_percent": (
            cnn["mean_station_enstrophy_relative_error"] < 0.15
        ),
        "every_station_enstrophy_ratio_between_0p75_and_1p25": all(
            0.75 <= ratio <= 1.25 for ratio in ratios
        ),
        "exact_no_slip": cnn["solid_speed_max"] <= 1.0e-7,
    }


def _write_station_figure(
    metrics: dict[str, Any], destination: Path, title: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for model_name, label, color, marker in (
        ("persistence", "persistence", "#737373", "s"),
        ("prediction", "multi-scale CNN", "#0072B2", "o"),
    ):
        rows = metrics["models"][model_name]["stationwise"]
        x = [row["x_over_d"] for row in rows]
        axes[0].plot(x, [row["enstrophy_ratio"] for row in rows], marker=marker, color=color, label=label)
        axes[1].plot(x, [row["vorticity_profile_relative_l2"] for row in rows], marker=marker, color=color, label=label)
        axes[2].plot(x, [row["normalized_psd_relative_l2"] for row in rows], marker=marker, color=color, label=label)
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].fill_between([2, 8], 0.75, 1.25, color="#009E73", alpha=0.12)
    axes[0].set(ylabel="predicted / LBM enstrophy", ylim=(0, None))
    axes[1].set(ylabel="vorticity-profile relative L2")
    axes[2].set(ylabel="normalized transverse-PSD relative L2")
    for axis in axes:
        axis.set_xlabel(r"$(x-x_c)/D$")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def _write_video(
    products: dict[str, Any],
    case: dict[str, Any],
    metrics: dict[str, Any],
    destination: Path,
    poster_destination: Path,
) -> None:
    truth = cylinder_cnn.vorticity(
        products["truth"][..., 0], products["truth"][..., 1], diameter=float(GRID["diameter"])
    )
    prediction = cylinder_cnn.vorticity(
        products["prediction"][..., 0], products["prediction"][..., 1], diameter=float(GRID["diameter"])
    )
    error = prediction - truth
    center_x, center_y = GRID["center"]
    diameter = float(GRID["diameter"])
    x = (np.asarray(case["x"]) - center_x) / diameter
    y = (np.asarray(case["y"]) - center_y) / diameter
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    vlimit = float(np.percentile(np.abs(truth), 99.5))
    elimit = float(np.percentile(np.abs(error), 99.5))
    display = np.arange(0, truth.shape[0], 2)
    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), constrained_layout=True)
    images = []
    for axis, values, title, limit, cmap in zip(
        axes,
        (truth, prediction, error),
        ("LBM target", "four-frame multi-scale CNN", "signed error: CNN - LBM"),
        (vlimit, vlimit, elimit),
        ("RdBu_r", "RdBu_r", "PuOr_r"),
    ):
        image = axis.imshow(
            values[display[0]], origin="lower", extent=extent, cmap=cmap,
            vmin=-limit, vmax=limit, interpolation="nearest", aspect="equal"
        )
        axis.add_patch(
            Circle((0.0, 0.0), 0.5, facecolor="white", edgecolor="black", linewidth=1.8)
        )
        axis.set(title=title, xlabel=r"$(x-x_c)/D$", ylabel=r"$(y-y_c)/D$", xlim=(-1.2, 12.0), ylim=(-3.5, 3.5))
        images.append(image)
    fig.colorbar(images[1], ax=list(axes[:2]), shrink=0.78, label=r"vorticity $\omega_zD/U_\infty$")
    fig.colorbar(images[2], ax=axes[2], shrink=0.78, label=r"vorticity error")
    cnn = metrics["models"]["prediction"]
    title = fig.suptitle(
        f"Unseen Re={metrics['reynolds']} | one-step dt*={SNAPSHOT_STRIDE*GRID['inflow_velocity']/GRID['diameter']:.3f} | "
        f"E_omega={100*cnn['vorticity_relative_l2']:.2f}%"
    )

    def update(frame_number: int):
        index = int(display[frame_number])
        for image, values in zip(images, (truth, prediction, error)):
            image.set_data(values[index])
        title.set_text(
            f"Unseen Re={metrics['reynolds']} | tU/D={products['times'][index]*GRID['inflow_velocity']/GRID['diameter']:.2f} | "
            f"four previous LBM frames -> next frame"
        )
        return images + [title]

    update(len(display) // 2)
    fig.savefig(poster_destination, dpi=180)
    animation = FuncAnimation(fig, update, frames=len(display), interval=50, blit=False)
    animation.save(
        destination,
        writer=FFMpegWriter(fps=20, bitrate=5000, metadata={"title": "FlowMLLab cylinder CNN blind comparison"}),
        dpi=100,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cylinder_cnn")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--filters", type=int, default=24)
    parser.add_argument("--run-blind", action="store_true")
    parser.add_argument(
        "--reuse-weights",
        action="store_true",
        help="load the frozen validation-selected weights instead of retraining",
    )
    args = parser.parse_args()
    if args.patch_size % 4:
        raise ValueError("patch-size must be divisible by four")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "cache"

    required = DEVELOPMENT_RE + (VALIDATION_RE,)
    cases = _ensure_cases(required, cache_dir, args.workers)
    development = {value: cases[value] for value in DEVELOPMENT_RE}
    field_scales, omega_scale = _training_scales(development)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    model = cylinder_cnn.build_multiscale_predictor(filters=args.filters)
    loss = cylinder_cnn.composite_flow_loss(
        field_scales=field_scales,
        diameter=float(GRID["diameter"]),
        vorticity_scale=omega_scale,
        gradient_weight=LOSS_WEIGHTS["gradient"],
        vorticity_weight=LOSS_WEIGHTS["vorticity"],
        divergence_weight=LOSS_WEIGHTS["divergence"],
    )
    weights_path = args.output / "multiscale_cnn.weights.h5"
    history_path = args.output / "training_history.csv"
    report_path = args.output / "multiscale_cnn_metrics.json"
    if args.reuse_weights:
        if not weights_path.exists() or not history_path.exists():
            raise FileNotFoundError("frozen weights/history are unavailable; train first")
        model.load_weights(weights_path)
        history_values = pd.read_csv(history_path).drop(columns=["epoch"]).to_dict("list")
        previous_report = (
            json.loads(report_path.read_text()) if report_path.exists() else {}
        )
        training_seconds = float(
            previous_report.get("model", {}).get("training_seconds", float("nan"))
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5.0e-4, clipnorm=1.0), loss=loss
        )
        train_data = _dataset(
            tf, development, batch_size=args.batch_size, patch_size=args.patch_size, seed=SEED
        )
        validation_data = _dataset(
            tf,
            {VALIDATION_RE: cases[VALIDATION_RE]},
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            seed=SEED + 1000,
        )
        started = time.perf_counter()
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=24,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, min_delta=1.0e-4, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=2.0e-5
                ),
            ],
            verbose=2,
        )
        training_seconds = time.perf_counter() - started
        history_values = history.history
        model.save_weights(weights_path)
        pd.DataFrame(history_values).to_csv(history_path, index_label="epoch")

    validation_products = _predict_case(model, cases[VALIDATION_RE], VALIDATION_RE)
    validation_metrics = _case_metrics(
        validation_products, cases[VALIDATION_RE], VALIDATION_RE
    )
    gates = _validation_gates(validation_metrics)
    _write_station_figure(
        validation_metrics,
        args.output / "re100_validation_downstream.png",
        "Re=100 validation: downstream preservation",
    )

    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "development_reynolds": list(DEVELOPMENT_RE),
            "validation_reynolds": VALIDATION_RE,
            "blind_reynolds": BLIND_RE,
            "history_frames": HISTORY,
            "prediction_horizon_frames": 1,
            "snapshot_stride_lattice_steps": SNAPSHOT_STRIDE,
            "dimensionless_snapshot_spacing": SNAPSHOT_STRIDE * GRID["inflow_velocity"] / GRID["diameter"],
            "casewise_split": True,
            "teacher_forced_one_step": True,
            "claim_scope": "educational one-step Reynolds-generalization; not autonomous rollout or grid-converged DNS",
        },
        "grid": GRID,
        "model": {
            "name": model.name,
            "parameters": int(model.count_params()),
            "filters": int(args.filters),
            "patch_size": int(args.patch_size),
            "field_scales": field_scales.tolist(),
            "vorticity_scale": float(omega_scale),
            "loss_weights": LOSS_WEIGHTS,
            "optimizer": "Adam(lr=5e-4, clipnorm=1)",
            "epochs_completed": len(history_values["loss"]),
            "training_seconds": float(training_seconds),
            "weights_reused": bool(args.reuse_weights),
            "seed": SEED,
        },
        "validation": validation_metrics,
        "validation_gates": gates,
        "validation_pass": bool(all(gates.values())),
        "historical_pod_failure": {
            "artifact": "../cylinder_ml/blind_re100_lbm_vs_neural.mp4",
            "vorticity_relative_l2": 0.1670338626686869,
            "status": "retained baseline/failure example; Re=100 is now validation, not blind",
        },
    }

    if args.run_blind:
        if not report["validation_pass"]:
            raise RuntimeError(
                "Re=100 validation gates failed; the untouched Re=105 blind case was not opened"
            )
        blind_cases = _ensure_cases((BLIND_RE,), cache_dir, args.workers)
        blind_products = _predict_case(model, blind_cases[BLIND_RE], BLIND_RE)
        blind_metrics = _case_metrics(blind_products, blind_cases[BLIND_RE], BLIND_RE)
        report["blind"] = blind_metrics
        _write_station_figure(
            blind_metrics,
            args.output / "re105_blind_downstream.png",
            "Re=105 blind test: downstream preservation",
        )
        _write_video(
            blind_products,
            blind_cases[BLIND_RE],
            blind_metrics,
            args.output / "re105_lbm_vs_multiscale_cnn.mp4",
            args.output / "re105_lbm_vs_multiscale_cnn_poster.png",
        )

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation_pass": report["validation_pass"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
