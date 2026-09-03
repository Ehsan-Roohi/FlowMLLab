#!/usr/bin/env python3
"""Generate the bounded probabilistic-UQ evidence from the cavity archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.w4utils import field_physics_metrics  # noqa: E402
from flowmllab.probabilistic_uq import (  # noqa: E402
    GaussianPrediction,
    calibration_curve,
    fit_pod_gaussian_process,
    gaussian_crps,
    gaussian_negative_log_likelihood,
    interpolate_complete_cases,
    predict_pod_gaussian_process,
    relative_l2_per_case,
    rescale_prediction,
    validation_scale_factor,
)


TRAIN_REYNOLDS = np.array([100.0, 150.0, 200.0, 225.0, 250.0, 350.0, 400.0])
VALIDATION_REYNOLDS = 300.0
BLIND_REYNOLDS = np.array([175.0, 275.0, 375.0])
POD_RANK = 4
LENGTH_SCALE = 0.75
NOISE_LEVEL = 1.0e-8
CALIBRATION_LEVEL = 0.9
CALIBRATION_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _indices(reynolds: np.ndarray, requested: np.ndarray) -> np.ndarray:
    found = []
    for value in requested:
        matches = np.flatnonzero(np.isclose(reynolds, value))
        if matches.size != 1:
            raise ValueError(f"Expected exactly one cavity case at Re={value:g}")
        found.append(int(matches[0]))
    return np.asarray(found, dtype=int)


def _interior(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[..., 1:-1, 1:-1]


def _safe_scale(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 1.0e-12)


def generate(root: Path) -> dict[str, object]:
    data_path = root / "data" / "cavity_data.npz"
    output = root / "results" / "probabilistic_uq"
    output.mkdir(parents=True, exist_ok=True)

    with np.load(data_path, allow_pickle=False) as archive:
        reynolds = np.asarray(archive["Re"], dtype=float)
        split = np.asarray(archive["split"]).astype(str)
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        fields = np.stack(
            [np.asarray(archive["u"], dtype=float), np.asarray(archive["v"], dtype=float)],
            axis=1,
        )

    train_index = _indices(reynolds, TRAIN_REYNOLDS)
    validation_index = _indices(reynolds, np.array([VALIDATION_REYNOLDS]))
    blind_index = _indices(reynolds, BLIND_REYNOLDS)
    if not np.all(split[blind_index] == "test"):
        raise ValueError("The declared blind Reynolds cases do not match the archive split")
    if np.any(np.isin(TRAIN_REYNOLDS, BLIND_REYNOLDS)):
        raise ValueError("A blind Reynolds case leaked into training")

    model = fit_pod_gaussian_process(
        TRAIN_REYNOLDS,
        fields[train_index],
        rank=POD_RANK,
        length_scale=LENGTH_SCALE,
        noise_level=NOISE_LEVEL,
    )
    validation_raw = predict_pod_gaussian_process(model, VALIDATION_REYNOLDS)
    validation_truth = fields[validation_index[0]]
    scale_factor = validation_scale_factor(
        _interior(validation_truth),
        GaussianPrediction(
            mean=_interior(validation_raw.mean), std=_interior(validation_raw.std)
        ),
        target_level=CALIBRATION_LEVEL,
    )

    blind_truth = fields[blind_index]
    blind_raw = predict_pod_gaussian_process(model, BLIND_REYNOLDS)
    blind_calibrated = rescale_prediction(blind_raw, scale_factor)
    baseline = interpolate_complete_cases(
        TRAIN_REYNOLDS, fields[train_index], BLIND_REYNOLDS
    )
    gp_errors = relative_l2_per_case(blind_truth, blind_raw.mean)
    baseline_errors = relative_l2_per_case(blind_truth, baseline)

    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for case, reynolds_value in enumerate(BLIND_REYNOLDS):
        truth = _interior(blind_truth[case])
        raw_mean = _interior(blind_raw.mean[case])
        raw_std = _safe_scale(_interior(blind_raw.std[case]))
        calibrated_std = _safe_scale(_interior(blind_calibrated.std[case]))
        raw_curve = calibration_curve(
            truth,
            GaussianPrediction(mean=raw_mean, std=raw_std),
            levels=CALIBRATION_LEVELS,
        )
        calibrated_curve = calibration_curve(
            truth,
            GaussianPrediction(mean=raw_mean, std=calibrated_std),
            levels=CALIBRATION_LEVELS,
        )
        correlation = spearmanr(
            np.abs(truth - raw_mean).reshape(-1), calibrated_std.reshape(-1)
        ).statistic
        physics = field_physics_metrics(
            x,
            y,
            blind_raw.mean[case, 0],
            blind_raw.mean[case, 1],
        )
        metric_rows.append(
            {
                "Re": float(reynolds_value),
                "pod_gp_relative_L2_uv": float(gp_errors[case]),
                "interpolation_relative_L2_uv": float(baseline_errors[case]),
                "raw_gaussian_nll": gaussian_negative_log_likelihood(
                    truth, raw_mean, raw_std
                ),
                "calibrated_gaussian_nll": gaussian_negative_log_likelihood(
                    truth, raw_mean, calibrated_std
                ),
                "raw_gaussian_crps": gaussian_crps(truth, raw_mean, raw_std),
                "calibrated_gaussian_crps": gaussian_crps(
                    truth, raw_mean, calibrated_std
                ),
                "raw_90_coverage": float(raw_curve.observed[2]),
                "calibrated_90_coverage": float(calibrated_curve.observed[2]),
                "calibrated_90_mean_width": float(calibrated_curve.mean_width[2]),
                "error_spread_spearman": float(correlation),
                **physics,
            }
        )
        for level_index, level in enumerate(CALIBRATION_LEVELS):
            calibration_rows.append(
                {
                    "Re": float(reynolds_value),
                    "nominal_level": float(level),
                    "raw_coverage": float(raw_curve.observed[level_index]),
                    "calibrated_coverage": float(calibrated_curve.observed[level_index]),
                    "raw_mean_width": float(raw_curve.mean_width[level_index]),
                    "calibrated_mean_width": float(
                        calibrated_curve.mean_width[level_index]
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    calibration = pd.DataFrame(calibration_rows)
    metrics.to_csv(output / "blind_metrics.csv", index=False, lineterminator="\n")
    calibration.to_csv(output / "calibration.csv", index=False, lineterminator="\n")

    aggregate_raw = calibration_curve(
        _interior(blind_truth),
        GaussianPrediction(
            mean=_interior(blind_raw.mean), std=_safe_scale(_interior(blind_raw.std))
        ),
        levels=CALIBRATION_LEVELS,
    )
    aggregate_calibrated = calibration_curve(
        _interior(blind_truth),
        GaussianPrediction(
            mean=_interior(blind_calibrated.mean),
            std=_safe_scale(_interior(blind_calibrated.std)),
        ),
        levels=CALIBRATION_LEVELS,
    )
    summary: dict[str, object] = {
        "status": "pass",
        "training_cases": TRAIN_REYNOLDS.tolist(),
        "validation_case": VALIDATION_REYNOLDS,
        "blind_cases": BLIND_REYNOLDS.tolist(),
        "pod_rank": POD_RANK,
        "length_scale_normalized_Re": LENGTH_SCALE,
        "gp_noise_level": NOISE_LEVEL,
        "validation_scale_factor": float(scale_factor),
        "mean_pod_gp_relative_L2_uv": float(gp_errors.mean()),
        "mean_interpolation_relative_L2_uv": float(baseline_errors.mean()),
        "aggregate_raw_90_coverage": float(aggregate_raw.observed[2]),
        "aggregate_calibrated_90_coverage": float(aggregate_calibrated.observed[2]),
        "max_wall_error": float(metrics["wall_max_error"].max()),
        "max_divergence_L2": float(metrics["div_l2_pred"].max()),
        "coverage_interpretation": (
            "Descriptive pointwise coverage; spatial CFD errors are correlated, so this is "
            "not an independent-sample coverage guarantee."
        ),
    }
    protocol = {
        "module": "Week 2.1 — Probabilistic UQ for CFD surrogates",
        "dataset": "data/cavity_data.npz",
        "dataset_sha256": _digest(data_path),
        "split_unit": "complete Reynolds-number case",
        "training_cases": TRAIN_REYNOLDS.tolist(),
        "validation_case": VALIDATION_REYNOLDS,
        "blind_cases": BLIND_REYNOLDS.tolist(),
        "frozen_model": {
            "representation": "velocity-only centered POD",
            "rank": POD_RANK,
            "coefficient_model": "independent fixed-hyperparameter Gaussian processes",
            "kernel": "1.0 * RBF(length_scale=0.75) + WhiteKernel(1e-8)",
        },
        "calibration": {
            "method": "single multiplicative standard-deviation scale",
            "fit_case": VALIDATION_REYNOLDS,
            "target_level": CALIBRATION_LEVEL,
            "claim_boundary": (
                "Validation-only descriptive rescaling; no finite-sample guarantee is claimed "
                "for correlated grid nodes or shifted physical regimes."
            ),
        },
        "baseline": "piecewise-linear complete-case interpolation",
        "public_sources": [
            "https://gaussianprocess.org/gpml/",
            "https://doi.org/10.1198/016214506000001437",
            "https://doi.org/10.1111/1467-9868.00294",
        ],
        "originality_declaration": (
            "All prose, equations, code, figures, and exercises in this Week-2.1 module were created "
            "for FlowMLLab from public sources and FlowMLLab-owned data. No restricted course "
            "handout, solution, figure, or code was incorporated."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    representative = int(np.flatnonzero(np.isclose(BLIND_REYNOLDS, 275.0))[0])
    truth_u = blind_truth[representative, 0]
    truth_v = blind_truth[representative, 1]
    mean_u = blind_raw.mean[representative, 0]
    mean_v = blind_raw.mean[representative, 1]
    truth_speed = np.hypot(truth_u, truth_v)
    mean_speed = np.hypot(mean_u, mean_v)
    vector_error = np.hypot(mean_u - truth_u, mean_v - truth_v)
    spread = np.hypot(
        blind_calibrated.std[representative, 0],
        blind_calibrated.std[representative, 1],
    )

    figure, axes = plt.subplots(1, 5, figsize=(18, 3.8), constrained_layout=True)
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    panels = (
        (truth_speed, "CFD speed"),
        (mean_speed, "POD-GP mean speed"),
        (vector_error, "Vector error"),
        (spread, "Calibrated marginal spread"),
    )
    for axis, (values, title) in zip(axes[:4], panels):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="equal")
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title(title)
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[4].plot(
        aggregate_raw.nominal,
        aggregate_raw.observed,
        "o--",
        label="raw GP",
    )
    axes[4].plot(
        aggregate_calibrated.nominal,
        aggregate_calibrated.observed,
        "s-",
        label="validation-scaled",
    )
    axes[4].plot([0.45, 1.0], [0.45, 1.0], "k:", label="ideal")
    axes[4].set(xlim=(0.45, 1.0), ylim=(0.0, 1.0), xlabel="Nominal", ylabel="Observed")
    axes[4].set_title("Blind pointwise coverage")
    axes[4].legend(fontsize=8)
    figure.suptitle("FlowMLLab probabilistic UQ evidence (field panels: blind Re=275)")
    figure.savefig(output / "probabilistic_uq_validation.png", dpi=180)
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = generate(args.root.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
