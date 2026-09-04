#!/usr/bin/env python3
"""Build the fair scattered-data baseline for the Week-8 scaling audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.gas_dynamics import shock_tube_pressure_ratio_general  # noqa: E402


RANGES = {
    "log_p4_p1": (np.log(1.5), np.log(300.0)),
    "log_t4_t1": (np.log(0.5), np.log(2.0)),
    "gamma": (1.25, 1.67),
    "gamma1": (1.25, 1.67),
    "gamma4": (1.25, 1.67),
    "log_r4_r1": (np.log(0.5), np.log(8.0)),
}


def specification(dimension: int) -> list[str]:
    return {
        2: ["log_p4_p1", "log_t4_t1"],
        3: ["log_p4_p1", "log_t4_t1", "gamma"],
        4: ["log_p4_p1", "log_t4_t1", "gamma1", "gamma4"],
        5: ["log_p4_p1", "log_t4_t1", "gamma1", "gamma4", "log_r4_r1"],
    }[dimension]


def sample(dimension: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = specification(dimension)
    lower = np.array([RANGES[name][0] for name in names])
    upper = np.array([RANGES[name][1] for name in names])
    unit = qmc.LatinHypercube(d=dimension, seed=seed).random(count)
    return qmc.scale(unit, lower, upper), lower, upper


def exact(values: np.ndarray, dimension: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(values, dtype=float))
    p4 = np.exp(x[:, 0])
    t4 = np.exp(x[:, 1])
    if dimension == 2:
        gamma_1 = gamma_4 = np.full(len(x), 1.4)
        gas_ratio = np.ones(len(x))
    elif dimension == 3:
        gamma_1 = gamma_4 = x[:, 2]
        gas_ratio = np.ones(len(x))
    elif dimension == 4:
        gamma_1, gamma_4 = x[:, 2], x[:, 3]
        gas_ratio = np.ones(len(x))
    else:
        gamma_1, gamma_4 = x[:, 2], x[:, 3]
        gas_ratio = np.exp(x[:, 4])
    return np.array([
        shock_tube_pressure_ratio_general(p, t, g1, g4, ratio)
        for p, t, g1, g4, ratio in zip(
            p4, t4, gamma_1, gamma_4, gas_ratio, strict=True
        )
    ])


def run(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dimension in range(2, 6):
        train, lower, upper = sample(dimension, 4096, 41 + 10 * dimension)
        test, _, _ = sample(dimension, 2048, 41 + 100 + dimension)
        train_target = exact(train, dimension)
        test_target = exact(test, dimension)
        train_fraction = (train_target - 1.0) / (np.exp(train[:, 0]) - 1.0)
        train_scaled = (train - lower) / (upper - lower)
        test_scaled = (test - lower) / (upper - lower)
        interpolator = RBFInterpolator(
            train_scaled,
            train_fraction,
            kernel="thin_plate_spline",
            neighbors=128,
            smoothing=1.0e-8,
        )
        predicted_fraction = interpolator(test_scaled)
        prediction = 1.0 + predicted_fraction * (np.exp(test[:, 0]) - 1.0)
        relative_l2 = np.linalg.norm(prediction - test_target) / np.linalg.norm(test_target)
        relative_linf = np.max(np.abs(prediction - test_target)) / np.max(
            np.abs(test_target)
        )
        rows.append({
            "dimension": dimension,
            "physical_inputs": ";".join(specification(dimension)),
            "training_states": 4096,
            "blind_test_states": 2048,
            "training_seed": 41 + 10 * dimension,
            "blind_seed": 41 + 100 + dimension,
            "method": "local_thin_plate_RBF",
            "neighbors": 128,
            "smoothing": 1.0e-8,
            "relative_l2": float(relative_l2),
            "relative_linf": float(relative_linf),
            "same_random_training_budget_as_mlp": True,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/gas_dynamics_week8/scattered_baseline.csv",
    )
    args = parser.parse_args()
    rows = run(args.output)
    for row in rows:
        print(
            f"d={row['dimension']}: RBF relative L2 = "
            f"{100.0 * float(row['relative_l2']):.4f}%"
        )


if __name__ == "__main__":
    main()
