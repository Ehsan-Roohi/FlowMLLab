#!/usr/bin/env python3
"""Build the fair scattered-data baseline for the Week-8 scaling audit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.gas_dynamics import shock_tube_pressure_ratio_general  # noqa: E402


# Reproduction tolerance only: platform LAPACK/BLAS changes can affect the last
# digits of the fitted RBF error norms. At the retained errors (1e-4 to 3e-2),
# this admits only about 2e-14 to 3e-12 absolute drift. It does not replace or
# relax any physical-accuracy gate in validate_week8_evidence.
REPRODUCTION_RTOL = 1.0e-10
REPRODUCTION_ATOL = 1.0e-14
METRIC_COLUMNS = ("relative_l2", "relative_linf")
BASELINE_COLUMNS = (
    "dimension", "physical_inputs", "training_states", "blind_test_states",
    "training_seed", "blind_seed", "method", "neighbors", "smoothing",
    *METRIC_COLUMNS, "same_random_training_budget_as_mlp",
)


def _baseline_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        records = list(csv.reader(stream))
    if not records or tuple(records[0]) != BASELINE_COLUMNS:
        raise ValueError(f"Baseline schema mismatch: {path}")
    if len(records) != 5:
        raise ValueError(f"Baseline row count mismatch: {path}; expected four cases")
    if any(len(row) != len(BASELINE_COLUMNS) for row in records[1:]):
        raise ValueError(f"Baseline row width mismatch: {path}")
    rows = [dict(zip(BASELINE_COLUMNS, row, strict=True)) for row in records[1:]]
    if [row["dimension"] for row in rows] != ["2", "3", "4", "5"]:
        raise ValueError(f"Baseline case order mismatch: {path}")
    return rows


def compare_baseline(generated: Path, reference: Path) -> dict[str, object]:
    """Compare replay errors within roundoff; all other CSV values are exact.

    Neither file is modified. Header/order, every case and every metadata value
    (including seeds, budgets, inputs and RBF settings) must match exactly. Only
    the two finite, nonnegative error norms admit the fixed reproduction bound
    abs(new - reference) <= atol + rtol * abs(reference).
    """
    expected = _baseline_rows(reference)
    actual = _baseline_rows(generated)
    maximum_delta = dict.fromkeys(METRIC_COLUMNS, 0.0)
    for row_number, (wanted, found) in enumerate(zip(expected, actual, strict=True), 1):
        for column in BASELINE_COLUMNS:
            if column not in METRIC_COLUMNS:
                if found[column] != wanted[column]:
                    raise ValueError(f"Baseline metadata mismatch: row {row_number}, {column}")
                continue
            try:
                old, new = float(wanted[column]), float(found[column])
            except ValueError as exc:
                raise ValueError(f"Non-numeric baseline metric: row {row_number}, {column}") from exc
            if not (math.isfinite(old) and math.isfinite(new) and old >= 0 and new >= 0):
                raise ValueError(f"Non-finite or negative baseline metric: row {row_number}, {column}")
            delta = abs(new - old)
            allowed = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(old)
            if delta > allowed:
                raise ValueError(
                    f"Baseline metric drift: row {row_number}, {column}; "
                    f"difference {delta:.17g} exceeds roundoff bound {allowed:.17g}"
                )
            maximum_delta[column] = max(maximum_delta[column], delta)
    return {
        "status": "pass", "rows": len(actual),
        "rtol": REPRODUCTION_RTOL, "atol": REPRODUCTION_ATOL,
        "numeric_columns": list(METRIC_COLUMNS),
        "exact_metadata_columns": [c for c in BASELINE_COLUMNS if c not in METRIC_COLUMNS],
        "maximum_absolute_delta": maximum_delta,
    }


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/gas_dynamics_week8/scattered_baseline.csv",
    )
    parser.add_argument(
        "--compare-to", type=Path,
        help="Check an isolated regeneration against this untouched frozen CSV.",
    )
    args = parser.parse_args(argv)
    if args.compare_to is not None and (
        args.output.resolve() == args.compare_to.resolve()
        or (args.output.exists() and args.compare_to.exists()
            and args.output.samefile(args.compare_to))
    ):
        parser.error("--output must be separate from the --compare-to reference")
    rows = run(args.output)
    if args.compare_to is not None:
        report = compare_baseline(args.output, args.compare_to)
        print(
            "PASS: exact case metadata; metric reproduction within "
            f"rtol={REPRODUCTION_RTOL:g}, atol={REPRODUCTION_ATOL:g}. "
            f"Maximum absolute differences: {report['maximum_absolute_delta']}"
        )
    for row in rows:
        print(
            f"d={row['dimension']}: RBF relative L2 = "
            f"{100.0 * float(row['relative_l2']):.4f}%"
        )


if __name__ == "__main__":
    main()
