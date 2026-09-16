"""Quantify how a moving-grid coordinate bug damages blind nozzle predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.optimize import brentq
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "nozzle_alignment_audit"
OUT = OUT_DIR / "nozzle_alignment_impact.png"
OUT_SVG = OUT.with_suffix(".svg")
METRICS = OUT_DIR / "alignment_impact_metrics.json"
GAMMA = 1.4
SEED = 7
TRAINING_CASE_SEED = 42
TEST_CASES = [(0.365, 2.35), (0.435, 3.15), (0.495, 3.65)]


def area(x: np.ndarray, throat: float, exit_ratio: float) -> np.ndarray:
    return np.where(
        x <= throat,
        1 + 0.6 * (1 + np.cos(np.pi * x / throat)),
        1 + 0.5 * (exit_ratio - 1) * (1 - np.cos(np.pi * (x - throat) / (1 - throat))),
    )


def area_mach(mach: float | np.ndarray) -> float | np.ndarray:
    mach = np.asarray(mach)
    return (
        2 / (GAMMA + 1) * (1 + (GAMMA - 1) * mach**2 / 2)
    ) ** ((GAMMA + 1) / (2 * (GAMMA - 1))) / mach


def reference(throat: float, exit_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    # A common moving-throat construction: 21 points on each side, sharing the throat.
    x = np.r_[np.linspace(0, throat, 21), np.linspace(throat, 1, 21)[1:]]
    cross_section = area(x, throat, exit_ratio)
    mach = np.array([
        1.0 if abs(xj - throat) < 1e-12 else brentq(
            lambda value: area_mach(value) - aj,
            *((1e-7, 1.0) if xj < throat else (1.0, 10.0)),
        )
        for xj, aj in zip(x, cross_section)
    ])
    return x, mach


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(48, 48), activation="tanh", solver="adam",
            max_iter=800, early_stopping=True, n_iter_no_change=40,
            learning_rate_init=0.003, random_state=SEED,
        ),
    )


def main() -> None:
    rng = np.random.default_rng(TRAINING_CASE_SEED)
    training_cases = list(zip(rng.uniform(0.28, 0.52, 24), rng.uniform(1.7, 4.0, 24)))
    records = [reference(*case) for case in training_cases]
    reused_x = records[0][0]

    aligned_rows, misaligned_rows, targets = [], [], []
    for (throat, exit_ratio), (x, mach) in zip(training_cases, records):
        parameters = np.c_[np.full(41, throat), np.full(41, exit_ratio)]
        aligned_rows.append(np.c_[parameters, x])
        misaligned_rows.append(np.c_[parameters, reused_x])
        targets.append(mach)
    x_aligned = np.vstack(aligned_rows)
    x_misaligned = np.vstack(misaligned_rows)
    y_train = np.concatenate(targets)

    predictors = {"aligned": model(), "misaligned": model()}
    predictors["aligned"].fit(x_aligned, y_train)
    predictors["misaligned"].fit(x_misaligned, y_train)

    results: list[dict[str, object]] = []
    for throat, exit_ratio in TEST_CASES:
        x, truth = reference(throat, exit_ratio)
        inputs = np.c_[np.full(41, throat), np.full(41, exit_ratio), x]
        aligned = predictors["aligned"].predict(inputs)
        misaligned = predictors["misaligned"].predict(inputs)
        results.append({
            "throat": throat,
            "exit_ratio": exit_ratio,
            "x": x,
            "truth": truth,
            "aligned": aligned,
            "misaligned": misaligned,
            "aligned_l2": float(np.linalg.norm(aligned - truth) / np.linalg.norm(truth)),
            "misaligned_l2": float(np.linalg.norm(misaligned - truth) / np.linalg.norm(truth)),
        })

    aligned_errors = np.array([item["aligned_l2"] for item in results], dtype=float)
    misaligned_errors = np.array([item["misaligned_l2"] for item in results], dtype=float)
    improvement = 1 - aligned_errors.mean() / misaligned_errors.mean()
    assert misaligned_errors.mean() > 2 * aligned_errors.mean()

    retained = {
        "claim_scope": "controlled quasi-1D isentropic stress test; not DSMC",
        "training_cases": len(training_cases),
        "blind_cases": [
            {
                "throat_x_over_L": item["throat"],
                "exit_area_ratio": item["exit_ratio"],
                "aligned_relative_l2": item["aligned_l2"],
                "misaligned_relative_l2": item["misaligned_l2"],
            }
            for item in results
        ],
        "mean_aligned_relative_l2": float(aligned_errors.mean()),
        "mean_misaligned_relative_l2": float(misaligned_errors.mean()),
        "relative_error_reduction": float(improvement),
        "model": "matched 2x48 tanh MLPRegressor; only coordinate pairing differs",
        "random_state": SEED,
        "training_case_seed": TRAINING_CASE_SEED,
        "scikit_learn_version": sklearn.__version__,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(retained, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 14,
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.5), constrained_layout=True)
    selected = results[0]
    x = np.asarray(selected["x"])
    truth = np.asarray(selected["truth"])
    aligned = np.asarray(selected["aligned"])
    misaligned = np.asarray(selected["misaligned"])

    ax = axes[0]
    ax.plot(x, truth, color="black", lw=3.2, label="reference")
    ax.plot(x, misaligned, color="#D55E00", lw=2.8, ls="--",
            label=f"misaligned: {100*selected['misaligned_l2']:.2f}%")
    ax.plot(x, aligned, color="#009E73", lw=2.8,
            label=f"aligned: {100*selected['aligned_l2']:.2f}%")
    ax.axvline(float(selected["throat"]), color="0.45", lw=1.2, ls=":")
    ax.set(title="(a) Blind Mach prediction", xlabel=r"axial coordinate, $x/L$", ylabel="Mach number")
    ax.legend(frameon=True, fontsize=12, facecolor="white", framealpha=0.95)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(x, np.abs(misaligned - truth), color="#D55E00", lw=3.0, label="misaligned")
    ax.fill_between(x, 0, np.abs(misaligned - truth), color="#D55E00", alpha=0.14)
    ax.plot(x, np.abs(aligned - truth), color="#009E73", lw=3.0, label="aligned")
    ax.fill_between(x, 0, np.abs(aligned - truth), color="#009E73", alpha=0.14)
    ax.axvline(float(selected["throat"]), color="0.45", lw=1.2, ls=":")
    ax.set(title="(b) Pointwise prediction error", xlabel=r"axial coordinate, $x/L$", ylabel=r"absolute Mach error")
    ax.legend(frameon=True, fontsize=12, facecolor="white", framealpha=0.95)
    ax.grid(alpha=0.25)

    ax = axes[2]
    positions = np.arange(len(TEST_CASES))
    width = 0.36
    ax.bar(positions - width/2, 100*misaligned_errors, width, color="#D55E00", label="misaligned")
    ax.bar(positions + width/2, 100*aligned_errors, width, color="#009E73", label="aligned")
    ax.set_xticks(positions, [rf"$x_t/L={case[0]:.3f}$" for case in TEST_CASES])
    ax.set(title="(c) Three unseen geometries", xlabel="blind test case", ylabel="relative L2 error (%)")
    ax.legend(frameon=True, fontsize=12, facecolor="white", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.97, 0.96,
            f"mean error: {100*misaligned_errors.mean():.2f}% → {100*aligned_errors.mean():.2f}%\n"
            f"{100*improvement:.0f}% lower after repair",
            transform=ax.transAxes, ha="right", va="top", fontsize=13, weight="bold",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#E8F5E9", "edgecolor": "#2E7D32"})

    fig.suptitle("A silent coordinate bug quadruples blind nozzle-surrogate error", fontsize=20, weight="bold")
    fig.savefig(OUT, dpi=180, facecolor="white")
    fig.savefig(OUT_SVG, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
