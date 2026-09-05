"""Regenerate compact, executed evidence for the Week-7.1 cylinder lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from flowmllab.hypersonic_cylinder import (
    TARGET_NAMES,
    case_interpolation_baseline,
    casewise_split_masks,
    fit_cylinder_mlp,
    load_cylinder_teaching_data,
    relative_l2,
    validate_hypersonic_cylinder_evidence,
)


def error_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(TARGET_NAMES, values, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = (args.output_dir or root / "tmp/hypersonic_cylinder_week7_1").resolve()
    if output == root / "results" or root / "results" in output.parents:
        raise ValueError("Use scratch output; retain reviewed results separately")
    output.mkdir(parents=True, exist_ok=True)
    data = load_cylinder_teaching_data(root)
    masks = casewise_split_masks(data.mach_inf)

    started = time.perf_counter()
    model = fit_cylinder_mlp(data, masks["train"], masks["validation"])
    fit_seconds = time.perf_counter() - started

    metrics: dict[str, object] = {
        "schema_version": 2,
        "status": "pass",
        "evidence_contract": validate_hypersonic_cylinder_evidence(root),
        "teaching_model": {
            "name": "trained 3x96 tanh MLP",
            "seed": 760,
            "max_epochs": 300,
            "selected_epoch": model.best_epoch,
            "validation_standardized_mse": model.validation_history,
            "fit_seconds": fit_seconds,
            "claim": "teaching analog; not the published Fusion-DeepONet",
        },
        "splits": {},
    }

    for name, mask in masks.items():
        mean = model.predict(data.mach_inf[mask], data.x[mask], data.y[mask])
        baseline = case_interpolation_baseline(data, masks["train"], mask)
        truth = data.targets[mask]
        metrics["splits"][name] = {
            "cases": sorted(np.unique(data.mach_inf[mask]).tolist()),
            "points": int(np.count_nonzero(mask)),
            "linear_case_baseline_relative_l2": error_dict(relative_l2(truth, baseline)),
            "teaching_operator_relative_l2": error_dict(relative_l2(truth, mean)),
        }

    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    case_mask = np.isclose(data.mach_inf, 8.5)
    truth = data.targets[case_mask]
    baseline = case_interpolation_baseline(data, masks["train"], case_mask)
    x = data.x[case_mask]
    y = data.y[case_mask]
    labels = (r"Local $M$", "source temperature (TOV)", "source pressure (P)")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)
    for column, label in enumerate(labels):
        top = axes[0, column].tricontourf(x, y, truth[:, column], levels=28, cmap="viridis")
        axes[0, column].set_title(f"DSMC derivative: {label}")
        fig.colorbar(top, ax=axes[0, column], shrink=0.82)
        error = np.abs(baseline[:, column] - truth[:, column])
        bottom = axes[1, column].tricontourf(x, y, error, levels=28, cmap="magma")
        axes[1, column].set_title(f"Mach interpolation |error|: {label}")
        fig.colorbar(bottom, ax=axes[1, column], shrink=0.82)
        for row in range(2):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
    fig.suptitle(r"Week 7.1 retained interpolation case: $M_\infty=8.5$")
    figure_path = output / "mach85_baseline_audit.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
