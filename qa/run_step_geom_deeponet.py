#!/usr/bin/env python3
"""Train and evaluate the leakage-free micro-step Geom-DeepONet.

The program enforces the existing FlowMLLab 5/2/2 geometry protocol.  It opens
only the seven-case learning archive for development, selects the zonal weight
on H33/H58, refits frozen models on all seven permitted cases, and only then
opens the physically separate H44/H67 test archive.

This is a development benchmark.  It does not overwrite retained release
evidence unless an explicit output directory is supplied.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
    STEP_SOURCE_COMMIT,
    load_step_height_archive,
)
from flowmllab.step_geom_deeponet import (
    GEOM_DEEPONET_DOI,
    evaluate_step_geom_deeponet,
    fit_step_geom_deeponet,
    infer_step_domain,
)


def mean_percent(rows: list[dict[str, float | int]], key: str) -> float:
    return 100.0 * float(np.mean([float(row[key]) for row in rows]))


def serializable_rows(rows: list[dict[str, float | int]]) -> list[dict[str, Any]]:
    return [
        {
            key: int(value) if isinstance(value, (int, np.integer)) else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--points-per-case", type=int, default=4096)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--omega-0", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=690)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--global-guard-percent", type=float, default=2.0)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for JSON report and final zonal weights",
    )
    args = parser.parse_args()
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas must be unique")
    if any(not 0.0 < alpha < 1.0 for alpha in args.alphas):
        raise ValueError("every --alphas value must lie strictly between zero and one")

    output: Path | None = None
    if args.output_dir is not None:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf

    def clear_finished_model() -> None:
        tf.keras.backend.clear_session()
        gc.collect()

    root = args.root.resolve()
    start = time.perf_counter()
    selection_rule = (
        "Minimize mean validation vortex relative L2 among candidates whose "
        f"mean validation global relative L2 is at most {args.global_guard_percent:g} "
        "percentage points above the unweighted baseline."
    )

    def persist_selection_checkpoint(
        status: str,
        rows: list[dict[str, Any]],
    ) -> None:
        checkpoint = {
            "status": status,
            "source_commit": STEP_SOURCE_COMMIT,
            "reference_doi": GEOM_DEEPONET_DOI,
            "split": {
                "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
                "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
                "held_out_test_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
            },
            "file_level_test_isolation": True,
            "test_archive_opened": False,
            "test_used_for_selection": False,
            "configuration": {
                "epochs": args.epochs,
                "points_per_case": args.points_per_case,
                "width": args.width,
                "omega_0": args.omega_0,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "alphas": [float(alpha) for alpha in args.alphas],
                "global_guard_percent": args.global_guard_percent,
            },
            "selection_rule": selection_rule,
            "selection": rows,
            "software": {"numpy": np.__version__, "tensorflow": tf.__version__},
            "elapsed_seconds": time.perf_counter() - start,
        }
        print(f"SELECTION_STATUS={status}", flush=True)
        if output is not None:
            selection_path = output / "step_geom_deeponet_selection.json"
            temporary_path = selection_path.with_suffix(".json.tmp")
            temporary_path.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(selection_path)
            print(f"selection: {selection_path}", flush=True)

    # Selection phase: the separate H44/H67 archive is not touched here.
    learning_cases = load_step_height_archive(root, split="learning")
    first = learning_cases[int(STEP_HEIGHT_DEVELOPMENT_PERCENT[0])]
    domain = infer_step_domain(first["x"], first["y"])
    fit_options = {
        "domain": domain,
        "points_per_case": args.points_per_case,
        "width": args.width,
        "omega_0": args.omega_0,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "verbose": args.verbose,
    }

    baseline = fit_step_geom_deeponet(
        learning_cases,
        STEP_HEIGHT_DEVELOPMENT_PERCENT,
        alpha=None,
        **fit_options,
    )
    baseline_validation = evaluate_step_geom_deeponet(
        baseline,
        learning_cases,
        STEP_HEIGHT_VALIDATION_PERCENT,
    )
    baseline_global = mean_percent(baseline_validation, "full_relative_l2")
    selection: list[dict[str, Any]] = [
        {
            "model": "unweighted",
            "alpha": None,
            "validation_global_relative_l2_percent": baseline_global,
            "validation_vortex_relative_l2_percent": mean_percent(
                baseline_validation, "vortex_relative_l2"
            ),
            "eligible_global_guard": True,
            "selected": False,
            "final_training_loss": float(baseline.history["loss"][-1]),
            "minimum_training_loss": float(min(baseline.history["loss"])),
            "validation_by_geometry": serializable_rows(baseline_validation),
        }
    ]
    print(
        "VALIDATION_SELECTION=" + json.dumps(selection[-1], sort_keys=True),
        flush=True,
    )
    persist_selection_checkpoint("selection_in_progress", selection)
    del baseline
    clear_finished_model()

    for alpha in args.alphas:
        fitted = fit_step_geom_deeponet(
            learning_cases,
            STEP_HEIGHT_DEVELOPMENT_PERCENT,
            alpha=float(alpha),
            **fit_options,
        )
        rows = evaluate_step_geom_deeponet(
            fitted,
            learning_cases,
            STEP_HEIGHT_VALIDATION_PERCENT,
        )
        global_error = mean_percent(rows, "full_relative_l2")
        selection.append(
            {
                "model": "zonal",
                "alpha": float(alpha),
                "validation_global_relative_l2_percent": global_error,
                "validation_vortex_relative_l2_percent": mean_percent(
                    rows, "vortex_relative_l2"
                ),
                "eligible_global_guard": global_error
                <= baseline_global + args.global_guard_percent,
                "selected": False,
                "final_training_loss": float(fitted.history["loss"][-1]),
                "minimum_training_loss": float(min(fitted.history["loss"])),
                "validation_by_geometry": serializable_rows(rows),
            }
        )
        print(
            "VALIDATION_SELECTION=" + json.dumps(selection[-1], sort_keys=True),
            flush=True,
        )
        persist_selection_checkpoint("selection_in_progress", selection)
        del fitted
        clear_finished_model()

    eligible = [
        row for row in selection[1:] if bool(row["eligible_global_guard"])
    ]
    if not eligible:
        persist_selection_checkpoint("no_eligible_zonal_candidate", selection)
        print("NO_ELIGIBLE_ZONAL_CANDIDATE", flush=True)
        raise RuntimeError("no zonal model satisfies the predeclared global-error guard")
    winner = min(
        eligible,
        key=lambda row: float(row["validation_vortex_relative_l2_percent"]),
    )
    selected_alpha = float(winner["alpha"])
    winner["selected"] = True
    persist_selection_checkpoint("selection_complete", selection)

    # Freeze architecture, optimization budget, and alpha; refit on all seven cases.
    final_heights = np.concatenate(
        (STEP_HEIGHT_DEVELOPMENT_PERCENT, STEP_HEIGHT_VALIDATION_PERCENT)
    )
    clear_finished_model()
    final_unweighted = fit_step_geom_deeponet(
        learning_cases,
        final_heights,
        alpha=None,
        **fit_options,
    )
    final_zonal = fit_step_geom_deeponet(
        learning_cases,
        final_heights,
        alpha=selected_alpha,
        **fit_options,
    )

    # This is the first access to the physically separate sealed-test file.
    test_cases = load_step_height_archive(root, split="test")
    test_metrics = {
        "unweighted": serializable_rows(
            evaluate_step_geom_deeponet(
                final_unweighted,
                test_cases,
                STEP_HEIGHT_HELD_OUT_PERCENT,
            )
        ),
        f"zonal_alpha_{selected_alpha:.2f}": serializable_rows(
            evaluate_step_geom_deeponet(
                final_zonal,
                test_cases,
                STEP_HEIGHT_HELD_OUT_PERCENT,
            )
        ),
    }

    report = {
        "status": "development benchmark; not retained release evidence",
        "source_commit": STEP_SOURCE_COMMIT,
        "reference_doi": GEOM_DEEPONET_DOI,
        "split": {
            "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
            "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
            "held_out_test_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
        },
        "file_level_test_isolation": True,
        "test_archive_opened_after_selection_and_final_fit": True,
        "test_used_for_selection": False,
        "allowed_model_inputs": [
            "geometry branch: h/H",
            "point trunk: normalized x, normalized y, analytic SDF/H",
        ],
        "forbidden_model_inputs": [
            "U or V patch",
            "target-derived mask",
            "held-out flow field",
        ],
        "domain": asdict(domain),
        "architecture": {
            "kind": "2-D Geom-DeepONet adaptation",
            "intermediate_branch_trunk_fusion": True,
            "global_point_pooling": "mean",
            "post_fusion_trunk": "SIREN",
            "output_fields": ["U", "V"],
            **final_zonal.configuration,
        },
        "selection_rule": selection_rule,
        "selection": selection,
        "selected_alpha": selected_alpha,
        "test_metrics": test_metrics,
        "software": {"numpy": np.__version__, "tensorflow": tf.__version__},
        "elapsed_seconds": time.perf_counter() - start,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        report_path = output / "step_geom_deeponet_report.json"
        weights_path = output / "step_geom_deeponet_zonal.weights.h5"
        report_path.write_text(rendered, encoding="utf-8")
        final_zonal.model.save_weights(weights_path)
        print(f"report: {report_path}")
        print(f"weights: {weights_path}")
    print(rendered, end="")


if __name__ == "__main__":
    main()
