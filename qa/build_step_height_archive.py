#!/usr/bin/env python3
"""Build the permitted compact nine-case micro-step archive for FlowMLLab."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (  # noqa: E402
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
    STEP_SOURCE_COMMIT,
    load_step_height_cases,
    validate_step_height_dataset,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_archive(
    output: Path,
    selected_heights: np.ndarray,
    cases: dict[int, dict[str, np.ndarray]],
) -> str:
    """Write a lossless-row, file-level split of the permitted derivative."""
    selected_cases = [cases[int(height)] for height in selected_heights]
    offsets = np.concatenate(
        ([0], np.cumsum([len(case["u"]) for case in selected_cases], dtype=np.int64))
    )

    np.savez_compressed(
        output,
        height_percent=selected_heights,
        height_ratio=selected_heights.astype(float) / 100.0,
        case_offset=offsets,
        x_m=np.concatenate([case["x"] for case in selected_cases]),
        y_m=np.concatenate([case["y"] for case in selected_cases]),
        u=np.concatenate([case["u"] for case in selected_cases]).astype(np.float32),
        v=np.concatenate([case["v"] for case in selected_cases]).astype(np.float32),
        fixed_knudsen=np.asarray(0.01),
        step_x_m=np.asarray(25.0e-9),
        source_commit=np.asarray(STEP_SOURCE_COMMIT),
    )
    return digest(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Checkout/cache of Ehsan-Roohi/roohi-step-dnn-mahdavi",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    report = validate_step_height_dataset(args.source)
    cases = load_step_height_cases(args.source)
    first = cases[int(STEP_HEIGHT_PERCENT[0])]
    if len(np.unique(first["x"])) != 200 or len(np.unique(first["y"])) != 120:
        raise ValueError("unexpected source parent grid")
    result_dir = args.root.resolve() / "results" / "mahdavi_deeponet"
    result_dir.mkdir(parents=True, exist_ok=True)
    learning_heights = np.concatenate(
        (STEP_HEIGHT_DEVELOPMENT_PERCENT, STEP_HEIGHT_VALIDATION_PERCENT)
    )
    archive_specs = {
        "step_height_learning_7cases.npz": learning_heights,
        "step_height_test_2cases.npz": STEP_HEIGHT_HELD_OUT_PERCENT,
    }
    archive_hashes = {
        name: write_archive(result_dir / name, heights, cases)
        for name, heights in archive_specs.items()
    }
    print(json.dumps({"archives": archive_hashes, **report}, indent=2))


if __name__ == "__main__":
    main()
