#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate exported cubic-FP neural closure parameters on a complete dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fp_project_utils import TARGET_NAMES, q_norm_from_features


def load_params(path):
    d = np.load(path)
    required = ["X_mean", "X_scale", "y_mean", "y_scale"] + [f"W{i}" for i in range(1, 6)] + [f"b{i}" for i in range(1, 6)]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{path}: missing deployment arrays {missing}")
    return {k: np.asarray(d[k], dtype=np.float64) for k in d.files}


def predict(X, p):
    h = (np.asarray(X, dtype=np.float64) - p["X_mean"]) / p["X_scale"]
    for i in range(1, 5):
        h = np.maximum(h @ p[f"W{i}"] + p[f"b{i}"], 0.0)
    y_s = h @ p["W5"] + p["b5"]
    return y_s * p["y_scale"] + p["y_mean"]


def metrics(a, b):
    den = max(np.linalg.norm(a), 1.0e-300)
    return {
        "relative_L2": float(np.linalg.norm(a - b) / den),
        "RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
        "MAE": float(np.mean(np.abs(a - b))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--outdir", default="fp_closure_evaluation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X = np.asarray(d["inputs"], dtype=np.float64)
    y = np.asarray(d["targets"], dtype=np.float64)
    params = load_params(args.model)
    pred = predict(X, params)

    rows = []
    groups = {
        "all": slice(0, 9),
        "C_block": slice(0, 6),
        "Gamma_block": slice(6, 9),
    }
    for name, sl in groups.items():
        row = {"group": name, **metrics(y[:, sl], pred[:, sl])}
        rows.append(row)

    qn = q_norm_from_features(X)
    q_threshold = float(np.percentile(qn, 80.0))
    hard = qn >= q_threshold
    if np.any(hard):
        rows.append({"group": "Gamma_block_high_q80", **metrics(y[hard, 6:9], pred[hard, 6:9])})

    for j, name in enumerate(TARGET_NAMES):
        rows.append({"group": name, **metrics(y[:, j], pred[:, j])})

    metrics_csv = outdir / f"{args.label}_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "relative_L2", "RMSE", "MAE"])
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        outdir / f"{args.label}_predictions.npz",
        inputs=X.astype(np.float32),
        targets=y.astype(np.float32),
        predictions=pred.astype(np.float32),
        q_norm=qn.astype(np.float32),
        high_q_mask=hard,
    )

    summary = {
        "label": args.label,
        "model": args.model,
        "data": args.data,
        "n_samples": int(len(X)),
        "q80_threshold": q_threshold,
        "metrics": rows,
    }
    (outdir / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Evaluation for {args.label}")
    for row in rows[:4]:
        print(row)
    print("Saved:", metrics_csv)


if __name__ == "__main__":
    main()
