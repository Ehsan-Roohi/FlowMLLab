#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the 16-to-9 cubic-FP closure surrogate used in Track 6.

Two controlled loss choices are supported:

  uniform   : all nine standardized coefficients have equal weight;
  qweighted : the heat-flux closure block Gamma_i receives a larger weight.

The network architecture is fixed to 16-hidden-hidden-hidden-hidden-9 so that
its exported arrays are directly compatible with the supplied CuPy cavity
solver.  This script does not require scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

TARGET_NAMES = [
    "Cxx", "Cxy", "Cxz", "Cyy", "Cyz", "Czz",
    "Gamma_x", "Gamma_y", "Gamma_z",
]


class SimpleStandardScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1.0e-30] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def load_many(paths):
    Xs, ys = [], []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if "inputs" not in d or "targets" not in d:
            raise KeyError(f"{path}: expected inputs and targets arrays; keys={d.files}")
        X = np.asarray(d["inputs"], dtype=np.float32)
        y = np.asarray(d["targets"], dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 16:
            raise ValueError(f"{path}: input shape must be (?,16), got {X.shape}")
        if y.ndim != 2 or y.shape[1] != 9:
            raise ValueError(f"{path}: target shape must be (?,9), got {y.shape}")
        keep = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
        Xs.append(X[keep])
        ys.append(y[keep])
        print(f"Loaded {path}: X={X[keep].shape}, y={y[keep].shape}", flush=True)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def block_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1.0e-300))

    rows = []
    groups = {
        "all": slice(0, 9),
        "C_block": slice(0, 6),
        "Gamma_block": slice(6, 9),
    }
    for name, sl in groups.items():
        a = y_true[:, sl]
        b = y_pred[:, sl]
        rows.append(
            dict(
                group=name,
                relative_L2=rel(a, b),
                RMSE=float(np.sqrt(np.mean((a - b) ** 2))),
                MAE=float(np.mean(np.abs(a - b))),
            )
        )
    return rows


def save_native_params(model, x_scaler, y_scaler, out_path):
    from tensorflow.keras import layers

    dense_layers = [layer for layer in model.layers if isinstance(layer, layers.Dense)]
    if len(dense_layers) != 5:
        raise RuntimeError(
            f"The deployment solver expects 5 Dense layers (4 hidden + output); found {len(dense_layers)}"
        )
    params = {
        "X_mean": x_scaler.mean_.astype(np.float64),
        "X_scale": x_scaler.scale_.astype(np.float64),
        "y_mean": y_scaler.mean_.astype(np.float64),
        "y_scale": y_scaler.scale_.astype(np.float64),
    }
    for i, layer in enumerate(dense_layers, start=1):
        W, b = layer.get_weights()
        params[f"W{i}"] = W.astype(np.float64)
        params[f"b{i}"] = b.astype(np.float64)
    np.savez(out_path, **params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss-mode", choices=["uniform", "qweighted"], default="qweighted")
    ap.add_argument("--q-weight", type=float, default=6.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--learning-rate", type=float, default=1.0e-4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    log_path = out_path.with_name(stem + "_history.csv")
    metrics_path = out_path.with_name(stem + "_validation_metrics.csv")
    metadata_path = out_path.with_name(stem + "_metadata.json")
    keras_path = out_path.with_name(stem + ".keras")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    X_train, y_train = load_many(args.train)
    X_val, y_val = load_many(args.val)

    x_scaler = SimpleStandardScaler()
    y_scaler = SimpleStandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    model = keras.Sequential(name=f"fp_closure_{args.loss_mode}")
    model.add(layers.Input(shape=(16,), name="low_order_moments"))
    for i in range(4):
        model.add(layers.Dense(args.hidden, activation="relu", name=f"hidden_{i+1}"))
    model.add(layers.Dense(9, activation="linear", name="closure_coefficients"))

    if args.loss_mode == "uniform":
        output_weights_np = np.ones(9, dtype=np.float32)
    else:
        # The z-associated components are typically weaker/noisier in a nominally
        # 2D cavity, so the two in-plane heat-flux coefficients receive the largest
        # emphasis, following the instructor research workflow.
        output_weights_np = np.asarray(
            [1.0, 1.0, 0.5, 1.0, 0.5, 1.0,
             args.q_weight, args.q_weight, max(2.0, args.q_weight / 3.0)],
            dtype=np.float32,
        )
    output_weights = tf.constant(output_weights_np, dtype=tf.float32)

    def weighted_standardized_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred) * output_weights, axis=-1)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=weighted_standardized_mse,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=max(4, args.patience // 2), min_lr=1.0e-6, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_path)),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    elapsed = time.perf_counter() - t0

    pred_val_s = model.predict(X_val_s, batch_size=args.batch, verbose=0)
    pred_val = pred_val_s * y_scaler.scale_[None, :] + y_scaler.mean_[None, :]

    rows = block_metrics(y_val, pred_val)
    per_target = []
    for j, name in enumerate(TARGET_NAMES):
        den = max(np.linalg.norm(y_val[:, j]), 1.0e-300)
        per_target.append(
            dict(
                group=name,
                relative_L2=float(np.linalg.norm(pred_val[:, j] - y_val[:, j]) / den),
                RMSE=float(np.sqrt(np.mean((pred_val[:, j] - y_val[:, j]) ** 2))),
                MAE=float(np.mean(np.abs(pred_val[:, j] - y_val[:, j]))),
            )
        )
    rows.extend(per_target)

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "relative_L2", "RMSE", "MAE"])
        writer.writeheader()
        writer.writerows(rows)

    save_native_params(model, x_scaler, y_scaler, out_path)
    model.save(keras_path)

    metadata = {
        "loss_mode": args.loss_mode,
        "output_weights": output_weights_np.tolist(),
        "architecture": f"16-{args.hidden}-{args.hidden}-{args.hidden}-{args.hidden}-9",
        "seed": args.seed,
        "train_files": args.train,
        "validation_files": args.val,
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "epochs_requested": args.epochs,
        "epochs_run": int(len(history.history.get("loss", []))),
        "elapsed_s": elapsed,
        "deployment_file": str(out_path),
        "keras_file": str(keras_path),
        "scope": (
            "Educational reduced-data model. Closed-loop cavity testing and complete-condition "
            "validation are required before making a physical claim."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved deployment parameters:", out_path)
    print("Saved Keras model:", keras_path)
    print("Saved metrics:", metrics_path)
    for row in rows[:3]:
        print(row)


if __name__ == "__main__":
    main()
