#!/usr/bin/env python3
"""Train the Week-15 fixed-domain DeepONet baseline on the OpenFOAM archive.

The test geometries and case counts match ``diverse_geometry_v1``.  This
baseline deliberately receives only Reynolds number in its branch and x,y in
its trunk: no mask, signed distance, geometry ID, or test-field information is
provided.  It therefore measures ordinary fixed-domain DeepONet transfer when
the domain itself changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


TEST_GEOMETRIES = (9, 23, 36, 48)
VALIDATION_GEOMETRIES = (10, 15, 28, 35, 50)
SEEDS = (17, 29, 43)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry_ids(masks: np.ndarray) -> np.ndarray:
    fingerprints = [hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() for mask in masks]
    order = list(dict.fromkeys(fingerprints))
    if len(order) != 51:
        raise RuntimeError(f"Expected 51 masks, found {len(order)}")
    return np.asarray([order.index(value) + 1 for value in fingerprints], dtype=np.int32)


def build_model(tf, seed: int):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    keras = tf.keras
    branch = keras.Input((1,), name="reynolds_number_only")
    trunk = keras.Input((2,), name="fixed_xy_trunk")
    b, t = branch, trunk
    for _ in range(2):
        b = keras.layers.Dense(128, activation="tanh")(b)
        t = keras.layers.Dense(128, activation="tanh")(t)
    b = keras.layers.Dense(64 * 3)(b)
    t = keras.layers.Dense(64)(t)
    b = keras.layers.Reshape((3, 64))(b)
    y = keras.layers.Lambda(lambda q: tf.einsum("bij,bj->bi", q[0], q[1]),
                            name="branch_trunk_inner_product")([b, t])
    return keras.Model((branch, trunk), y, name="ordinary_fixed_domain_deeponet")


def relative_l2(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(100.0 * np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def case_metrics(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y, p = truth[mask], pred[mask]
    velocity = relative_l2(y[:, :2], p[:, :2])
    yc = y[:, 2] - y[:, 2].mean()
    pc = p[:, 2] - p[:, 2].mean()
    pressure = relative_l2(yc[:, None], pc[:, None])
    truth_reverse, pred_reverse = y[:, 0] < 0, p[:, 0] < 0
    union = np.count_nonzero(truth_reverse | pred_reverse)
    iou = float(np.count_nonzero(truth_reverse & pred_reverse) / union) if union else 1.0
    bias = float(100.0 * (pred_reverse.mean() - truth_reverse.mean()))
    return {"velocity_percent": velocity, "pressure_percent": pressure,
            "reverse_iou": iou, "reverse_area_bias_percent": bias}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--steps-per-epoch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        raw = archive["raw"].astype(np.float32)
        queries = archive["queries"].astype(np.float32)
        masks = archive["masks"].astype(bool)
        reynolds = archive["Re"].astype(np.float32)
        shape = archive["shape"].astype(np.int64)
    gids = geometry_ids(masks)
    test_case = np.isin(gids, TEST_GEOMETRIES)
    val_case = np.isin(gids, VALIDATION_GEOMETRIES)
    train_case = ~(test_case | val_case)
    if (int(train_case.sum()), int(val_case.sum()), int(test_case.sum())) != (107, 11, 12):
        raise RuntimeError("Unexpected 107/11/12 case split")

    train_rows = np.flatnonzero((train_case[:, None] & masks).ravel())
    val_rows = np.flatnonzero((val_case[:, None] & masks).ravel())
    case_index = np.repeat(np.arange(len(raw)), raw.shape[1])
    flat_raw = raw.reshape(-1, 3)
    flat_xy = queries[:, :, :2].reshape(-1, 2)
    flat_re = np.repeat(reynolds, raw.shape[1]).astype(np.float32)[:, None]

    re_mean, re_std = flat_re[train_rows].mean(0), flat_re[train_rows].std(0)
    xy_mean, xy_std = flat_xy[train_rows].mean(0), flat_xy[train_rows].std(0)
    y_mean, y_std = flat_raw[train_rows].mean(0), flat_raw[train_rows].std(0)
    re_std[re_std == 0] = 1
    xy_std[xy_std == 0] = 1
    y_std[y_std == 0] = 1
    x_branch = ((flat_re - re_mean) / re_std).astype(np.float32)
    x_trunk = ((flat_xy - xy_mean) / xy_std).astype(np.float32)
    targets = ((flat_raw - y_mean) / y_std).astype(np.float32)

    all_rows: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    start_all = time.perf_counter()
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        model = build_model(tf, seed)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        best = {"loss": float("inf"), "epoch": 0, "weights": None}
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            losses = []
            for _ in range(args.steps_per_epoch):
                idx = rng.choice(train_rows, args.batch_size, replace=False)
                with tf.GradientTape() as tape:
                    prediction = model((x_branch[idx], x_trunk[idx]), training=True)
                    loss = tf.reduce_mean(tf.square(prediction - targets[idx]))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                losses.append(float(loss))
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                # Stable fixed validation sample, never used for updates.
                vrng = np.random.default_rng(100_000 + seed)
                vidx = vrng.choice(val_rows, min(65536, len(val_rows)), replace=False)
                vpred = model((x_branch[vidx], x_trunk[vidx]), training=False).numpy()
                vloss = float(np.mean((vpred - targets[vidx]) ** 2))
                record = {"epoch": epoch, "train_mse_scaled": float(np.mean(losses)),
                          "validation_mse_scaled": vloss}
                history.append(record)
                if vloss < best["loss"]:
                    best = {"loss": vloss, "epoch": epoch, "weights": model.get_weights()}
                print(json.dumps({"seed": seed, **record}), flush=True)
        if best["weights"] is None:
            raise RuntimeError("No validation checkpoint")
        model.set_weights(best["weights"])
        seed_dir = args.output / "predictions" / "geometry_holdout" / "ordinary_deeponet" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for ci in np.flatnonzero(test_case):
            rows = np.arange(ci * raw.shape[1], (ci + 1) * raw.shape[1])
            pred_scaled = model((x_branch[rows], x_trunk[rows]), training=False).numpy()
            pred = pred_scaled * y_std + y_mean
            name = f"g{gids[ci]:03d}_Re{int(reynolds[ci])}_medium"
            metric = case_metrics(raw[ci], pred, masks[ci])
            all_rows.append({"protocol": "diverse_geometry_v1", "case": name,
                             "geometry": f"g{gids[ci]:03d}", "Re": int(reynolds[ci]),
                             "model": "ordinary_DeepONet", "seed": seed,
                             "checkpoint_epoch": int(best["epoch"]), **metric})
            np.savez_compressed(seed_dir / f"{name}_prediction.npz",
                                prediction=pred.astype(np.float32), truth=raw[ci],
                                coordinates=queries[ci, :, :2].astype(np.float64), mask=masks[ci],
                                shape=shape, row_index=np.int64(ci), Re=np.float32(reynolds[ci]),
                                evaluation_group=np.asarray("test"),
                                pressure_convention=np.asarray("stored target p*=Re*p; compare after mean centering"))
        histories[str(seed)] = history
        model.save_weights(args.output / f"ordinary_deeponet_seed{seed}.weights.h5")

    with (args.output / "case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    manifest = {
        "model": "ordinary fixed-domain DeepONet", "branch_inputs": ["Re"],
        "trunk_inputs": ["normalized x", "normalized y"],
        "excluded_geometry_inputs": ["mask", "SDF", "geometry ID"],
        "dataset_sha256": sha256(args.dataset), "dataset_cases": len(raw),
        "split": {"train_cases": 107, "validation_cases": 11, "test_cases": 12,
                  "validation_geometry_ids": list(VALIDATION_GEOMETRIES),
                  "test_geometry_ids": list(TEST_GEOMETRIES)},
        "seeds": args.seeds, "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
        "batch_size": args.batch_size, "target": ["u", "v", "Re*p_kinematic"],
        "selection": "minimum scaled validation MSE at 10-epoch audits",
        "training_seconds": time.perf_counter() - start_all,
        "software": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "numpy": np.__version__}, "history": histories,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("WEEK15_ORDINARY_DEEPONET_PASS", flush=True)


if __name__ == "__main__":
    main()
