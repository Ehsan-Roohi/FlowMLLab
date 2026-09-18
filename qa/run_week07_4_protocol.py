#!/usr/bin/env python3
"""Run the frozen Week 7.4 diverse-wake pretraining and lift-decoding protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab import diverse_wake_pretraining as dw  # noqa: E402
from flowmllab import masked_pretraining as mp  # noqa: E402
from flowmllab.cylinder_ml import fit_pod  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/week07_4_wakes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = (Path(tempfile.mkdtemp(prefix="flowmllab-week74-quick-")) if args.quick
                       else ROOT / "results/week07_4_diverse_pretraining")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, __import__("os").cpu_count() or 1))
    manifest = dw.load_manifest(args.data); split = manifest["split_contract"]
    for case in manifest['cases']:
        if dw.sha256(args.data / case['file']) != case['sha256']:
            raise ValueError(f"Dataset hash mismatch: {case['file']}")
    cases = {re: dw.load_case(args.data, re) for role in split.values() for re in role}
    development_fields = np.concatenate([cases[re][0] for re in split["development"]])
    development_lift = np.concatenate([cases[re][1] for re in split["development"]])
    validation_fields, validation_lift, _ = cases[split["validation"][0]]
    layout = mp.PatchLayout(); mean = float(development_fields.mean()); scale = float(development_fields.std())
    train_scaled = (development_fields - mean) / scale
    val_scaled = (validation_fields - mean) / scale
    steps = 1500 if args.quick else 6000
    seeds = (0,) if args.quick else (0, 1, 2)
    ks = (4, 16, 64) if args.quick else (2, 4, 8, 16, 32, 64, 128)

    model = mp.build_mae(layout, dim=64, depth=2, heads=4, decoder_dim=32,
                         decoder_depth=1, seed=74)
    val_visible = mp.random_patch_masks(len(validation_fields), layout.n_patches, 0.75, seed=7401)
    started = time.perf_counter()
    history, best_step = mp.train_masked_model(
        model, layout.patchify(train_scaled), layout, mask_ratio=0.75, steps=steps,
        batch_size=16, lr=2e-3, seed=74, val_patches=layout.patchify(val_scaled),
        val_visible=val_visible, val_every=100,
    )
    pretrain_seconds = time.perf_counter() - started
    random_model = mp.build_mae(layout, dim=64, depth=2, heads=4, decoder_dim=32,
                                decoder_depth=1, seed=74)

    def encoder_features(net, frames):
        scaled = (frames - mean) / scale
        visible = np.ones((len(frames), layout.n_patches), bool)
        return mp.encoder_features(net, layout.patchify(scaled), visible, pooling="visible_mean")

    pod = fit_pod(development_fields.reshape(len(development_fields), -1), rank=32)

    def pod_features(frames):
        return (frames.reshape(len(frames), -1) - pod.mean) @ pod.modes

    methods = {"pretrained_encoder": lambda f: encoder_features(model, f),
               "random_encoder": lambda f: encoder_features(random_model, f),
               "pod32": pod_features}
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def fit_predict(x_train, y_train, x_test, alpha):
        return mp.RidgeReadout(alpha).fit(x_train, y_train[:, None]).predict(x_test).ravel()

    selected, validation_scores = {}, {}
    for name, transform in methods.items():
        xdev, xval = transform(development_fields), transform(validation_fields)
        scores = {}
        for alpha in alphas:
            prediction = fit_predict(xdev, development_lift, xval, alpha)
            scores[alpha] = dw.normalized_rmse(prediction, validation_lift)
        selected[name] = min(scores, key=scores.get)
        validation_scores[name] = {str(alpha): value for alpha, value in scores.items()}

    records, example = [], None
    for reynolds in split["test"]:
        frames, lift, times = cases[reynolds]
        pool = np.arange(0, 150); test_idx = np.arange(180, len(frames))
        cache = {name: transform(frames) for name, transform in methods.items()}
        for seed in seeds:
            for k in ks:
                labelled = mp.sample_labelled_frames(pool, k, 10000 + 100 * reynolds + 10 * seed + k)
                for name in methods:
                    prediction = fit_predict(cache[name][labelled], lift[labelled], cache[name][test_idx], selected[name])
                    records.append({"reynolds": reynolds, "seed": seed, "k": k, "method": name,
                                    "nrmse": dw.normalized_rmse(prediction, lift[test_idx])})
                    if reynolds == 115 and seed == 0 and k == (16 if args.quick else 32):
                        if example is None: example = {"time": times[test_idx].tolist(), "truth": lift[test_idx].tolist()}
                        example[name] = prediction.tolist()
    curves = dw.summarize_trajectory_records(records)

    colors = {"pretrained_encoder": "#D62728", "random_encoder": "#9467BD", "pod32": "#1F77B4"}
    labels = {"pretrained_encoder": "pretrained encoder + ridge", "random_encoder": "random encoder + ridge",
              "pod32": "POD-32 coefficients + ridge"}
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for name in methods:
        x = np.asarray(ks); y = np.asarray([curves[name][k]["mean"] for k in ks])
        lo = y - np.asarray([curves[name][k]["min"] for k in ks]); hi = np.asarray([curves[name][k]["max"] for k in ks]) - y
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", capsize=3, color=colors[name], label=labels[name])
    ax.set(xscale="log", yscale="log", xlabel="labelled frames from the new trajectory (k)",
           ylabel="lift decoding NRMSE (%)", title="Week 7.4 label efficiency on four unseen LBM trajectories")
    ax.grid(True, which="both", alpha=.25); ax.legend(); fig.savefig(args.output / "label_efficiency.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.plot(example["time"], example["truth"], color="black", lw=2, label="LBM lift")
    for name in methods:
        ax.plot(example["time"], example[name], lw=1.4, color=colors[name], label=labels[name])
    ax.set(xlabel="lattice timestep", ylabel="lift coefficient", title=f"Re=115 adaptation, k={16 if args.quick else 32} labels")
    ax.grid(alpha=.25); ax.legend(ncol=2); fig.savefig(args.output / "re115_lift_decoding.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.semilogy([h["step"] for h in history], [h["val_loss"] for h in history], color="#D62728")
    ax.set(xlabel="pretraining step", ylabel="masked validation MSE", title="Diverse-wake masked pretraining")
    ax.grid(alpha=.25); fig.savefig(args.output / "pretraining_loss.png", dpi=180); plt.close(fig)

    data_hashes = {case["file"]: case["sha256"] for case in manifest["cases"]}
    payload = {"protocol": {"development": split["development"], "validation": split["validation"],
                             "test": split["test"], "label_pool": [0, 150], "test_window": [180, 251],
                             "ks": list(ks), "seeds": list(seeds), "mask_ratio": .75, "pretrain_steps": steps,
                             "field": "v/U", "downstream_label": "instantaneous lift coefficient",
                             "selection_rule": "ridge alpha on Re105 only", "quick": args.quick},
               "model": {"parameters": sum(p.numel() for p in model.parameters()), "feature_dimension": 64,
                         "best_step": best_step, "pretraining_seconds": pretrain_seconds},
               "validation_scores": validation_scores, "selected_alpha": selected,
               "curves": {m: {str(k): v for k, v in curve.items()} for m, curve in curves.items()},
               "records": records, "example": example, "data_hashes": data_hashes,
               "pretraining_history": history,
               "label_accounting": {"development_selection_labels": len(development_lift),
                                    "validation_selection_labels": len(validation_lift),
                                    "target_labels_per_trajectory": list(ks)},
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    mp.write_json(args.output / "metrics.json", payload)
    print(json.dumps({m: {k: round(v["mean"], 3) for k, v in c.items()} for m, c in curves.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
