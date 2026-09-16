#!/usr/bin/env python3
"""Recompute Week 4.2 on 51x51 nodes using the retained case manifest.

The original 25x25 tests were inspected, so these remain regression cases.
All numerical labels are newly solved at 51x51; no field is upsampled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.cavity_diversity import evaluate, solve_cases
from flowmllab.stokes_correction import solve_stokes
from flowmllab.stokes_refined import train_one


OUT = ROOT / "results/stokes_grid51"
CASES = ROOT / "results/stokes_refined/expanded/expanded_cases.json"
FIELDS = ("psi", "omega", "u", "v", "lid", "residual")


def generate(chunk_size: int = 8, start_case: int = 0, end_case: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads(CASES.read_text())
    if (OUT / "dataset.npz").exists() and (OUT / "cases.json").exists():
        if json.loads((OUT / "cases.json").read_text()) != cases:
            raise ValueError("Saved 51-node dataset uses a different case manifest")
        print("Using existing 51-node dataset; remove it explicitly to regenerate.", flush=True)
        return
    parts = OUT / "parts"
    parts.mkdir(exist_ok=True)
    end_case = len(cases) if end_case is None else min(end_case, len(cases))
    if start_case % chunk_size or end_case % chunk_size and end_case != len(cases):
        raise ValueError("Shard boundaries must align with chunk size")
    for start in range(start_case, end_case, chunk_size):
        stop = min(start + chunk_size, len(cases))
        path = parts / f"cases_{start:03d}_{stop:03d}.npz"
        if path.exists():
            continue
        result = solve_cases(cases[start:stop], n=51)
        np.savez_compressed(path, **{key: result[key] for key in FIELDS})
        print(f"Solved {stop}/{len(cases)}: {result['steps']} steps, "
              f"max residual {result['residual'].max():.3g}", flush=True)
    filenames = [parts / f"cases_{start:03d}_{min(start + chunk_size, len(cases)):03d}.npz"
                 for start in range(0, len(cases), chunk_size)]
    if all(path.exists() for path in filenames) and not (OUT / "dataset.npz").exists():
        arrays = {key: np.concatenate([np.load(path)[key] for path in filenames])
                  for key in FIELDS}
        assert arrays["psi"].shape == (len(cases), 51, 51)
        np.savez_compressed(OUT / "dataset.npz", **arrays)
        (OUT / "cases.json").write_text(json.dumps(cases, indent=2))
        (OUT / "solver_audit.json").write_text(json.dumps({
            "grid": 51, "case_count": len(cases),
            "max_steady_rhs": float(arrays["residual"].max()),
            "note": "Fresh 51x51 finite-difference solves; unchanged, previously inspected cases."
        }, indent=2))


def train() -> None:
    torch.set_num_threads(2)
    cases = json.loads((OUT / "cases.json").read_text())
    data = np.load(OUT / "dataset.npz")
    low_path = OUT / "stokes.npz"
    if not low_path.exists():
        low = solve_stokes(cases, n=51)
        assert np.max(low["linear_relative_residual"]) < 1e-9
        np.savez_compressed(low_path, **low)
    low = np.load(low_path)
    re = np.array([case["Re"] for case in cases])
    x = low["psi"]
    y = data["psi"]
    selected = dict(rank=24, width=96, omega_weight=.02, epochs=1800)
    rows = []
    for family in ("constant", "diverse"):
        tr = np.array([i for i, case in enumerate(cases)
                       if case["family"] == family and case["split"] == "train"])
        va = np.array([i for i, case in enumerate(cases)
                       if case["family"] == family and case["split"] == "val"])
        for seed in (7, 17, 27):
            path = OUT / f"{family}_seed{seed}.pt"
            if not path.exists():
                model, record, history = train_one(x, y, re, tr, va, seed=seed, **selected)
                torch.save(dict(state_dict=model.state_dict(), config=record,
                                n=51, family=family, train_indices=tr.tolist()), path)
                pd.DataFrame(history).to_csv(OUT / f"{family}_seed{seed}_history.csv", index=False)
                print(f"Trained {family} seed {seed}: validation={record['validation']:.5g}", flush=True)
    # Loading through PODCorrection requires its train-only basis and mean to be
    # reconstructed. Those buffers are overwritten by the saved state dictionary.
    from flowmllab.stokes_refined import PODCorrection
    for family in ("constant", "diverse"):
        tr = np.array([i for i, case in enumerate(cases)
                       if case["family"] == family and case["split"] == "train"])
        lo = torch.tensor(x, dtype=torch.float64)
        rr = torch.tensor(re, dtype=torch.float64)
        ensemble = []
        for seed in (7, 17, 27):
            saved = torch.load(OUT / f"{family}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = PODCorrection(x[tr], y[tr], re[tr], rank=24, width=96)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            ensemble.append(model)
        for test_family in ("constant", "diverse", "ood_shape", "ood_re"):
            ids = np.array([i for i, case in enumerate(cases)
                            if case["family"] == test_family and case["split"] == "test"])
            with torch.no_grad():
                prediction = np.mean([model(lo[ids], rr[ids]).numpy()
                                      for model in ensemble], axis=0)
            np.savez_compressed(OUT / f"prediction_{family}_{test_family}.npz",
                                psi=prediction, case_ids=np.array([cases[i]["id"] for i in ids]))
            metrics = evaluate(prediction, y[ids], data["lid"][ids])
            for j, i in enumerate(ids):
                rows.append(dict(train_family=family, test_family=test_family,
                                 case=cases[i]["id"], Re=cases[i]["Re"],
                                 **{key: float(value[j]) for key, value in metrics.items()}))
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    summary = pd.DataFrame(rows).groupby(["train_family", "test_family"]).velocity_rel_l2.agg(
        ["mean", "max", "std"])
    summary.to_csv(OUT / "summary.csv")
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "train"))
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--start-case", type=int, default=0)
    parser.add_argument("--end-case", type=int)
    args = parser.parse_args()
    if args.stage == "generate":
        generate(args.chunk_size, args.start_case, args.end_case)
    else:
        train()
