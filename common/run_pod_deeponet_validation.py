#!/usr/bin/env python3
"""Execute the Week-4 POD-DeepONet cavity project on CPU.

The operator is written in explicit branch--trunk form.  A development-only
POD supplies an interpretable spatial trunk and a tanh branch network maps
Reynolds number to modal coefficients.  Rank and branch width are selected on
one complete development case; three untouched Reynolds cases are opened only
after the protocol is frozen.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = ROOT / "results" / "pod_deeponet"
DATASET = ROOT / "data" / "cavity_data.npz"
sys.path.insert(0, str(HERE))
import w4utils  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"


SEEDS = (690, 691, 692)
VALIDATION_RE = 225
RANKS = (3, 4)
HIDDEN_CANDIDATES = ((16, 16), (32, 32), (64, 64))


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATASET, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def case_index(data: dict[str, np.ndarray], re_value: float) -> int:
    return int(np.where(np.isclose(data["Re"], re_value))[0][0])


def state_matrix(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.hstack(
        (
            data["u"][indices].reshape(len(indices), -1),
            data["v"][indices].reshape(len(indices), -1),
        )
    )


def make_trunk(data: dict[str, np.ndarray], indices: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    states = state_matrix(data, indices)
    mean = np.mean(states, axis=0)
    _, singular_values, modes = np.linalg.svd(states - mean, full_matrices=False)
    return {
        "mean": mean,
        "modes": modes[:rank],
        "singular_values": singular_values,
        "energy_fraction": np.asarray(
            np.sum(singular_values[:rank] ** 2) / np.sum(singular_values**2)
        ),
    }


def coefficients(
    data: dict[str, np.ndarray], indices: np.ndarray, trunk: dict[str, np.ndarray]
) -> np.ndarray:
    return (state_matrix(data, indices) - trunk["mean"]) @ trunk["modes"].T


def fit_branch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    trunk: dict[str, np.ndarray],
    hidden: tuple[int, ...],
    seed: int,
) -> dict[str, object]:
    re_values = data["Re"][indices, None]
    coeff = coefficients(data, indices, trunk)
    x_scaler = StandardScaler().fit(re_values)
    y_scaler = StandardScaler().fit(coeff)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="tanh",
        solver="lbfgs",
        alpha=1.0e-7,
        max_iter=5000,
        max_fun=100000,
        tol=1.0e-10,
        random_state=seed,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_scaler.transform(re_values), y_scaler.transform(coeff))
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "trunk": trunk,
        "hidden": hidden,
        "seed": seed,
        "iterations": int(model.n_iter_),
        "training_seconds": float(time.perf_counter() - started),
    }


def predict(
    bundle: dict[str, object], re_value: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    scaled = bundle["model"].predict(
        bundle["x_scaler"].transform(np.asarray([[re_value]]))
    ).reshape(1, -1)
    coeff = bundle["y_scaler"].inverse_transform(scaled)[0]
    state = bundle["trunk"]["mean"] + coeff @ bundle["trunk"]["modes"]
    count = int(np.prod(shape))
    u = state[:count].reshape(shape).copy()
    v = state[count:].reshape(shape).copy()
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[-1, 1:-1] = 1.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    return u, v


def ensemble_predict(
    bundles: list[dict[str, object]], re_value: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    fields = [predict(bundle, re_value, shape) for bundle in bundles]
    return (
        np.mean(np.stack([field[0] for field in fields]), axis=0),
        np.mean(np.stack([field[1] for field in fields]), axis=0),
    )


def field_report(
    data: dict[str, np.ndarray], re_value: float, prediction: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    idx = case_index(data, re_value)
    return w4utils.field_validation_report(
        data["x"], data["y"], data["u"][idx], data["v"][idx],
        prediction[0], prediction[1]
    )


def select_model(data: dict[str, np.ndarray]) -> tuple[int, tuple[int, ...], pd.DataFrame]:
    development = np.where(data["split"] == "train")[0]
    validation_idx = case_index(data, VALIDATION_RE)
    selection_train = development[development != validation_idx]
    rows = []
    for rank in RANKS:
        trunk = make_trunk(data, selection_train, rank)
        for hidden in HIDDEN_CANDIDATES:
            seed_errors = []
            iterations = []
            for seed in SEEDS:
                bundle = fit_branch(data, selection_train, trunk, hidden, seed)
                prediction = predict(bundle, VALIDATION_RE, data["u"].shape[1:])
                report = field_report(data, VALIDATION_RE, prediction)
                seed_errors.append(float(report["relative_L2_uv"]))
                iterations.append(int(bundle["iterations"]))
            rows.append(
                {
                    "rank": rank,
                    "hidden": "x".join(map(str, hidden)),
                    "trunk_energy_fraction": float(trunk["energy_fraction"]),
                    "mean_validation_relative_L2_uv": float(np.mean(seed_errors)),
                    "min_validation_relative_L2_uv": float(np.min(seed_errors)),
                    "max_validation_relative_L2_uv": float(np.max(seed_errors)),
                    "mean_iterations": float(np.mean(iterations)),
                }
            )
    selection = pd.DataFrame(rows).sort_values("mean_validation_relative_L2_uv")
    best = selection.iloc[0]
    hidden = tuple(int(value) for value in str(best["hidden"]).split("x"))
    return int(best["rank"]), hidden, selection


def ghia_error(
    re_value: int, u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    reference = w4utils.GHIA[re_value]
    mid = len(x) // 2
    u_sample = np.interp(reference["y"], y, u[:, mid])
    v_sample = np.interp(reference["x"], x, v[mid, :])
    return (
        float(np.linalg.norm(u_sample - reference["u"]) / np.linalg.norm(reference["u"])),
        float(np.linalg.norm(v_sample - reference["v"]) / np.linalg.norm(reference["v"])),
    )


def result_figure(
    data: dict[str, np.ndarray],
    bundles: list[dict[str, object]],
    inference_ms: float,
    cfd_seconds: float,
    blind_error: float,
) -> None:
    x, y = data["x"], data["y"]
    xg, yg = np.meshgrid(x, y)
    pred400 = ensemble_predict(bundles, 400, data["u"].shape[1:])
    pred275 = ensemble_predict(bundles, 275, data["u"].shape[1:])
    idx275 = case_index(data, 275)
    speed400 = np.hypot(*pred400)
    speed275 = np.hypot(*pred275)
    truth275 = np.hypot(data["u"][idx275], data["v"][idx275])
    error275 = np.hypot(pred275[0] - data["u"][idx275], pred275[1] - data["v"][idx275])
    reference = w4utils.GHIA[400]
    mid = len(x) // 2
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=True)
    im = axes[0, 0].contourf(xg, yg, speed400, 29, cmap="viridis")
    axes[0, 0].streamplot(x, y, pred400[0], pred400[1], color="white", density=1.05, linewidth=0.7)
    axes[0, 0].set(title=r"(a) POD-DeepONet, $Re=400$", xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.87, label=r"$|\mathbf{u}|/U_{lid}$")
    axes[0, 1].plot(data["u"][case_index(data, 400), :, mid], y, "k-", label="CFD")
    axes[0, 1].plot(pred400[0][:, mid], y, "--", color="#DC2626", label="POD-DeepONet")
    axes[0, 1].scatter(reference["u"], reference["y"], color="#F59E0B", edgecolor="white", zorder=3, label="Ghia et al.")
    axes[0, 1].set(title="(b) Vertical centerline", xlabel=r"$u/U_{lid}$", ylabel=r"$y/L$")
    axes[0, 2].plot(x, data["v"][case_index(data, 400), mid, :], "k-", label="CFD")
    axes[0, 2].plot(x, pred400[1][mid, :], "--", color="#DC2626", label="POD-DeepONet")
    axes[0, 2].scatter(reference["x"], reference["v"], color="#F59E0B", edgecolor="white", zorder=3, label="Ghia et al.")
    axes[0, 2].set(title="(c) Horizontal centerline", xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$")
    for axis in axes[0, 1:]:
        axis.grid(alpha=0.25)
        axis.legend()
    levels = np.linspace(0, max(truth275.max(), speed275.max()), 29)
    im = axes[1, 0].contourf(xg, yg, truth275, levels=levels, cmap="viridis")
    axes[1, 0].set(title=r"(d) CFD blind field, $Re=275$", xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.87)
    im = axes[1, 1].contourf(xg, yg, speed275, levels=levels, cmap="viridis")
    axes[1, 1].set(title=rf"(e) Blind prediction; $E_{{uv}}={100*blind_error:.3f}\%$", xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.87)
    im = axes[1, 2].contourf(xg, yg, error275, 29, cmap="magma")
    axes[1, 2].set(title=rf"(f) Error; {inference_ms:.2f} ms vs CFD {cfd_seconds:.1f} s", xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.87, label=r"$\|\Delta\mathbf{u}\|/U_{lid}$")
    fig.suptitle("Executed DeepONet project: Ghia fidelity, blind accuracy, and inference advantage", fontsize=18, fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"pod_deeponet_ghia_validation.{suffix}", bbox_inches="tight")
    compact_jpeg = FIGURES / "pod_deeponet_ghia_validation.tmp.jpg"
    fig.savefig(
        compact_jpeg,
        format="jpg",
        dpi=110,
        bbox_inches="tight",
        pil_kwargs={"quality": 76, "optimize": True},
    )
    encoded = base64.b64encode(compact_jpeg.read_bytes()).decode("ascii")
    (FIGURES / "pod_deeponet_ghia_validation.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" '
        'viewBox="0 0 1600 900" role="img" aria-label="Executed POD-DeepONet validation">'
        f'<image width="1600" height="900" preserveAspectRatio="xMidYMid meet" '
        f'href="data:image/jpeg;base64,{encoded}"/></svg>\n',
        encoding="utf-8",
    )
    compact_jpeg.unlink()
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_data()
    rank, hidden, selection = select_model(data)
    selection.to_csv(FIGURES / "deeponet_selection.csv", index=False)
    development = np.where(data["split"] == "train")[0]
    trunk = make_trunk(data, development, rank)
    bundles = [fit_branch(data, development, trunk, hidden, seed) for seed in SEEDS]
    blind_indices = np.where(data["split"] == "test")[0]
    rows = []
    for bundle in bundles:
        for idx in blind_indices:
            re_value = float(data["Re"][idx])
            report = field_report(data, re_value, predict(bundle, re_value, data["u"].shape[1:]))
            rows.append({"method": "individual POD-DeepONet", "seed": bundle["seed"], "Re": re_value, **report})
    predictions = {}
    for idx in blind_indices:
        re_value = float(data["Re"][idx])
        prediction = ensemble_predict(bundles, re_value, data["u"].shape[1:])
        predictions[re_value] = prediction
        rows.append({"method": "three-seed POD-DeepONet ensemble", "seed": -1, "Re": re_value, **field_report(data, re_value, prediction)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(FIGURES / "deeponet_metrics.csv", index=False)

    repetitions = 1000
    started = time.perf_counter()
    for _ in range(repetitions):
        ensemble_predict(bundles, 275, data["u"].shape[1:])
    inference_ms = 1000.0 * (time.perf_counter() - started) / repetitions
    cfd_started = time.perf_counter()
    cfd_result = w4utils.run_cavity(Re=275, N=65, dt=8.0e-4, max_steps=30000, tol=1.0e-6, verbose=False)
    cfd_seconds = time.perf_counter() - cfd_started

    ghia_rows = []
    for re_value in (100, 400):
        idx = case_index(data, re_value)
        prediction = ensemble_predict(bundles, re_value, data["u"].shape[1:])
        deep_eu, deep_ev = ghia_error(re_value, prediction[0], prediction[1], data["x"], data["y"])
        cfd_eu, cfd_ev = ghia_error(re_value, data["u"][idx], data["v"][idx], data["x"], data["y"])
        ghia_rows.append({"Re": re_value, "CFD_Ghia_Eu": cfd_eu, "CFD_Ghia_Ev": cfd_ev, "POD_DeepONet_Ghia_Eu": deep_eu, "POD_DeepONet_Ghia_Ev": deep_ev})
    ghia = pd.DataFrame(ghia_rows)
    ghia.to_csv(FIGURES / "deeponet_ghia_metrics.csv", index=False)
    timing = {
        "POD_DeepONet_ensemble_inference_ms": inference_ms,
        "CFD_Re275_seconds": cfd_seconds,
        "speedup": cfd_seconds / (inference_ms / 1000.0),
        "CFD_steps": int(cfd_result["steps"]),
        "CFD_final_residual": float(cfd_result["final_residual"]),
        "training_seconds_by_seed": {str(bundle["seed"]): bundle["training_seconds"] for bundle in bundles},
        "selected_rank": rank,
        "selected_hidden": list(hidden),
        "trunk_energy_fraction": float(trunk["energy_fraction"]),
        "development_Re": data["Re"][development].tolist(),
        "validation_Re_for_selection": VALIDATION_RE,
        "blind_Re": data["Re"][blind_indices].tolist(),
        "boundary_output_transform": True,
    }
    (FIGURES / "deeponet_protocol_and_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    np.savez_compressed(
        FIGURES / "deeponet_predictions.npz",
        Re=data["Re"][blind_indices],
        u=np.stack([predictions[float(data["Re"][idx])][0] for idx in blind_indices]),
        v=np.stack([predictions[float(data["Re"][idx])][1] for idx in blind_indices]),
        seeds=np.asarray(SEEDS), rank=np.asarray(rank), hidden=np.asarray(hidden),
    )
    iy, ix = np.indices(data["u"].shape[1:])
    prediction_table = {
        "iy": iy.ravel(),
        "ix": ix.ravel(),
        "x": data["x"][ix.ravel()],
        "y": data["y"][iy.ravel()],
    }
    for re_value, (u_pred, v_pred) in predictions.items():
        label = str(int(re_value))
        prediction_table[f"u_Re{label}"] = u_pred.ravel()
        prediction_table[f"v_Re{label}"] = v_pred.ravel()
    pd.DataFrame(prediction_table).to_csv(
        FIGURES / "deeponet_predictions.csv", index=False, float_format="%.9g"
    )
    blind275 = float(metrics[(metrics["method"].str.startswith("three-seed")) & np.isclose(metrics["Re"], 275)]["relative_L2_uv"].iloc[0])
    result_figure(data, bundles, inference_ms, cfd_seconds, blind275)
    print("selection\n", selection.to_string(index=False), flush=True)
    print("blind metrics\n", metrics[["method", "seed", "Re", "relative_L2_uv", "div_l2_pred", "wall_rms_error"]].to_string(index=False), flush=True)
    print("Ghia\n", ghia.to_string(index=False), flush=True)
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
