#!/usr/bin/env python3
"""Execute the Week-4 multi-output POD-DeepONet cavity project on CPU.

The operator uses separate POD trunks for the divergence-free velocity state
``[u, v]`` and the zero-mean pressure field ``p``.  Two small branch heads map
Reynolds number to their modal coefficients.  Keeping the trunks separate
prevents pressure scaling from changing the velocity modes while making
pressure a direct learned output rather than a post-processed reconstruction.
Rank, branch width, and the declared scalar input transform are selected on one
complete development case; the retained test Reynolds cases are not used by
the selection rule.
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
VELOCITY_RANKS = (3, 4)
VELOCITY_HIDDEN_CANDIDATES = ((16, 16), (32, 32), (64, 64))
PRESSURE_RANKS = (2, 3, 4)
PRESSURE_HIDDEN_CANDIDATES = ((4,), (8,), (8, 8), (16, 16), (32, 32))
PRESSURE_INPUT_TRANSFORMS = ("linear", "log")


def load_data() -> dict[str, np.ndarray]:
    with np.load(DATASET, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def case_index(data: dict[str, np.ndarray], re_value: float) -> int:
    return int(np.where(np.isclose(data["Re"], re_value))[0][0])


def state_matrix(
    data: dict[str, np.ndarray], indices: np.ndarray, field: str = "velocity"
) -> np.ndarray:
    """Return complete-case states for one POD trunk."""
    if field == "velocity":
        return np.hstack(
            (
                data["u"][indices].reshape(len(indices), -1),
                data["v"][indices].reshape(len(indices), -1),
            )
        )
    if field == "pressure":
        pressure = data["p"][indices].reshape(len(indices), -1).copy()
        pressure -= np.mean(pressure, axis=1, keepdims=True)
        return pressure
    raise ValueError(f"Unknown field {field!r}; choose 'velocity' or 'pressure'")


def make_trunk(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    rank: int,
    field: str = "velocity",
) -> dict[str, np.ndarray | str]:
    states = state_matrix(data, indices, field)
    mean = np.mean(states, axis=0)
    _, singular_values, modes = np.linalg.svd(states - mean, full_matrices=False)
    return {
        "field": field,
        "mean": mean,
        "modes": modes[:rank],
        "singular_values": singular_values,
        "energy_fraction": np.asarray(
            np.sum(singular_values[:rank] ** 2) / np.sum(singular_values**2)
        ),
    }


def coefficients(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    trunk: dict[str, np.ndarray | str],
) -> np.ndarray:
    states = state_matrix(data, indices, str(trunk["field"]))
    return (states - trunk["mean"]) @ trunk["modes"].T


def branch_features(re_values: np.ndarray, transform: str) -> np.ndarray:
    """Apply a declared scalar feature transform before standardization."""
    values = np.asarray(re_values, dtype=float).reshape(-1, 1)
    if transform == "linear":
        return values
    if transform == "log":
        if np.any(values <= 0.0):
            raise ValueError("The logarithmic Reynolds feature requires Re > 0")
        return np.log(values)
    raise ValueError(f"Unknown input transform {transform!r}")


def fit_branch(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    trunk: dict[str, np.ndarray | str],
    hidden: tuple[int, ...],
    seed: int,
    input_transform: str = "linear",
) -> dict[str, object]:
    re_values = data["Re"][indices]
    coeff = coefficients(data, indices, trunk)
    features = branch_features(re_values, input_transform)
    x_scaler = StandardScaler().fit(features)
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
        model.fit(x_scaler.transform(features), y_scaler.transform(coeff))
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "trunk": trunk,
        "field": str(trunk["field"]),
        "input_transform": input_transform,
        "hidden": hidden,
        "seed": seed,
        "iterations": int(model.n_iter_),
        "training_seconds": float(time.perf_counter() - started),
    }


def predict_head(
    bundle: dict[str, object], re_value: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    features = branch_features(
        np.asarray([re_value]), str(bundle["input_transform"])
    )
    scaled = bundle["model"].predict(
        bundle["x_scaler"].transform(features)
    ).reshape(1, -1)
    coeff = bundle["y_scaler"].inverse_transform(scaled)[0]
    state = bundle["trunk"]["mean"] + coeff @ bundle["trunk"]["modes"]
    if bundle["field"] == "pressure":
        pressure = state.reshape(shape).copy()
        pressure -= np.mean(pressure)
        return pressure

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


def fit_operator(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    trunks: dict[str, dict[str, np.ndarray | str]],
    selected: dict[str, dict[str, object]],
    seed: int,
) -> dict[str, object]:
    """Fit the two branch heads of one multi-output POD-DeepONet member."""
    return {
        "seed": seed,
        "velocity": fit_branch(
            data,
            indices,
            trunks["velocity"],
            tuple(selected["velocity"]["hidden"]),
            seed,
            str(selected["velocity"]["input_transform"]),
        ),
        "pressure": fit_branch(
            data,
            indices,
            trunks["pressure"],
            tuple(selected["pressure"]["hidden"]),
            seed,
            str(selected["pressure"]["input_transform"]),
        ),
    }


def predict(
    bundle: dict[str, object], re_value: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict velocity and direct zero-mean pressure from one operator member."""
    velocity = predict_head(bundle["velocity"], re_value, shape)
    pressure = predict_head(bundle["pressure"], re_value, shape)
    assert isinstance(velocity, tuple) and isinstance(pressure, np.ndarray)
    return velocity[0], velocity[1], pressure


def ensemble_predict(
    bundles: list[dict[str, object]], re_value: float, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = [predict(bundle, re_value, shape) for bundle in bundles]
    return tuple(
        np.mean(np.stack([field[component] for field in fields]), axis=0)
        for component in range(3)
    )


def field_report(
    data: dict[str, np.ndarray],
    re_value: float,
    prediction: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    idx = case_index(data, re_value)
    return w4utils.field_validation_report(
        data["x"], data["y"], data["u"][idx], data["v"][idx],
        prediction[0], prediction[1], data["p"][idx], prediction[2]
    )


def _head_validation_error(
    data: dict[str, np.ndarray],
    re_value: float,
    field: str,
    prediction: tuple[np.ndarray, np.ndarray] | np.ndarray,
) -> float:
    idx = case_index(data, re_value)
    if field == "pressure":
        report = w4utils.pressure_errors(data["p"][idx], prediction)
        return float(report["relative_L2_p"])
    report = w4utils.field_validation_report(
        data["x"], data["y"], data["u"][idx], data["v"][idx],
        prediction[0], prediction[1]
    )
    return float(report["relative_L2_uv"])


def select_model(
    data: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    """Select velocity and pressure heads using only the complete Re=225 case."""
    development = np.where(data["split"] == "train")[0]
    validation_idx = case_index(data, VALIDATION_RE)
    selection_train = development[development != validation_idx]
    rows = []
    candidate_sets = {
        "velocity": (
            VELOCITY_RANKS,
            VELOCITY_HIDDEN_CANDIDATES,
            ("linear",),
        ),
        "pressure": (
            PRESSURE_RANKS,
            PRESSURE_HIDDEN_CANDIDATES,
            PRESSURE_INPUT_TRANSFORMS,
        ),
    }
    for field, (ranks, hidden_candidates, transforms) in candidate_sets.items():
        for rank in ranks:
            trunk = make_trunk(data, selection_train, rank, field)
            for hidden in hidden_candidates:
                for transform in transforms:
                    seed_errors = []
                    iterations = []
                    for seed in SEEDS:
                        bundle = fit_branch(
                            data, selection_train, trunk, hidden, seed, transform
                        )
                        prediction = predict_head(
                            bundle, VALIDATION_RE, data["u"].shape[1:]
                        )
                        seed_errors.append(
                            _head_validation_error(
                                data, VALIDATION_RE, field, prediction
                            )
                        )
                        iterations.append(int(bundle["iterations"]))
                    rows.append(
                        {
                            "head": field,
                            "rank": rank,
                            "hidden": "x".join(map(str, hidden)),
                            "input_transform": transform,
                            "trunk_energy_fraction": float(trunk["energy_fraction"]),
                            "mean_validation_relative_L2": float(np.mean(seed_errors)),
                            "min_validation_relative_L2": float(np.min(seed_errors)),
                            "max_validation_relative_L2": float(np.max(seed_errors)),
                            "mean_iterations": float(np.mean(iterations)),
                        }
                    )
    selection = pd.DataFrame(rows).sort_values(
        ["head", "mean_validation_relative_L2"]
    )
    selected: dict[str, dict[str, object]] = {}
    for field in ("velocity", "pressure"):
        best = selection[selection["head"] == field].iloc[0]
        selected[field] = {
            "rank": int(best["rank"]),
            "hidden": tuple(int(value) for value in str(best["hidden"]).split("x")),
            "input_transform": str(best["input_transform"]),
        }
    return selected, selection


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
    speed400 = np.hypot(pred400[0], pred400[1])
    speed275 = np.hypot(pred275[0], pred275[1])
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
    selected, selection = select_model(data)
    selection.to_csv(FIGURES / "deeponet_selection.csv", index=False)
    development = np.where(data["split"] == "train")[0]
    trunks = {
        field: make_trunk(data, development, int(config["rank"]), field)
        for field, config in selected.items()
    }
    bundles = [
        fit_operator(data, development, trunks, selected, seed) for seed in SEEDS
    ]
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
        ghia_rows.append({"Re": re_value, "CFD_Ghia_Eu": cfµ¨¥Â¸­yêë¢°k¢G§¦*^d_eu, "CFD_Ghia_Ev": cfd_ev, "POD_DeepONet_Ghia_Eu": deep_eu, "POD_DeepONet_Ghia_Ev": deep_ev})
    ghia = pd.DataFrame(ghia_rows)
    ghia.to_csv(FIGURES / "deeponet_ghia_metrics.csv", index=False)
    training_seconds = {
        str(bundle["seed"]): {
            field: float(bundle[field]["training_seconds"])
            for field in ("velocity", "pressure")
        }
        for bundle in bundles
    }
    timing = {
        "POD_DeepONet_ensemble_inference_ms": inference_ms,
        "CFD_Re275_seconds": cfd_seconds,
        "speedup": cfd_seconds / (inference_ms / 1000.0),
        "CFD_steps": int(cfd_result["steps"]),
        "CFD_final_residual": float(cfd_result["final_residual"]),
        "training_seconds_by_seed_and_head": training_seconds,
        "selected_heads": {
            field: {
                "rank": int(config["rank"]),
                "hidden": list(config["hidden"]),
                "input_transform": str(config["input_transform"]),
                "trunk_energy_fraction": float(trunks[field]["energy_fraction"]),
            }
            for field, config in selected.items()
        },
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
        p=np.stack([predictions[float(data["Re"][idx])][2] for idx in blind_indices]),
        seeds=np.asarray(SEEDS),
        velocity_rank=np.asarray(selected["velocity"]["rank"]),
        velocity_hidden=np.asarray(selected["velocity"]["hidden"]),
        pressure_rank=np.asarray(selected["pressure"]["rank"]),
        pressure_hidden=np.asarray(selected["pressure"]["hidden"]),
    )
    iy, ix = np.indices(data["u"].shape[1:])
    prediction_table = {
        "iy": iy.ravel(),
        "ix": ix.ravel(),
        "x": data["x"][ix.ravel()],
        "y": data["y"][iy.ravel()],
    }
    for re_value, (u_pred, v_pred, p_pred) in predictions.items():
        label = str(int(re_value))
        prediction_table[f"u_Re{label}"] = u_pred.ravel()
        prediction_table[f"v_Re{label}"] = v_pred.ravel()
        prediction_table[f"p_Re{label}"] = p_pred.ravel()
    pd.DataFrame(prediction_table).to_csv(
        FIGURES / "deeponet_predictions.csv", index=False, float_format="%.9g"
    )
    blind275 = float(metrics[(metrics["method"].str.startswith("three-seed")) & np.isclose(metrics["Re"], 275)]["relative_L2_uv"].iloc[0])
    result_figure(data, bundles, inference_ms, cfd_seconds, blind275)
    print("selection\n", selection.to_string(index=False), flush=True)
    print(
        "blind metrics\n",
        metrics[
            [
                "method", "seed", "Re", "relative_L2_uv", "relative_L2_p",
                "div_l2_pred", "wall_rms_error",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("Ghia\n", ghia.to_string(index=False), flush=True)
    print(json.dumps(timing, indent=2), flush=True)


if __name__ == "__main__":
    main()
