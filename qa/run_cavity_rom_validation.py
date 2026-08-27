#!/usr/bin/env python3
"""Generate the frozen validation package for the Week-4.1 cavity ROM lab."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))
from flowmllab import cavity_rom  # noqa: E402
import w4utils  # noqa: E402


RESULTS = ROOT / "results" / "cavity_rom"
TRAIN_RE = [100, 150, 200, 225, 250, 350, 400]
VALIDATION_RE = 300
BLIND_RE = [175, 275, 375]
N = 33
DT = 2.0e-3
STEPS = 2500
SNAPSHOT_STRIDE = 25
CANDIDATE_RANKS = [4, 8, 12, 16]
ERROR_GATE = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _velocity_vector(result: dict, prefix: str = "final") -> np.ndarray:
    return np.concatenate(
        [np.asarray(result[f"{prefix}_u"]).reshape(-1), np.asarray(result[f"{prefix}_v"]).reshape(-1)]
    )


def _relative(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(prediction) - np.asarray(reference))
        / (np.linalg.norm(reference) + 1.0e-30)
    )


def reproduce_fixed_cavity_archive() -> pd.DataFrame:
    """Prove that the snapshot-enabled FOM reproduces the accepted Week-4 solver."""
    with np.load(ROOT / "data" / "cavity_data.npz", allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    quality = pd.read_csv(ROOT / "data" / "case_quality.csv").set_index("Re")
    rows = []
    for reynolds in (100, 400):
        steps = int(quality.loc[reynolds, "steps"])
        result = cavity_rom.simulate_fom(
            reynolds,
            n=65,
            dt=1.0e-3,
            steps=steps,
            snapshot_stride=steps,
        )
        index = int(np.where(data["Re"] == reynolds)[0][0])
        reference = np.concatenate(
            [data["u"][index].reshape(-1), data["v"][index].reshape(-1), data["omega"][index].reshape(-1)]
        )
        prediction = np.concatenate(
            [result["final_u"].reshape(-1), result["final_v"].reshape(-1), result["final_omega"].reshape(-1)]
        )
        ghia_input = {
            "u": result["final_u"],
            "v": result["final_v"],
            "x": data["x"],
            "y": data["y"],
            "U": 1.0,
            "Re": reynolds,
        }
        ghia_u, ghia_v = w4utils.ghia_errors(ghia_input)
        rows.append(
            {
                "Re": reynolds,
                "N": 65,
                "dt": 1.0e-3,
                "steps": steps,
                "archive_relative_L2_u_v_omega": _relative(reference, prediction),
                "Ghia_relative_L2_u": ghia_u,
                "Ghia_relative_L2_v": ghia_v,
            }
        )
    table = pd.DataFrame(rows)
    assert table["archive_relative_L2_u_v_omega"].max() < 5.0e-13, table
    assert table[["Ghia_relative_L2_u", "Ghia_relative_L2_v"]].to_numpy().max() < 0.20, table
    return table


def convergence_studies() -> pd.DataFrame:
    """Run independent grid and time-step refinement at Re=275 and t=5."""
    rows = []
    grid_runs = {
        n: cavity_rom.simulate_fom(
            275, n=n, dt=DT, steps=STEPS, snapshot_stride=STEPS
        )
        for n in (25, 33, 49, 65)
    }
    fine = grid_runs[65]
    fine_grid = np.linspace(0.0, 1.0, 65)
    for n in (25, 33, 49):
        grid = np.linspace(0.0, 1.0, n)
        y_mesh, x_mesh = np.meshgrid(grid, grid, indexing="ij")
        points = np.column_stack([y_mesh.reshape(-1), x_mesh.reshape(-1)])
        interpolated = []
        for field in (fine["final_u"], fine["final_v"]):
            interpolated.append(
                RegularGridInterpolator((fine_grid, fine_grid), field)(points).reshape(n, n)
            )
        reference = np.concatenate([value.reshape(-1) for value in interpolated])
        prediction = _velocity_vector(grid_runs[n])
        rows.append(
            {
                "study": "grid",
                "control": float(n),
                "reference": "N=65",
                "relative_L2_uv": _relative(reference, prediction),
            }
        )

    temporal_reference = cavity_rom.simulate_fom(
        275, n=N, dt=5.0e-4, steps=10000, snapshot_stride=10000
    )
    reference_vector = _velocity_vector(temporal_reference)
    for dt in (4.0e-3, 2.0e-3, 1.0e-3):
        steps = int(round(5.0 / dt))
        result = cavity_rom.simulate_fom(
            275, n=N, dt=dt, steps=steps, snapshot_stride=steps
        )
        rows.append(
            {
                "study": "time_step",
                "control": dt,
                "reference": "dt=0.0005",
                "relative_L2_uv": _relative(reference_vector, _velocity_vector(result)),
            }
        )
    table = pd.DataFrame(rows)
    grid_errors = table[table["study"] == "grid"]["relative_L2_uv"].to_numpy()
    time_errors = table[table["study"] == "time_step"]["relative_L2_uv"].to_numpy()
    assert np.all(np.diff(grid_errors) < 0.0), table
    assert np.all(np.diff(time_errors) < 0.0), table
    return table


def _trajectory_metrics(reference: dict, prediction: dict, n: int) -> dict[str, float]:
    velocity_errors = cavity_rom.velocity_error_trajectory(
        reference["states"], prediction["states"], n
    )
    vorticity_errors = np.asarray(
        [
            cavity_rom.relative_l2(truth, estimate)
            for truth, estimate in zip(reference["states"], prediction["states"])
        ]
    )
    truth_diagnostics = cavity_rom.physical_diagnostics(reference["states"][-1], n)
    diagnostics = cavity_rom.physical_diagnostics(prediction["states"][-1], n)
    return {
        "final_relative_L2_uv": float(velocity_errors[-1]),
        "max_time_relative_L2_uv": float(velocity_errors.max()),
        "final_relative_L2_omega": float(vorticity_errors[-1]),
        # Relative vorticity error is undefined at t=0 because the reference
        # interior vorticity is exactly zero.  The maximum therefore begins at
        # the first positive retained time.
        "max_time_relative_L2_omega": float(vorticity_errors[1:].max()),
        "wall_rms_error": diagnostics["wall_rms_error"],
        "divergence_l2": diagnostics["divergence_l2"],
        "vortex_position_error": float(
            np.hypot(
                diagnostics["vortex_x"] - truth_diagnostics["vortex_x"],
                diagnostics["vortex_y"] - truth_diagnostics["vortex_y"],
            )
        ),
    }


def _timed_query(callable_, repeats: int = 7) -> tuple[float, object]:
    callable_()  # warm-up
    samples = []
    last = None
    for _ in range(repeats):
        started = perf_counter()
        last = callable_()
        if "states" in last and "final_u" not in last:
            cavity_rom.state_to_fields(last["states"][-1], N)
        samples.append(perf_counter() - started)
    return float(np.median(samples)), last


def build_rom_evidence() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Select on Re=300, then open the three predeclared blind cases."""
    offline_started = perf_counter()
    training = {
        reynolds: cavity_rom.simulate_fom(
            reynolds,
            n=N,
            dt=DT,
            steps=STEPS,
            snapshot_stride=SNAPSHOT_STRIDE,
        )
        for reynolds in TRAIN_RE
    }
    maximum_pod = cavity_rom.fit_pod(
        [training[reynolds]["states"] for reynolds in TRAIN_RE], max(CANDIDATE_RANKS)
    )
    nonlinear_snapshots = cavity_rom.convection_snapshots(
        [training[reynolds]["states"] for reynolds in TRAIN_RE], 1.0, N
    )
    nonlinear_basis = cavity_rom.fit_nonlinear_basis(
        nonlinear_snapshots, max_dimension=max(CANDIDATE_RANKS)
    )
    offline_seconds = perf_counter() - offline_started

    validation = cavity_rom.simulate_fom(
        VALIDATION_RE,
        n=N,
        dt=DT,
        steps=STEPS,
        snapshot_stride=SNAPSHOT_STRIDE,
    )
    selection_rows = []
    frozen = None
    for rank in CANDIDATE_RANKS:
        pod = cavity_rom.truncate_pod(maximum_pod, rank)
        deim = cavity_rom.fit_deim(pod, nonlinear_basis, rank)
        galerkin = cavity_rom.simulate_pod_galerkin(
            pod, VALIDATION_RE, DT, STEPS, SNAPSHOT_STRIDE
        )
        hyper = cavity_rom.simulate_pod_deim(
            deim, VALIDATION_RE, DT, STEPS, SNAPSHOT_STRIDE
        )
        galerkin_metrics = _trajectory_metrics(validation, galerkin, N)
        deim_metrics = _trajectory_metrics(validation, hyper, N)
        accepted = (
            galerkin_metrics["max_time_relative_L2_uv"] < ERROR_GATE
            and deim_metrics["max_time_relative_L2_uv"] < ERROR_GATE
        )
        selection_rows.append(
            {
                "rank": rank,
                "deim_dimension": rank,
                "POD_cumulative_energy": maximum_pod.cumulative_energy[rank - 1],
                "POD_Galerkin_max_time_relative_L2_uv": galerkin_metrics[
                    "max_time_relative_L2_uv"
                ],
                "POD_DEIM_max_time_relative_L2_uv": deim_metrics[
                    "max_time_relative_L2_uv"
                ],
                "passes_one_percent_gate": accepted,
            }
        )
        if accepted and frozen is None:
            frozen = (pod, deim)
    selection = pd.DataFrame(selection_rows)
    assert frozen is not None, selection
    pod, deim = frozen
    assert pod.rank == 16 and deim.deim_dimension == 16, selection

    blind_rows = []
    retained = {}
    for reynolds in BLIND_RE:
        reference = cavity_rom.simulate_fom(
            reynolds,
            n=N,
            dt=DT,
            steps=STEPS,
            snapshot_stride=SNAPSHOT_STRIDE,
        )
        galerkin = cavity_rom.simulate_pod_galerkin(
            pod, reynolds, DT, STEPS, SNAPSHOT_STRIDE
        )
        hyper = cavity_rom.simulate_pod_deim(
            deim, reynolds, DT, STEPS, SNAPSHOT_STRIDE
        )
        for method, prediction in (("POD-Galerkin", galerkin), ("POD-DEIM", hyper)):
            metrics = _trajectory_metrics(reference, prediction, N)
            blind_rows.append({"Re": reynolds, "method": method, **metrics})
        if reynolds == 275:
            retained = {"FOM": reference, "POD-Galerkin": galerkin, "POD-DEIM": hyper}
    blind = pd.DataFrame(blind_rows)
    assert blind["max_time_relative_L2_uv"].max() < ERROR_GATE, blind
    assert blind["final_relative_L2_omega"].max() < ERROR_GATE, blind
    assert blind["wall_rms_error"].max() == 0.0, blind
    assert blind["divergence_l2"].max() < 1.0e-12, blind

    fom_seconds, _ = _timed_query(
        lambda: cavity_rom.simulate_fom(
            275, n=N, dt=DT, steps=STEPS, snapshot_stride=SNAPSHOT_STRIDE
        )
    )
    galerkin_seconds, _ = _timed_query(
        lambda: cavity_rom.simulate_pod_galerkin(
            pod, 275, DT, STEPS, SNAPSHOT_STRIDE
        )
    )
    deim_seconds, _ = _timed_query(
        lambda: cavity_rom.simulate_pod_deim(
            deim, 275, DT, STEPS, SNAPSHOT_STRIDE
        )
    )
    saving = fom_seconds - deim_seconds
    timing = {
        "timing_scope": "complete t=0-to-5 query plus one final full-field reconstruction",
        "repeats": 7,
        "statistic": "median wall time after one warm-up",
        "platform": platform.platform(),
        "processor": platform.processor() or "not reported by container",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "offline_training_seconds": float(offline_seconds),
        "FOM_query_seconds": fom_seconds,
        "POD_Galerkin_query_seconds": galerkin_seconds,
        "POD_DEIM_query_seconds": deim_seconds,
        "POD_Galerkin_speedup": fom_seconds / galerkin_seconds,
        "POD_DEIM_speedup": fom_seconds / deim_seconds,
        "break_even_query_count": float(offline_seconds / saving) if saving > 0.0 else None,
    }
    assert timing["POD_DEIM_speedup"] > 2.0, timing

    summary = {
        "status": "pass",
        "selected_rank": pod.rank,
        "selected_deim_dimension": deim.deim_dimension,
        "selection_Re": VALIDATION_RE,
        "blind_Re": BLIND_RE,
        "maximum_blind_velocity_error": float(blind["max_time_relative_L2_uv"].max()),
        "maximum_blind_final_vorticity_error": float(
            blind["final_relative_L2_omega"].max()
        ),
        "maximum_blind_divergence_l2": float(blind["divergence_l2"].max()),
        "maximum_blind_wall_rms_error": float(blind["wall_rms_error"].max()),
        "POD_Galerkin_speedup": timing["POD_Galerkin_speedup"],
        "POD_DEIM_speedup": timing["POD_DEIM_speedup"],
        "break_even_query_count": timing["break_even_query_count"],
    }
    return selection, blind, timing, {
        "summary": summary,
        "retained": retained,
        "model": deim,
    }


def make_figure(
    selection: pd.DataFrame,
    blind: pd.DataFrame,
    timing: dict,
    retained: dict,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2))
    axis = axes[0, 0]
    axis.plot(
        selection["rank"],
        100.0 * selection["POD_Galerkin_max_time_relative_L2_uv"],
        "o-",
        label="POD-Galerkin",
    )
    axis.plot(
        selection["rank"],
        100.0 * selection["POD_DEIM_max_time_relative_L2_uv"],
        "s-",
        label="POD-DEIM",
    )
    axis.axhline(100.0 * ERROR_GATE, color="black", linestyle="--", label="1% gate")
    axis.set(xlabel="POD rank = DEIM dimension", ylabel="validation max-time $E_{uv}$ (%)", title="Leakage-free selection at Re=300")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for method, marker in (("POD-Galerkin", "o"), ("POD-DEIM", "s")):
        subset = blind[blind["method"] == method]
        axis.plot(subset["Re"], 100.0 * subset["max_time_relative_L2_uv"], marker + "-", label=method)
    axis.axhline(100.0 * ERROR_GATE, color="black", linestyle="--")
    axis.set(xlabel="blind Reynolds number", ylabel="max-time $E_{uv}$ (%)", title="Blind trajectories")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[0, 2]
    labels = ["FOM", "POD-Galerkin", "POD-DEIM"]
    values = [
        timing["FOM_query_seconds"],
        timing["POD_Galerkin_query_seconds"],
        timing["POD_DEIM_query_seconds"],
    ]
    axis.bar(labels, values, color=["0.35", "#4472C4", "#ED7D31"])
    axis.set_yscale("log")
    axis.set(ylabel="median query time (s)", title="Online cost at Re=275")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)

    fom_fields = cavity_rom.state_to_fields(retained["FOM"]["states"][-1], N)
    speed = np.hypot(fom_fields["u"], fom_fields["v"])
    x = np.linspace(0.0, 1.0, N)
    levels = np.linspace(0.0, float(speed.max()), 18)
    contour = axes[1, 0].contourf(x, x, speed, levels=levels, cmap="viridis")
    axes[1, 0].streamplot(x, x, fom_fields["u"], fom_fields["v"], color="white", density=0.8, linewidth=0.45)
    axes[1, 0].set(title="FOM speed, blind Re=275", xlabel="x/L", ylabel="y/L", aspect="equal")
    fig.colorbar(contour, ax=axes[1, 0], fraction=0.046)

    for axis, method in zip(axes[1, 1:], ("POD-Galerkin", "POD-DEIM")):
        fields = cavity_rom.state_to_fields(retained[method]["states"][-1], N)
        error = np.hypot(fields["u"] - fom_fields["u"], fields["v"] - fom_fields["v"])
        contour = axis.contourf(x, x, error, levels=18, cmap="magma")
        axis.set(title=f"{method} vector error", xlabel="x/L", ylabel="y/L", aspect="equal")
        fig.colorbar(contour, ax=axis, fraction=0.046)
    fig.suptitle("FlowMLLab Week 4.1: classical and hyper-reduced ROM validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS / "cavity_rom_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fom_validation = reproduce_fixed_cavity_archive()
    convergence = convergence_studies()
    selection, blind, timing, bundle = build_rom_evidence()
    protocol = {
        "protocol_version": "cavity-rom-1.0",
        "module_version": cavity_rom.CAVITY_ROM_VERSION,
        "problem": "two-dimensional nondimensional square lid-driven cavity",
        "state": "interior vorticity with streamfunction recovered by DST Poisson solve",
        "integrator": "forward Euler, matched to the existing Week-4 FOM",
        "spatial_scheme": "second-order centered differences with Thom wall vorticity",
        "training_Re": TRAIN_RE,
        "selection_Re": VALIDATION_RE,
        "blind_Re": BLIND_RE,
        "grid_N": N,
        "dt": DT,
        "steps": STEPS,
        "final_time": DT * STEPS,
        "snapshot_stride": SNAPSHOT_STRIDE,
        "candidate_ranks": CANDIDATE_RANKS,
        "selection_rule": "minimum rank with POD-Galerkin and POD-DEIM max-time relative L2 velocity error below 1%",
        "core_dataset_sha256": _sha256(ROOT / "data" / "cavity_data.npz"),
        "copyright_note": "implemented independently for FlowMLLab; no mini-rom source code copied",
    }
    fom_validation.to_csv(RESULTS / "fom_validation.csv", index=False, float_format="%.12g")
    convergence.to_csv(RESULTS / "convergence.csv", index=False, float_format="%.12g")
    selection.to_csv(RESULTS / "selection.csv", index=False, float_format="%.12g")
    blind.to_csv(RESULTS / "blind_metrics.csv", index=False, float_format="%.12g")
    (RESULTS / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (RESULTS / "validation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (RESULTS / "validation_summary.json").write_text(
        json.dumps(bundle["summary"], indent=2), encoding="utf-8"
    )
    cavity_rom.save_deim_model(RESULTS / "cavity_rom_model.npz", bundle["model"])
    make_figure(selection, blind, timing, bundle["retained"])
    print("CAVITY_ROM_VALIDATION_PASS")
    print(json.dumps(bundle["summary"], indent=2))


if __name__ == "__main__":
    main()
