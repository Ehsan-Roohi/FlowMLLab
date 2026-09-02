#!/usr/bin/env python3
"""Generate or verify the retained Week-5 cylinder LBM teaching evidence.

The default command is cheap: it verifies the committed evidence.  Passing
``--regenerate`` performs the Reynolds-number sweep.  The retained ``quick``
profile is a regime demonstration, not a grid-converged external-cylinder DNS;
the more expensive ``validation`` profile is provided for student refinement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.cylinder_lbm import (  # noqa: E402
    CYLINDER_LBM_VERSION,
    recommended_parameters,
    simulate_cylinder,
)


CASES = (5, 20, 40, 100, 180)
REFERENCE = {
    20: {"Cd_low": 1.95, "Cd_high": 2.12, "St_low": np.nan, "St_high": np.nan,
         "Lr_low": 0.88, "Lr_high": 1.02},
    40: {"Cd_low": 1.46, "Cd_high": 1.58, "St_low": np.nan, "St_high": np.nan,
         "Lr_low": 2.15, "Lr_high": 2.32},
    100: {"Cd_low": 1.27, "Cd_high": 1.38, "St_low": 0.158, "St_high": 0.171,
          "Lr_low": 1.30, "Lr_high": 1.55},
    180: {"Cd_low": 1.25, "Cd_high": 1.38, "St_low": 0.185, "St_high": 0.200,
          "Lr_low": 0.82, "Lr_high": 1.05},
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _one_case(job: tuple[int, str]) -> dict[str, Any]:
    reynolds, fidelity = job
    settings = recommended_parameters(reynolds, fidelity)
    # A finite deterministic seed exposes the antisymmetric mode without
    # continuously forcing the wake.  Longer runs are used for shedding cases.
    settings.update(perturbation=1.0e-2, seed=690, collision_model="trt")
    if fidelity == "quick" and reynolds >= 100:
        settings["steps"] = max(settings["steps"], 15000)
    started = time.perf_counter()
    result = simulate_cylinder(reynolds, **settings)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _metrics(result: dict[str, Any], fidelity: str) -> dict[str, Any]:
    n = len(result["time"])
    tail = slice(n // 2, None)
    cl = result["lift_coefficient"][tail]
    cd = result["drag_coefficient"][tail]
    cl_rms = float(np.std(cl))
    st = float(result["strouhal"])
    reynolds = int(result["metadata"]["reynolds"])
    expected = (
        "attached"
        if reynolds < 6.3
        else "steady recirculating"
        if reynolds < 47
        else "periodic shedding"
    )
    observed = (
        "periodic shedding"
        if np.isfinite(st) and 0.08 < st < 0.30 and cl_rms > 4.0e-3
        else "steady recirculating"
        if np.isfinite(result["recirculation_length_over_diameter"])
        else "attached"
    )
    density_drift = float(np.max(np.abs(result["mean_density_ratio"] - 1.0)))
    row: dict[str, Any] = {
        "Re": reynolds,
        "fidelity": fidelity,
        "expected_regime": expected,
        "observed_regime": observed,
        "regime_pass": observed == expected,
        "tau": result["metadata"]["relaxation_time"],
        "Mach": result["metadata"]["lattice_mach"],
        "blockage": result["metadata"]["blockage_ratio"],
        "steps": result["metadata"]["config"]["steps"],
        "observed_time_D_over_U": (
            result["metadata"]["config"]["steps"]
            * result["metadata"]["config"]["inflow_velocity"]
            / result["metadata"]["config"]["diameter"]
        ),
        "density_drift": density_drift,
        "Cd_mean": float(np.mean(cd)),
        "Cl_rms": cl_rms,
        "St": st if observed == "periodic shedding" else np.nan,
        "Lr_over_D": float(result["recirculation_length_over_diameter"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "stability_pass": bool(density_drift < 0.01 and result["metadata"]["lattice_mach"] < 0.1),
    }
    reference = REFERENCE.get(reynolds)
    if reference:
        row["reference_Cd_pass"] = bool(reference["Cd_low"] <= row["Cd_mean"] <= reference["Cd_high"])
        row["reference_Lr_pass"] = bool(reference["Lr_low"] <= row["Lr_over_D"] <= reference["Lr_high"])
        row["reference_St_pass"] = (
            np.nan
            if not np.isfinite(reference["St_low"])
            else bool(reference["St_low"] <= row["St"] <= reference["St_high"])
        )
    else:
        row.update(reference_Cd_pass=np.nan, reference_Lr_pass=np.nan, reference_St_pass=np.nan)
    return row


def _save_case(result: dict[str, Any], directory: Path) -> None:
    reynolds = int(result["metadata"]["reynolds"])
    np.savez_compressed(
        directory / f"re{reynolds}_teaching_case.npz",
        x=np.asarray(result["x"], dtype=np.float32),
        y=np.asarray(result["y"], dtype=np.float32),
        solid=np.asarray(result["solid"], dtype=bool),
        rho=np.asarray(result["rho"], dtype=np.float32),
        u=np.asarray(result["u"], dtype=np.float32),
        v=np.asarray(result["v"], dtype=np.float32),
        p=np.asarray(result["p"], dtype=np.float32),
        vorticity=np.asarray(result["vorticity"], dtype=np.float32),
        time=np.asarray(result["time"], dtype=np.float32),
        drag_coefficient=np.asarray(result["drag_coefficient"], dtype=np.float32),
        lift_coefficient=np.asarray(result["lift_coefficient"], dtype=np.float32),
        mean_density_ratio=np.asarray(result["mean_density_ratio"], dtype=np.float32),
        metadata=json.dumps(result["metadata"], sort_keys=True),
    )


def _plot_regimes(results: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13})
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), constrained_layout=True)
    field_axes = axes.flat[:5]
    maximum = max(float(np.percentile(np.abs(r["vorticity"][~r["solid"]]), 99)) for r in results)
    image = None
    for axis, result in zip(field_axes, results):
        meta = result["metadata"]
        diameter = meta["config"]["diameter"]
        cx, cy = meta["cylinder_center"]
        extent = ((result["x"][0] - cx) / diameter, (result["x"][-1] - cx) / diameter,
                  (result["y"][0] - cy) / diameter, (result["y"][-1] - cy) / diameter)
        image = axis.imshow(result["vorticity"], origin="lower", extent=extent, aspect="auto",
                            cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        axis.add_patch(Circle((0.0, 0.0), 0.5, facecolor="#F7F7F7",
                              edgecolor="black", linewidth=1.2, zorder=5))
        axis.set_title(f"Re = {int(meta['reynolds'])}")
        axis.set_xlabel(r"$(x-x_c)/D$")
        axis.set_ylabel(r"$(y-y_c)/D$")
        axis.set_aspect("equal", adjustable="box")
    fig.colorbar(image, ax=list(field_axes), shrink=0.82, label=r"lattice vorticity $\omega_z$")

    signal_axis = axes.flat[5]
    for result in results:
        reynolds = int(result["metadata"]["reynolds"])
        if reynolds < 40:
            continue
        diameter = result["metadata"]["config"]["diameter"]
        speed = result["metadata"]["config"]["inflow_velocity"]
        signal_axis.plot(result["time"] * speed / diameter, result["lift_coefficient"],
                         lw=1.15, label=f"Re={reynolds}")
    signal_axis.set_xlabel(r"$tU_\infty/D$")
    signal_axis.set_ylabel(r"$C_L$")
    signal_axis.set_title("Lift: decay versus sustained shedding")
    signal_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=True, ncol=3
    )
    fig.suptitle("FlowMLLab Week 5: D2Q9 cylinder regimes (quick teaching profile)", fontsize=16)
    fig.savefig(output / "cylinder_lbm_regimes.png", dpi=200)
    fig.savefig(output / "cylinder_lbm_regimes.pdf")
    plt.close(fig)


def regenerate(output: Path, fidelity: str, workers: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(value, fidelity) for value in CASES]
    if workers == 1:
        results = [_one_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one_case, jobs))
    results.sort(key=lambda item: item["metadata"]["reynolds"])
    for result in results:
        _save_case(result, output)
    rows = [_metrics(result, fidelity) for result in results]
    table = pd.DataFrame(rows)
    table.to_csv(output / "regime_metrics.csv", index=False)
    pd.DataFrame([{"Re": key, **value} for key, value in REFERENCE.items()]).to_csv(
        output / "reference_ranges.csv", index=False
    )
    _plot_regimes(results, output)
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "solver_version": CYLINDER_LBM_VERSION,
        "profile": fidelity,
        "cases": list(CASES),
        "collision": "TRT D2Q9; BGK remains an explicit teaching comparison",
        "boundaries": {"inlet": "Zou-He velocity", "outlet": "convective",
                       "cylinder": "Bouzidi interpolated circular wall",
                       "transverse": "periodic"},
        "regime_gate": "mandatory for quick and validation profiles",
        "reference_gate": "reported but not enforced for quick; mandatory only after grid/domain/time refinement",
        "reference_sources": {
            "Re20_40": "Gautier, Biau & Lamballais, Computers & Fluids 75 (2013)",
            "Re50_180": "Qu et al., Journal of Fluids and Structures 39 (2013)",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "limitations": [
            "The retained quick profile is qualitative and is not grid-converged DNS evidence.",
            "Periodic transverse boundaries represent a weakly interacting cylinder array.",
            "A coarsely resolved interpolated circular wall still requires diameter refinement.",
            "Re=180 is a two-dimensional teaching solution and does not model Mode A or B.",
        ],
    }
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = verify(output)
    summary["regeneration_elapsed_seconds"] = float(table["elapsed_seconds"].sum())
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify(output: Path) -> dict[str, Any]:
    metrics_path = output / "regime_metrics.csv"
    protocol_path = output / "validation_protocol.json"
    figure_path = output / "cylinder_lbm_regimes.png"
    if not metrics_path.is_file() or not protocol_path.is_file() or not figure_path.is_file():
        raise FileNotFoundError("retained cylinder evidence is incomplete; run with --regenerate")
    metrics = pd.read_csv(metrics_path)
    if metrics["Re"].astype(int).tolist() != list(CASES):
        raise AssertionError("retained Reynolds cases do not match the frozen teaching protocol")
    if not metrics["stability_pass"].astype(bool).all():
        raise AssertionError("one or more cylinder cases failed the low-Mach/density stability gate")
    if not metrics["regime_pass"].astype(bool).all():
        failed = metrics.loc[~metrics["regime_pass"].astype(bool), "Re"].tolist()
        raise AssertionError(f"cylinder regime gate failed at Re={failed}")
    summary = {
        "status": "pass",
        "profile": str(metrics["fidelity"].iloc[0]),
        "cases": metrics["Re"].astype(int).tolist(),
        "all_stability_gates_pass": True,
        "all_regime_gates_pass": True,
        "quantitative_reference_gate_enforced": str(metrics["fidelity"].iloc[0]) == "validation",
        "max_density_drift": float(metrics["density_drift"].max()),
        "figure_sha256": _digest(figure_path),
    }
    if summary["quantitative_reference_gate_enforced"]:
        for column in ("reference_Cd_pass", "reference_Lr_pass", "reference_St_pass"):
            values = metrics[column].dropna().astype(bool)
            if not values.all():
                raise AssertionError(f"quantitative cylinder validation failed: {column}")
    return summary


def verify_cnn(output: Path) -> dict[str, Any]:
    """Verify frozen four-frame CNN evidence without importing TensorFlow."""
    metrics_path = output / "multiscale_cnn_metrics.json"
    video_path = output / "re105_lbm_vs_multiscale_cnn.mp4"
    if not metrics_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("retained cylinder CNN evidence is incomplete")
    report = json.loads(metrics_path.read_text())
    protocol = report["protocol"]
    if protocol["development_reynolds"] != [60, 80, 90, 110, 120, 140]:
        raise AssertionError("CNN development split changed")
    if protocol["validation_reynolds"] != 100 or protocol["blind_reynolds"] != 105:
        raise AssertionError("CNN validation/blind split changed")
    if not report["validation_pass"] or not all(report["validation_gates"].values()):
        raise AssertionError("CNN validation gates failed")
    blind = report["blind"]["models"]
    if not (
        blind["prediction"]["vorticity_relative_l2"]
        < blind["persistence"]["vorticity_relative_l2"]
    ):
        raise AssertionError("blind CNN does not beat persistence on vorticity")
    if video_path.stat().st_size <= 1_000_000:
        raise AssertionError("blind CNN video is unexpectedly small")
    return {
        "status": "pass",
        "validation_reynolds": 100,
        "blind_reynolds": 105,
        "blind_vorticity_relative_l2": float(
            blind["prediction"]["vorticity_relative_l2"]
        ),
        "persistence_vorticity_relative_l2": float(
            blind["persistence"]["vorticity_relative_l2"]
        ),
        "video_sha256": _digest(video_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--fidelity", choices=("quick", "validation"), default="quick")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    output = args.root.resolve() / "results" / "cylinder_lbm"
    report = regenerate(output, args.fidelity, max(1, args.workers)) if args.regenerate else verify(output)
    report["four_frame_cnn"] = verify_cnn(
        args.root.resolve() / "results" / "cylinder_cnn"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
