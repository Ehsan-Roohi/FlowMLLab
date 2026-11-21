#!/usr/bin/env python3
"""Build the two static scientific figures used by the Week-8 teaching module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.gas_dynamics import (  # noqa: E402
    area_mach,
    fanno_ratios,
    nozzle_back_pressure,
    oblique_detachment,
    oblique_theta,
    rayleigh_ratios,
    shock_tube_pressure_ratio,
    validate_week8_evidence,
)


RESULTS = ROOT / "results" / "gas_dynamics_week8"
NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
RED = "#C44536"
GRAY = "#667085"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_hashes() -> None:
    provenance = json.loads((RESULTS / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in provenance["copied_files"].items():
        actual = _digest(RESULTS / filename)
        if actual != expected:
            raise ValueError(f"Week-8 evidence hash mismatch for {filename}: {actual}")


def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", color=NAVY, fontweight="bold", fontsize=11)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#D9E2EC", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def build_physics_map() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True)

    mach_sub = np.linspace(0.15, 0.995, 240)
    mach_sup = np.linspace(1.005, 3.0, 260)
    for mach, color, label in (
        (mach_sub, BLUE, "subsonic branch"),
        (mach_sup, RED, "supersonic branch"),
    ):
        axes[0, 0].plot(mach, rayleigh_ratios(mach)[:, 4], color=color, lw=2.3, label=label)
    axes[0, 0].axvline(1.0, color=NAVY, ls="--", lw=1.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0, 0], "A  Rayleigh heat-addition branches", "Mach number", r"$T_0/T_0^*$")

    for mach, color in ((mach_sub, BLUE), (mach_sup, RED)):
        axes[0, 1].plot(mach, fanno_ratios(mach)[:, 4], color=color, lw=2.3)
    axes[0, 1].axvline(1.0, color=NAVY, ls="--", lw=1.2)
    axes[0, 1].set_ylim(0.0, 4.0)
    _style_axis(axes[0, 1], "B  Fanno friction-to-choking", "Mach number", r"$4fL^*/D$")

    mach = 2.0
    mu = np.arcsin(1.0 / mach)
    beta = np.linspace(mu + 1.0e-4, 0.5 * np.pi - 1.0e-4, 500)
    theta = np.degrees(oblique_theta(mach, beta))
    beta_peak, theta_peak = oblique_detachment(mach)
    axes[0, 2].plot(np.degrees(beta), theta, color=TEAL, lw=2.4)
    axes[0, 2].scatter([np.degrees(beta_peak)], [np.degrees(theta_peak)], color=RED, s=45, zorder=3)
    axes[0, 2].annotate("detachment", (np.degrees(beta_peak), np.degrees(theta_peak)),
                        xytext=(-42, -22), textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "color": GRAY})
    _style_axis(axes[0, 2], r"C  Oblique shock, $M_1=2$", r"shock angle $\beta$ (deg)", r"turn angle $\theta$ (deg)")

    exit_area_ratio = 2.5
    shock_area = np.linspace(1.0001, exit_area_ratio - 0.0001, 260)
    back_pressure = np.array([nozzle_back_pressure(exit_area_ratio, area) for area in shock_area])
    axes[1, 0].plot(shock_area, back_pressure, color=GOLD, lw=2.4)
    _style_axis(axes[1, 0], r"D  Normal shock in a C-D nozzle", r"$A_s/A_t$", r"$P_b/P_{01}$")

    pressure_driver = np.geomspace(1.05, 50.0, 180)
    pressure_driven = np.array([shock_tube_pressure_ratio(value) for value in pressure_driver])
    axes[1, 1].plot(pressure_driver, pressure_driven, color=BLUE, lw=2.4)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    _style_axis(axes[1, 1], "E  Ideal shock-tube compatibility", r"driver ratio $P_4/P_1$", r"star pressure $P_2/P_1$")

    mach_area = np.r_[np.linspace(0.12, 0.99, 220), np.linspace(1.01, 4.0, 260)]
    area_ratio = area_mach(mach_area)
    axes[1, 2].plot(mach_area[mach_area < 1.0], area_ratio[mach_area < 1.0], color=BLUE, lw=2.3, label="subsonic")
    axes[1, 2].plot(mach_area[mach_area > 1.0], area_ratio[mach_area > 1.0], color=RED, lw=2.3, label="supersonic")
    axes[1, 2].axvline(1.0, color=NAVY, ls="--", lw=1.2)
    axes[1, 2].set_ylim(0.9, 5.5)
    axes[1, 2].legend(frameon=False, fontsize=8)
    _style_axis(axes[1, 2], "F  Area-Mach ambiguity", "Mach number", r"$A/A^*$")

    fig.suptitle(
        "Week 8: exact gas-dynamics maps before scientific machine learning",
        color=NAVY,
        fontweight="bold",
        fontsize=16,
    )
    fig.savefig(RESULTS / "week8_exact_physics.png", dpi=200, bbox_inches="tight")
    fig.savefig(RESULTS / "week8_exact_physics.pdf", bbox_inches="tight")
    plt.close(fig)


def build_evidence_overview() -> None:
    """Current classroom branch profiles; research aggregates remain numerical tables."""
    source=ROOT / "results/course_middle_refresh/week08_branch_predictions.npz"
    if not source.is_file():
        raise FileNotFoundError("Execute the Week 8 Lab 2 classroom fit with FLOWMLLAB_COURSE_REFRESH_EXPORT=1 before rebuilding this figure.")
    with np.load(source, allow_pickle=False) as data:
        q={k:data[k].copy() for k in data.files}
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.4),sharex="col")
    for j,branch_id in enumerate((0,1)):
        ids=np.flatnonzero(q["branch"]==branch_id)
        ids=ids[np.argsort(q["input_ratio"][ids])]
        x=q["input_ratio"][ids];truth=q["reference"][ids]
        for key,label,color,style in (("single","single MLP",RED,"--"),("experts","branch expert",BLUE,"-.")):
            pred=q[key][ids]
            axes[0,j].plot(x,pred,style,color=color,lw=1.8,label=label)
            axes[1,j].plot(x,pred-truth,style,color=color,lw=1.8,label=label)
        axes[0,j].plot(x,truth,"k.",ms=4,label="exact Rayleigh")
        axes[0,j].set(title=("Subsonic branch" if j==0 else "Supersonic branch"),ylabel="Mach number")
        axes[1,j].axhline(0,color="0.5",lw=.8)
        axes[1,j].set(xlabel=r"$T_0/T_0^*$",ylabel="predicted - exact Mach")
        for ax in axes[:,j]:ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(RESULTS / "week8_model_evidence.png",dpi=200,bbox_inches="tight")
    fig.savefig(RESULTS / "week8_model_evidence.pdf",bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    verify_source_hashes()
    report = validate_week8_evidence(ROOT)
    build_physics_map()
    build_evidence_overview()
    print("FLOWMLLAB_WEEK8_FIGURES_PASS")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
