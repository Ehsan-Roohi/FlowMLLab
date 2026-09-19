#!/usr/bin/env python3
"""Render full-height Week 9 nozzle fields by stated-plane parity reflection.

The retained DSMC export supplies one half of the nozzle, ending at its stated
symmetry plane (y=92 micrometres).  These *derived visualisations* complete the
other half for teaching: density, U, temperature, Mach and pressure are even;
V is odd and is set to zero at the plane.  They do not alter raw arrays, model
weights, selection, or raw-label error metrics.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.nozzle_transport import FIELDS
from flowmllab.mahdavi_deeponet import load_nozzle_fields

OUT = ROOT / "results" / "nozzle_transport"
LABELS = ("Density (source units)", r"$U$ (m/s)", r"$V$ (m/s)",
          "Temperature (K)", "Mach number", "Pressure (source units)")
ODD_CHANNEL = 2


def reflect_grid(values: np.ndarray, *, odd: bool) -> np.ndarray:
    """Reflect rows about the final (stated symmetry) row without duplicating it."""
    complete = np.array(values, copy=True)
    if odd:
        complete[-1, :] = 0.0
    reflected = complete[-2::-1, :]
    if odd:
        reflected = -reflected
    return np.concatenate((complete, reflected), axis=0)


def full_coordinates(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = y_m[-1, :]
    return (np.concatenate((x_m, x_m[-2::-1, :]), axis=0),
            np.concatenate((y_m, 2.0 * center[None, :] - y_m[-2::-1, :]), axis=0))


def main() -> None:
    data = load_nozzle_fields(ROOT)
    saved = np.load(OUT / "predictions.npz", allow_pickle=False)
    x_um, y_um = full_coordinates(data["x_m"], data["y_m"])
    x_um *= 1e6
    y_um *= 1e6
    for pressure in saved["pressure_kpa"].astype(int):
        raw_index = int(np.flatnonzero(data["pressure_kpa"] == pressure)[0])
        predicted_index = int(np.flatnonzero(saved["pressure_kpa"] == pressure)[0])
        reference = np.stack([data[field][raw_index] for field in FIELDS], axis=-1)
        prediction = saved["symmetry_constrained"][predicted_index]
        fig, axes = plt.subplots(6, 3, figsize=(13.2, 21.0), layout="constrained")
        for channel in range(6):
            reference_full = reflect_grid(reference[:, :, channel], odd=channel == ODD_CHANNEL)
            prediction_full = reflect_grid(prediction[:, :, channel], odd=channel == ODD_CHANNEL)
            error_full = np.abs(prediction_full - reference_full)
            lo = min(float(reference_full.min()), float(prediction_full.min()))
            hi = max(float(reference_full.max()), float(prediction_full.max()))
            cmap = "RdBu_r" if channel == ODD_CHANNEL else ("inferno" if channel == 3 else "viridis")
            for col, values in enumerate((reference_full, prediction_full, error_full)):
                ax = axes[channel, col]
                image = ax.pcolormesh(x_um, y_um, values, shading="auto",
                                      cmap="magma" if col == 2 else cmap,
                                      vmin=0 if col == 2 else lo,
                                      vmax=max(float(error_full.max()), 1e-12) if col == 2 else hi,
                                      rasterized=True)
                ax.axhline(92.0, color="white", lw=0.75, ls="--", alpha=0.9)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(r"$x$ ($\mu$m)")
                if col == 0:
                    ax.set_ylabel(LABELS[channel] + "\n" + r"$y$ ($\mu$m)")
                if channel == 0:
                    ax.set_title(("DSMC half-domain, parity-completed", "Selected model, parity-completed", "Absolute difference")[col])
                fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
        fig.suptitle(
            f"Week 9 micro-nozzle at {pressure} kPa | full-height parity-completed view\n"
            r"dashed line: stated symmetry plane $y=92\,\mu$m; $V=0$ imposed only for this visualisation",
            fontsize=15,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(OUT / f"nozzle_P{pressure}_fields_full_domain.{suffix}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
