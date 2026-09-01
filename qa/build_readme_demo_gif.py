"""Build the compact README animation from retained blind-case evidence."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.demo import available_blind_reynolds, load_blind_demo_case


def main() -> None:
    cases = [load_blind_demo_case(value, ROOT) for value in available_blind_reynolds(ROOT)]
    output = ROOT / "assets" / "flowmllab_blind_demo.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.25), constrained_layout=True)
    colorbars = []

    def render(frame: int):
        case = cases[frame]
        for colorbar in colorbars:
            colorbar.remove()
        colorbars.clear()
        for axis in axes.ravel():
            axis.clear()

        field_limit = max(
            float(np.max(case.reference_speed)),
            float(np.max(case.prediction_speed)),
        )
        extent = [float(case.x[0]), float(case.x[-1]), float(case.y[0]), float(case.y[-1])]
        speed_images = []
        for axis, values, title in (
            (axes[0, 0], case.reference_speed, "Validated CFD velocity"),
            (axes[0, 1], case.prediction_speed, "POD–DeepONet blind velocity"),
        ):
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="YlGnBu_r",
                vmin=0.0,
                vmax=field_limit,
                interpolation="nearest",
            )
            axis.set_title(title, weight="bold")
            axis.set_xlabel("$x/L$")
            axis.set_ylabel("$y/L$")
            axis.set_aspect("equal")
            speed_images.append(image)

        colorbars.append(
            figure.colorbar(
                speed_images[-1],
                ax=axes[0, :].tolist(),
                fraction=0.026,
                pad=0.02,
                label="$|\\mathbf{u}|/U_{lid}$",
            )
        )

        pressure_limit = max(
            float(np.max(np.abs(case.reference_p))),
            float(np.max(np.abs(case.prediction_p))),
        )
        pressure_images = []
        for axis, values, title in (
            (axes[1, 0], case.reference_p, "CFD pressure (zero-mean gauge)"),
            (axes[1, 1], case.prediction_p, "POD–DeepONet blind pressure"),
        ):
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                norm=SymLogNorm(
                    linthresh=0.02,
                    vmin=-pressure_limit,
                    vmax=pressure_limit,
                    base=10,
                ),
                interpolation="nearest",
            )
            axis.set_title(title, weight="bold")
            axis.set_xlabel("$x/L$")
            axis.set_ylabel("$y/L$")
            axis.set_aspect("equal")
            pressure_images.append(image)

        colorbars.append(
            figure.colorbar(
                pressure_images[-1],
                ax=axes[1, :].tolist(),
                fraction=0.026,
                pad=0.02,
                label="Dimensionless pressure (symmetric-log colors)",
            )
        )

        figure.suptitle(
            f"Retained blind case: Re = {case.reynolds:g}   •   "
            f"velocity $L_2$ = {100 * case.relative_l2_uv:.4f}%   •   "
            f"pressure $L_2$ = {100 * case.relative_l2_p:.4f}%",
            fontsize=12.5,
            weight="bold",
        )
        return tuple(axes.ravel())

    animation = FuncAnimation(
        figure,
        render,
        frames=len(cases),
        interval=1800,
        repeat=True,
        blit=False,
    )
    temporary_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    animation.save(temporary_output, writer=PillowWriter(fps=1), dpi=100)
    temporary_output.replace(output)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
