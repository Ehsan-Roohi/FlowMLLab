"""Solve and render the physical-time Week 1.2 cavity movie companion.

Examples
--------
python qa/build_cavity_startup.py solve --reynolds 100 --cells 32 --end-time 10 --dt .02 --output results/cavity_startup/preview
python qa/build_cavity_startup.py render --input results/cavity_startup/preview --output results/cavity_startup/preview

The solve subcommand never renders. It can run in one allocated Unity Workspace
case directory per Slurm task. Render requires a complete matched set of files.
"""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import html as html_module
import json
import math
import os
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "cavity_startup" / "matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize, SymLogNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

from flowmllab.cavity_startup import (observation_times, simulate_startup,
    save_trajectory, load_trajectory, case_summary, frame_fields,
    common_field_bounds, compare_trajectories)
from flowmllab.pressure_velocity_experiments import ExperimentalConfig


LABELS = {"speed": r"$|\mathbf{u}|/U_{\mathrm{lid}}$", "omega": r"$\omega L/U_{\mathrm{lid}}$",
          "p": r"$(p-\langle p\rangle)/(\rho U_{\mathrm{lid}}^2)$"}
NAMES = {"speed": "Speed and instantaneous streamlines", "omega": "Vorticity and instantaneous streamlines",
         "p": "Pressure and instantaneous streamlines"}
FIELD_COLORS = ["#1539df", "#008bff", "#00d8d8", "#16c94b", "#f4e725", "#ff8a14", "#d7191c"]
RGB = LinearSegmentedColormap.from_list("startup_fields", FIELD_COLORS, N=256)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def source_hashes():
    files = ["flowmllab/pressure_velocity.py", "flowmllab/pressure_velocity_experiments.py",
             "flowmllab/cavity_startup.py", "qa/build_cavity_startup.py"]
    return {p: sha256(ROOT / p) for p in files}


def solve(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"re{args.reynolds:g}.npz"
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite retained scientific evidence: {path}; choose a new experiment directory")
    config = ExperimentalConfig(method=args.method, cells=args.cells, reynolds=args.reynolds,
        convection=args.convection, dt=args.dt, end_time=args.end_time,
        outer_correctors=args.outer_correctors, pressure_correctors=args.pressure_correctors,
        max_iterations=math.ceil(args.end_time / args.dt) + math.ceil(args.end_time / args.interval) + 5)
    result = simulate_startup(config, observation_times(args.end_time, args.interval), progress=args.progress)
    digest = save_trajectory(result, path)
    summary = case_summary(result)
    summary.update(trajectory=path.name, trajectory_sha256=digest, source_sha256=source_hashes(),
                   environment={"python": platform.python_version(), "numpy": np.__version__,
                                "platform": platform.platform()},
                   initial_condition=result["initial_condition"], pressure_gauge=result["pressure_gauge"])
    write_json(destination / f"re{args.reynolds:g}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return result


def load_cases(folder, reynolds=(100, 400, 1000)):
    cases = []
    for re in reynolds:
        path = Path(folder) / f"re{re:g}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if sha256(path) != receipt["trajectory_sha256"]:
            raise ValueError(f"Trajectory hash mismatch: {path}")
        case = load_trajectory(path)
        if int(case["config"]["reynolds"]) != re:
            raise ValueError(f"Mislabeled Reynolds number in {path}")
        cases.append(case)
    base = cases[0]
    for case in cases[1:]:
        if not np.array_equal(base["times"], case["times"]):
            raise ValueError("All movie cases must have the same actual physical frame times")
        for key in ("cells", "dt", "method", "convection", "outer_correctors", "pressure_correctors"):
            if base["config"][key] != case["config"][key]:
                raise ValueError(f"Movie cases differ in matched control {key}")
    return cases


def streamfunction_levels(cases):
    """One shared set of physical streamfunction levels for the entire movie."""
    lo = min(float(frame_fields(c, k)["psi"].min()) for c in cases for k in range(len(c["times"])))
    hi = max(float(frame_fields(c, k)["psi"].max()) for c in cases for k in range(len(c["times"])))
    bound = max(abs(lo), abs(hi), 1e-6)
    # Log-spaced contours expose resolved secondary cells without changing data.
    positive = np.geomspace(bound * 1e-5, bound, 17)
    return np.r_[-positive[::-1], positive]


def normalizer(field, bounds):
    limits = bounds[field]
    if field == "speed":
        return Normalize(vmin=limits["vmin"], vmax=limits["vmax"])
    threshold = 1.0 if field == "omega" else 0.01
    return SymLogNorm(linthresh=min(threshold, limits["vmax"]), linscale=1.0,
                      vmin=limits["vmin"], vmax=limits["vmax"], base=10)


def color_ticks(field, norm):
    if field == "speed":
        return np.linspace(0, norm.vmax, 5)
    threshold = min(1.0 if field == "omega" else .01, norm.vmax)
    ticks = sorted(set([-norm.vmax, -threshold, 0.0, threshold, norm.vmax]))
    if norm.vmax / threshold >= 20:
        ticks = sorted(set([*ticks, -10 * threshold, 10 * threshold]))
    return ticks


def draw_panels(cases, field, frame_index, bounds=None, streamline_levels=None, axes=None):
    """Draw native-grid fields; only the contour renderer interpolates cell values."""
    bounds = common_field_bounds(cases) if bounds is None else bounds
    levels = streamfunction_levels(cases) if streamline_levels is None else streamline_levels
    norm = normalizer(field, bounds)
    columns = 4 if len(cases) == 7 else min(3, len(cases))
    rows = math.ceil(len(cases) / columns)
    if axes is None:
        fig, grid = plt.subplots(rows, columns, figsize=(4.15 * columns, 4.15 * rows + 1.85), squeeze=False)
        axes = list(grid.ravel())
    else:
        fig = axes[0].figure
    for ax in axes:
        ax.clear()
    contour_levels = norm.inverse(np.linspace(0, 1, 41))
    image = None
    for ax, result in zip(axes, cases):
        values = frame_fields(result, frame_index)
        n = result["config"]["cells"]
        edges = np.linspace(0, 1, n + 1)
        # The thin boundary half-cells keep their finite-volume cell values.
        image = ax.pcolormesh(edges, edges, values[field], cmap=RGB, norm=norm, shading="flat", rasterized=True)
        if np.ptp(values[field]) > 0:
            ax.contourf(values["x"], values["y"], values[field], levels=contour_levels,
                        cmap=RGB, norm=norm, extend="neither")
        psi = values["psi"]
        selected = levels[(levels > psi.min()) & (levels < psi.max())]
        if len(selected):
            ax.contour(edges, edges, psi, levels=selected, colors="#151515", linewidths=.52,
                       linestyles=["solid" if q < 0 else "dashed" for q in selected])
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$")
        ax.set_title(f"Re = {result['config']['reynolds']:g}", fontsize=16, pad=23)
        ax.xaxis.label.set_size(14); ax.yaxis.label.set_size(14)
        ax.tick_params(labelsize=12)
        ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
        ax.annotate("", xy=(.68, 1.018), xytext=(.32, 1.018), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#202020", "lw": 1.0}, annotation_clip=False)
    for ax in axes[len(cases):]:
        ax.set_visible(False)
    return fig, axes, image, norm


def frame_title(cases, field, frame_index):
    separator = "\n" if len(cases) < 3 else "  |  "
    return NAMES[field] + separator + f"$tU_{{\\mathrm{{lid}}}}/L={cases[0]['times'][frame_index]:g}$"


def comparison_figure(cases, field, frame_index, bounds=None, streamline_levels=None):
    fig, axes, image, norm = draw_panels(cases, field, frame_index, bounds, streamline_levels)
    height = fig.get_figheight()
    top_inches = 1.30 if len(cases) < 3 else 1.05
    fig.subplots_adjust(left=.10 if len(cases) < 3 else .055, right=.98,
                        top=1-top_inches/height, bottom=1.75/height, wspace=.23, hspace=.43)
    color_ax = fig.add_axes([.25, .65/height, .5, .10/height])
    ticks = color_ticks(field, norm)
    if len(cases) < 3:
        ticks = np.linspace(norm.vmin, norm.vmax, 3)
    bar = fig.colorbar(image, cax=color_ax, orientation="horizontal", ticks=ticks)
    bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.3g}"))
    bar.ax.set_xlabel(LABELS[field], labelpad=5, fontsize=14)
    bar.ax.tick_params(labelsize=12)
    title = fig.suptitle(frame_title(cases, field, frame_index), y=1-.13/height,
                        fontsize=12 if len(cases) < 3 else 18)
    c = cases[0]["config"]
    separator = "\n" if len(cases) < 3 else " · "
    footer = f"{c['cells']} × {c['cells']} MAC cells · {c['method'].upper()}" + separator + f"Δt* ≤ {c['dt']:g} · startup from rest"
    fig.text(.5, (.90 if len(cases) < 3 else 1.10)/height, footer, ha="center",
             fontsize=9 if len(cases) < 3 else 11)
    return fig, axes, title


def ffmpeg_path():
    explicit = os.environ.get("FLOWMLLAB_FFMPEG")
    path = explicit or shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg is needed for MP4 export. Install it or set FLOWMLLAB_FFMPEG; scientific simulation itself does not need it.")
    return path


def render(args):
    if min(args.fps, args.dpi, args.stride) < 1:
        raise ValueError("fps, dpi and stride must be positive")
    cases = load_cases(args.input, args.reynolds)
    qualification = "Finite-time 2D startup demonstration; sensitivity evidence is separate; no steady Ghia/3D validation claim."
    selection_file = args.input.parent / "presentation.json"
    if selection_file.is_file():
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        if (args.input.parent / selection["trajectory_directory"]).resolve() == args.input.resolve():
            qualification = selection["qualification"]
    if max(args.reynolds) >= 3200:
        qualification += " High-Re panels are provisional, mesh-dependent teaching examples, not resolved flow predictions; inspect the matched-grid profiles and sensitivity values."
    bounds = common_field_bounds(cases); levels = streamfunction_levels(cases)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#323232", "axes.linewidth": .8})
    records = {}
    for field in ("speed", "omega", "p"):
        fig, axes, title = comparison_figure(cases, field, 0, bounds, levels)
        def update(index):
            draw_panels(cases, field, index, bounds, levels, axes=axes)
            title.set_text(frame_title(cases, field, index))
            return axes
        selected = list(range(0, len(cases[0]["times"]), args.stride))
        if selected[-1] != len(cases[0]["times"]) - 1:
            selected.append(len(cases[0]["times"]) - 1)
        animation = FuncAnimation(fig, update, frames=selected, interval=1000 / args.fps, blit=False)
        movie = output / f"cavity_startup_{field}.mp4"
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path()
        animation.save(movie, writer=FFMpegWriter(fps=args.fps, codec="libx264",
                       extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart"]), dpi=args.dpi)
        plt.close(fig)
        snapshots = []
        for requested in (.2, 1., 4., cases[0]["times"][-1]):
            index = int(np.argmin(np.abs(cases[0]["times"] - requested)))
            if index in snapshots:
                continue
            snapshots.append(index)
            fig, _, _ = comparison_figure(cases, field, index, bounds, levels)
            stem = f"{field}_t{cases[0]['times'][index]:g}"
            fig.savefig(output / (stem + ".png"), dpi=170)
            fig.savefig(output / (stem + ".pdf"))
            plt.close(fig)
        records[field] = {"movie": movie.name, "movie_sha256": sha256(movie),
                          "frames": len(selected), "fps": args.fps,
                          "displayed_physical_times": cases[0]["times"][selected].tolist(),
                          "limits": bounds[field],
                          "scale": "linear" if field == "speed" else "symmetric logarithmic",
                          "linear_threshold": None if field == "speed" else (1.0 if field == "omega" else .01)}
        print(f"Rendered {movie.name}", flush=True)
    receipts = {f"re{re:g}": json.loads((args.input / f"re{re:g}.json").read_text(encoding="utf-8"))
                for re in args.reynolds}
    write_json(output / "movies.json", {"cases": [case_summary(c) for c in cases], "fields": records,
        "trajectory_receipts": receipts,
        "field_palette": {"name": "startup_fields", "stops": FIELD_COLORS},
        "source_hash_scope": "source_sha256 identifies current rendering/admission code; trajectory_receipts preserve the actual simulation source hashes",
        "streamfunction_levels": levels.tolist(), "source_sha256": source_hashes(),
        "streamlines": "Instantaneous streamfunction contours of each native MAC field, not particle pathlines.",
        "comparison": "Same physical frame times, mesh, timestep controls, initial and wall conditions; only Reynolds number varies.",
        "color_limits": "Fixed across all cases and all frames, retaining full recorded extrema; signed fields use the stated symmetric-log transform.",
        "qualification": qualification})
    tags = "\n".join(f'<section><h2>{NAMES[k]}</h2><video controls loop preload="metadata" width="100%" src="{records[k]["movie"]}"></video></section>' for k in records)
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Cavity startup</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;font:16px system-ui;color:#182433}video{background:white}section{margin:30px 0}aside{padding:14px 18px;border-left:4px solid #596675;background:#f3f5f7}</style><h1>Cavity startup from rest</h1><p>Same physical times and fixed field scales across Reynolds numbers. Black curves are instantaneous streamlines, not particle tracks. Pressure has zero spatial mean.</p><aside><strong>Numerical qualification.</strong> ' + html_module.escape(qualification) + '</aside>' + tags + '<p>Signed vorticity uses a symmetric-log scale, linear for |ωL/U| ≤ 1. Pressure uses a symmetric-log scale, linear for |p/(ρU²)| ≤ 0.01. Full recorded extrema are retained.</p><p>See the companion notebook and lecture for matched-grid physical profiles and quantified sensitivity.</p></html>'
    (output / "index.html").write_text(html, encoding="utf-8", newline="\n")
    return records


def sensitivity(args):
    a, b = load_trajectory(args.candidate), load_trajectory(args.reference)
    report = compare_trajectories(a, b)
    write_json(args.output, report)
    print(json.dumps({"final_frame": report["frames"][-1], "meaning": report["meaning"]}, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("solve")
    command.add_argument("--reynolds", type=int, choices=[100, 400, 1000, 3200, 5000, 7500, 10000], required=True)
    command.add_argument("--cells", type=int, default=32)
    command.add_argument("--dt", type=float, default=.02)
    command.add_argument("--end-time", type=float, default=10.)
    command.add_argument("--interval", type=float, default=.1)
    command.add_argument("--method", choices=["piso", "pimple"], default="pimple")
    command.add_argument("--convection", choices=["central", "upwind"], default="central")
    command.add_argument("--outer-correctors", type=int, default=3)
    command.add_argument("--pressure-correctors", type=int, default=2)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--progress", action="store_true")
    command.set_defaults(function=solve)
    command = commands.add_parser("render")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--reynolds", type=int, nargs="+", default=[100, 400, 1000])
    command.add_argument("--fps", type=int, default=15)
    command.add_argument("--dpi", type=int, default=120)
    command.add_argument("--stride", type=int, default=1)
    command.set_defaults(function=render)
    command = commands.add_parser("compare")
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(function=sensitivity)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.function(args)
