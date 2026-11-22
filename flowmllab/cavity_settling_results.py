"""Read and plot the closed cavity campaign without starting a CFD solve.

All fields are selected native snapshots. Every opened payload is checked
against the publication manifest; spatial contour rendering is not smoothing.
"""
from functools import cached_property
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

from flowmllab.cavity_startup import frame_fields

REYNOLDS = (100, 400, 1000, 3200, 5000, 7500, 10000)
LABELS = {"speed": r"$|\mathbf{u}|/U_{\rm lid}$",
          "p": r"$(p-\langle p\rangle)/(\rho U_{\rm lid}^2)$",
          "omega": r"$\omega_z L/U_{\rm lid}$"}
NAMES = {"speed": "Speed and instantaneous streamlines", "p": "Signed pressure",
         "omega": "Signed vorticity"}


class ResultBundle:
    """A lazy, checksum-checked view of the published scientific evidence."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "flowmllab.cavity.final-results.v1":
            raise ValueError("Unsupported final-results manifest")
        self.cases = {c["id"]: c for c in self.manifest["cases"]}
        self.confirmations = {c["reynolds"]: c for c in self.manifest["confirmations"]}
        self._payloads = {}

    def payload(self, relative):
        if relative not in self.manifest["files"]:
            raise ValueError("File is not admitted by the result manifest")
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Result path escapes the bundle")
        if relative not in self._payloads:
            raw = path.read_bytes()
            identity = self.manifest["files"][relative]
            if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError(f"Scientific payload checksum mismatch: {relative}")
            self._payloads[relative] = raw
        return self._payloads[relative]

    def json(self, relative):
        return json.loads(self.payload(relative))

    def verify_all(self):
        for relative in self.manifest["files"]:
            self.payload(relative)
        return len(self.manifest["files"])

    @cached_property
    def analysis(self):
        raw = gzip.decompress(self.payload("data/analysis.json.gz"))
        if hashlib.sha256(raw).hexdigest() != self.manifest["original_analysis_sha256"]:
            raise ValueError("Decoded analysis differs from the remote source")
        result = json.loads(raw)
        if result["errors"] or result["coverage"]["verified_checkpoint"] != 22:
            raise ValueError("The analysis does not cover the complete admitted campaign")
        return result

    def baseline(self, reynolds):
        matches = [c for c in self.cases.values() if c["purpose"] == "continue" and c["config"]["reynolds"] == reynolds]
        if len(matches) != 1:
            raise ValueError("Select one of the seven retained Reynolds numbers")
        return matches[0]

    def windows(self, case_id):
        return self.json(self.cases[case_id]["windows_file"])["windows"]

    def native(self, entry, time=None):
        time = entry.get("time", entry["fields"][-1]["time"]) if time is None else float(time)
        records = [r for r in entry["fields"] if r["time"] == time]
        if len(records) != 1:
            raise ValueError(f"No exact retained native field at t*={time:g}; no time substitution is allowed")
        record = records[0]
        raw = self.payload(record["file"])
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Native field record identity mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"]))
            arrays = {k: archive[k].copy() for k in ("u_faces", "v_faces", "p")}
        if meta["config"] != entry["config"] or meta["time"] != time or meta["contract_sha256"] != record["contract_sha256"]:
            raise ValueError("Native field configuration/time/contract mismatch")
        n = entry["config"]["cells"]
        for key, shape in {"u_faces": (n, n + 1), "v_faces": (n + 1, n), "p": (n, n)}.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError("Invalid native field shape or values")
        fields = frame_fields({"config": entry["config"], **{k: v[None, ...] for k, v in arrays.items()}}, 0)
        return fields, meta

    def outcomes(self):
        rows = []
        for re in REYNOLDS:
            case = self.baseline(re)
            windows = self.windows(case["id"])
            complete = [w for w in windows if not w.get("partial_initial_window", False)]
            last = complete[-1]
            confirmation = self.confirmations.get(re)
            report = self.json(confirmation["confirmation_file"]) if confirmation else None
            passed = bool(report and report["complete"] and report["independent_confirmation_passed"] and not report["failed_metrics"])
            rows.append({"Re": re, "baseline_end_time": case["time"],
                         "last_window_start": last["start_time"], "last_window_end": last["end_time"],
                         "failed_gates": len(last["failed_metrics"]),
                         "consecutive_passes": case["consecutive_passes"],
                         "confirmation_end_time": report["actual_time"] if report else None,
                         "fixed_grid_confirmed": passed,
                         "conclusion": "Fixed-grid temporal stationarity confirmed" if passed else "Unresolved at the recorded endpoint"})
        return rows

    def comparisons(self):
        rows = []
        for pair in self.analysis["comparisons"]:
            frames = [f for f in pair["frames"] if f["time"] == 10.0]
            if pair["status"] != "matched" or len(frames) != 1:
                raise ValueError("Every retained control must have one exact t*=10 comparison")
            frame = frames[0]
            if frame["candidate_recorded_time"] != 10 or frame["reference_recorded_time"] != 10:
                raise ValueError("Control comparison times differ")
            rows.append({"id": pair["id"], "kind": pair["kind"],
                         "Re": self.cases[pair["candidate_id"]]["config"]["reynolds"],
                         "time": 10., "candidate_id": pair["candidate_id"], "reference_id": pair["reference_id"],
                         "velocity_relative_l2": frame["velocity_relative_l2"],
                         "pressure_rms_difference": frame["pressure_rms_difference"]})
        return rows


def _norm(field, fields):
    lo = min(float(f[field].min()) for f in fields)
    hi = max(float(f[field].max()) for f in fields)
    if field == "speed":
        return Normalize(0, max(1., hi)), np.linspace(0, max(1., hi), 5)
    bound = max(abs(lo), abs(hi), 1e-12)
    threshold = min(1. if field == "omega" else .01, bound)
    norm = SymLogNorm(linthresh=threshold, vmin=-bound, vmax=bound)
    ticks = [-bound, -threshold, 0., threshold, bound]
    return norm, sorted(set(ticks))


def _panel(ax, fields, field, norm, *, streamlines=False, streamline_levels=None):
    n = len(fields["x"])
    edges = np.linspace(0., 1., n + 1)
    # Preserve the finite-volume half-cells at the boundary; contours use
    # native cell centers. No filtering, clipping or gauge modification.
    ax.pcolormesh(edges, edges, fields[field], cmap="turbo", norm=norm, shading="flat", rasterized=True)
    image = ax.contourf(fields["x"], fields["y"], fields[field],
                        levels=norm.inverse(np.linspace(0, 1, 49)), cmap="turbo", norm=norm)
    if streamlines:
        psi = fields["psi"]
        levels = np.asarray(streamline_levels)
        levels = levels[(levels > psi.min()) & (levels < psi.max())]
        if levels.size:
            ax.contour(edges, edges, psi, levels=levels, colors="#161616", linewidths=.55,
                       linestyles=["solid" if x < 0 else "dashed" for x in levels])
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=r"$x/L$", ylabel=r"$y/L$")
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    return image


def overview(bundle, field="speed", time=None):
    """Seven native fields at their own endpoints, or exact common t*=10/100."""
    if field not in LABELS:
        raise ValueError("Choose speed, p or omega")
    entries = [bundle.baseline(re) for re in REYNOLDS]
    pairs = [bundle.native(c, time) for c in entries]
    values = [f for f, _ in pairs]
    norm, ticks = _norm(field, values)
    bound = max(max(abs(f["psi"].min()), abs(f["psi"].max())) for f in values)
    positive = np.geomspace(max(bound, 1e-6) * 1e-5, max(bound, 1e-6), 19)
    levels = np.r_[-positive[::-1], positive]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 9.1))
    fig.subplots_adjust(left=.055, right=.98, top=.86, bottom=.19, wspace=.27, hspace=.38)
    for ax, c, (f, meta) in zip(axes.flat, entries, pairs):
        image = _panel(ax, f, field, norm, streamlines=field == "speed", streamline_levels=levels)
        ax.set_title(f"Re = {c['config']['reynolds']:g}   |   t* = {meta['time']:g}", fontsize=12, pad=10)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(.04, .80, "128 × 128 cells\nΔt* = 0.01\nPIMPLE: 3 outer / 2 pressure\nCentral convection\nImpulsive unit lid; rest initially", va="top", fontsize=11, linespacing=1.6)
    axes.flat[-1].text(.04, .28, "Confirmed separately with 6/4:\nRe = 100, 400, 1000\nHigher-Re endpoints: unresolved", va="top", fontsize=10, linespacing=1.5)
    label = "Own baseline endpoints (different physical times)" if time is None else f"Matched physical time t* = {time:g}"
    fig.suptitle(NAMES[field] + "\n" + label, fontsize=16, y=.975)
    cax = fig.add_axes([.29, .10, .42, .018])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
    cb.set_label(LABELS[field], labelpad=6)
    scale = "Linear speed scale." if field == "speed" else "Symmetric-log scale; linear for |value| ≤ " + ("1." if field == "omega" else "0.01.")
    fig.text(.5, .018, "Native values, full extrema, one scale across all seven panels. " + scale, ha="center", fontsize=10)
    return fig


def grid_fields(bundle, reynolds):
    """Fine reference fill and coarse contours at identical physical levels."""
    pairs = [p for p in bundle.analysis["comparisons"] if p["kind"] == "grid" and bundle.cases[p["candidate_id"]]["config"]["reynolds"] == reynolds]
    if len(pairs) != 1:
        raise ValueError("Choose Re = 3200, 5000, 7500 or 10000 for the retained mesh control")
    pair = pairs[0]
    a, ma = bundle.native(bundle.cases[pair["candidate_id"]], 10.)
    b, mb = bundle.native(bundle.cases[pair["reference_id"]], 10.)
    if (ma["config"]["cells"], mb["config"]["cells"]) != (128, 256):
        raise ValueError("Unexpected coarse/fine grid mapping")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.9))
    fig.subplots_adjust(left=.07, right=.95, top=.79, bottom=.23, wspace=.40)
    for ax, field in zip(axes, ("speed", "p")):
        norm, ticks = _norm(field, [a, b])
        im = _panel(ax, b, field, norm)
        levels = np.linspace(.1, .9, 9) if field == "speed" else np.array([-.1, -.03, -.01, 0., .01, .03, .1])
        levels = levels[(levels > max(a[field].min(), b[field].min())) & (levels < min(a[field].max(), b[field].max()))]
        for f, color, style, width in ((b, "#151515", "solid", .8), (a, "#6b21a8", "dashed", 1.1)):
            lines = ax.contour(f["x"], f["y"], f[field], levels=levels, colors=color, linestyles=style, linewidths=width)
            if style == "dashed":
                lines.set_path_effects([path_effects.Stroke(linewidth=2.2, foreground="white"), path_effects.Normal()])
        ax.set_title(NAMES[field].replace(" and instantaneous streamlines", ""), fontsize=13, pad=12)
        bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.04, ticks=ticks)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
        bar.set_label(LABELS[field], labelpad=8)
    fig.suptitle(f"Matched mesh comparison: Re = {reynolds:g}, t* = 10\nΔt* = 0.0025; PIMPLE 3/2; central convection", fontsize=15, y=.97)
    legend = fig.legend(handles=[Line2D([], [], color="#151515", lw=1, label="256 × 256"),
                                Line2D([], [], color="#6b21a8", ls="--", lw=1.4, label="128 × 128")],
                        loc="lower center", bbox_to_anchor=(.5, .11), ncol=2, frameon=False)
    fig.text(.5, .048, "Filled field: 256 × 256. Lines use identical physical levels; each grid retains its native values.\nEarly-time numerical sensitivity, not late-time grid convergence. Pressure uses a symmetric-log scale (threshold 0.01).", ha="center", fontsize=10)
    return fig


def late_profiles(bundle, reynolds):
    """Physical profiles at explicit late times; no stationarity inference by eye."""
    case = bundle.baseline(reynolds)
    samples = [(case, t, f"Baseline t* = {t:g}") for t in sorted({100., case["time"]})]
    if reynolds in bundle.confirmations:
        entry = bundle.confirmations[reynolds]
        t = entry["fields"][-1]["time"]
        samples.append((entry, t, f"6/4 confirmation t* = {t:g}"))
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.9))
    fig.subplots_adjust(left=.065, right=.98, top=.68, bottom=.24, wspace=.42)
    colors = ("#1c67a7", "#c53d21", "#6b21a8")
    styles = ("-", "--", "-.")
    specs = [("u", "vertical", r"$y/L$", r"$u/U_{\rm lid}$"),
             ("v", "horizontal", r"$x/L$", r"$v/U_{\rm lid}$"),
             ("p", "vertical", r"$y/L$", LABELS["p"]),
             ("p", "horizontal", r"$x/L$", LABELS["p"])]
    handles = []
    for (entry, time, label), color, style in zip(samples, colors, styles):
        fields, _ = bundle.native(entry, time)
        handles.append(Line2D([], [], color=color, ls=style, lw=1.6, label=label))
        for ax, (key, direction, xlabel, ylabel) in zip(axes, specs):
            array = fields[key]
            # Both grids are even: x/L or y/L=0.5 lies halfway between the
            # adjacent native cell centers. No extrapolation to a wall.
            middle = array.shape[0] // 2
            values = .5 * (array[:, middle - 1] + array[:, middle]) if direction == "vertical" else .5 * (array[middle - 1, :] + array[middle, :])
            coordinate = fields["y" if direction == "vertical" else "x"]
            ax.plot(coordinate, values, color=color, ls=style, lw=1.6)
            ax.set(xlim=(0, 1), xlabel=xlabel, ylabel=ylabel)
            ax.set_title(direction.capitalize() + " centerline", fontsize=11)
            ax.grid(alpha=.22)
    fig.suptitle(f"Late physical profiles: Re = {reynolds:g}\n128 × 128 cells; Δt* = 0.01; PIMPLE; central convection", fontsize=15, y=.97)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .82), ncol=3, frameon=False)
    fig.text(.5, .065, "Actual times are stated separately; these curves show temporal evolution, not a matched-time mesh test.\nCell-center samples are retained; pressure has zero area mean, with no extrapolated wall-pressure values.", ha="center", fontsize=10)
    return fig
