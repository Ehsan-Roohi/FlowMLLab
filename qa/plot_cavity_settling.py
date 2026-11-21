"""Plot admitted cavity settling evidence without solving or altering fields.

The input is qa/analyze_cavity_settling.py's compact scientific JSON. Native
sample positions, signs and physical times are retained. No curve is drawn
across a missing interval, and a numerical comparator is never called truth.

Examples::

    python qa/plot_cavity_settling.py --analysis analysis.json --output figures
    python qa/plot_cavity_settling.py --analysis analysis.json --output figures \
        --profile-times 10 25 --histories continuation
    python qa/plot_cavity_settling.py --synthetic-smoke --output tmp/plot_smoke

The last command is a labelled rendering test, not a CFD experiment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "cavity_settling_plot" / "matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


INPUT_SCHEMA = "flowmllab.cavity.settling.analysis.v1"
OUTPUT_SCHEMA = "flowmllab.cavity.settling.plots.v1"
COLORS = {"blue": "#1557b0", "red": "#cc3929", "green": "#158267", "purple": "#8245a6"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
         "axes.titlesize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
         "legend.fontsize": 9, "axes.edgecolor": "#39424d", "axes.linewidth": .8,
         "text.color": "#17212c", "axes.labelcolor": "#17212c", "figure.facecolor": "white",
         "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8", newline="\n")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError("Empty figure identifier")
    return cleaned


def _table(case, name):
    table = case.get(name) or {}
    columns = table.get("columns", [])
    rows = table.get("rows", [])
    if not rows:
        return {}
    if len(set(columns)) != len(columns) or "time" not in columns:
        raise ValueError(f"Invalid {name} column names")
    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(f"Invalid {name} table shape")
    time = array[:, columns.index("time")]
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"{name} physical times must increase strictly")
    return {key: array[:, i] for i, key in enumerate(columns)}


def _policy(case):
    return (case.get("stationarity") or {}).get("policy") or {}


def _split(time, values, interval):
    """Split missing values and time gaps; return actual samples only."""
    result, start = [], None
    good = np.isfinite(time) & np.isfinite(values)
    for i in range(len(time)):
        gap = i > 0 and interval is not None and time[i] - time[i - 1] > 1.5 * interval + 1e-10
        if not good[i] or gap:
            if start is not None:
                result.append((time[start:i], values[start:i])); start = None
        if good[i] and start is None:
            start = i
    if start is not None:
        result.append((time[start:], values[start:]))
    return result


def _interval(time, declared):
    if declared is not None and np.isfinite(declared) and declared > 0:
        return float(declared)
    differences = np.diff(time)
    return float(np.median(differences)) if len(differences) else None


def _series(case, key):
    """Prefer fine actual-step diagnostics, retaining a labelled native prefix."""
    diagnostic, frames = _table(case, "diagnostics"), _table(case, "frame_history")
    policy = _policy(case)
    diagnostic_metadata = case.get("diagnostics") or {}
    stride = diagnostic_metadata.get("stride", 1)
    if type(stride) is not int or stride < 1:
        raise ValueError("Diagnostic display stride must be a positive integer")
    declared = diagnostic_metadata.get("continuation_interval", policy.get("diagnostic_interval"))
    if declared is not None:
        declared = float(declared) * stride
    output = []
    if key in diagnostic and np.isfinite(diagnostic[key]).any():
        dtime, dvalues = diagnostic["time"], diagnostic[key]
        finite = np.isfinite(dvalues)
        first, last = dtime[finite][[0, -1]]
        if key in frames:
            outside = (frames["time"] < first - 1e-10) | (frames["time"] > last + 1e-10)
            values = np.where(outside, frames[key], np.nan)
            for x, y in _split(frames["time"], values, _interval(frames["time"], policy.get("snapshot_interval"))):
                output.append((x, y, "native frame"))
        for x, y in _split(dtime, dvalues, _interval(dtime, declared)):
            output.append((x, y, "actual-step diagnostic"))
    elif key in frames:
        for x, y in _split(frames["time"], frames[key], _interval(frames["time"], policy.get("snapshot_interval"))):
            output.append((x, y, "native frame"))
    return sorted(output, key=lambda item: float(item[0][0]))


def _draw_series(ax, segments, label, color, pattern="-", marker=None):
    labelled = False
    for time, values, _source in segments:
        # Two isolated samples are observations, not evidence of a resolved curve.
        line_style = pattern if len(time) > 2 else "None"
        sample_marker = marker if len(time) > 2 else "o"
        ax.plot(time, values, color=color, ls=line_style, lw=1.65,
                marker=sample_marker, ms=3, markevery=max(1, len(time) // 13),
                label=label if not labelled else "_nolegend_")
        labelled = True
    return labelled


def _outside_legend(ax, columns=2):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1.025),
                           ncol=min(columns, len(handles)), frameon=False, borderaxespad=0,
                           columnspacing=1.25, handlelength=2.7)
        legend.set_in_layout(False)
        return legend
    return None


def _base_axes(ax, title, ylabel, zero=False):
    ax.set_title(title, y=1.22, pad=0)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e3e8ed", lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out")
    if zero:
        ax.axhline(0., color="#7c8791", lw=.65, zorder=0)


def _conditions(case):
    config = case.get("config") or {}
    entry = case.get("case") or {}
    reynolds = config.get("reynolds", entry.get("reynolds", "?"))
    n = config.get("cells", entry.get("cells", "?"))
    dt = config.get("dt", entry.get("dt", "?"))
    outer = config.get("outer_correctors", entry.get("outer", "?"))
    pressure = config.get("pressure_correctors", entry.get("pressure", "?"))
    method = str(config.get("method", "PIMPLE")).upper()
    return f"Re = {reynolds:g}  |  {n} x {n} cells  |  dt* = {dt:g}  |  {method}: {outer} outer / {pressure} pressure corrections"


def _short_conditions(case):
    config = case["config"]
    return f"N={config['cells']}, dt*={config['dt']:g}, {config['outer_correctors']} outer/{config['pressure_correctors']} pressure"


def _probe_position(case, name):
    point = (case.get("probe_locations") or {}).get(name)
    if point and point.get("actual_xy"):
        return point["actual_xy"]
    # This is the continuation's fixed nearest-cell definition, not a search
    # for the instantaneous vortex center. Report the actual sampled position.
    n = case["config"]["cells"]
    requested = (.05, .95) if name == "upper_left" else (.5, .5)
    return [(min(n - 1, int(x * n)) + .5) / n for x in requested]


def _footer(fig, text):
    wrapped = "\n".join(textwrap.wrap(text, width=160, break_long_words=False))
    fig.text(.065, .018, wrapped, ha="left", va="bottom", fontsize=9, color="#43515f", linespacing=1.4)


def _synthetic_stamp(fig, fixture):
    if fixture:
        fig.text(.5, .993, "SYNTHETIC RENDERING FIXTURE - NOT CFD RESULTS", ha="center", va="top",
                 fontsize=11, weight="bold", color="#923a31")


def history_figure(case_id, case, fixture=False):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.7))
    fig.subplots_adjust(left=.07, right=.975, bottom=.145, top=.80, wspace=.28, hspace=.9)
    fig.text(.5, .949, "Cavity startup: physical-time diagnostic history", ha="center", fontsize=16, weight="bold")
    fig.text(.5, .914, _conditions(case), ha="center", fontsize=11)
    _synthetic_stamp(fig, fixture)
    records, legends = {}, []
    panels = [
        ("Upper-left probe velocity", r"$u/U_{\rm lid},\ v/U_{\rm lid}$",
         [("upper_left_u", r"$u/U_{\rm lid}$", COLORS["blue"], "-"), ("upper_left_v", r"$v/U_{\rm lid}$", COLORS["red"], "--")], True),
        ("Central probe velocity", r"$u/U_{\rm lid},\ v/U_{\rm lid}$",
         [("central_u", r"$u/U_{\rm lid}$", COLORS["blue"], "-"), ("central_v", r"$v/U_{\rm lid}$", COLORS["red"], "--")], True),
        ("Signed probe vorticity", r"$\omega L/U_{\rm lid}$",
         [("upper_left_omega", "Upper left", COLORS["green"], "-"), ("central_omega", "Central", COLORS["purple"], "--")], True),
        ("Kinetic energy", r"$\frac{1}{2}\langle u^{*2}+v^{*2}\rangle$",
         [("kinetic_energy", "Native velocity field", COLORS["blue"], "-")], False),
        ("Signed streamfunction extrema", r"$\psi/(U_{\rm lid}L)$",
         [("psi_min", r"$\min(\psi^*)$", COLORS["blue"], "-"), ("psi_max", r"$\max(\psi^*)$", COLORS["red"], "--")], True),
        ("Independent momentum defects", r"$R_{\infty}L/U_{\rm lid}^2$",
         [("steady_momentum_defect_linf", "Steady equations", COLORS["blue"], "-"), ("time_step_momentum_defect_linf", "Time-step equations", COLORS["red"], "--")], False),
    ]
    all_times = []
    for ax, (title, ylabel, curves, zero) in zip(axes.flat, panels):
        _base_axes(ax, title, ylabel, zero)
        found = False
        for key, label, color, pattern in curves:
            segments = _series(case, key)
            if "momentum_defect" in key and any(np.any(values < 0) for _, values, _ in segments):
                raise ValueError("A momentum-defect norm cannot be negative")
            found |= _draw_series(ax, segments, label, color, pattern)
            records[key] = [{"source": source, "start_time": float(time[0]), "end_time": float(time[-1]),
                             "samples": len(time), "min": float(values.min()), "max": float(values.max())}
                            for time, values, source in segments]
            all_times.extend(float(x) for time, _, _ in segments for x in (time[0], time[-1]))
        if not found:
            ax.text(.5, .5, "No admitted samples for this quantity", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#697684")
        ax.set_xlabel(r"$t^*=tU_{\rm lid}/L$")
        if title == "Independent momentum defects":
            ax.set_yscale("symlog", linthresh=1e-10, linscale=1, base=10)
            thresholds = _policy(case).get("thresholds", {})
            steady_gate = thresholds.get("steady_momentum_defect_linf")
            step_gate = thresholds.get("time_step_momentum_defect_linf")
            if steady_gate is not None and steady_gate == step_gate:
                ax.axhline(steady_gate, color="#43484d", ls=":", lw=1.1, label=f"Frozen gate: {steady_gate:g}")
            ax.set_ylim(bottom=0.)
        legend = _outside_legend(ax, columns=2)
        if legend:
            legends.append(legend)
    if all_times:
        lo, hi = min(all_times), max(all_times)
        if lo == hi:
            lo, hi = lo - .01, hi + .01
        for ax in axes.flat:
            ax.set_xlim(lo, hi)
    status = case.get("admission", "unavailable")
    stationarity = case.get("stationarity") or {}
    flag = "yes" if stationarity.get("candidate") is True else "no" if stationarity.get("candidate") is False else "unavailable"
    ul, center = _probe_position(case, "upper_left"), _probe_position(case, "central")
    admission_label = {"verified_checkpoint": "verified native checkpoint", "seed_only": "seed prefix only"}.get(status, status)
    foot = (f"Evidence: {admission_label}; available data through t*={max(all_times):g}. " if all_times else f"Evidence: {admission_label}; no time samples. ")
    foot += (f"Upper-left/central probes at (x/L,y/L)=({ul[0]:.4f},{ul[1]:.4f}) / ({center[0]:.4f},{center[1]:.4f}); these are fixed positions, not tracked vortex centers. "
             f"Frozen-window candidate: {flag}; a checkpoint or candidate is not grid convergence. Native frames supply the prefix where available; continuation diagnostics are preferred. "
             "Impulsive unit lid; initially stationary fluid. Gaps are not joined. Momentum defects use a symmetric-log axis, linear below 1e-10; zero is retained.")
    _footer(fig, foot)
    return fig, {"kind": "history", "case_id": case_id, "admission": status, "series": records,
                 "diagnostic_display_stride": (case.get("diagnostics") or {}).get("stride", 1),
                 "probe_positions": {"upper_left": ul, "central": center}, "momentum_scale": {"type": "symlog", "linthresh": 1e-10},
                 "qualification": case.get("qualification", "No steady or spatial convergence claim")}, legends


def _profile_at(case, time):
    matches = [item for item in case.get("profiles", []) if np.isclose(item["time"], time, rtol=0, atol=1e-10)]
    if len(matches) != 1:
        raise ValueError(f"Exactly one actual-time profile at t*={time:g} is required; nearest-time substitution is forbidden")
    return matches[0]


def _profile_array(profile, direction, coordinate, key):
    data = profile[direction]
    x, y = np.asarray(data[coordinate], dtype=float), np.asarray(data[key], dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or np.any(np.diff(x) <= 0)
            or np.any(x < 0) or np.any(x > 1)):
        raise ValueError("Invalid native centerline coordinates or component coverage")
    return x, y


def profile_figure(comparison, candidate, reference, time, fixture=False):
    if candidate["config"]["reynolds"] != reference["config"]["reynolds"]:
        raise ValueError("A numerical sensitivity comparison must keep Reynolds number fixed")
    fixed = {"time": ("cells", "outer_correctors", "pressure_correctors"),
             "grid": ("dt", "outer_correctors", "pressure_correctors"),
             "coupling": ("dt", "cells")}.get(comparison["kind"])
    if fixed is None:
        raise ValueError("Unknown numerical control kind")
    for name in ("method", "convection", *fixed):
        if candidate["config"].get(name) != reference["config"].get(name):
            raise ValueError(f"A {comparison['kind']} comparison changes the supposedly fixed {name}")
    a, b = _profile_at(candidate, time), _profile_at(reference, time)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 6.4))
    fig.subplots_adjust(left=.058, right=.985, bottom=.25, top=.65, wspace=.34)
    reynolds = candidate["config"]["reynolds"]
    title = {"time": "Time-step", "grid": "Grid", "coupling": "Coupling"}.get(comparison["kind"], comparison["kind"])
    fig.text(.5, .927, f"{title} sensitivity: Re = {reynolds:g},  t* = {time:g}", ha="center", fontsize=16, weight="bold")
    config = candidate["config"]
    model = f"{str(config['method']).upper()}, {config.get('convection', 'unspecified')} convection; impulsive unit lid"
    fig.text(.5, .878, model + " | Native centerline velocities and zero-area-mean pressure", ha="center", fontsize=11)
    _synthetic_stamp(fig, fixture)
    handles = [Line2D([], [], color=COLORS["blue"], lw=1.9, ls="-", label="Candidate: " + _short_conditions(candidate)),
               Line2D([], [], color=COLORS["red"], lw=1.7, ls="--", label="Numerical comparator: " + _short_conditions(reference))]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .832), frameon=False,
                        ncol=2, handlelength=3.4, columnspacing=2, fontsize=10)
    pressure_label = r"$(p-\langle p\rangle)/(\rho U_{\rm lid}^2)$"
    specs = [("vertical", "y", "u", "Vertical centerline velocity", r"$y/L$", r"$u/U_{\rm lid}$"),
             ("horizontal", "x", "v", "Horizontal centerline velocity", r"$x/L$", r"$v/U_{\rm lid}$"),
             ("vertical", "y", "p", "Vertical centerline pressure", r"$y/L$", pressure_label),
             ("horizontal", "x", "p", "Horizontal centerline pressure", r"$x/L$", pressure_label)]
    record = {"kind": "profiles", "comparison_id": comparison["id"], "comparison_kind": comparison["kind"],
              "candidate_id": comparison["candidate_id"], "reference_id": comparison["reference_id"],
              "time": float(time), "meaning": "Matched numerical comparator, not truth", "profiles": {}}
    for ax, (direction, coordinate, key, label, xlabel, ylabel) in zip(axes, specs):
        ax.set_title(label, fontsize=11, pad=14)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1))
        ax.grid(True, color="#e3e8ed", lw=.6); ax.set_axisbelow(True)
        ax.axhline(0, color="#7c8791", lw=.65, zorder=0)
        record["profiles"][direction + "_" + key] = {}
        for profile, case, name, color, pattern in ((a, candidate, "candidate", COLORS["blue"], "-"), (b, reference, "comparator", COLORS["red"], "--")):
            x, y = _profile_array(profile, direction, coordinate, key)
            for xx, yy in _split(x, y, 1. / case["config"]["cells"]):
                ax.plot(xx, yy, color=color, ls=pattern if len(xx) > 2 else "None", lw=1.7,
                        marker="o" if len(xx) <= 2 else None, ms=3)
            finite = np.isfinite(y)
            record["profiles"][direction + "_" + key][name] = {"samples": len(x), "finite_samples": int(finite.sum()),
                "coordinate_min": float(x.min()) if len(x) else None, "coordinate_max": float(x.max()) if len(x) else None,
                "value_min": float(y[finite].min()) if finite.any() else None, "value_max": float(y[finite].max()) if finite.any() else None}
    _footer(fig, "Both profiles use the same actual physical time; no nearest-time replacement or smoothing. Each curve retains its native spatial samples. "
            "Pressure has each field's zero-area-mean gauge, with no extrapolated wall values. The comparator is a numerical control, not an exact solution; these profiles alone do not establish steady or grid-independent flow.")
    return fig, record, [legend]


def check_layout(fig, legends):
    """Focused style check: legends outside data axes, with visible text bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    axes_boxes = [ax.get_window_extent(renderer) for ax in fig.axes if ax.get_visible()]
    for legend in legends:
        box = legend.get_window_extent(renderer)
        if any(box.overlaps(data_box) for data_box in axes_boxes):
            raise ValueError("A legend overlaps a data axes rectangle")
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            raise ValueError("Legend is clipped outside the figure")
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.x1 > bounds.x1 + 1 or box.y0 < bounds.y0 - 1 or box.y1 > bounds.y1 + 1:
            raise ValueError("Figure header/footer text is clipped")
    return {"legends_outside_data_axes": True, "figure_text_within_bounds": True}


def _save(fig, output, stem, dpi, legends, allowed_existing):
    style_check = check_layout(fig, legends)
    files = []
    for extension in ("png", "pdf"):
        path = output / (stem + "." + extension)
        if path.exists() and path.name not in allowed_existing:
            raise FileExistsError(f"Refusing to overwrite an unrelated figure: {path}")
        temporary = output / (path.name + "." + uuid.uuid4().hex + ".part")
        try:
            fig.savefig(temporary, format=extension, dpi=dpi, facecolor="white")
            if extension == "png":
                from PIL import Image
                with Image.open(temporary) as img:
                    img.verify()
            else:
                from pypdf import PdfReader
                reader = PdfReader(temporary)
                if len(reader.pages) != 1:
                    raise ValueError("A scientific figure must occupy exactly one PDF page")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        files.append({"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    plt.close(fig)
    return files, style_check


def render(document, output, *, case_ids=None, comparison_ids=None, histories="continuation",
           profile_times=None, dpi=180, input_path=None):
    if document.get("schema") != INPUT_SCHEMA:
        raise ValueError("Expected the admitted cavity settling analysis schema")
    if dpi < 80:
        raise ValueError("Use at least 80 dpi for readable inspection output")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "plot_receipt.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else None
    if previous and previous.get("schema") != OUTPUT_SCHEMA:
        raise ValueError("Existing output directory belongs to another renderer")
    allowed = {item["name"] for fig in (previous or {}).get("figures", []) for item in fig["files"]}
    figures, skipped = [], []
    fixture = document.get("test_fixture") is True
    cases = document.get("cases") or {}
    admissible = {key: value for key, value in cases.items() if value.get("admission") in ("verified_checkpoint", "seed_only")}
    for key, value in cases.items():
        if key not in admissible:
            skipped.append({"case_id": key, "reason": value.get("admission", "unavailable")})
    if case_ids:
        unknown = set(case_ids) - set(cases)
        if unknown:
            raise ValueError("Unknown requested cases: " + ", ".join(sorted(unknown)))
    with plt.rc_context(STYLE):
        if histories != "none":
            for key, case in admissible.items():
                if case_ids and key not in case_ids:
                    continue
                if not case_ids and histories == "continuation" and case.get("case", {}).get("purpose") != "continue":
                    continue
                fig, record, legends = history_figure(key, case, fixture)
                record["files"], record["style_check"] = _save(fig, output, "history_" + _name(key), dpi, legends, allowed)
                figures.append(record)
        comparisons = document.get("comparisons") or []
        if comparison_ids:
            unknown = set(comparison_ids) - {item["id"] for item in comparisons}
            if unknown:
                raise ValueError("Unknown requested comparisons: " + ", ".join(sorted(unknown)))
        for pair in comparisons:
            if comparison_ids and pair["id"] not in comparison_ids:
                continue
            if pair.get("status") != "matched":
                skipped.append({"comparison_id": pair["id"], "reason": pair.get("status", "unavailable")}); continue
            if pair["candidate_id"] not in admissible or pair["reference_id"] not in admissible:
                skipped.append({"comparison_id": pair["id"], "reason": "case not admitted"}); continue
            available = sorted(set(float(t) for t in pair.get("profile_times", [])))
            selected = [available[-1]] if available and profile_times is None else []
            if profile_times is not None:
                for requested in profile_times:
                    matches = [time for time in available if np.isclose(time, requested, rtol=0, atol=1e-10)]
                    if matches:
                        selected.append(matches[0])
                    else:
                        skipped.append({"comparison_id": pair["id"], "time": requested, "reason": "no exact matched native profiles"})
            if not selected:
                skipped.append({"comparison_id": pair["id"], "reason": "no selected matched profile time"}); continue
            for time in sorted(set(selected)):
                fig, record, legends = profile_figure(pair, admissible[pair["candidate_id"]], admissible[pair["reference_id"]], time, fixture)
                stem = "profiles_" + _name(pair["id"]) + "_t" + _name(f"{time:g}")
                record["files"], record["style_check"] = _save(fig, output, stem, dpi, legends, allowed)
                figures.append(record)
    document_hash = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    receipt = {"schema": OUTPUT_SCHEMA, "test_fixture": fixture,
               "input": {"path": str(input_path) if input_path else "labelled in-memory synthetic rendering fixture",
                         "sha256": sha256(input_path) if input_path else None, "canonical_json_sha256": document_hash},
               "renderer_sha256": sha256(__file__), "figures": figures, "skipped": skipped,
               "comparison_meaning": "Numerical comparator, not truth; same actual time and native spatial coverage",
               "data_policy": "No new CFD solve, smoothing, pressure re-gauging, nearest-time substitution or fabricated missing curve",
               "visual_review": "Automated bounds checks and file read-back passed; inspect rendered PNG/PDF pages before scientific publication"}
    _write_json(receipt_path, receipt)
    return receipt


def synthetic_fixture():
    """Small deliberately gapped curves, signed quantities and missing-case test."""
    config = {"method": "pimple", "convection": "central", "reynolds": 1000, "cells": 8, "dt": .01,
              "outer_correctors": 3, "pressure_correctors": 2}
    names = ["time", "upper_left_u", "upper_left_v", "central_u", "central_v", "upper_left_omega", "central_omega",
             "kinetic_energy", "psi_min", "psi_max", "steady_momentum_defect_linf", "time_step_momentum_defect_linf"]
    def rows(times):
        answer = []
        for t in times:
            phase = t - 10
            answer.append([float(t), .03 * np.sin(phase * 4), -.01 + .01 * np.cos(phase * 4),
                           -.05 + .01 * np.sin(phase), .02 * np.cos(phase), 2 * np.sin(phase * 3), -.6 + .1 * np.cos(phase),
                           .03 + .001 * np.sin(phase), -.09 + .003 * np.cos(phase), .001 + .0002 * np.sin(phase),
                           1e-3 * np.exp(-phase), 2e-6 * np.exp(-phase)])
        return answer
    times = np.arange(10.05, 11.001, .05)
    times = times[(times < 10.4) | (times > 10.65)]
    coordinates = (np.arange(8) + .5) / 8
    def profiles(offset):
        y, x = coordinates, coordinates
        return [{"time": 10., "vertical": {"y": y.tolist(), "u": (np.sin(np.pi * y) * (.15 + offset)).tolist(),
                    "v": np.zeros(8).tolist(), "p": ((y - .5) * (.2 + offset)).tolist(), "omega": np.sin(y).tolist()},
                "horizontal": {"x": x.tolist(), "v": (np.sin(2 * np.pi * x) * (.07 + offset)).tolist(),
                    "u": np.zeros(8).tolist(), "p": ((x - .5) * (-.1 + offset)).tolist(), "omega": np.cos(x).tolist()}}]
    common = {"admission": "verified_checkpoint", "qualification": "SYNTHETIC RENDERING FIXTURE ONLY",
              "config": config, "case": {"purpose": "continue", "reynolds": 1000},
              "stationarity": {"candidate": False, "policy": {"diagnostic_interval": .05, "snapshot_interval": .5,
                  "thresholds": {"steady_momentum_defect_linf": 1e-7, "time_step_momentum_defect_linf": 1e-7}}},
              "frame_history": {"columns": names, "rows": rows([9., 9.5, 10.])},
              "diagnostics": {"columns": names, "rows": rows(times)}, "profiles": profiles(0)}
    comparator = json.loads(json.dumps(common))
    comparator["case"]["purpose"] = "coupling_control"
    comparator["config"].update(outer_correctors=6, pressure_correctors=4)
    comparator["profiles"] = profiles(.01)
    return {"schema": INPUT_SCHEMA, "test_fixture": True,
            "cases": {"fixture_baseline": common, "fixture_comparator": comparator, "fixture_unavailable": {"admission": "missing"}},
            "comparisons": [{"id": "fixture_coupling", "kind": "coupling", "candidate_id": "fixture_baseline",
                "reference_id": "fixture_comparator", "status": "matched", "profile_times": [10.], "frames": [],
                "meaning": "Synthetic numerical comparator example, not truth"}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--analysis", type=Path)
    choice.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--histories", choices=["continuation", "all", "none"], default="continuation")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--comparison-ids", nargs="+")
    parser.add_argument("--profile-times", nargs="+", type=float)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    document = synthetic_fixture() if args.synthetic_smoke else json.loads(args.analysis.read_text(encoding="utf-8"))
    receipt = render(document, args.output, case_ids=args.case_ids, comparison_ids=args.comparison_ids,
                     histories=args.histories, profile_times=args.profile_times, dpi=args.dpi, input_path=args.analysis)
    if args.synthetic_smoke:
        history = next(item for item in receipt["figures"] if item["kind"] == "history")
        assert len(history["series"]["upper_left_u"]) == 3, "Missing interval must remain a visible curve break"
        assert any(item.get("case_id") == "fixture_unavailable" for item in receipt["skipped"])
        assert len(receipt["figures"]) == 2
        strided = json.loads(json.dumps(document["cases"]["fixture_baseline"]))
        strided["frame_history"]["rows"] = []
        strided["diagnostics"]["rows"] = strided["diagnostics"]["rows"][:6:2]
        strided["diagnostics"].update(stride=2, continuation_interval=.05)
        assert len(_series(strided, "upper_left_u")) == 1, "Declared display thinning is not a missing interval"
        profile = next(item for item in receipt["figures"] if item["kind"] == "profiles")
        assert profile["profiles"]["vertical_p"]["candidate"]["coordinate_min"] > 0
        assert profile["profiles"]["vertical_p"]["candidate"]["coordinate_max"] < 1
        try:
            _profile_at(document["cases"]["fixture_baseline"], 10.001)
        except ValueError:
            pass
        else:
            raise AssertionError("Nearest-time substitution must be refused")
    print(json.dumps({"figures": len(receipt["figures"]), "skipped": receipt["skipped"],
                      "test_fixture": receipt["test_fixture"], "receipt": str(args.output / "plot_receipt.json")}, indent=2))


if __name__ == "__main__":
    main()
