"""Post-freeze upper-left-corner audit of the retained Week 4 field models.

No training or model selection occurs here. The diagnostic window was chosen
to inspect the requested flow structure, after the original tests were scored.
It is therefore a post-hoc physical diagnostic, not an additional blind test.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.ndimage import label

from . import cavity_field_operator as operator

WINDOW = (0.0, 0.25, 0.65, 1.0)
PROFILE_X = 0.06
PROFILE_Y = 0.90
METHODS = ("CFD", "POD-DeepONet", "Linear in log Re")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_npz(path, expected, checked):
    path = Path(path)
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"Frozen evidence hash mismatch: {path.name}")
    checked[str(path)] = digest
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_frozen_cases(root, reynolds=(3200, 7500)):
    """Reconstruct saved predictions before opening their matched test targets."""
    root = Path(root)
    folder = root / "results/cavity_field_deeponet"
    score = json.loads((folder / "scoring_receipt.json").read_text())
    selection_path = folder / "frozen_selection.json"
    if sha256(selection_path) != score["frozen_selection_sha256"]:
        raise ValueError("Frozen field selection changed")
    selection = json.loads(selection_path.read_text())
    checked = {str(selection_path): sha256(selection_path)}
    trunk = checked_npz(folder / "trunk.npz", selection["trunk_sha256"], checked)
    saved = checked_npz(folder / "frozen_predictions.npz",
                        selection["frozen_predictions_sha256"], checked)
    training = []
    for re in operator.TRAIN:
        training.append(checked_npz(root / f"results/cavity_highre_ghia/n512/re{re}.npz",
                                    selection["data_read_during_training"][str(re)], checked))
    cases = {}
    for re in reynolds:
        if re not in selection["test_re"]:
            raise ValueError("This diagnostic accepts the recorded held-out cases only")
        index = list(saved["test_re"]).index(re)
        coefficients = saved["candidate_coefficients"][selection["selected_indices"], index].mean(axis=0)
        baseline = operator.baseline_coefficients(trunk, [re], coordinate="log")[0]
        predicted = {}
        for name, coef in (("POD-DeepONet", coefficients), ("Linear in log Re", baseline)):
            flat = operator.reconstruct(trunk, coef)
            shape = training[0]["u"].shape
            n = training[0]["u"].size
            weights = operator.mixing_weights(trunk, coef)
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
            predicted[name] = {
                "x": training[0]["x"], "y": training[0]["y"],
                "u": flat[:n].reshape(shape), "v": flat[n:].reshape(shape),
                # The same affine combination gives the streamfunction associated
                # with these velocity modes; this is not a separately fitted field.
                "psi": sum(w * f["psi"] for w, f in zip(weights, training)),
            }
        target_path = root / f"results/cavity_highre_ghia/n512/re{re}.npz"
        truth = checked_npz(target_path, score["heldout_field_sha256"][str(re)], checked)
        cases[re] = {"CFD": truth, **predicted}
    return cases, {"checked_sha256": {Path(k).relative_to(root).as_posix(): v for k, v in checked.items()},
                   "training": False, "selection_changed": False,
                   "diagnostic_scope": "post-hoc corner diagnostic of already frozen, scored whole-Re tests",
                   "window_x0_x1_y0_y1": list(WINDOW), "profile_x": PROFILE_X, "profile_y": PROFILE_Y}


def crop(field, bounds=WINDOW):
    x0, x1, y0, y1 = bounds
    ix = (field["x"] >= x0) & (field["x"] <= x1)
    iy = (field["y"] >= y0) & (field["y"] <= y1)
    return {"x": field["x"][ix], "y": field["y"][iy],
            **{key: field[key][np.ix_(iy, ix)] for key in ("u", "v", "psi")}}


def vortex_peak(field, bounds=WINDOW, threshold=1e-10):
    """Return a resolved positive streamfunction island, never a wall maximum.

    The cavity's main clockwise vortex has negative psi under u=psi_y,
    v=-psi_x. A positive interior local maximum in this upper-left window is a
    counter-rotating island. The reported centre is a native grid node, not a
    sub-grid fitted location; the boundary corner itself is excluded.
    """
    z = crop(field, bounds)
    p = z["psi"]
    if min(p.shape) < 3:
        raise ValueError("Need at least three native nodes along each window axis")
    candidate = p.copy()
    candidate[[0, -1], :] = -np.inf
    candidate[:, [0, -1]] = -np.inf
    j, i = np.unravel_index(np.argmax(candidate), candidate.shape)
    peak = float(p[j, i])
    is_max = np.all(peak >= p[j-1:j+2, i-1:i+2])
    if not is_max or peak <= threshold:
        return {"resolved_positive_island": False, "psi_peak_UL": None,
                "x_peak_L": None, "y_peak_L": None, "positive_area_L2": 0.0,
                "island_intersects_window_edge": False}
    regions, _ = label(p > threshold)
    region = regions == regions[j, i]
    # Native-node trapezoid area, with half weights on the plotted window edges.
    wx = np.gradient(z["x"]); wy = np.gradient(z["y"])
    wx[[0, -1]] *= 0.5; wy[[0, -1]] *= 0.5
    area = float(np.sum(region * wy[:, None] * wx[None, :]))
    touches = bool(region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any())
    return {"resolved_positive_island": True, "psi_peak_UL": peak,
            "x_peak_L": float(z["x"][i]), "y_peak_L": float(z["y"][j]),
            "positive_area_L2": area, "island_intersects_window_edge": touches}


def diagnostic_rows(cases):
    rows = []
    for re, methods in cases.items():
        truth = methods["CFD"]
        tc = crop(truth)
        reference_peak = vortex_peak(truth)
        for name in METHODS:
            field = methods[name]; local = crop(field)
            err2 = (field["u"]-truth["u"])**2 + (field["v"]-truth["v"])**2
            ec2 = (local["u"]-tc["u"])**2 + (local["v"]-tc["v"])**2
            ref2 = truth["u"]**2 + truth["v"]**2
            c2 = tc["u"]**2 + tc["v"]**2
            peak = vortex_peak(field)
            shift = None
            strength = None
            if peak["resolved_positive_island"] and reference_peak["resolved_positive_island"]:
                shift = float(np.hypot(peak["x_peak_L"]-reference_peak["x_peak_L"],
                                       peak["y_peak_L"]-reference_peak["y_peak_L"]))
                strength = (peak["psi_peak_UL"]-reference_peak["psi_peak_UL"])/reference_peak["psi_peak_UL"]
            rows.append({"Re": re, "role": "whole-Re held-out field", "method": name,
                         "field_relative_l2": float(np.sqrt(err2.sum()/ref2.sum())),
                         "corner_relative_l2": float(np.sqrt(ec2.sum()/c2.sum())),
                         "corner_vector_rms_U": float(np.sqrt(ec2.mean())),
                         "corner_vector_max_U": float(np.sqrt(ec2.max())),
                         **peak, "peak_shift_L": shift, "peak_strength_signed_relative_error": strength})
    return rows


def plot_corner(methods, reynolds):
    """Same native contour levels, full visible extrema, and linear error scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    truth = methods["CFD"]
    t = crop(truth)
    n = crop(methods["POD-DeepONet"])
    b = crop(methods["Linear in log Re"])
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4.25), layout="constrained")
    psi = t["psi"]
    limit = max(float(np.abs(psi).max()), 1e-8)
    threshold = max(1e-5, float(np.max(psi))/20)
    norm = SymLogNorm(threshold, vmin=float(psi.min()), vmax=float(psi.max()))
    levels = norm.inverse(np.linspace(0, 1, 41))
    image = axes[0].contourf(t["x"], t["y"], psi, levels=levels, norm=norm, cmap="jet", extend="neither")
    positive = max(float(psi.max()), threshold)
    contour_levels = np.unique(np.r_[-.02, -.01, -.003, -.001, 0., positive*np.array([.08, .2, .4, .65, .85])])
    for z, color, style, width in ((t, "black", "solid", .7), (n, "#f40000", "dashed", 1.05),
                                   (b, "#65268d", "dotted", .9)):
        axes[0].contour(z["x"], z["y"], z["psi"], levels=contour_levels,
                        colors=color, linestyles=style, linewidths=width)
    ticks = sorted(set([float(psi.min()), float(psi.max()), 0.0]
                       + [v for v in (-.01, -.001, -.0001, .0001, .001) if psi.min() < v < psi.max()]))
    number_format = FuncFormatter(lambda value, _: f"{value:.2g}" if value else "0")
    fig.colorbar(image, ax=axes[0], label=r"$\psi/(UL)$", shrink=.89,
                 ticks=ticks, format=number_format)
    errors = [np.hypot(z["u"]-t["u"], z["v"]-t["v"]) for z in (n, b)]
    maxerr = max(float(a.max()) for a in errors)
    maxerr = max(maxerr, 1e-15)
    err_levels = np.linspace(0, maxerr, 31)
    for ax, error in zip(axes[1:], errors):
        im = ax.contourf(t["x"], t["y"], error, levels=err_levels, cmap="jet", vmin=0, vmax=maxerr)
    fig.colorbar(im, ax=list(axes[1:]), label=r"$\|\hat{\mathbf{u}}-\mathbf{u}_{CFD}\|_2/U$", shrink=.89,
                 ticks=np.linspace(0, maxerr, 6), format=number_format)
    for ax, title in zip(axes, (f"Re = {reynolds}: upper-left flow", "POD-DeepONet error", "Linear in log Re error")):
        ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal", title=title)
        ax.tick_params(labelsize=9)
    handles = [Line2D([0], [0], color=c, ls=s, lw=1.3) for c, s in (("black", "-"), ("#f40000", "--"), ("#65268d", ":"))]
    axes[0].legend(handles, METHODS, loc="upper center", bbox_to_anchor=(.5, -.21), frameon=False, fontsize=9)
    return fig


def plot_corner_profiles(methods, reynolds):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex="col", layout="constrained")
    truth = methods["CFD"]
    cuts = {}
    for name, field in methods.items():
        cuts[name] = (np.array([np.interp(PROFILE_X, field["x"], row) for row in field["u"]]),
                      np.array([np.interp(PROFILE_Y, field["y"], col) for col in field["v"].T]))
    styles = {"CFD": ("black", "-"), "POD-DeepONet": ("#cf1b1b", "--"), "Linear in log Re": ("#4b27a7", ":")}
    for name in METHODS:
        color, ls = styles[name]
        for j, coordinate in enumerate((truth["y"], truth["x"])):
            mask = (coordinate >= (.65 if j == 0 else 0)) & (coordinate <= (1 if j == 0 else .25))
            axes[0, j].plot(coordinate[mask], cuts[name][j][mask], color=color, ls=ls, lw=1.6, label=name)
            if name != "CFD":
                axes[1, j].plot(coordinate[mask], (cuts[name][j]-cuts["CFD"][j])[mask], color=color, ls=ls, lw=1.6)
    axes[0, 0].set(ylabel=r"$u/U$", title=f"Re = {reynolds}; x/L = {PROFILE_X:g}")
    axes[0, 1].set(ylabel=r"$v/U$", title=f"Re = {reynolds}; y/L = {PROFILE_Y:g}")
    axes[1, 0].set(xlabel=r"$y/L$", ylabel=r"$(\hat u-u_{CFD})/U$")
    axes[1, 1].set(xlabel=r"$x/L$", ylabel=r"$(\hat v-v_{CFD})/U$")
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.axhline(0, color=".6", lw=.7, zorder=0)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.14), ncol=3, frameon=False, fontsize=9)
    return fig
