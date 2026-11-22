"""Physical plots for Week 4 teaching; all values come from supplied arrays.

The helpers never fit models, alter boundaries, smooth fields, or trim extrema.
Native arrays are indexed [y, x].  Pressure is plotted only when supplied.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def checked_fields(x, y, **fields):
    """Reject malformed spatial pairings instead of silently transposing them."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 3:
        raise ValueError("One-dimensional x/y arrays with at least three nodes are required")
    if not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError("Coordinates must strictly increase")
    for name, values in fields.items():
        values = np.asarray(values, float)
        if values.shape != (len(y), len(x)) or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite field[y, x]")
    return x, y


def _limits(*arrays, signed=False):
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    if signed:
        hi = max(abs(lo), abs(hi), 1e-15)
        return -hi, hi
    if lo == hi:
        hi = lo + max(abs(lo) * 1e-6, 1e-15)
    return lo, hi


def _axis(ax, x, y):
    ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", xlim=(x[0], x[-1]),
           ylim=(y[0], y[-1]), aspect="equal")
    ax.tick_params(labelsize=10)


def _stream(ax, x, y, u, v, color="black", style="solid", density=1.25):
    # Matplotlib streamplot is only defined on an equally spaced grid.
    if not (np.allclose(np.diff(x), np.diff(x)[0]) and
            np.allclose(np.diff(y), np.diff(y)[0])):
        raise ValueError("Streamlines require an equally spaced grid; do not interpolate silently")
    lines = ax.streamplot(x, y, u, v, color=color, density=density,
                          linewidth=.75, arrowsize=.7, zorder=3)
    lines.lines.set_linestyle(style)
    return lines


def velocity_comparison(x, y, u, v, predictions, reynolds, context=""):
    """One full-range physical comparison per method, with equal physical levels.

    Returns figures, so a notebook can display/save every solver or model rather
    than silently retaining only the last Matplotlib axes.
    """
    x, y = checked_fields(x, y, u=u, v=v)
    figures = []
    reference_speed = np.hypot(u, v)
    for method, (up, vp) in predictions.items():
        checked_fields(x, y, u_pred=up, v_pred=vp)
        prediction_speed = np.hypot(up, vp)
        error = np.hypot(up-u, vp-v)
        _, vmax = _limits(reference_speed, prediction_speed)
        levels = np.linspace(0, vmax, 41)
        comparison_levels = np.linspace(0, vmax, 9)[1:-1]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), layout="constrained")
        im = axes[0].contourf(x, y, reference_speed, levels=levels, cmap="turbo",
                              norm=Normalize(0, vmax))
        axes[0].contour(x, y, reference_speed, levels=comparison_levels,
                        colors="black", linewidths=.7)
        axes[0].contour(x, y, prediction_speed, levels=comparison_levels,
                        colors="magenta", linestyles="--", linewidths=.95)
        axes[0].set_title("Matched speed contours", fontsize=11)
        fig.colorbar(im, ax=axes[0], label=r"$|\mathbf{u}|/U_{lid}$", shrink=.88)
        axes[1].contourf(x, y, reference_speed, levels=levels, cmap="turbo",
                         norm=Normalize(0, vmax), alpha=.35)
        _stream(axes[1], x, y, u, v)
        _stream(axes[1], x, y, up, vp, color="magenta", style="dashed")
        axes[1].set_title("Velocity streamlines on the same axes", fontsize=11)
        max_error = max(float(error.max()), 1e-15)
        er = axes[2].contourf(x, y, error, levels=np.linspace(0, max_error, 41),
                              cmap="turbo", norm=Normalize(0, max_error))
        fig.colorbar(er, ax=axes[2], label=r"$|\widehat{\mathbf{u}}-\mathbf{u}|/U_{lid}$",
                     shrink=.88)
        axes[2].set_title("Vector discrepancy (linear scale)", fontsize=11)
        for ax in axes:
            _axis(ax, x, y)
        handles = [Line2D([], [], color="black", label="CFD"),
                   Line2D([], [], color="magenta", ls="--", label=method)]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
        fig.suptitle(f"Re = {reynolds:g}; {len(x)} × {len(y)} nodes" +
                     (f"; {context}" if context else ""), fontsize=12)
        figures.append(fig)
    return figures


def velocity_profiles(x, y, u, v, predictions, reynolds, context=""):
    """Native centreline profiles plus signed differences, no clipped limits."""
    x, y = checked_fields(x, y, u=u, v=v)
    uc = np.array([np.interp(.5, x, row) for row in u])
    vc = np.array([np.interp(.5, y, column) for column in np.asarray(v).T])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    ax[0, 0].plot(uc, y, color="black", lw=1.8, label="CFD")
    ax[0, 1].plot(x, vc, color="black", lw=1.8, label="CFD")
    colors = ["#d62728", "#0077bb", "#009988", "#990099"]
    patterns = ["--", "-.", ":", (0, (5, 2, 1, 2))]
    for i, (name, (up, vp)) in enumerate(predictions.items()):
        checked_fields(x, y, u_pred=up, v_pred=vp)
        pu = np.array([np.interp(.5, x, row) for row in up])
        pv = np.array([np.interp(.5, y, column) for column in np.asarray(vp).T])
        style = dict(color=colors[i % len(colors)], ls=patterns[i % len(patterns)], lw=1.4)
        ax[0, 0].plot(pu, y, label=name, **style)
        ax[0, 1].plot(x, pv, **style)
        ax[1, 0].plot(pu-uc, y, **style)
        ax[1, 1].plot(x, pv-vc, **style)
    ax[0, 0].set(xlabel=r"$u/U_{lid}$ at $x/L=0.5$", ylabel=r"$y/L$")
    ax[0, 1].set(xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$ at $y/L=0.5$")
    ax[1, 0].set(xlabel=r"$(\hat{u}-u)/U_{lid}$", ylabel=r"$y/L$")
    ax[1, 1].set(xlabel=r"$x/L$", ylabel=r"$(\hat{v}-v)/U_{lid}$")
    ax[1, 0].axvline(0, color=".6", lw=.6)
    ax[1, 1].axhline(0, color=".6", lw=.6)
    ax[1, 0].xaxis.set_major_locator(MaxNLocator(5))
    ax[1, 0].ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=True)
    ax[1, 1].yaxis.set_major_locator(MaxNLocator(5))
    ax[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    for a in ax.flat:
        a.grid(alpha=.18)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncol=min(4, 1+len(predictions)), frameon=False)
    fig.suptitle(f"Re = {reynolds:g}; physical profiles and signed discrepancies" +
                 (f"; {context}" if context else ""), fontsize=12)
    return fig


def pressure_comparison(x, y, pressure, prediction, reynolds, method="POD-DeepONet"):
    """Compare supplied zero-mean pressure labels with a matched-gauge prediction.

    This measures agreement with recovered CFD pressure, not independent
    validation by the Ghia velocity table.
    """
    x, y = checked_fields(x, y, pressure=pressure, prediction=prediction)
    p = np.asarray(pressure)-np.mean(pressure)
    pp = np.asarray(prediction)-np.mean(prediction)
    lo, hi = _limits(p, pp, signed=True)
    pnorm = SymLogNorm(linthresh=.01, vmin=lo, vmax=hi, base=10)
    levels = pnorm.inverse(np.linspace(0, 1, 49))
    lines = pnorm.inverse(np.linspace(0, 1, 15))[1:-1]
    error = pp-p
    elo, ehi = _limits(error, signed=True)
    fig, ax = plt.subplots(1, 3, figsize=(12.8, 4), layout="constrained")
    im = ax[0].contourf(x, y, p, levels=levels, cmap="turbo", norm=pnorm)
    ax[0].contour(x, y, p, levels=lines, colors="black", linewidths=.7)
    ax[0].contour(x, y, pp, levels=lines, colors="magenta", linestyles="--", linewidths=.95)
    fig.colorbar(im, ax=ax[0], label=r"$(p-\langle p\rangle)/(\rho U_{lid}^{2})$", shrink=.88)
    er = ax[1].contourf(x, y, error, levels=np.linspace(elo, ehi, 41), cmap="turbo")
    fig.colorbar(er, ax=ax[1], label=r"$(\hat p-p)/(\rho U_{lid}^{2})$", shrink=.88)
    ax[0].set_title("Pressure: symlog, linear for |p*| ≤ 0.01", fontsize=10)
    ax[1].set_title("Signed pressure discrepancy", fontsize=11)
    for a in ax[:2]:
        _axis(a, x, y)
    pc = np.array([np.interp(.5, y, column) for column in p.T])
    ppc = np.array([np.interp(.5, y, column) for column in pp.T])
    ax[2].plot(x, pc, color="black", label="CFD")
    ax[2].plot(x, ppc, color="magenta", ls="--", label=method)
    ax[2].set(xlabel=r"$x/L$ at $y/L=0.5$", ylabel=r"$(p-\langle p\rangle)/(\rho U_{lid}^{2})$")
    ax[2].grid(alpha=.18)
    fig.legend(*ax[2].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               frameon=False)
    fig.suptitle(f"Re = {reynolds:g}; retained zero-mean CFD pressure labels", fontsize=12)
    return fig


def cfd_diagnostics(result):
    """Fields, convergence, Ghia points and signed discrepancies for a live solve."""
    from .ghia import GHIA, valid_mask
    x, y = checked_fields(result["x"], result["y"], u=result["u"], v=result["v"],
                           psi=result["psi"], omega=result["omega"])
    u, v = result["u"]/result["U"], result["v"]/result["U"]
    re = int(round(result["Re"]))
    speed, omega = np.hypot(u, v), result["omega"]/result["U"]
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1), layout="constrained")
    im = ax[0].contourf(x, y, speed, levels=np.linspace(0, max(speed.max(), 1e-15), 41), cmap="turbo")
    _stream(ax[0], x, y, u, v)
    fig.colorbar(im, ax=ax[0], label=r"$|\mathbf{u}|/U_{lid}$", shrink=.88)
    ps = result["psi"]/result["U"]
    plo, phi = _limits(ps)
    ax[1].contour(x, y, ps, levels=np.linspace(plo, min(0, phi), 18)[1:-1],
                  colors="black", linewidths=.8, linestyles="solid")
    if phi > 0:
        ax[1].contour(x, y, ps, levels=np.geomspace(max(phi*1e-4, 1e-15), phi, 10),
                      colors="#0055aa", linewidths=.8)
    wlo, whi = _limits(omega, signed=True)
    norm = SymLogNorm(linthresh=1, vmin=wlo, vmax=whi, base=10)
    levels = norm.inverse(np.linspace(0, 1, 49))
    om = ax[2].contourf(x, y, omega, levels=levels, norm=norm, cmap="turbo")
    fig.colorbar(om, ax=ax[2], label=r"$\omega L/U_{lid}$ (symlog; linear below 1)", shrink=.88)
    for a, title in zip(ax, ["Speed and streamlines", "Signed streamfunction contours", "Native-node vorticity"]):
        _axis(a, x, y); a.set_title(title, fontsize=11)
    fig.suptitle(f"Re = {re}; {len(x)} × {len(y)} nodes; step {result['steps']}; "
                 f"t* = {result['steps']*result['dt']:g}", fontsize=12)

    diag, axes = plt.subplots(2, 3, figsize=(13, 7.7), layout="constrained")
    uc = np.array([np.interp(.5, x, row) for row in u])
    vc = np.array([np.interp(.5, y, column) for column in v.T])
    axes[0, 0].plot(uc, y, color="black", label="CFD")
    axes[0, 1].plot(x, vc, color="black", label="CFD")
    if re in GHIA:
        for k, component, coord, values in [(0, "u", y, uc), (1, "v", x, vc)]:
            ref = GHIA[re]; pos = np.asarray(ref["y" if component == "u" else "x"])
            truth = np.asarray(ref[component]); keep = valid_mask(re, component)
            if k == 0:
                axes[0, k].plot(truth[keep], pos[keep], "o", mfc="white", color="#d62728", label="Ghia")
            else:
                axes[0, k].plot(pos[keep], truth[keep], "o", mfc="white", color="#d62728", label="Ghia")
            difference = np.interp(pos, coord, values)-truth
            axes[1, k].plot(pos[keep], difference[keep], "o", color="#0077bb", label="CFD minus Ghia")
            if (~keep).any():
                if k == 0:
                    axes[0, k].plot(truth[~keep], pos[~keep], "x", color="#990099", label="Flagged printed sample")
                else:
                    axes[0, k].plot(pos[~keep], truth[~keep], "x", color="#990099", label="Flagged printed sample")
                axes[1, k].plot(pos[~keep], difference[~keep], "x", color="#990099", label="Flagged printed sample")
    else:
        for a in axes[1, :2]:
            a.text(.5, .5, f"No Ghia samples at Re={re}", ha="center", transform=a.transAxes)
    axes[0, 0].set(xlabel=r"$u/U_{lid}$ at $x/L=0.5$", ylabel=r"$y/L$")
    axes[0, 1].set(xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$ at $y/L=0.5$")
    axes[1, 0].set(xlabel=r"$y/L$", ylabel=r"$(u_{CFD}-u_{Ghia})/U_{lid}$")
    axes[1, 1].set(xlabel=r"$x/L$", ylabel=r"$(v_{CFD}-v_{Ghia})/U_{lid}$")
    for a in axes[1, :2]: a.axhline(0, color=".5", lw=.7)
    residual = np.asarray(result["residual_values"])
    steps = np.asarray(result["residual_steps"])
    axes[0, 2].semilogy(steps, residual, "o-", ms=3, color="black")
    axes[0, 2].set(xlabel="Iteration step", ylabel="Relative vorticity change")
    divergence = np.gradient(u, x, axis=1)+np.gradient(v, y, axis=0)
    axes[1, 2].plot(x[1:-1], divergence[len(y)//2, 1:-1], color="black")
    axes[1, 2].set(xlabel=r"$x/L$ at $y/L=0.5$", ylabel=r"$(\nabla\!\cdot\!\mathbf{u}) L/U_{lid}$")
    for a in axes.flat: a.grid(alpha=.18)
    handles, labels = [], []
    for a in axes.flat:
        for h, label in zip(*a.get_legend_handles_labels()):
            if label not in labels: handles.append(h); labels.append(label)
    diag.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    diag.suptitle(f"Re = {re}; solver and independent centreline checks", fontsize=12)
    return fig, diag


def profile_trial(model, reynolds=400):
    """A new Lab 6 fit is assessed at its validation case, never a hidden refit."""
    from .ghia import GHIA, valid_mask
    from .ghia_surrogate import predict, linear_baseline
    ref = GHIA[int(reynolds)]
    grid = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for j, component in enumerate(("u", "v")):
        pos = np.asarray(ref["y" if component == "u" else "x"])
        keep = valid_mask(int(reynolds), component)
        values = predict(model, reynolds, grid, component)
        baseline = linear_baseline(reynolds, grid, component)
        axes[0, j].plot(grid, values, color="#d62728", label="Current student model")
        axes[0, j].plot(grid, baseline, color="#0077bb", ls="--", label="Training-case interpolation")
        axes[0, j].plot(pos[keep], ref[component][keep], "ko", mfc="white", label="Ghia")
        for label, prediction, color, marker in [
            ("Current student model", predict(model, reynolds, pos, component), "#d62728", "o"),
            ("Training-case interpolation", linear_baseline(reynolds, pos, component), "#0077bb", "s")]:
            axes[1, j].plot(pos[keep], (prediction-ref[component])[keep], marker, color=color, label=label)
        if (~keep).any():
            axes[0, j].plot(pos[~keep], ref[component][~keep], "x", color="#990099", label="Flagged printed sample")
        axes[0, j].set(xlabel="y/L" if component == "u" else "x/L", ylabel=component+"/U_lid")
        axes[1, j].set(xlabel="y/L" if component == "u" else "x/L", ylabel="Signed discrepancy / U_lid")
        axes[1, j].axhline(0, color=".5", lw=.7)
    for a in axes.flat: a.grid(alpha=.18)
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Validation Re = {reynolds:g}; centreline regression only", fontsize=12)
    return fig


def reference_speed_labels(fig):
    """Use U_ref for variable lids, whose velocity scale is the prescribed max."""
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        text.set_text(text.get_text().replace("U_{lid}", "U_{ref}"))
    return fig


def historical_pilot(root, train_family="constant", test_family="constant", case_index=0):
    """Load exactly matched retained pilot predictions; never fit or relabel them."""
    import json
    from pathlib import Path
    from .cavity_diversity import fields
    root = Path(root)
    folder = root / "results/cavity_diversity_pilot"
    cases = json.loads((folder/"manifest.json").read_text())["cases"]
    if train_family not in ("constant", "diverse"):
        raise ValueError("Training family must be constant or diverse")
    indices = [i for i, c in enumerate(cases) if c["family"] == test_family and c["split"] == "test"]
    if not isinstance(case_index, int) or not 0 <= case_index < len(indices):
        raise ValueError("Choose a retained case index in the displayed family")
    index = indices[case_index]
    case = cases[index]
    with np.load(folder/"dataset.npz", allow_pickle=False) as z:
        reference = {k: z[k][index].copy() for k in ("psi", "u", "v", "lid")}
    n = reference["psi"].shape[-1]
    reference["x"] = reference["y"] = np.linspace(0, 1, n)
    reconstructed = fields(reference["psi"][None], reference["lid"][None])
    for q, actual in zip(("u", "v"), reconstructed):
        np.testing.assert_allclose(actual[0], reference[q], rtol=1e-12, atol=1e-13)
    methods = [("nearest", 0, "Nearest case"), ("rbf", 0, "RBF"),
               ("coordinate_mlp", 7, "Coordinate MLP"), ("pod_deeponet", 7, "POD-DeepONet"),
               ("unet", 7, "U-Net"), ("fno", 7, "FNO"), ("fno_physics", 7, "Physics-weighted FNO")]
    predictions = {}
    with np.load(folder/"predictions.npz", allow_pickle=False) as z:
        for key, seed, name in methods:
            arrays = z[f"{train_family}_{key}_{seed}_{test_family}"]
            if len(arrays) != len(indices):
                raise ValueError("Prediction count differs from ordered manifest test cases")
            psi = arrays[case_index]
            up, vp = fields(psi[None], reference["lid"][None])
            predictions[name] = dict(psi=psi.copy(), u=up[0], v=vp[0])
    return case, reference, predictions


def stokes_refinement(root, family="constant", case_index=0):
    """Pair original POD and two correction ensembles at the same retained case."""
    from pathlib import Path
    import json
    from .cavity_diversity import fields
    root = Path(root)
    case, reference, pilot = historical_pilot(root, family, family, case_index)
    folder = root / "results/stokes_refined"
    expanded_cases = json.loads((folder/"expanded/expanded_cases.json").read_text())
    expanded_tests = [c for c in expanded_cases if c["family"] == family and c["split"] == "test"]
    # The retained refinement reused the original test cases; verify that claim.
    matched = expanded_tests[case_index]
    for key in ("id", "Re", "coefficients", "U_ref", "L", "nu"):
        if matched[key] != case[key]:
            raise ValueError(f"Refinement test case changed: {key}")
    predictions = {"Original POD (seed 7)": pilot["POD-DeepONet"]}
    for subfolder, name in [("same_data", "Correction / same data"),
                            ("expanded", "Correction / 64 train")]:
        with np.load(folder/subfolder/"predictions.npz", allow_pickle=False) as z:
            psi = z[f"{family}_{family}_ensemble"][case_index].copy()
        up, vp = fields(psi[None], reference["lid"][None])
        predictions[name] = dict(psi=psi, u=up[0], v=vp[0])
    with np.load(folder/"velocity_pressure_fields.npz", allow_pickle=False) as z:
        for name in ("u", "v"):
            np.testing.assert_allclose(reference[name], z[f"reference_{family}_{name}"][case_index],
                                       rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(predictions["Correction / 64 train"][name],
                                       z[f"{family}_{family}_{name}"][case_index], rtol=1e-12, atol=1e-13)
        reference["p"] = z[f"reference_{family}_p"][case_index].copy()
        predictions["Correction / 64 train"]["p"] = z[f"{family}_{family}_p"][case_index].copy()
    return case, reference, predictions


def streamfunction_vorticity_comparison(reference, predictions, reynolds, context=""):
    """Native psi contours and signed interior curl profiles on a matched mesh."""
    x, y = reference["x"], reference["y"]
    psi = reference["psi"]
    checked_fields(x, y, psi=psi)
    h = x[1]-x[0]
    def omega(p):
        return -(p[1:-1, 2:]+p[1:-1, :-2]+p[2:, 1:-1]+p[:-2, 1:-1]-4*p[1:-1, 1:-1])/h**2
    wr = omega(psi)
    plo, phi = _limits(psi)
    bound = max(abs(plo), abs(phi), 1e-15)
    norm = SymLogNorm(linthresh=1e-5, vmin=-bound, vmax=bound)
    levels = norm.inverse(np.linspace(0, 1, 49))
    lines = np.r_[-np.geomspace(max(bound*1e-5, 1e-15), bound, 17)[::-1],
                   np.geomspace(max(bound*1e-6, 1e-15), bound*.05, 12)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), layout="constrained")
    colors = ["#0077bb", "#cc3311", "#ee00cc"]
    styles = ["--", "-.", ":"]
    for ax in axes[0]:
        im = ax.contourf(x, y, psi, levels=levels, cmap="turbo", norm=norm)
        ax.contour(x, y, psi, levels=lines, colors="black", linewidths=.9, linestyles="solid")
        for (name, values), color, style in zip(predictions.items(), colors, styles):
            ax.contour(x, y, values["psi"], levels=lines, colors=color, linewidths=.8, linestyles=style)
        _axis(ax, x, y)
    axes[0, 0].set_title("Signed streamfunction, common levels", fontsize=11)
    axes[0, 1].set(xlim=(.6, 1), ylim=(0, .4), title="Lower-right corner, same levels")
    fig.colorbar(im, ax=axes[0], label=r"$\psi/(U_{ref}L)$ (symlog; linear below $10^{-5}$)", shrink=.8)
    j = int(np.argmin(abs(y[1:-1]-.5)))
    axes[1, 0].plot(x[1:-1], wr[j], color="black", label="CFD")
    for (name, values), color, style in zip(predictions.items(), colors, styles):
        wp = omega(values["psi"])
        axes[1, 0].plot(x[1:-1], wp[j], color=color, ls=style, label=name)
        axes[1, 1].plot(x[1:-1], wp[j]-wr[j], color=color, ls=style)
    for ax in axes[1]: ax.set_xlabel(r"$x/L$ at native $y/L=0.5$"); ax.grid(alpha=.18)
    axes[1, 0].set_ylabel(r"$\omega L/U_{ref}$ from $-\nabla_h^2\psi$")
    axes[1, 1].set_ylabel(r"$(\hat\omega-\omega)L/U_{ref}$")
    axes[1, 1].axhline(0, color=".5", lw=.7)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Re = {reynolds:.3f}; {len(x)} × {len(y)} native nodes; {context}", fontsize=12)
    return fig
