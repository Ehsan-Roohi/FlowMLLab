"""Complete-case neural comparisons and physical figures for Week 4 Lab 2.

The train/development split is made before coordinates are expanded.  All
training routines receive only those two partitions: test labels are not an
argument.  TensorFlow is imported lazily so data/plot utilities work on CPU
installations without an ML backend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import os
import json

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class NetworkConfig:
    widths: tuple[int, ...] = (48, 48, 48)
    activation: str = "tanh"
    adam_steps: int = 400
    lbfgs_steps: int = 600
    learning_rate: float = 0.002
    record_every: int = 20
    boundary_aware: bool = False
    boundary_lifting: str = "product"
    seed: int = 690


def case_split(data, validation_re=(225.0, 350.0)):
    """Reserve complete development cases; never promote a public test case."""
    re = np.asarray(data["Re"], dtype=float)
    labels = np.asarray(data["split"]).astype(str)
    if len(np.unique(re)) != len(re):
        raise ValueError("Each Reynolds number must identify one complete case.")
    public_test = labels == "test"
    if not np.any(public_test):
        raise ValueError("Supply an explicitly labelled test partition.")
    requested = np.isclose(re[:, None], np.asarray(validation_re)[None, :]).any(axis=1)
    if requested.sum() != len(validation_re) or np.any(requested & public_test):
        raise ValueError("Choose existing training Reynolds numbers for validation.")
    fit = (labels == "train") & ~requested
    if fit.sum() < 3 or not requested.any():
        raise ValueError("Need at least three fitting cases and one validation case.")
    if re[fit].min() >= re[~fit].min() or re[fit].max() <= re[~fit].max():
        raise ValueError("Keep the minimum and maximum Re in fitting for interpolation.")
    return fit, requested, public_test


def interpolate_rows(rstar, rtrain, ytrain):
    rr = np.asarray(rtrain, dtype=float)
    yy = np.asarray(ytrain, dtype=float)
    order = np.argsort(rr)
    rr, yy = rr[order], yy[order]
    exact = np.flatnonzero(np.isclose(rr, rstar, rtol=0, atol=1e-12))
    if exact.size:
        return yy[exact[0]].copy()
    hi = np.searchsorted(rr, rstar)
    if hi == 0 or hi == len(rr):
        raise ValueError("Interpolation requires two bracketing training cases.")
    weight = (rstar - rr[hi - 1]) / (rr[hi] - rr[hi - 1])
    return (1 - weight) * yy[hi - 1] + weight * yy[hi]


def primary_targets(data):
    rows = []
    for psi in data["psi"]:
        j, i = np.unravel_index(np.argmin(psi), psi.shape)
        rows.append((psi[j, i], data["x"][i], data["y"][j]))
    return np.asarray(rows)


def point_samples(data, mask, rmin, rmax, stride=2):
    """Uniform coarse points plus all wall-adjacent rows/columns, no test access."""
    x, y = np.asarray(data["x"]), np.asarray(data["y"])
    ii = np.unique(np.r_[np.arange(0, len(x), stride), 0, 1, len(x)-2, len(x)-1])
    jj = np.unique(np.r_[np.arange(0, len(y), stride), 0, 1, len(y)-2, len(y)-1])
    xx, yy = np.meshgrid(x[ii], y[jj])
    features, targets = [], []
    for k in np.flatnonzero(mask):
        rr = np.full(xx.size, 2 * (data["Re"][k] - rmin) / (rmax - rmin) - 1)
        features.append(np.c_[rr, xx.ravel(), yy.ravel()])
        targets.append(np.c_[data["u"][k][np.ix_(jj, ii)].ravel(),
                             data["v"][k][np.ix_(jj, ii)].ravel()])
    return np.vstack(features), np.vstack(targets)


def make_network(ninput, noutput, config):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    inputs = tf.keras.Input(shape=(ninput,), dtype="float64")
    if config.boundary_aware:
        # Smooth interior extension of a discontinuous lid. Corner velocity is
        # zero, exactly matching the supplied CFD discretisation.
        class BoundaryOutput(tf.keras.layers.Layer):
            def call(self, pair):
                features, raw = pair
                x, y = features[:, 1:2], features[:, 2:3]
                side = x * (1-x)
                if config.boundary_lifting == "corner":
                    denominator = side**2 + (1-y)**2
                    lift = tf.math.divide_no_nan(y * side**2, denominator)
                    envelope = tf.math.divide_no_nan(4*y*side*(1-y), denominator)
                elif config.boundary_lifting == "product":
                    denominator = side + (1-y)**2
                    lift = tf.math.divide_no_nan(y * side, denominator)
                    envelope = 16 * side * y * (1-y)
                else:
                    raise ValueError("boundary_lifting must be 'product' or 'corner'")
                return tf.concat([lift + envelope * raw[:, :1],
                                  envelope * raw[:, 1:2]], axis=1)
        z = tf.keras.layers.Lambda(lambda t: tf.concat([t[:, :1], 2*t[:, 1:]-1], axis=1),
                                   dtype="float64")(inputs)
    else:
        z = inputs
        if ninput == 3:
            z = tf.keras.layers.Lambda(lambda t: tf.concat([t[:, :1], 2*t[:, 1:]-1], axis=1),
                                       dtype="float64")(z)
    for width in config.widths:
        z = tf.keras.layers.Dense(width, activation=config.activation, dtype="float64")(z)
    z = tf.keras.layers.Dense(noutput, dtype="float64")(z)
    if config.boundary_aware:
        z = BoundaryOutput(dtype="float64")([inputs, z])
    return tf.keras.Model(inputs, z)


def fit_network(xfit, yfit, xval, yval, config):
    """Adam then L-BFGS; restore the lowest complete-case validation MSE.

    Both terms are ordinary unweighted component MSE in the supplied output
    coordinates. There is no test argument, gradient residual, or PINN claim.
    """
    import tensorflow as tf
    model = make_network(np.shape(xfit)[1], np.shape(yfit)[1], config)
    xf, yf, xv, yv = [tf.constant(a, dtype=tf.float64) for a in (xfit, yfit, xval, yval)]
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    variables = model.trainable_variables
    shapes = [tuple(v.shape) for v in variables]
    sizes = [int(np.prod(shape)) for shape in shapes]
    offsets = np.cumsum([0] + sizes)

    @tf.function
    def loss_gradient():
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(xf, training=True) - yf))
        gradients = tape.gradient(loss, variables)
        return loss, gradients

    @tf.function
    def adam_step():
        loss, gradients = loss_gradient()
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function
    def losses():
        return (tf.reduce_mean(tf.square(model(xf, training=False)-yf)),
                tf.reduce_mean(tf.square(model(xv, training=False)-yv)))

    history, best = [], {"validation": np.inf, "weights": None}
    start = perf_counter()
    def record(stage, step):
        training, validation = [float(z) for z in losses()]
        if not np.isfinite(training + validation):
            raise FloatingPointError("Non-finite neural loss; inspect learning rate and data.")
        history.append(dict(stage=stage, step=int(step), seconds=perf_counter()-start,
                            training_mse=training, validation_mse=validation))
        if os.environ.get("FLOWMLLAB_TRAIN_PROGRESS") == "1" and (step == 0 or step % 200 == 0):
            print(f"{stage} {step}: fit MSE={training:.6g}, validation MSE={validation:.6g}, "
                  f"elapsed={perf_counter()-start:.1f}s", flush=True)
        progress_path = os.environ.get("FLOWMLLAB_TRAIN_PROGRESS_JSON")
        if progress_path and (step == 0 or step % 200 == 0):
            progress = {"config": asdict(config), **history[-1]}
            destination = Path(progress_path)
            temporary = destination.with_suffix(".writing.json")
            temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            temporary.replace(destination)
        if validation < best["validation"]:
            best.update(validation=validation, weights=model.get_weights(), stage=stage, step=int(step))
    record("initial", 0)
    for step in range(1, config.adam_steps+1):
        adam_step()
        if step % config.record_every == 0 or step == config.adam_steps:
            record("Adam", step)

    def assign(theta):
        for j, variable in enumerate(variables):
            variable.assign(theta[offsets[j]:offsets[j+1]].reshape(shapes[j]))
    def objective(theta):
        assign(theta)
        loss, gradients = loss_gradient()
        return float(loss), np.concatenate([g.numpy().ravel() for g in gradients])
    calls = 0
    def callback(theta):
        nonlocal calls
        calls += 1
        if calls % config.record_every == 0:
            assign(theta)
            record("L-BFGS", calls)
    result = None
    if config.lbfgs_steps:
        result = minimize(objective, np.concatenate([v.numpy().ravel() for v in variables]),
                          jac=True, method="L-BFGS-B", callback=callback,
                          options={"maxiter": config.lbfgs_steps, "maxls": 30,
                                   "maxcor": 20, "ftol": 1e-12, "gtol": 1e-9})
        assign(result.x)
        record("L-BFGS", int(result.nit))
    model.set_weights(best.pop("weights"))
    metadata = {"config": asdict(config), "parameters": model.count_params(),
                "seconds": perf_counter()-start, "selected": best,
                "adam_gradient_steps": config.adam_steps,
                "lbfgs_iterations": int(result.nit) if result is not None else 0,
                "lbfgs_objective_evaluations": int(result.nfev) if result is not None else 0,
                "lbfgs_gradient_evaluations": int(result.njev) if result is not None else 0,
                "lbfgs_message": str(result.message) if result is not None else "not run"}
    return model, history, metadata


def predict_field(model, reynolds, data, rmin, rmax):
    xx, yy = np.meshgrid(data["x"], data["y"])
    rr = np.full(xx.size, 2*(reynolds-rmin)/(rmax-rmin)-1)
    q = np.asarray(model(np.c_[rr, xx.ravel(), yy.ravel()], training=False))
    return q[:, 0].reshape(xx.shape), q[:, 1].reshape(xx.shape)


def field_metrics(x, y, reference_u, reference_v, predicted_u, predicted_v):
    x, y = np.asarray(x), np.asarray(y)
    u, v, up, vp = map(np.asarray, (reference_u, reference_v, predicted_u, predicted_v))
    error = np.hypot(up-u, vp-v)
    wall = np.zeros(error.shape, bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    divergence = np.gradient(up, x, axis=1, edge_order=2)+np.gradient(vp, y, axis=0, edge_order=2)
    reference_div = np.gradient(u, x, axis=1, edge_order=2)+np.gradient(v, y, axis=0, edge_order=2)
    # Central finite differences near the singular lid corner are sensitive to
    # resolution. A stated two-cell rim is excluded from the interior diagnostic.
    core = np.s_[2:-2, 2:-2]
    xx, yy = np.meshgrid(x, y)
    corners = {"upper_left": (xx > 0) & (xx <= .2) & (yy >= .8) & (yy < 1),
               "upper_right": (xx >= .8) & (xx < 1) & (yy >= .8) & (yy < 1),
               "lower_left": (xx > 0) & (xx <= .2) & (yy > 0) & (yy <= .2),
               "lower_right": (xx >= .8) & (xx < 1) & (yy > 0) & (yy <= .2)}
    return {"vector_relative_l2": float(np.linalg.norm(error)/np.linalg.norm(np.hypot(u, v))),
            "vector_rms": float(np.sqrt(np.mean(error**2))), "vector_max": float(error.max()),
            "wall_rms": float(np.sqrt(np.mean(error[wall]**2))), "wall_max": float(error[wall].max()),
            "divergence_core_rms": float(np.sqrt(np.mean(divergence[core]**2))),
            "reference_divergence_core_rms": float(np.sqrt(np.mean(reference_div[core]**2))),
            **{f"{name}_rms": float(np.sqrt(np.mean(error[mask]**2))) if np.any(mask) else float("nan")
               for name, mask in corners.items()}}


def figure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11,
                         "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 180,
                         "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(directory/f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def plot_scalar_trends(re, targets, fit, val, test, dense_re, predictions):
    import matplotlib.pyplot as plt
    labels = (r"$\psi_{\min}/(U_{lid}L)$", r"$x_v/L$", r"$y_v/L$")
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), layout="constrained")
    palette = ("#0066aa", "#c24600", "#762a83")
    for j, ax in enumerate(axes):
        ax.plot(re[fit], targets[fit, j], "ko", label="CFD: fit", ms=5)
        ax.plot(re[val], targets[val, j], "ks", mfc="white", label="CFD: validation", ms=6)
        ax.plot(re[test], targets[test, j], "k*", mfc="white", label="CFD: test", ms=9)
        for (name, values), color, style in zip(predictions.items(), palette, ("-", "--", "-.")):
            ax.plot(dense_re, values[:, j], color=color, ls=style, label=name)
        ax.set(xlabel="Reynolds number", ylabel=labels[j])
        ax.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=3, frameon=False)
    return fig


def plot_histories(histories):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(histories), figsize=(max(7, 4.3*len(histories)), 4),
                             squeeze=False, layout="constrained")
    for ax, (name, history) in zip(axes[0], histories.items()):
        ax.semilogy([h["seconds"] for h in history], [h["training_mse"] for h in history],
                    color="#0066aa", label="Fit cases")
        ax.semilogy([h["seconds"] for h in history], [h["validation_mse"] for h in history],
                    color="#c24600", ls="--", label="Validation cases")
        selected = min(history, key=lambda row: row["validation_mse"])
        ax.plot(selected["seconds"], selected["validation_mse"], marker="*", color="black",
                markerfacecolor="#e6b800", ms=10, ls="none", label="Restored checkpoint")
        switch = next((h["seconds"] for h in history if h["stage"] == "L-BFGS"), None)
        if switch is not None:
            ax.axvline(switch, color=".5", ls=":", label="First recorded L-BFGS iterate")
        ax.set(title=name, xlabel="Elapsed training time (s)", ylabel="Component MSE")
        ax.grid(alpha=.2)
    handles, names = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2, frameon=False)
    return fig


def plot_field_comparison(data, index, predictions):
    """Full extrema and common scales, with field and streamline comparisons."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    x, y = data["x"], data["y"]
    u, v = data["u"][index], data["v"][index]
    speed = np.hypot(u, v)
    fields = [("CFD", (u, v))] + list(predictions.items())
    vmax = max(np.hypot(a, b).max() for _, (a, b) in fields)
    levels = np.linspace(0, vmax, 25)
    maximum_error = max(np.hypot(a-u, b-v).max() for _, (a, b) in predictions.items())
    error_levels = np.linspace(0, max(maximum_error, 1e-12), 25)
    fig, axes = plt.subplots(2, len(fields), figsize=(3.8*len(fields), 7.8), layout="constrained")
    for j, (name, (a, b)) in enumerate(fields):
        field = axes[0, j].contourf(x, y, np.hypot(a, b), levels=levels, cmap="turbo")
        axes[0, j].streamplot(x, y, a, b, density=1.15, color="black", linewidth=.55, arrowsize=.65)
        axes[0, j].set_title(f"{name}; Re = {data['Re'][index]:g}")
        if j:
            error = axes[1, j].contourf(x, y, np.hypot(a-u, b-v), levels=error_levels, cmap="turbo")
            axes[1, j].set_title(f"{name}: vector error")
        else:
            axes[1, j].contourf(x, y, speed, levels=levels, cmap="turbo", alpha=.78)
            contour_levels = np.array([.05, .12, .22, .4, .65, .85])
            axes[1, j].contour(x, y, speed, levels=contour_levels, colors="black", linewidths=.8)
            for (method, (ap, bp)), color, style in zip(predictions.items(),
                                                       (".4", "#cc00cc", "#111111"),
                                                       ("--", "-.", ":")):
                axes[1, j].contour(x, y, np.hypot(ap, bp), levels=contour_levels,
                                   colors=color, linestyles=style, linewidths=.9)
            axes[1, j].set_title("Matched speed contours")
    for ax in axes.ravel():
        ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.spines[["top", "right"]].set_visible(True)
    fig.colorbar(field, ax=axes[0, :], shrink=.8, label=r"$|\mathbf{u}|/U_{lid}$")
    fig.colorbar(error, ax=axes[1, 1:], shrink=.8,
                 label=r"$\|\widehat{\mathbf{u}}-\mathbf{u}_{CFD}\|/U_{lid}$ (linear)")
    handles = [Patch(facecolor=plt.get_cmap("turbo")(.6), label="CFD"),
               Line2D([], [], color="black", lw=1, label="CFD speed contours")]
    for name, color, style in zip(predictions, (".4", "#cc00cc", "#111111"), ("--", "-.", ":")):
        handles.append(Line2D([], [], color=color, ls=style, label=name))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return fig


def plot_profiles(data, index, predictions):
    import matplotlib.pyplot as plt
    x, y = data["x"], data["y"]
    u, v = data["u"][index], data["v"][index]
    midx, midy = np.argmin(abs(x-.5)), np.argmin(abs(y-.5))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    axes[0, 0].plot(u[:, midx], y, "k-", lw=2, label="CFD")
    axes[0, 1].plot(x, v[midy, :], "k-", lw=2, label="CFD")
    for (name, (up, vp)), color, ls in zip(predictions.items(),
                                           ("#0066aa", "#c24600", "#762a83"), ("--", "-.", ":")):
        axes[0, 0].plot(up[:, midx], y, color=color, ls=ls, label=name)
        axes[0, 1].plot(x, vp[midy, :], color=color, ls=ls, label=name)
        axes[1, 0].plot(up[:, midx]-u[:, midx], y, color=color, ls=ls)
        axes[1, 1].plot(x, vp[midy, :]-v[midy, :], color=color, ls=ls)
    axes[0, 0].set(xlabel=r"$u/U_{lid}$", ylabel=r"$y/L$", title=f"x/L = {x[midx]:g}; Re = {data['Re'][index]:g}")
    axes[0, 1].set(xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$", title=f"y/L = {y[midy]:g}; Re = {data['Re'][index]:g}")
    axes[1, 0].set(xlabel=r"$(\widehat{u}-u_{CFD})/U_{lid}$", ylabel=r"$y/L$")
    axes[1, 1].set(xlabel=r"$x/L$", ylabel=r"$(\widehat{v}-v_{CFD})/U_{lid}$")
    axes[1, 0].axvline(0, color=".6", lw=.7)
    axes[1, 1].axhline(0, color=".6", lw=.7)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return fig


def plot_wall_diagnostics(data, index, predictions):
    import matplotlib.pyplot as plt
    x, y = data["x"], data["y"]
    u, v = data["u"][index], data["v"][index]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    for (name, (up, vp)), color, ls in zip(predictions.items(),
                                           ("#0066aa", "#c24600", "#762a83"), ("--", "-.", ":")):
        axes[0].plot(x, up[-1]-u[-1], color=color, ls=ls, label=name)
        axes[1].plot(y, np.hypot(up[:, 0]-u[:, 0], vp[:, 0]-v[:, 0]), color=color, ls=ls)
        divergence = np.gradient(up, x, axis=1, edge_order=2)+np.gradient(vp, y, axis=0, edge_order=2)
        axes[2].plot(y[2:-2], np.sqrt(np.mean(divergence[2:-2, 2:-2]**2, axis=1)), color=color, ls=ls)
    refdiv = np.gradient(u, x, axis=1, edge_order=2)+np.gradient(v, y, axis=0, edge_order=2)
    axes[2].plot(y[2:-2], np.sqrt(np.mean(refdiv[2:-2, 2:-2]**2, axis=1)), "k-", label="CFD")
    axes[0].set(xlabel=r"$x/L$ along lid", ylabel=r"$(\widehat{u}-u_{CFD})/U_{lid}$")
    axes[1].set(xlabel=r"$y/L$ along left wall", ylabel="Wall vector error / lid speed")
    axes[2].set(xlabel=r"$y/L$", ylabel=r"Row RMS of $\nabla\cdot\widehat{\mathbf{u}}$", title="Interior; two-cell rim excluded")
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[2].get_legend_handles_labels()
    fig.legend(handles+handles2, labels+labels2, loc="outside lower center", ncol=4, frameon=False)
    return fig
