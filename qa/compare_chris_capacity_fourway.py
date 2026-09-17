"""Re=100, D/W=5: compare two retained PINNs with two CFD solvers.

The runs may have unequal optimization histories. This is an observed field
comparison, not a controlled neural-capacity or convergence experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from audit_week13_nektar import structured, integrate_paths
from audit_week13_cfd import audit as foam_audit, foam_list
from check_week13_nektar_vtu import read_vtu

def accepted_source(chunk, reynolds):
    chunk = chunk.resolve()
    marker = json.loads((chunk/"accepted.json").read_text())
    spec = json.loads((chunk.parent/"campaign.json").read_text())
    if (not marker.get("clean_exit") or spec["re"] != reynolds
            or spec["order"] != 6 or spec["depth_over_width"] != 5):
        raise ValueError("Nektar++ case provenance mismatch")
    source = (chunk/marker["attempt"]).resolve()
    if source.parent != chunk:
        raise ValueError("Accepted attempt escapes its chunk")
    digest = hashlib.sha256((source/"cavity.vtu").read_bytes()).hexdigest()
    if digest != marker["primitive_vtu_sha256"]:
        raise ValueError("Accepted VTU hash mismatch")
    return source, marker, spec


def interpolate(x, y, field, xx, yy):
    q = np.column_stack((yy.ravel(), xx.ravel()))
    return RegularGridInterpolator((y, x), field, bounds_error=False,
                                   fill_value=None)(q).reshape(xx.shape)


def gauge(p):
    """Remove the arbitrary incompressible-pressure constant on a shared grid."""
    return p - np.mean(p)


def relative_l2(a, b):
    denominator = np.sum(np.square(b))
    return float(np.sqrt(np.sum(np.square(a-b))/denominator)) if denominator else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pinn-small", type=Path, required=True)
    ap.add_argument("--pinn-large", type=Path, required=True)
    ap.add_argument("--chunk", type=Path, required=True)
    ap.add_argument("--foam", type=Path, required=True)
    ap.add_argument("--loss-small", type=Path)
    ap.add_argument("--loss-large", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    audits = [json.loads((p.parent/"audit.json").read_text())
              for p in (args.pinn_small, args.pinn_large)]
    for audit in audits:
        case = audit["case"]
        if case["Re"] != 100 or case["depth_over_width"] != 5 or case["tri"] != 0:
            raise ValueError("Both PINNs must be the Re100 D5 tri0 case")
    source, marker, spec = accepted_source(args.chunk, 100)
    if marker["time"] < 100:
        raise ValueError("Nektar++ accepted time is too early")
    xn, yn, nf, _ = structured(read_vtu(source/"cavity.vtu"), depth=5)
    frec, (xf, yf, vf, _, _) = foam_audit(args.foam)
    if frec["re"] != 100 or not np.isclose(frec["depth_over_width"], 5):
        raise ValueError("OpenFOAM case mismatch")
    # A common physical grid; avoid assigning a CFD mesh refinement to a PINN.
    x = np.linspace(.005, .995, 100)
    y = np.linspace(.005, 4.995, 500)
    xx, yy = np.meshgrid(x, y)
    solvers = []
    titles = ["Nektar++", "OpenFOAM", "PINN width 32", "PINN width 64"]
    foam_pressure = foam_list(args.foam/str(frec["time"])/"p",
                              field=True, components=1).reshape(len(yf), len(xf))
    for xs, ys, fields in ((xn, yn, nf),
                           (xf, yf, dict(u=vf[:,:,0], v=vf[:,:,1], p=foam_pressure))):
        solvers.append({k: interpolate(xs, ys, fields[k], xx, yy)
                        for k in ("u", "v", "p")})
    for path in (args.pinn_small, args.pinn_large):
        with np.load(path) as data:
            solvers.append({k: interpolate(data["x"], data["y"], data[k], xx, yy)
                            for k in ("u", "v", "p")})
    for fields in solvers:
        fields["p"] = gauge(fields["p"])
        fields["speed"] = np.hypot(fields["u"], fields["v"])
        fields["omega"] = np.gradient(fields["v"], x, axis=1) - np.gradient(fields["u"], y, axis=0)
        fields["psi"], _ = integrate_paths(x, y, fields["u"], fields["v"])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"case": "Re=100 D/W=5 tri0", "nektar_time": marker["time"],
              "nektar_grid": [spec["nx"], spec["ny"]],
              "foam_grid": [frec["nx"], frec["ny"]],
              "pinn_audits": audits, "comparison_grid": [len(x), len(y)],
              "limitations": ["PINN widths and optimization histories differ; inspect individual provenance.",
                              "PINN checkpoints have unequal training budgets.",
                              "The width-64 vector may be an unconverged intermediate optimizer state.",
                              "Pressure is independently zero-mean gauged on the shared grid.",
                              "The very weak bottom vortex requires separate mesh and integration checks."]}
    for i in range(1, 4):
        report[titles[i]] = {
            "velocity_L2_vs_Nektar": relative_l2(
                np.stack((solvers[i]["u"],solvers[i]["v"]),axis=-1),
                np.stack((solvers[0]["u"],solvers[0]["v"]),axis=-1)),
            "pressure_L2_vs_Nektar": relative_l2(solvers[i]["p"], solvers[0]["p"]),
            "velocity_L2_bottom_y_lt_1_vs_Nektar": relative_l2(
                np.stack((solvers[i]["u"][y<1],solvers[i]["v"][y<1]),axis=-1),
                np.stack((solvers[0]["u"][y<1],solvers[0]["v"][y<1]),axis=-1)),
            "velocity_RMS_bottom_Ulid": float(np.sqrt(np.mean(
                (solvers[i]["u"][y<1]-solvers[0]["u"][y<1])**2 +
                (solvers[i]["v"][y<1]-solvers[0]["v"][y<1])**2)))}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    quantities = [("u", r"$u/U_{lid}$", "RdBu_r"),
                  ("v", r"$v/U_{lid}$", "RdBu_r"),
                  ("speed", r"$|u|/U_{lid}$", "viridis"),
                  ("p", r"$(p-\bar p)/(\rho U_{lid}^2)$", "RdBu_r"),
                  ("omega", r"$\omega_z W/U_{lid}$", "RdBu_r"),
                  ("psi", r"$\psi/(U_{lid}W)$", "RdBu_r")]
    for key, label, cmap in quantities:
        values = [f[key] for f in solvers]
        if key == "speed":
            norm = None
            lo, hi = 0, max(float(np.nanpercentile(v, 99.5)) for v in values)
        else:
            limit = max(float(np.nanpercentile(np.abs(v), 99.5)) for v in values)
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
            lo = hi = None
        fig, axes = plt.subplots(1, 4, figsize=(15, 16), layout="constrained")
        for ax, title, field, solver in zip(axes, titles, values, solvers):
            im = ax.pcolormesh(x, y, field, shading="auto", cmap=cmap,
                               vmin=lo, vmax=hi, norm=norm, rasterized=True)
            if key in ("speed", "psi"):
                levels = np.unique(np.r_[-.1,-.05,-.02,-.005,-.001,
                                         -np.logspace(-8,-4,5),0,
                                         np.logspace(-8,-4,5),.001,.005])
                ax.contour(x, y, solver["psi"], levels=levels,
                           colors="white" if key=="speed" else "black",
                           linewidths=.55, alpha=.75)
            ax.set(xlim=(0,1), ylim=(0,5), xlabel="x/W", ylabel="y/W",
                   title=title, aspect="equal")
        fig.colorbar(im, ax=axes.tolist(), shrink=.64, label=label)
        fig.suptitle(f"Re=100, D/W=5 | {label} | shared colour scale", fontsize=18)
        fig.savefig(args.output/f"fourway_{key}.png", dpi=210)
        fig.savefig(args.output/f"fourway_{key}.pdf")
        plt.close(fig)

    # Lower-cavity streamfunction needs its own levels: full-depth contours
    # hide the 1e-9--1e-6 recirculation cells.
    fig, axes = plt.subplots(1, 4, figsize=(15, 12), layout="constrained")
    lower_levels = np.unique(np.r_[-np.logspace(-9,-3,13), 0,
                                    np.logspace(-9,-3,13)])
    for ax, title, field in zip(axes, titles, solvers):
        ax.contour(x, y, field["psi"], levels=lower_levels,
                   colors="black", linewidths=.8)
        ax.set(xlim=(0,1), ylim=(0,3), xlabel="x/W", ylabel="y/W",
               title=title, aspect="equal")
    fig.suptitle("Lower cavity: identical signed log-spaced streamfunction levels", fontsize=17)
    fig.savefig(args.output/"lower_cells_streamlines.png", dpi=240)
    fig.savefig(args.output/"lower_cells_streamlines.pdf")
    plt.close(fig)

    if args.loss_small is not None and args.loss_large is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        loss_report = {}
        for label, path in (("PINN width 32", args.loss_small),
                            ("PINN width 64", args.loss_large)):
            data = np.atleast_2d(np.loadtxt(path, comments="#"))
        # DeepXDE writes two train-loss columns followed by two test-loss
        # columns in these retained files. Do not mix train and test losses.
            step, components = data[:,0], data[:,1:3]
            total = np.sum(components, axis=1)
            mask = np.isfinite(total) & (total>0)
            ax.semilogy(step[mask], total[mask], marker="o", markersize=3,
                        label=f"{label}: retained segment ({mask.sum()} samples)")
            loss_report[label] = {"samples":int(mask.sum()),
                                  "step_first":float(step[mask][0]),
                                  "step_last":float(step[mask][-1]),
                                  "loss_first":float(total[mask][0]),
                                  "loss_last":float(total[mask][-1])}
        ax.set(xlabel="Step reported by each retained loss.dat",
               ylabel="Sum of recorded training-loss components",
               title="Recorded training loss (unequal runs; not a matched-budget test)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(args.output/"pinn_loss_retained.png", dpi=220)
        fig.savefig(args.output/"pinn_loss_retained.pdf")
        plt.close(fig)
        report["loss"] = loss_report
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"output":str(args.output), "metrics":report["PINN width 64"]}))


if __name__ == "__main__":
    main()
