"""Retained evidence for the Week 10.1 scattering lab (CPU, deterministic).

Writes metrics.json and two figures into --output (must not be an existing
retained directory). Nothing here touches the article's DSMC runs or checkpoint.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flowmllab.scattering_lab import (build_table, transport_cross_sections, omega22, fit_surrogate,
    predict_surrogate, collision_weighted_audit, deflection_angle, omega22_star, B_MAX_DEFAULT)

HCB_OMEGA22_STAR = {1.0: 1.587, 1.5: 1.314, 2.0: 1.175, 3.0: 1.039, 4.0: 0.9700, 6.0: 0.8963, 10.0: 0.8242}

ENERGIES = np.geomspace(0.6, 120.0, 40)       # E / epsilon; covers x=E/kT up to 20 at T*=6
IMPACT = np.linspace(0.0, B_MAX_DEFAULT, 61)   # b / sigma
TEST_ENERGIES = np.sqrt(ENERGIES[:-1] * ENERGIES[1:])   # geometric midpoints: never in training
TEMPERATURES = (1.5, 3.0, 6.0)                 # T* = kT / epsilon (argon: ~180, 360, 720 K)


def main(output):
    out = Path(output)
    if out.exists():
        raise SystemExit("Refusing to write into an existing directory: " + str(out))
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    table = build_table(ENERGIES, IMPACT)
    test_table = build_table(TEST_ENERGIES, IMPACT)
    fitted = fit_surrogate(table, hidden=(48, 48), seed=0)
    pred_train = predict_surrogate(fitted, table.energies, table.impact)
    pred_test = predict_surrogate(fitted, test_table.energies, test_table.impact)
    q1, q2 = transport_cross_sections(IMPACT, test_table.cos_chi)
    q1s, q2s = transport_cross_sections(IMPACT, pred_test)
    metrics = {
        "sampling_scope": "Gamma(3/2) random Maxwellian pairs; energy-truncated; no collision-rate weighting",
        "potential": "Lennard-Jones 12-6, reduced units (teaching analog; not the Jaeger Ar-Ar potential)",
        "table": {"energies": ENERGIES.tolist(), "impact_count": len(IMPACT), "b_max": B_MAX_DEFAULT},
        "surrogate": {"architecture": "MLP 2x48 tanh, lbfgs, inputs (log E, b), output cos chi", "seed": 0},
        "train_rms_cos_error": float(np.sqrt(np.mean((pred_train - table.cos_chi) ** 2))),
        "test_energy_rms_cos_error": float(np.sqrt(np.mean((pred_test - test_table.cos_chi) ** 2))),
        "test_energy_max_abs_cos_error": float(np.max(np.abs(pred_test - test_table.cos_chi))),
        "viscosity_cross_section_relative_error_per_test_energy": (np.abs(q2s - q2) / q2).tolist(),
        "diffusion_cross_section_relative_error_per_test_energy": (np.abs(q1s - q1) / q1).tolist(),
        "omega22": {str(T): {"exact": float(omega22(TEST_ENERGIES, q2, T)),
                             "surrogate": float(omega22(TEST_ENERGIES, q2s, T))} for T in TEMPERATURES},
        "collision_weighted_audit": {str(T): collision_weighted_audit(fitted, T, count=3000, seed=1) for T in TEMPERATURES},
        "omega22_star_vs_hirschfelder_table": {str(T): {"computed_exact": float(omega22_star(ENERGIES, transport_cross_sections(IMPACT, table.cos_chi)[1], T)),
                                                        "table": ref} for T, ref in HCB_OMEGA22_STAR.items()},
    }
    for T in metrics["omega22_star_vs_hirschfelder_table"]:
        d = metrics["omega22_star_vs_hirschfelder_table"][T]; d["relative_difference"] = (d["computed_exact"] - d["table"]) / d["table"]
    for T in metrics["omega22"]:
        e, s = metrics["omega22"][T]["exact"], metrics["omega22"][T]["surrogate"]
        metrics["omega22"][T]["relative_error"] = abs(s - e) / e
    metrics["elapsed_seconds"] = time.perf_counter() - t0
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for e in (0.6, 1.5, 5.0, 20.0):
        chi = [deflection_angle(float(b), e) for b in IMPACT]
        ax[0].plot(IMPACT, np.degrees(chi), label=f"$E/\\epsilon={e:g}$")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("$b/\\sigma$"); ax[0].set_ylabel("$\\chi$ (deg)")
    ax[0].set_title("Exact classical deflection (LJ)"); ax[0].legend(fontsize=8)
    im = ax[1].pcolormesh(IMPACT, TEST_ENERGIES, pred_test - test_table.cos_chi, cmap="RdBu_r", vmin=-0.1, vmax=0.1, shading="auto")
    ax[1].set_yscale("log"); ax[1].set_xlabel("$b/\\sigma$"); ax[1].set_ylabel("$E/\\epsilon$ (unseen energies)")
    ax[1].set_title("Surrogate error in $\\cos\\chi$"); fig.colorbar(im, ax=ax[1], extend="both")
    ax[2].plot(TEST_ENERGIES, q2, "k-", label="exact $Q^{(2)}$"); ax[2].plot(TEST_ENERGIES, q2s, "r--", label="surrogate $Q^{(2)}$")
    ax[2].set_xscale("log"); ax[2].set_xlabel("$E/\\epsilon$"); ax[2].set_ylabel("$Q^{(2)}/\\sigma^2$"); ax[2].legend(fontsize=8)
    ax[2].set_title("Viscosity cross-section")
    fig.tight_layout(); fig.savefig(out / "scattering_lab_overview.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for T in TEMPERATURES:
        a = metrics["collision_weighted_audit"][str(T)]
        ax.bar(str(T), a["rms_cos_error"], color="steelblue")
    ax.set_xlabel("$T^*=kT/\\epsilon$"); ax.set_ylabel("Maxwellian-pair RMS error in $\\cos\\chi$")
    ax.set_title("Conditional random-pair audit (energy-truncated)")
    fig.tight_layout(); fig.savefig(out / "scattering_lab_weighted_error.png", dpi=180); plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ("train_rms_cos_error", "test_energy_rms_cos_error", "test_energy_max_abs_cos_error", "omega22", "elapsed_seconds")}, indent=1))
    print({T: round(a["rms_cos_error"], 4) for T, a in metrics["collision_weighted_audit"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
