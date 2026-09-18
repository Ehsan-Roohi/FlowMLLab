#!/usr/bin/env python3
"""Re-plot the Week 12 first-seed heat-flux fields with a robust colour scale.

The retained `cavity_q{x,y}_hero.png` use the full value range, so the lid band
sets the colour limits and the interior appears almost blank. This script draws
the same retained arrays (`results/week12_research/first_seed_fields.npz`) with
limits at the 1st and 99th percentiles of the reference field and an extended
colour bar, and writes `cavity_q{x,y}_hero_robust.png` beside the originals.
No value is changed; only the mapping from value to colour is clipped, and the
caption says so. Errors in the panel titles are read from `metrics.csv`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/week12_research"

PANELS = [("reference", "Independent DSMC reference"),
          ("raw_b3", "Raw DSMC: 3 blocks"),
          ("promoted_full_hierarchy", "Archived observation-conditioned estimator")]


def main() -> None:
    fields = dict(np.load(EVIDENCE / "first_seed_fields.npz", allow_pickle=False))
    metrics = pd.read_csv(EVIDENCE / "metrics.csv")
    seed = int(metrics.seed.min())
    first = metrics[metrics.seed == seed]
    for component in ("qx", "qy"):
        reference = fields[f"{component}_reference"]
        lo, hi = np.percentile(reference, [1, 99])
        limit = float(max(abs(lo), abs(hi)))
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
        for ax, (key, title) in zip(axes, PANELS):
            array = fields[f"{component}_{key}"]
            im = ax.imshow(array, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                           extent=(0, 1, 0, 1), interpolation="nearest")
            label = title
            if key != "reference":
                error = float(first[(first.field == component) & (first.method == key)].reference_nrmse.iloc[0])
                label += f"\nreference NRMSE: {100*error:.2f}%"
            ax.set_title(label, fontsize=10)
            ax.set(xlabel="normalized column position", ylabel="normalized row position")
        cbar = fig.colorbar(im, ax=axes, extend="both", shrink=0.9)
        cbar.set_label(f"{component} (archive units)")
        fig.suptitle(f"DSMC heat-flux reconstruction | {component} | Kn = 0.085 | seed {seed} | "
                     "robust colour scale (values outside the limits are saturated, not removed)", fontsize=11)
        fig.savefig(EVIDENCE / f"cavity_{component}_hero_robust.png", dpi=160)
        plt.close(fig)
        print(EVIDENCE / f"cavity_{component}_hero_robust.png", f"limits +/-{limit:.3f}")


if __name__ == "__main__":
    main()
