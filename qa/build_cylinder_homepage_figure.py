"""Render the original 400x400 DSMC fields with continuous, masked contours.

Run with --archive /path/to/AllMachNNCylinder.zip. The compact classroom data
and its retained metrics are unchanged; this figure has a separate provenance
record and full-grid Mach-8.5 baseline score.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from flowmllab.hypersonic_cylinder import relative_l2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "data/hypersonic_cylinder/manifest.json").read_text())
    with args.archive.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive_hash != contract["source_archive_sha256"]:
        raise ValueError("Source archive does not match the reviewed data contract")

    fields, entries = [], []
    with zipfile.ZipFile(args.archive) as archive:
        for token in ("8", "85", "9"):
            name = f"structured/StructuredGridM{token}.dat"
            raw = archive.read(name)
            table = np.loadtxt(io.BytesIO(raw), skiprows=3)
            if table.shape != (160000, 20):
                raise ValueError(f"Unexpected source shape: {name}: {table.shape}")
            fields.append(table.reshape(400, 400, 20))
            entries.append({"entry": name, "sha256": hashlib.sha256(raw).hexdigest()})

    left, center, right = fields
    if not all(np.allclose(center[:, :, :2], f[:, :, :2]) for f in (left, right)):
        raise ValueError("The three physical grids are not aligned")
    x, y = center[0, :, 0], center[:, 0, 1]
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("Source grid must be increasing")
    if not (np.allclose(center[:, :, 0], x[None, :]) and
            np.allclose(center[:, :, 1], y[:, None])):
        raise ValueError("Expected a Cartesian structured grid")
    targets = [f[:, :, [11, 10, 18]] for f in fields]
    valid = [np.isfinite(q).all(axis=-1) & (np.abs(q) < 1e20).all(axis=-1) for q in targets]
    common_valid = np.logical_and.reduce(valid)
    truth = targets[1]
    prediction = .5 * targets[0] + .5 * targets[2]
    error = np.abs(prediction - truth)
    errors = relative_l2(truth[common_valid], prediction[common_valid])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), layout="constrained")
    labels = ["Local Mach", "Temperature · source TOV", "Pressure · source P"]
    for j, label in enumerate(labels):
        for i, values in enumerate((truth[:, :, j], error[:, :, j])):
            ax = axes[i, j]
            mask = valid[1] if i == 0 else common_valid
            field = np.ma.array(values, mask=~mask)
            lo, hi = float(field.min()), float(field.max())
            levels = np.linspace(lo, hi, 161)
            ax.set_facecolor("#cbd5e1")
            artist = ax.contourf(x, y, field, levels=levels,
                                 cmap="viridis" if i == 0 else "magma",
                                 corner_mask=True, antialiased=False)
            cb = fig.colorbar(artist, ax=ax, shrink=.82, pad=.025)
            cb.locator = MaxNLocator(5)
            cb.update_ticks()
            ax.set(aspect="equal", xlabel="x (source coordinates)",
                   ylabel="y (source coordinates)")
            ax.set_title(label if i == 0 else
                         f"Absolute interpolation error\nRelative L2: {100*errors[j]:.3f}%")
    fig.suptitle("Rarefied cylinder | freestream Mach 8.5\nOriginal 400 × 400 DSMC fields",
                 fontsize=17, fontweight="bold")
    fig.supxlabel("Gray: masked solid/sentinel region. Interpolation uses Mach 8 and 9; no spatial smoothing.",
                  fontsize=10)
    output = root / "results/hypersonic_cylinder_week7_1"
    image_path = output / "cylinder_homepage.png"
    fig.savefig(image_path, dpi=300, facecolor="white")
    plt.close(fig)
    metadata = {
        "source_archive_sha256": archive_hash, "source_entries": entries,
        "grid_shape": [400, 400], "freestream_mach": 8.5,
        "baseline_mach": [8, 9], "common_valid_points": int(common_valid.sum()),
        "full_grid_baseline_relative_l2": dict(zip(
            ["local_mach", "source_TOV", "source_P"], errors.tolist())),
        "rendering": {"method": "masked contourf", "levels": 161,
                      "spatial_smoothing": False, "dpi": 300, "pixels": [3900, 1980]},
        "note": "Full-grid single-case scores; compact classroom metrics remain unchanged.",
        "figure_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    (output / "cylinder_homepage_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
