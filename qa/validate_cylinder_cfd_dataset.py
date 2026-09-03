#!/usr/bin/env python3
"""Fail closed if a cylinder CFD release is incomplete or corrupted."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED = {
    60: "development", 80: "development", 90: "development",
    95: "fresh_test", 100: "validation", 105: "retained_test",
    110: "development", 120: "development", 140: "development",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    rows = {int(row["reynolds"]): row for row in manifest["cases"]}
    assert set(rows) == set(EXPECTED), "missing or unexpected Reynolds case"
    for reynolds, split in EXPECTED.items():
        row = rows[reynolds]
        assert row["split"] == split, f"split leakage at Re={reynolds}"
        path = args.dataset / row["file"]
        assert path.is_file() and digest(path) == row["sha256"], f"checksum failed: {path}"
        with np.load(path, allow_pickle=False) as data:
            assert str(data["split"]) == split
            assert float(data["reynolds"]) == reynolds
            assert data["u"].shape == data["v"].shape == data["p"].shape
            assert data["u"].ndim == 3 and data["u"].shape[0] >= 250
            assert np.isfinite(data["u"]).all() and np.isfinite(data["v"]).all()
            assert np.isfinite(data["p"]).all()
            strouhal = float(data["strouhal"])
            # Low-amplitude development wakes may legitimately fail the
            # conservative spectral-peak gate.  Quantitative validation and
            # both held-out shedding cases must always resolve a finite St.
            if reynolds >= 90:
                assert np.isfinite(strouhal), f"unresolved Strouhal at Re={reynolds}"
        print(f"PASS Re={reynolds} split={split} sha256={row['sha256'][:12]}")
    print("PASS: 9/9 CFD cases, checksums, arrays, and leakage-free split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
