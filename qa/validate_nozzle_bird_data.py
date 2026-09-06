#!/usr/bin/env python3
"""Validate the recovered full-resolution micro-nozzle DSMC fields."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nozzle_bird_pr"
EXPECTED = [15, 16, 18, 19, 20, *range(22, 31), 33]


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["case_count"] == 15
    assert [entry["pressure_ratio"] for entry in manifest["files"]] == EXPECTED
    actual = sorted(DATA.glob("P=*full.dat"), key=lambda p: int(re.search(r"P=(\d+)", p.name).group(1)))
    assert [int(re.search(r"P=(\d+)", p.name).group(1)) for p in actual] == EXPECTED
    for path, entry in zip(actual, manifest["files"]):
        payload = path.read_bytes()
        assert path.name == entry["file"]
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        head = payload[:4096].decode("ascii", errors="replace")
        assert 'VARIABLES = "X"' in head
        assert "ZONETYPE=Ordered" in head
        assert entry["zone_dimensions"] == {"I": 101, "J": 31, "K": 1}
    print("NOZZLE_BIRD_DATA_PASS: 15 files; hashes and 101x31x1 headers verified")


if __name__ == "__main__":
    main()
