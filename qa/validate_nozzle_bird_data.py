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
    assert manifest["schema_version"] == 2
    assert manifest["case_count"] == 15
    assert [entry["back_pressure_kpa"] for entry in manifest["files"]] == EXPECTED
    transform = manifest["full_domain_transform"]
    assert transform["y_translation_m"] == -0.000092
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
        full_text = payload.decode("ascii", errors="replace")
        assert full_text.count("ZONE T=") == 4
        assert full_text.count("VARSHARELIST = ([1,3-12]=") == 2
    print(
        "NOZZLE_BIRD_DATA_PASS: 15 back-pressure files; hashes, four-zone "
        "headers and mirror variable sharing verified"
    )


if __name__ == "__main__":
    main()
