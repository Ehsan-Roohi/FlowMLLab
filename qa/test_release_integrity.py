from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.9.0"


def test_release_metadata_is_synchronized() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert f'version = "{VERSION}"' in pyproject
    assert f'version: "{VERSION}"' in citation
    assert f"releases/tag/v{VERSION}" in citation
    assert f"Current release: **v{VERSION}**" in readme
    assert zenodo["version"] == VERSION
    assert f"v{VERSION}" in zenodo["description"]
    assert any(
        item["identifier"].endswith(f"/releases/tag/v{VERSION}")
        for item in zenodo["related_identifiers"]
    )


def test_week15_figure_provenance_matches_published_pngs() -> None:
    directory = ROOT / "results/week15_postaudit"
    provenance = json.loads((directory / "figure_provenance.json").read_text())
    assert len(provenance) == 6
    for item in provenance:
        path = directory / item["figure"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
