#!/usr/bin/env python3
"""Embed freshly generated validation figures/metrics in the two owner notebooks."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "article_figures"


def cell_id(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]


def stream(text: str) -> dict:
    return {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}


def image_output(path: Path) -> dict:
    path.open("rb").close()  # explicit readability check before encoding
    return {
        "data": {
            "image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
            "text/plain": [f"<article validation figure: {path.name}>"],
        },
        "metadata": {},
        "output_type": "display_data",
    }


def table_output(record: dict) -> dict:
    frame = pd.DataFrame([record])
    return {
        "data": {
            "text/html": frame.to_html(index=False),
            "text/plain": frame.to_string(index=False).splitlines(),
        },
        "metadata": {},
        "output_type": "display_data",
    }


def set_outputs(notebook: dict, label: str, outputs: list[dict], count: int) -> None:
    target = cell_id(label)
    for cell in notebook["cells"]:
        if cell.get("id") == target:
            cell["execution_count"] = count
            cell["outputs"] = outputs
            return
    raise KeyError(f"Managed cell {label!r} was not found.")


def update_ghia() -> None:
    path = ROOT / "notebooks" / "week01" / "03_cavity_ghia.ipynb"
    notebook = json.loads(path.read_text())
    velocity = json.loads((RESULTS / "fig02_cavity_benchmark_metrics.json").read_text())
    pressure = json.loads((RESULTS / "fig08_pressure_recovery_metrics.json").read_text())
    set_outputs(notebook, "ghia-v3-bootstrap", [stream(f"Course root: {ROOT}\n")], 1)
    set_outputs(
        notebook,
        "ghia-v3-build",
        [
            image_output(RESULTS / "fig02_cavity_benchmark.png"),
            table_output(velocity),
            image_output(RESULTS / "fig08_pressure_recovery.png"),
            table_output(pressure),
            stream(
                "Paper-ready files:\n"
                f" - {RESULTS / 'fig02_cavity_benchmark.pdf'}\n"
                f" - {RESULTS / 'fig08_pressure_recovery.pdf'}\n"
            ),
        ],
        2,
    )
    set_outputs(notebook, "ghia-v3-rerun", [stream("Using retained Re=1000 N=65 and N=129 solver outputs; set the switch to rerun them.\n")], 3)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def update_dsmc() -> None:
    path = ROOT / "notebooks" / "week03" / "AI_in_Fluids_Week3_Lab2_Mini_DSMC_Cavity_Revised_Student.ipynb"
    notebook = json.loads(path.read_text())
    metrics = json.loads((RESULTS / "fig10a_mohammadzadeh_validation_metrics.json").read_text())
    set_outputs(notebook, "dsmc-v3-bootstrap", [stream(f"Course root: {ROOT}\n")], 1)
    set_outputs(
        notebook,
        "dsmc-v3-build",
        [
            image_output(RESULTS / "fig10a_mohammadzadeh_validation.png"),
            table_output(metrics),
            stream(f"Paper-ready file: {RESULTS / 'fig10a_mohammadzadeh_validation.pdf'}\n"),
        ],
        2,
    )
    set_outputs(notebook, "dsmc-v3-rerun", [stream("Using four retained completed runs; set the switch to rerun the production validation.\n")], 3)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    update_ghia()
    update_dsmc()
    print("Embedded current article-validation outputs in the Week-1 and Week-3 notebooks.")


if __name__ == "__main__":
    main()
