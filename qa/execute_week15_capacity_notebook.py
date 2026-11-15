"""Execute capacity analysis cells using in-process IPython.

Requires nbformat and IPython. This is cell execution, not a hosted
Jupyter/Colab acceptance check, new training or a checkpoint replay.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/week15/W15_Larger_Networks_Analysis.ipynb"


def main() -> None:
    os.chdir(ROOT)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    shell = InteractiveShell.instance()
    started = time.monotonic()
    executed = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        executed += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        result.raise_error()
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = executed
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    report = {
        "scope": "Retained-summary analysis and figure display; no training or missing-checkpoint replay",
        "execution_mode": "Sequential cells in a fresh Python process using in-process IPython",
        "external_kernel_status": "Not verified: host denied socket binding for TCP and IPC Jupyter transports",
        "notebook": NOTEBOOK.relative_to(ROOT).as_posix(),
        "status": "passed",
        "code_cells_executed": executed,
        "error_outputs": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
    }
    (ROOT / "results/week15_capacity_study/notebook_execution.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
