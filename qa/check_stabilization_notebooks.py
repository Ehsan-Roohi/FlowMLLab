"""Execute the two reviewed notebooks and prove retained data/results unchanged.

Requires nbformat, nbclient, nbconvert and ipykernel in the QA environment.
Executed notebooks and reports go to tmp; HTML is published only on explicit
--publish-html. Never rewrites the student notebook or retained results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb",
    "notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb",
)


def retained_hashes():
    paths = [p.relative_to(ROOT).as_posix() for folder in ("results", "data")
             for p in (ROOT / folder).rglob("*") if p.is_file()]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-html", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="Execute against an isolated checkout when verifying retained evidence")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    scratch = ROOT / "tmp/stabilization"
    scratch.mkdir(parents=True, exist_ok=True)
    for variable, folder in (("IPYTHONDIR", "ipython"),
                             ("JUPYTER_CONFIG_DIR", "jupyter_config"),
                             ("JUPYTER_RUNTIME_DIR", "jupyter_runtime")):
        os.environ[variable] = str(scratch / folder)
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    before = retained_hashes()
    results = []
    for filename in NOTEBOOKS:
        notebook = nbformat.read(ROOT / filename, as_version=4)
        nbformat.validate(notebook)
        started = time.perf_counter()
        executed = NotebookClient(notebook, timeout=600, kernel_name="python3",
                                  resources={"metadata": {"path": str(ROOT)}}).execute()
        elapsed = time.perf_counter() - started
        # Keep machine-specific paths out of published teaching outputs.
        for cell in executed.cells:
            for output in cell.get("outputs", []):
                if "text" in output:
                    output["text"] = output["text"].replace(str(ROOT), "<FlowMLLab checkout>")
        nbformat.write(executed, scratch / Path(filename).name)
        if args.publish_html:
            destination = ROOT / "docs/notebooks"
            destination.mkdir(parents=True, exist_ok=True)
            html, _ = HTMLExporter().from_notebook_node(executed)
            (destination / (Path(filename).stem + ".html")).write_text(html, encoding="utf-8")
        results.append({"notebook": filename, "seconds": elapsed, "status": "pass"})
        print(json.dumps(results[-1]), flush=True)
        after = retained_hashes()
        if after != before:
            changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
            raise AssertionError(f"Retained data/results changed during execution: {changed}")
    report = {"runs": results, "retained_files_checked": len(before),
              "retained_hashes_unchanged": True}
    (scratch / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
