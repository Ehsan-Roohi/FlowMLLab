"""Execute the reviewed notebooks and prove retained data/results unchanged.

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
    "notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb",
    "notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb",
    "notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb",
)


def retained_hashes():
    paths = [p.relative_to(ROOT).as_posix() for folder in ("results", "data")
             for p in (ROOT / folder).rglob("*") if p.is_file()]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def execute_in_process(notebook):
    """Run trusted Python notebooks with an isolated IPython namespace, no sockets."""
    import base64
    import io
    import sys
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    shell.user_ns.update({"__name__": "__main__"})
    sys.path.insert(0, str(ROOT))
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        result.raise_error()
        cell.outputs = []
        for name in ("stdout", "stderr"):
            value = getattr(captured, name)
            if value:
                cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=value))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", bbox_inches="tight")
            cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode()}))
        plt.close("all")
        cell.execution_count = count
    InteractiveShell.clear_instance()
    return notebook


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-html", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Execute trusted local notebooks without kernel sockets")
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
        if args.in_process:
            executed = execute_in_process(notebook)
        else:
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
    report = {"execution_backend": "in_process" if args.in_process else "jupyter_kernel", "runs": results, "retained_files_checked": len(before),
              "retained_hashes_unchanged": True}
    (scratch / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
