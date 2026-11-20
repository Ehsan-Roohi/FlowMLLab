"""Run the full Week 4 Lab 2 on CPU and retain outputs plus a compact receipt."""
from pathlib import Path
import os, sys, json, time, hashlib, platform
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = ROOT / "tmp/week04_lab2"
TMP.mkdir(parents=True, exist_ok=True)
for key, part in [("MPLCONFIGDIR", "mpl"), ("IPYTHONDIR", "ipython"),
                  ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"), ("JUPYTER_CONFIG_DIR", "jupyter-config")]:
    folder = TMP/part
    folder.mkdir(exist_ok=True)
    os.environ[key] = str(folder)
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="-1",
                  PYTHONDONTWRITEBYTECODE="1", TF_CPP_MIN_LOG_LEVEL="2",
                  FLOWMLLAB_W4_LAB2_OUTPUT=str(ROOT/"results/week04_lab2"))
os.environ["FLOWMLLAB_W4_LAB2_DATA"] = str(ROOT/"data/cavity_data.npz")
os.environ["FLOWMLLAB_TRAIN_PROGRESS_JSON"] = str(TMP/"live_training.json")
os.environ["JUPYTER_PATH"] = str(Path(sys.prefix)/"share/jupyter")
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
path = ROOT/"notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb"
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
start = time.perf_counter()
client = NotebookClient(nb, timeout=7200, km=km, resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
elapsed = time.perf_counter()-start
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
figures = sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", []))
assert not errors
nbformat.validate(nb)
receipt = {"notebook": str(path.relative_to(ROOT)), "executed_on": datetime.now(timezone.utc).date().isoformat(),
           "execution_utc": datetime.now(timezone.utc).isoformat(), "execution_host": platform.node(),
           "elapsed_run_all_seconds": elapsed, "code_cells": sum(c.cell_type == "code" for c in nb.cells),
           "inline_figures": figures, "error_outputs": len(errors), "python": sys.version,
           "kernel": sys.executable, "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "executed_code_sha256": hashlib.sha256("\n".join(c.source for c in nb.cells if c.cell_type == "code").encode()).hexdigest(),
           "helper_source_sha256": hashlib.sha256((ROOT/"flowmllab/cavity_lab2.py").read_bytes()).hexdigest(),
           "cpu": platform.processor(), "tensorflow_intra_op_threads": 2,
           "claim": "Fresh complete CPU execution on the recorded host; hosted Colab execution not claimed."}
(ROOT/"qa/week04_lab2_execution.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
