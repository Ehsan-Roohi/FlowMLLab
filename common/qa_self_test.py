"""Instructor-facing smoke tests for the MIE 690A Week 5-6 pack (v1.3).

Run from the package root or from a flat track folder. TensorFlow-dependent
checks are performed only when TensorFlow is installed. Notebook cells are
located by content rather than fragile hard-coded indices, and the test uses
only Python's built-in JSON parser (no nbformat dependency).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "cavity_data.npz"
EXPECTED_SHA = "09b96b744ee4d18126d8dcc92feb60e128774a1b4d41bb3d8c90a63ccfbabc36"

sys.path.insert(0, str(HERE))
import w4utils
import w5_common
import mini_dsmc


def find_notebook(name: str) -> Path:
    candidates = [
        ROOT / "notebooks" / "week05_06" / name,
        ROOT / "Student_Notebooks" / name,
        ROOT / name,
        HERE / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def find_code_cell(nb: dict, *required_terms: str) -> tuple[int, str]:
    """Return the first code cell containing all required terms."""
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if all(term in src for term in required_terms):
            return idx, src
    raise AssertionError(f"No code cell contains all terms: {required_terms}")


def main() -> None:
    print("w4utils", w4utils.W4_UTILS_VERSION)
    print("w5_common", w5_common.W5_COMMON_VERSION)
    print("mini_dsmc", mini_dsmc.MINI_DSMC_VERSION)
    assert w4utils.W4_UTILS_VERSION == "6.3"
    assert w5_common.W5_COMMON_VERSION == "1.2"
    assert mini_dsmc.MINI_DSMC_VERSION == "1.2"

    defaults = mini_dsmc.MiniDSMCConfig()
    assert defaults.steps == 3200 and defaults.sample_start == 2400
    print("late sampling-window defaults: PASS")

    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    print("dataset_sha256", sha)
    assert sha == EXPECTED_SHA

    data = w4utils.load_dataset(DATA, search_variants=False)
    for r in (100, 275, 400):
        i = int(np.where(data["Re"].astype(float) == r)[0][0])
        m = w4utils.field_physics_metrics(data["x"], data["y"], data["u"][i], data["v"][i])
        assert m["wall_rms_error"] < 1e-12
        assert m["lid_u_mae"] < 1e-12
    print("corner-excluded truth wall metrics: PASS")

    eq = mini_dsmc.run_case(mini_dsmc.MiniDSMCConfig(
        Uwall=0.0, ppc=6, steps=1600, sample_start=800, seed=7
    ))
    mean_T = float(np.mean(eq["T"]))
    print("equilibrium mini_dsmc mean T", mean_T)
    assert 0.90 <= mean_T <= 1.10
    print("pooled temperature estimator: PASS")

    p1 = load_notebook(find_notebook("P1_Re_Generalization.ipynb"))
    p1_source = "\n".join(cell_source(c) for c in p1["cells"])
    assert "train_pointwise_fixed_epochs" in p1_source
    assert "ast.literal_eval" in p1_source
    assert "BEST=eval" not in p1_source
    print("Track-1 fair final training/static checks: PASS")

    p2_path = find_notebook("P2_Physics_Guided_DNN.ipynb")
    p2 = load_notebook(p2_path)
    p2_source = "\n".join(cell_source(c) for c in p2["cells"])
    assert "def compute_losses" in p2_source
    assert "def _losses" not in p2_source
    assert "GLOBAL_ERROR_TOL" in p2_source
    # Release-gate content matching must be checked even on the instructor build
    # machine where TensorFlow may be unavailable.  This prevents a stale search
    # token from hiding behind the TensorFlow skip branch.
    find_code_cell(p2, "import time, importlib", "assert w4utils.W4_UTILS_VERSION")
    find_code_cell(p2, "train_mask=w5_common.re_mask", "fit_standardizers")
    find_code_cell(p2, "class DivPenaltyTrainer", "def compute_losses")
    print("Track-2 Keras-name, tolerance, and content-location checks: PASS")

    # Rank-one shared POD produces one coefficient. scikit-learn returns a
    # 1-D prediction in this case, so branch_predict must reshape it before
    # StandardScaler.inverse_transform.
    re_small = np.array([100.0, 200.0, 300.0, 400.0])
    c_small = np.array([[0.0], [0.4], [0.7], [1.0]])
    branch_small = w5_common.train_branch(re_small, c_small, hidden=(4,), seed=7)
    pred_small = w5_common.branch_predict(branch_small, 250.0)
    assert np.asarray(pred_small).shape == (1,)
    print("Track-3 rank-one branch prediction smoke test: PASS")

    p3 = load_notebook(find_notebook("P3_POD_Study.ipynb"))
    p3_source = "\n".join(cell_source(c) for c in p3["cells"])
    assert "reconstruction_relative_L2_uv" in p3_source
    assert "coefficient_count" in p3_source
    assert "archive_compression" in p3_source
    print("Track-3 expanded reconstruction/compression checks: PASS")

    p5 = load_notebook(find_notebook("P5_Rarefied_Cavity.ipynb"))
    p5_source = "\n".join(cell_source(c) for c in p5["cells"])
    assert "D_single={k:np.array(v,copy=True)" in p5_source
    assert "P5_{VARIANT}_cases_v12.npz" in p5_source
    assert 'MINI_DSMC_VERSION == "1.2"' in p5_source
    print("Track-5 re-run-safe/static cache checks: PASS")

    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        print("TensorFlow smoke test: SKIPPED (TensorFlow unavailable in this runtime)")
        print("Reason:", exc)
    else:
        # Execute only the content-identified cells needed to define and fit
        # the revised custom trainer. This remains robust if markdown or cells
        # are inserted or reordered later.
        ns = {"__name__": "__qa__"}
        cell_specs = [
            ("import time, importlib", "assert w4utils.W4_UTILS_VERSION"),
            ("train_mask=w5_common.re_mask", "fit_standardizers"),
            ("class DivPenaltyTrainer", "def compute_losses"),
        ]
        for terms in cell_specs:
            idx, src = find_code_cell(p2, *terms)
            exec(compile(src, f"P2 content-matched cell {idx}", "exec"), ns)

        trainer = ns["DivPenaltyTrainer"](
            ns["w5_common"].make_dense_model(3, 3, (8, 8), "tanh", 5),
            ns["scalers"], 1e-4
        )
        trainer.compile(ns["tf"].keras.optimizers.Adam(1e-3))
        X = ns["scalers"].transform_x(ns["Xtr"][:256])
        Y = ns["scalers"].transform_y(ns["Ytr"][:256])
        Xv = ns["scalers"].transform_x(ns["Xva"][:128])
        Yv = ns["scalers"].transform_y(ns["Yva"][:128])
        hist = trainer.fit(X, Y, validation_data=(Xv, Yv), epochs=1, batch_size=64, verbose=0)
        assert "val_loss" in hist.history
        print("Track-2 Keras fit/val_loss smoke test: PASS")

    print("ALL AVAILABLE QA CHECKS PASSED")


if __name__ == "__main__":
    main()
