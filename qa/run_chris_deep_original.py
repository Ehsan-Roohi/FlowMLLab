"""Run the author-supplied deep cavity, without publishing the private sources.

The source's physics, parameter box and iteration budgets remain unchanged.
The compatibility adapter explicitly dispatches SSBroyden2 instead of silently
falling back to stock L-BFGS-B. Interrupted external phases warm-restart their
parameter vector (not their inverse Hessian); this is recorded in resume.json.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import zipfile

SOURCE_HASH = "b905a11536656780324a75111c5a1ec9a3aa135c8f49296e929528216d005cd2"
OPT_HASH = "d680a8ce4b1242d0630a3754062ead04b83e05bb6d5acaf43fd2c6e5ebfddf50"


def checked_extract(archive, target):
    names = {"CavityTrapREDepthSSB20.py", "_optimize.py", "optimizers.txt", "scipy_optimizer.txt"}
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != names:
            raise ValueError("Unexpected source archive members")
        for name in names:
            content = z.read(name)
            expected = {"CavityTrapREDepthSSB20.py": SOURCE_HASH, "_optimize.py": OPT_HASH}.get(name)
            if expected and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
            destination = target / name
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite different source: {name}")
            destination.write_bytes(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    archive = args.archive.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = out / "source"
    checked_extract(archive, src)
    os.environ["DDE_BACKEND"] = "tensorflow.compat.v1"
    import numpy as np
    import scipy
    import scipy.optimize as so
    import deepxde as dde
    import tensorflow as tf
    assert scipy.__version__ == "1.13.1", scipy.__version__
    assert tf.config.list_physical_devices("GPU"), "GPU required"
    dde.config.set_default_float("float64")
    spec = importlib.util.spec_from_file_location("scipy.optimize._chris_ssb", src / "_optimize.py")
    custom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom)
    assert "method_bfgs" in __import__("inspect").signature(custom._minimize_bfgs).parameters
    # Test the supplied implementation, not an optimizer with the same label.
    # The unmodified author's formula has an a_k=0 degeneracy at machine-epsilon
    # tolerance on tiny exactly-solvable objectives. Use a nondegenerate smoke
    # test; do not alter the original cavity's own epsilon-level tolerance.
    test = custom._minimize_bfgs(so.rosen, np.array([-1.2, 1.]),
                               jac=so.rosen_der, method_bfgs="SSBroyden2",
                               gtol=1e-8, maxiter=100)
    assert test.fun < 1e-12, test
    original = runpy.run_path(str(src / "CavityTrapREDepthSSB20.py"), run_name="author_source")
    expected = dict(ReMin=900, ReMax=1100, DMin=2.1, DMax=2.3,
                    epochsAdam=5000, epochsLBFGS=25000, NumBFGS=10)
    for k, v in expected.items():
        assert original[k] == v, (k, original[k])
    # Fail early if the installed geometry API cannot represent the original box.
    box = dde.geometry.Rectangle([0] * 5, [1] * 5)
    assert box.random_points(8).shape == (8, 5)
    print("PREFLIGHT PASS: original deep box; GPU; float64; actual SSBroyden2", flush=True)
    if args.preflight:
        return
    os.chdir(out)
    Path("model").mkdir(exist_ok=True)
    state_path = Path("resume.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": 0, "adam_done": 0}
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def record():
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)

    phase = [-1]
    original_train = dde.Model.train
    base_minimize = so.minimize

    def minimize(fun, x0, *a, **kw):
        options = dict(kw.get("options", {}))
        if "method_bfgs" not in options:
            return base_minimize(fun, x0, *a, **kw)
        assert options.pop("method_bfgs") == "SSBroyden2"
        assert options.pop("method", "BFGS") == "BFGS"
        assert kw.get("jac") is True
        assert not kw.get("constraints")
        bounds = kw.get("bounds")
        assert bounds is None or all(lo is None and hi is None for lo, hi in bounds)
        vector = Path(f"phase-{phase[0]}-vector.npy")
        if vector.exists():
            x0 = np.load(vector, allow_pickle=False)
            state["external_restart"] = "parameter warm restart; inverse Hessian reset"
            record()
        memo = custom.MemoizeJac(fun)
        previous_callback = kw.get("callback")
        iterations = [0]

        def callback(result):
            x = result.x
            iterations[0] += 1
            if previous_callback:
                previous_callback(x)
            if iterations[0] % 100 == 0 or stop[0]:
                temp = vector.with_suffix(".tmp")
                with temp.open("wb") as f:
                    np.save(f, x)
                temp.replace(vector)
            if stop[0]:
                raise SystemExit(99)

        # These L-BFGS-only options were ignored by the author's BFGS code too.
        ignored = {k: options.pop(k) for k in ("maxcor", "maxfun", "maxls", "ftol") if k in options}
        print("SSBroyden2 dispatch; unused L-BFGS options:", ignored, flush=True)
        result = custom._minimize_bfgs(memo, x0, jac=memo.derivative,
                                      callback=callback, method_bfgs="SSBroyden2", **options)
        print("SSBroyden2 result:", result.message, "loss=", result.fun, flush=True)
        if not np.isfinite(result.fun):
            raise FloatingPointError("Nonfinite external objective")
        return result

    so.minimize = minimize

    adam_offset = [0]

    class SaveAdam(dde.callbacks.Callback):
        def on_epoch_end(self):
            count = self.model.train_state.step
            if count % 500 == 0 or stop[0]:
                state["checkpoint"] = self.model.save("model/restart", verbose=0)
                state["adam_done"] = adam_offset[0] + count
                record()
                if stop[0]:
                    raise SystemExit(99)

    def train(model, *a, **kw):
        phase[0] += 1
        if phase[0] < state["phase"]:
            return model.losshistory, model.train_state
        if state.get("checkpoint"):
            kw["model_restore_path"] = state["checkpoint"]
        if phase[0] == 0:
            adam_offset[0] = state.get("adam_done", 0)
            kw["epochs"] = max(1, 5000 - state.get("adam_done", 0))
            kw["callbacks"] = list(kw.get("callbacks") or []) + [SaveAdam()]
        result = original_train(model, *a, **kw)
        state["checkpoint"] = model.save("model/restart", verbose=0)
        state["phase"] = phase[0] + 1
        record()
        if stop[0]:
            raise SystemExit(99)
        return result

    dde.Model.train = train
    Path("provenance.json").write_text(json.dumps(dict(source_sha256=SOURCE_HASH,
        optimizer_sha256=OPT_HASH, parameters=expected, protocol="original source with explicit SSB dispatch and checkpoint wrapper"), indent=2))
    original["main"]()


if __name__ == "__main__":
    main()
