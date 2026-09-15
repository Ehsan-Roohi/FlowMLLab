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
import math
import os
from pathlib import Path
import runpy
import signal
import sys
import zipfile

SOURCE_HASH = "b905a11536656780324a75111c5a1ec9a3aa135c8f49296e929528216d005cd2"
OPT_HASH = "d680a8ce4b1242d0630a3754062ead04b83e05bb6d5acaf43fd2c6e5ebfddf50"
DEEP_CASES = [(100, 5), (500, 5), (1000, 5), (500, 7), (1000, 7)]


def restart_safe_saveplot(saveplot):
    """Skip author export calls before any training in a restarted process.

    A skipped phase has no in-memory loss rows. DeepXDE cannot concatenate
    that empty history; exporting it would also overwrite retained evidence.
    Nonempty histories still use the unmodified exporter and propagate errors.
    """
    def save(history, train_state, *args, **kwargs):
        if len(history.steps) == 0:
            print("RESTART: skipping empty history export before resumed training", flush=True)
            return None
        return saveplot(history, train_state, *args, **kwargs)
    return save


def external_plateau(previous_loss, current_loss, message):
    """Detect an SSB phase that cannot improve the stored objective."""
    return (
        previous_loss is not None
        and math.isclose(current_loss, previous_loss, rel_tol=1e-13, abs_tol=0.0)
        and "precision loss" in str(message).lower()
    )


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
    p.add_argument("--case-index", type=int, choices=range(5))
    p.add_argument("--depth", type=float, help="Declared continuation depth")
    p.add_argument("--reynolds", type=float, help="Declared continuation Reynolds number")
    p.add_argument("--ssb-phases", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--fixed-sampling", action="store_true",
                   help="Keep deterministic collocation points across phases and restarts")
    p.add_argument("--refine-from", type=Path,
                   help="Complete checkpoint prefix used to start a separate refinement run")
    p.add_argument("--lower-anchors", type=int, default=0,
                   help="Extra residual points biased toward the lower 60% of the cavity")
    p.add_argument("--refine-adam-lr", type=float, default=1e-4)
    p.add_argument("--refine-adam-steps", type=int, default=5000,
                   help="Short restart-safe Adam budget for a refinement run")
    p.add_argument("--refine-ssb-steps", type=int, default=18000,
                   help="SSB iterations per restart-safe refinement phase")
    args = p.parse_args()
    explicit_case = args.depth is not None or args.reynolds is not None
    if explicit_case and (args.depth is None or args.reynolds is None or args.case_index is not None):
        p.error("Supply both --depth and --reynolds, without --case-index")
    if explicit_case and not (math.isfinite(args.depth) and args.depth > .1
                              and math.isfinite(args.reynolds) and args.reynolds > 0):
        p.error("Depth must exceed 0.1 and Reynolds number must be positive and finite")
    if args.ssb_phases < 1 or args.refine_ssb_steps < 1:
        p.error("SSB phases and steps must be positive")
    if args.refine_adam_steps < 1:
        raise ValueError("--refine-adam-steps must be positive")
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
    dde.config.set_random_seed(args.seed)
    expected = dict(ReMin=900, ReMax=1100, DMin=2.1, DMax=2.3,
                    epochsAdam=5000, epochsLBFGS=25000, NumBFGS=10)
    for k, v in expected.items():
        assert original[k] == v, (k, original[k])
    if args.case_index is not None or explicit_case:
        re, depth = ((args.reynolds, args.depth) if explicit_case else DEEP_CASES[args.case_index])
        expected.update(ReMin=.9*re, ReMax=1.1*re, DMin=depth-.1, DMax=depth+.1)
        expected["NumBFGS"] = args.ssb_phases
        if args.refine_from or explicit_case:
            expected["epochsLBFGS"] = args.refine_ssb_steps
        # run_path may return a copy; functions resolve the defining globals.
        original['main'].__globals__.update(expected)
        for function in ('main', 'pde', 'output_transform_cavity_flow'):
            assert all(original[function].__globals__[k] == v for k, v in expected.items())
    config_path = out / 'case-parameters.json'
    if config_path.exists() and json.loads(config_path.read_text()) != expected:
        raise ValueError('Refusing to reuse checkpoints with different physical parameters')
    config_path.write_text(json.dumps(expected, indent=2))
    training_config = dict(seed=args.seed, fixed_sampling=args.fixed_sampling,
                           adam_steps=args.refine_adam_steps,
                           adam_lr=args.refine_adam_lr if args.refine_from else original['lr'],
                           lower_anchors=args.lower_anchors)
    training_path = out / 'training-configuration.json'
    if training_path.exists() and json.loads(training_path.read_text()) != training_config:
        raise ValueError('Refusing to resume with different training settings')
    training_path.write_text(json.dumps(training_config, indent=2))
    # Fail early if the installed geometry API cannot represent the original box.
    box = dde.geometry.Rectangle([0] * 5, [1] * 5)
    assert box.random_points(8).shape == (8, 5)
    print("PREFLIGHT PASS: GPU; float64; actual SSBroyden2; parameters", expected, flush=True)
    if args.preflight:
        return
    os.chdir(out)
    Path("model").mkdir(exist_ok=True)
    state_path = Path("resume.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"phase": 0, "adam_done": 0}
    if args.refine_from and not state_path.exists():
        prefix = args.refine_from.resolve()
        for suffix in (".index", ".meta", ".data-00000-of-00001"):
            if not Path(str(prefix) + suffix).is_file():
                raise FileNotFoundError(str(prefix) + suffix)
        state["checkpoint"] = str(prefix)
        state["refinement"] = {
            "lower_anchors": args.lower_anchors,
            "adam_lr": args.refine_adam_lr,
            "adam_steps": args.refine_adam_steps,
            "ssb_steps_per_phase": args.refine_ssb_steps,
            "sampling": "deterministic target-case anchors; eta=0.6*Beta(1,2)",
        }
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
        previous_loss = state.get("last_external_loss")
        stalled = external_plateau(previous_loss, result.fun, result.message)
        state["last_external_loss"] = float(result.fun)
        if stalled:
            state["external_stalled"] = True
            state["external_stop_reason"] = (
                "SSBroyden2 repeated the previous objective and reported precision loss"
            )
            print("SSBroyden2 plateau detected; later external phases will be skipped.",
                  flush=True)
        record()
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
        if phase[0] > 0 and state.get("external_stalled"):
            state["phase"] = phase[0] + 1
            record()
            print(f"Skipping external phase {phase[0]}: "
                  f"{state['external_stop_reason']}", flush=True)
            return model.losshistory, model.train_state
        if state.get("checkpoint"):
            kw["model_restore_path"] = state["checkpoint"]
        if phase[0] == 0:
            adam_offset[0] = state.get("adam_done", 0)
            kw["epochs"] = max(1, args.refine_adam_steps - state.get("adam_done", 0))
            kw["callbacks"] = list(kw.get("callbacks") or []) + [SaveAdam()]
        if args.fixed_sampling:
            kw["callbacks"] = [cb for cb in kw.get("callbacks", [])
                               if not isinstance(cb, dde.callbacks.PDEPointResampler)]
        result = original_train(model, *a, **kw)
        state["checkpoint"] = model.save("model/restart", verbose=0)
        state["phase"] = phase[0] + 1
        record()
        if stop[0]:
            raise SystemExit(99)
        return result

    dde.Model.train = train
    dde.saveplot = restart_safe_saveplot(dde.saveplot)
    if args.lower_anchors:
        if args.lower_anchors < 1000:
            raise ValueError("Use at least 1000 refinement anchors")
        if args.case_index is None and not explicit_case:
            raise ValueError("Refinement anchors require declared case parameters")
        base_pde_init = dde.data.PDE.__init__

        def pde_init(instance, *a, **kw):
            base_pde_init(instance, *a, **kw)
            rng = np.random.default_rng(20260914)
            anchors = rng.random((args.lower_anchors, 5))
            anchors[:, 1] = 0.6 * rng.beta(1.0, 2.0, args.lower_anchors)
            anchors[:, 2] = (
                re - expected["ReMin"]
            ) / (expected["ReMax"] - expected["ReMin"])
            anchors[:, 3] = (
                depth - expected["DMin"]
            ) / (expected["DMax"] - expected["DMin"])
            anchors[:, 4] = (0.0 - original["triMin"]) / (original["triMax"] - original["triMin"])
            instance.add_anchors(anchors)

        dde.data.PDE.__init__ = pde_init
    if args.refine_from:
        base_compile = dde.Model.compile

        def compile_refinement(model, optimizer, *a, **kw):
            if str(optimizer).lower() == "adam":
                kw["lr"] = args.refine_adam_lr
            return base_compile(model, optimizer, *a, **kw)

        dde.Model.compile = compile_refinement
    Path("provenance.json").write_text(json.dumps(dict(source_sha256=SOURCE_HASH,
        optimizer_sha256=OPT_HASH, parameters=expected,
        refinement=state.get("refinement"),
        seed=args.seed, fixed_sampling=args.fixed_sampling,
        protocol="author source with explicit SSB dispatch and checkpoint wrapper; declared parameter-box adaptation" if args.case_index is not None else "original source with explicit SSB dispatch and checkpoint wrapper"), indent=2))
    original["main"]()


if __name__ == "__main__":
    main()
