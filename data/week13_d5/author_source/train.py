"""Train a deep-cavity PINN: SOAP warm-up, then limited-memory SSBroyden.

    python train.py --Re 100 --D 2 --seed 12345 --out runs/d2_re100 \
            [--width 64 --depth 6] [--envelope] \
            [--n-pde 262144 --n-test 262144 --chunk 262144] \
            [--soap-steps 10000] [--steps 10000] \
            [--init weights.pth] [--resume] [--max-hours 8] \
            [--ref reference_D2_Re100.json]

TWO STAGES, and both are needed.

  SOAP      a preconditioned first-order method, run on a fresh batch every 50
            steps.  It gets the loss to ~1e-3 quickly from a random start but
            stalls there.
  SSBroyden a quasi-Newton method on a FIXED collocation set, which takes the
            loss the remaining 4-6 decades.  This is where the accuracy comes
            from; see optimizer.py.

WHAT WE LEARNED ABOUT INITIALISATION, which matters more than anything else
here: at high Reynolds number, training from scratch converges to a WRONG
steady solution -- low loss, wrong flow.  Continuation fixes it: solve at a
lower Re (or smaller aspect ratio), then restart from those weights with
--init.  Expect the solution to get worse before it gets better after such a
jump; recovery took ~10k steps in our Re 10000 -> 20000 runs.  A run stopped
too early looks like a failure when it is merely mid-recovery, which is why
the eddy structure is tracked (and logged) at every log point rather than only
at the end.

MEMORY.  The collocation set is processed in chunks of `--chunk` points, each
chunk contributing its share of the loss and calling backward immediately, so
peak memory follows the chunk size and not the point count.  Cost tracks the
CHUNK COUNT, so use the largest chunk that fits.

CHECKPOINTS are written at every log point (weights, history, optimiser store)
and `--resume` continues from them, so a long run survives a job time limit.
A non-finite loss restores the last finite weights and stops.
"""
import argparse
import json
import math
import os
import time

import torch

from cavity import (CavityNet, Geometry, MLP, residual, residual_sq_sum,
                    sample_points)
from optimizer import LMStore, SSBroydenLM
import score as scoring

# Ratio of the corner-excluded held-out residual to the training loss above
# which the run is flagged.  THIS IS THE ONE RELIABLE AUTOMATIC CHECK we have:
# on every sound run measured it sat at 0.9-1.0x the training loss, while runs
# that converged to a WRONG faint eddy gave 21x, 6950x and 106000x.  It needs no
# reference solution, so it works where nothing else does -- and it catches the
# failure that the training loss actively hides (see README, "Never select a
# model by its training loss").
FAR_RATIO_WARN = 20.0

TEST_SEED_OFFSET = 7919          # so the test set never coincides with training
SOAP_SEED_OFFSET = 500009


class ChunkedClosure:
    """Full-batch loss and gradient, evaluated in chunks.

    Returns the MEAN over the whole point set as a Python float and leaves the
    gradient in p.grad, which is what the optimiser's closure protocol wants.
    """

    def __init__(self, params, sq_sum_fn, X, chunk):
        self.params = list(params)
        self.fn = sq_sum_fn
        self.X = X
        self.N = X.shape[0]
        self.chunk = int(chunk)
        self.n_calls = 0

    def __call__(self):
        self.n_calls += 1
        for q in self.params:
            q.grad = None
        tot = 0.0
        for s0 in range(0, self.N, self.chunk):
            Lc = self.fn(self.X[s0:s0 + self.chunk]) / self.N
            Lc.backward()
            tot += float(Lc.detach())
        return tot


def test_loss(net, g, X, chunk, far_mask=None):
    """Held-out loss over all points, and over points far from the top corners."""
    s = torch.zeros((), dtype=torch.float64, device=X.device)
    s_far = torch.zeros((), dtype=torch.float64, device=X.device)
    for s0 in range(0, X.shape[0], chunk):
        sl = slice(s0, s0 + chunk)
        r1, r2 = residual(net, X[sl], g)
        e = r1.detach()[:, 0]**2 + r2.detach()[:, 0]**2
        s += e.sum()
        if far_mask is not None:
            s_far += e[far_mask[sl]].sum()
    n_far = int(far_mask.sum()) if far_mask is not None else 0
    return (float(s) / X.shape[0],
            float(s_far) / n_far if n_far else None)


def far_from_top_corners(X, g, radius=0.1):
    x, y = X[:, 0], X[:, 1]
    d = torch.minimum((x**2 + (1 - y)**2).sqrt(),
                      ((g.W - x)**2 + (1 - y)**2).sqrt())
    return d >= radius


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def write_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def flat_params(net):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def set_flat_params(net, flat):
    with torch.no_grad():
        off = 0
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[off:off + n].view_as(p))
            off += n


def track(net, g, device, nx=129):
    """A compact snapshot of the eddy structure, cheap enough for every log point."""
    st = scoring.score(net, g, nx=nx, test_n=0, device=device)["structure"]
    return dict(n_large=st["n_large"], psi=st["psi"], centres=st["centres"],
                ratios=st["ratios"], spacings=st["spacings"],
                signs_alternate=st["signs_alternate"])


def track_line(t):
    return ("%d eddies  psi %s  ratios %s"
            % (t["n_large"], ["%+.4e" % p for p in t["psi"][:4]],
               ["%.1f" % r for r in t["ratios"][:3]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--D", type=float, default=2.0, help="depth (width = W)")
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6, help="hidden layers")
    ap.add_argument("--envelope", action="store_true",
                    help="amplitude envelope; recommended for D >= 2")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--y0-env", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--n-pde", type=int, default=1 << 18)
    ap.add_argument("--n-test", type=int, default=1 << 18)
    ap.add_argument("--chunk", type=int, default=1 << 18,
                    help="points per chunk; use the largest that fits")
    ap.add_argument("--soap-steps", type=int, default=10000)
    ap.add_argument("--soap-batch", type=int, default=1 << 18)
    ap.add_argument("--soap-resample", type=int, default=50)
    ap.add_argument("--soap-lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=10000, help="SSBroyden steps")
    ap.add_argument("--m", type=int, default=0,
                    help="curvature pairs to store (0 = steps + 1; never fewer"
                         " than the steps taken -- see optimizer.py)")
    ap.add_argument("--init", default=None, help="warm start from weights.pth")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=1e9,
                    help="stop cleanly before a job time limit")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--test-every", type=int, default=500)
    ap.add_argument("--track-nx", type=int, default=129)
    ap.add_argument("--score-nx", type=int, default=257)
    ap.add_argument("--far-radius", type=float, default=0.1)
    ap.add_argument("--ref", default=None, help="reference_fd.py JSON")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    torch.set_default_dtype(torch.float64)      # float64 throughout, deliberately
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    t_start = time.time()

    def log(msg):
        print(msg, flush=True)

    kw = {} if a.kappa is None else dict(kappa=a.kappa)
    g = Geometry(W=a.W, D=a.D, Re=a.Re, envelope=a.envelope,
                 y0_env=a.y0_env, beta=a.beta, **kw)
    torch.manual_seed(a.seed)
    mlp = MLP(width=a.width, depth=a.depth).to(dev)
    net = CavityNet(mlp, g).to(dev)
    P = mlp.n_params()

    if a.init:
        sd = torch.load(a.init, map_location=dev, weights_only=True)
        if any(k.startswith("mlp.") for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}
        mlp.load_state_dict(sd)
        log("[deep] warm start from %s" % a.init)

    X = sample_points(a.n_pde, g, a.seed, dev)
    X_test = sample_points(a.n_test, g, a.seed + TEST_SEED_OFFSET, dev)
    far_test = far_from_top_corners(X_test, g, a.far_radius)
    ref = None
    if a.ref:
        with open(a.ref) as fh:
            ref = json.load(fh)

    m_store = a.m if a.m > 0 else a.steps + 1
    log("[deep] D=%g W=%g Re=%g seed=%d net=%dx%d P=%d N=%d N_test=%d "
        "chunk=%d envelope=%s store=%.2f GiB out=%s"
        % (a.D, a.W, a.Re, a.seed, a.width, a.depth, P, X.shape[0],
           X_test.shape[0], a.chunk, a.envelope,
           LMStore.bytes_for(P, m_store) / 2**30, a.out))

    hpath = os.path.join(a.out, "history.json")
    wpath = os.path.join(a.out, "weights.pth")
    opath = os.path.join(a.out, "optimizer.pth")
    spath = os.path.join(a.out, "soap.pth")
    wopath = os.path.join(a.out, "weights_opt.pth")

    hist = dict(args=vars(a), P=P, N=int(X.shape[0]), N_test=int(X_test.shape[0]),
                geometry=g.to_dict(),
                soap_step=[], soap_loss=[], soap_done=0,
                step=[], loss=[], closures=[], wall=[],
                test_step=[], test_loss=[], test_loss_far=[],
                track_step=[], track=[], stop_reason=None)
    resumed = False
    if a.resume and os.path.exists(hpath):
        with open(hpath) as f:
            old = json.load(f)
        hist.update({k: old[k] for k in hist if k in old and k != "args"})
        # A wall guard describes the previous allocation, not the resumed run.
        # Keeping it set makes a SOAP-stage restart return immediately after
        # finishing its first resumed segment.
        hist["stop_reason"] = None
        resumed = True
        log("[deep] resuming: SOAP %d steps, SSB %d steps on record"
            % (hist["soap_done"], len(hist["step"])))

    def over_wall():
        return (time.time() - t_start) / 3600.0 >= a.max_hours

    # ----------------------------------------------------------------- SOAP
    if a.soap_steps > 0 and hist["soap_done"] < a.soap_steps:
        from pytorch_optimizer.optimizer.soap import SOAP
        opt_s = SOAP(net.parameters(), lr=a.soap_lr, betas=(0.99, 0.999),
                     weight_decay=0.0, precondition_frequency=1)
        k = 0
        if resumed and hist["soap_done"] > 0 and os.path.exists(spath):
            ck = torch.load(spath, map_location=dev, weights_only=False)
            net.load_state_dict(ck["model"])
            opt_s.load_state_dict(ck["opt"])
            k = int(ck["step"])
            keep = [i for i, s in enumerate(hist["soap_step"]) if s <= k]
            hist["soap_step"] = [hist["soap_step"][i] for i in keep]
            hist["soap_loss"] = [hist["soap_loss"][i] for i in keep]
            log("[deep] SOAP resumed at step %d" % k)
        closure_s, batch_blk = None, None
        last_good = flat_params(net)
        while k < a.soap_steps:
            blk = k // a.soap_resample
            if closure_s is None or blk != batch_blk:
                # a fresh batch every --soap-resample steps, from a
                # deterministic seed so a resumed run sees the same batches
                Xb = sample_points(a.soap_batch, g,
                                   a.seed * 1000003 + SOAP_SEED_OFFSET + blk,
                                   dev)
                closure_s = ChunkedClosure(net.parameters(),
                                           residual_sq_sum(net, g), Xb, a.chunk)
                batch_blk = blk
            f = closure_s()
            if not math.isfinite(f):
                set_flat_params(net, last_good)
                hist["stop_reason"] = "nonfinite_soap"
                log("[deep] non-finite SOAP loss at step %d; restored the last "
                    "finite weights" % k)
                break
            last_good = flat_params(net)
            opt_s.step()
            k += 1
            if k % a.log_every == 0 or k == a.soap_steps:
                hist["soap_step"].append(k)
                hist["soap_loss"].append(f)
                hist["soap_done"] = k
                write_json(hist, hpath)
                atomic_save(net.state_dict(), wpath)
                atomic_save(dict(model=net.state_dict(), opt=opt_s.state_dict(),
                                 step=k), spath)
                log("[deep] SOAP step %6d loss %.6e  %.3f s/step"
                    % (k, f, (time.time() - t_start) / k))
                if over_wall():
                    hist["stop_reason"] = "wall_guard_soap"
                    break
        hist["soap_done"] = k
        del opt_s
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        atomic_save(net.state_dict(), os.path.join(a.out, "soap_final.pth"))
        write_json(hist, hpath)
        if hist["stop_reason"] is not None:
            log("[deep] stopped during SOAP: %s" % hist["stop_reason"])
            return

    # ------------------------------------------------------------ SSBroyden
    closure = ChunkedClosure(net.parameters(), residual_sq_sum(net, g), X,
                             a.chunk)
    opt = SSBroydenLM(net.parameters(), m=m_store, device=dev)

    k0 = 0
    if resumed and hist["step"] and os.path.exists(opath):
        net.load_state_dict(torch.load(wopath, map_location=dev,
                                       weights_only=True))
        opt.load_state(opath)
        k0 = opt.state["k"]
        for key in ("step", "loss", "closures", "wall"):
            del hist[key][k0:]
        keep = [i for i, s in enumerate(hist["test_step"]) if s <= k0]
        for key in ("test_step", "test_loss", "test_loss_far"):
            hist[key] = [hist[key][i] for i in keep]
        keep = [i for i, s in enumerate(hist["track_step"]) if s <= k0]
        for key in ("track_step", "track"):
            hist[key] = [hist[key][i] for i in keep]
        log("[deep] SSBroyden resumed at step %d" % k0)
    elif resumed:
        # The SSB stage is starting over -- either it never began, or its
        # optimiser checkpoint is missing (a run killed before the first log
        # point, or one stopped by the non-finite guard, which deliberately
        # does not save the store).  Either way the SOAP warm-up is still
        # valid and must be reloaded: without this the run would silently
        # restart from the RANDOM initialisation and look like a
        # non-converging optimiser rather than a lost warm start.
        soap_final = os.path.join(a.out, "soap_final.pth")
        if os.path.exists(soap_final):
            net.load_state_dict(torch.load(soap_final, map_location=dev,
                                           weights_only=True))
            log("[deep] restarting the SSB stage from soap_final.pth")
        if hist["step"]:
            log("[deep] discarding %d SSB steps with no optimiser checkpoint"
                % len(hist["step"]))
        for key in ("step", "loss", "closures", "wall"):
            hist[key] = []
        hist["test_step"], hist["test_loss"], hist["test_loss_far"] = [], [], []
        hist["track_step"], hist["track"] = [], []

    wall0 = hist["wall"][-1] if hist["wall"] else 0.0
    t0 = time.time()
    if k0 == 0:
        tl, tf = test_loss(net, g, X_test, a.chunk, far_test)
        hist["test_step"].append(0)
        hist["test_loss"].append(tl)
        hist["test_loss_far"].append(tf)
        t = track(net, g, dev, a.track_nx)
        hist["track_step"].append(0)
        hist["track"].append(t)
        log("[deep] step      0 " + track_line(t))

    def save_all(with_opt):
        write_json(hist, hpath)
        atomic_save(net.state_dict(), wpath)
        if with_opt:
            opt.save_state(opath)
            atomic_save(net.state_dict(), wopath)

    last_good = flat_params(net)
    k = k0
    while k < a.steps:
        n_before = closure.n_calls
        f = float(opt.step(lambda: torch.tensor(closure(), dtype=torch.float64,
                                                device=dev)))
        cur = flat_params(net)
        if not (math.isfinite(f) and bool(torch.isfinite(cur).all())):
            set_flat_params(net, last_good)
            hist["stop_reason"] = "nonfinite"
            log("[deep] non-finite loss/parameters at step %d; restored the "
                "step-%d weights, optimiser store NOT saved" % (k + 1, k))
            break
        last_good = cur
        k += 1
        hist["step"].append(k)
        hist["loss"].append(f)
        hist["closures"].append(closure.n_calls - n_before)
        hist["wall"].append(wall0 + time.time() - t0)

        if k % a.test_every == 0:
            tl, tf = test_loss(net, g, X_test, a.chunk, far_test)
            hist["test_step"].append(k)
            hist["test_loss"].append(tl)
            hist["test_loss_far"].append(tf)
        if k % a.log_every == 0:
            t = track(net, g, dev, a.track_nx)
            hist["track_step"].append(k)
            hist["track"].append(t)
            save_all(False)
            n = min(100, k - k0)
            far = (hist["test_loss_far"][-1]
                   if hist["test_loss_far"][-1] is not None else float("nan"))
            ratio = far / f if f > 0 else float("nan")
            log("[deep] step %6d loss %.6e test %.6e (far %.6e, %.1fx) "
                "closures/step %.2f  %.3f s/step  | %s"
                % (k, f, hist["test_loss"][-1], far, ratio,
                   sum(hist["closures"][-n:]) / max(n, 1),
                   (time.time() - t0) / max(k - k0, 1), track_line(t)))
            if ratio == ratio and ratio > FAR_RATIO_WARN:
                log("[deep] WARNING: held-out residual away from the corners is"
                    " %.0fx the training loss. The network is fitting its own"
                    " collocation points; the faint eddies are probably wrong"
                    " even though the loss looks converged. Add points, or"
                    " switch the envelope on if this is a deep cavity."
                    % ratio)
        if over_wall():
            hist["stop_reason"] = "wall_guard"
            log("[deep] wall guard at step %d; checkpointing for --resume" % k)
            break

    if opt.H.dropped:
        # see optimizer.py: truncation destroys the stored Hessian, so this
        # invalidates the run rather than merely degrading it
        log("[deep] ERROR: the curvature store dropped %d pairs (m = %d). "
            "Re-run with --m at least the number of steps."
            % (opt.H.dropped, opt.H.m))
        hist["stop_reason"] = "store_overflow"
    if hist["stop_reason"] is None:
        hist["stop_reason"] = "completed"
    hist["wall_total"] = wall0 + time.time() - t0
    save_all(hist["stop_reason"] not in ("nonfinite", "store_overflow"))

    sc = scoring.score(net, g, nx=a.score_nx, test_n=0, device=dev)
    if ref is not None:
        sc["vs_ref"] = scoring.compare_ref(sc, ref)
    fl = [v for v in hist["loss"] if math.isfinite(v)]
    sc.update(weights=wpath, seed=a.seed, width=a.width, depth=a.depth,
              envelope=a.envelope, init=a.init, soap_steps=hist["soap_done"],
              final_loss=fl[-1] if fl else None,
              final_test_loss=hist["test_loss"][-1],
              final_test_loss_far=hist["test_loss_far"][-1],
              far_radius=a.far_radius, steps_done=len(hist["step"]),
              wall_total=hist["wall_total"], stop_reason=hist["stop_reason"])
    write_json(sc, os.path.join(a.out, "score.json"))
    log("\n".join("[deep] " + s for s in scoring.summary_lines(sc, g)))
    log("[deep] done: %d SSB steps (%s), final loss %s, %.1f min"
        % (len(hist["step"]), hist["stop_reason"],
           "%.6e" % sc["final_loss"] if sc["final_loss"] else "n/a",
           hist["wall_total"] / 60))


if __name__ == "__main__":
    main()
