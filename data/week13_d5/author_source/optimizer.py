"""Limited-memory self-scaling Broyden (SSBroyden), with a strong-Wolfe line search.

WHY THIS OPTIMISER.  A PINN loss has to be driven many decades down before the
weak features of a flow appear; Adam-class methods stall long before that, and
the second-order methods that do get there (BFGS, SSBroyden) normally carry a
dense P x P inverse Hessian.  At P = 21k parameters that matrix is 3.3 GB and
at P = 330k it is 813 GB -- a hard ceiling, reached well before the network is
large enough for a deep cavity.

WHAT IS STORED INSTEAD.  The dense SSBroyden update, applied to the inverse
Hessian H, is

    H <- H - u u'/q + phi q v v' ;   H <- H / tau ;   H <- H + rho s s'

with u = H y, q = y'H y, s the step, y the gradient difference.  Starting from
H_0 = I, H_k is therefore exactly

    H_k = S (sigma I + sum_j [u_j s_j] C_j [u_j s_j]')

i.e. two stored vectors and a 2x2 coefficient block per step, plus two global
scalars.  That is an IDENTITY, not an approximation -- the usual third vector
v = rho s - u/q is a linear combination of the other two.  Memory is 2 m P
doubles instead of P^2, and H v costs four (m, P) products instead of one
P x P product.

    C_uu = (phi - 1)/(q tau),   C_us = -phi rho/tau,
    C_ss = phi q rho^2/tau + rho

TRUNCATION IS NOT GRACEFUL.  H_k is an accumulated object whose generators
cancel against each other, so dropping the oldest pair does not degrade the
Hessian gently -- it destroys it (keeping 200 of 400 pairs measured a 63%
error).  `m` is therefore a CORRECTNESS parameter, not a memory knob: set it to
at least the number of steps you will take.  The store counts `dropped` and
train.py treats dropped > 0 as a configuration error.

SELF-SCALING.  tau divides the whole accumulated H every step.  Rather than
touching every stored coefficient, it is carried in one global scalar S, with
stored coefficients pre-divided by the S current when they were written, and
folded back in when S drifts past 10^+-60.

A NOTE ON REPRODUCIBILITY.  tau and theta below are defined through max/min
clamps, which makes the recursion mildly chaotic: two runs differing only in
floating-point association order drift apart over a few hundred steps.  That is
a property of SSBroyden itself, not of this limited-memory form -- comparisons
between implementations must be made on the objective reached, not by matching
trajectories step for step.
"""
import json
import math
import os

import torch

__all__ = ["strong_wolfe_line_search", "ssb_scalars", "LMStore", "SSBroydenLM"]

SAFE_DECADES = 60.0          # renormalise the global scale outside 10^+-60


# ---------------------------------------------------------------------------
# Strong-Wolfe line search (zoom by bisection)
# ---------------------------------------------------------------------------
def _phi_and_derphi(eval_fg, xk, pk, alpha):
    f, g = eval_fg(xk + alpha * pk)
    return f, g, torch.dot(g, pk)


def _zoom(eval_fg, xk, pk, phi0, derphi0, c1, c2, alo, ahi, phi_alo,
          derphi_alo, maxiter=20):
    for _ in range(maxiter):
        aj = 0.5 * (alo + ahi)
        phi_aj, g_aj, derphi_aj = _phi_and_derphi(eval_fg, xk, pk, aj)
        if (phi_aj > phi0 + c1 * aj * derphi0) or (phi_aj >= phi_alo):
            ahi = aj
        else:
            if abs(float(derphi_aj)) <= -c2 * float(derphi0):
                return aj, phi_aj, g_aj
            if float(derphi_aj) * (ahi - alo) >= 0:
                ahi = alo
            alo, phi_alo, derphi_alo = aj, phi_aj, derphi_aj
    phi_alo2, g_alo2, _ = _phi_and_derphi(eval_fg, xk, pk, alo)
    return alo, phi_alo2, g_alo2


def strong_wolfe_line_search(eval_fg, xk, pk, gfk, old_fval, c1=1e-4, c2=0.9,
                             amax=None, maxiter=10, zoom_maxiter=20,
                             alpha1=1.0):
    """Returns (alpha, x_new, f_new, g_new).

    eval_fg(x) -> (loss, flat gradient).  If the proposed direction is not a
    descent direction (which the clamped SSBroyden update can produce), it
    falls back to steepest descent for that step.
    """
    derphi0 = torch.dot(gfk, pk)
    if (not torch.isfinite(derphi0)) or float(derphi0) >= 0.0:
        pk = -gfk
        derphi0 = -torch.dot(gfk, gfk)
    phi0 = old_fval
    alpha0 = 0.0
    alpha1 = float(alpha1)
    if amax is not None:
        alpha1 = min(alpha1, float(amax))
    phi_a0, derphi_a0 = phi0, derphi0
    phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (i > 0 and phi_a1 >= phi_a0):
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0, c1, c2,
                                  alpha0, alpha1, phi_a0, derphi_a0,
                                  maxiter=zoom_maxiter)
            return float(a_s), xk + a_s * pk, p_s, g_s
        if abs(float(derphi_a1)) <= -c2 * float(derphi0):
            return float(alpha1), xk + alpha1 * pk, phi_a1, g_a1
        if float(derphi_a1) >= 0.0:
            a_s, p_s, g_s = _zoom(eval_fg, xk, pk, phi0, derphi0, c1, c2,
                                  alpha1, alpha0, phi_a1, derphi_a1,
                                  maxiter=zoom_maxiter)
            return float(a_s), xk + a_s * pk, p_s, g_s
        alpha0, phi_a0, derphi_a0 = alpha1, phi_a1, derphi_a1
        alpha1 *= 2.0
        if amax is not None:
            alpha1 = min(alpha1, float(amax))
        phi_a1, g_a1, derphi_a1 = _phi_and_derphi(eval_fg, xk, pk, alpha1)
    return float(alpha1), xk + alpha1 * pk, phi_a1, g_a1


# ---------------------------------------------------------------------------
# The SSBroyden scalars
# ---------------------------------------------------------------------------
def ssb_scalars(P, ys, ykHkyk, sg, alpha, eps=1e-30):
    """(rho, phi, tau) for one update.

    ys = y's, ykHkyk = y'Hy, sg = s'g, all Python floats; alpha is the step
    length the line search accepted.  This is the self-scaling Broyden family
    with the standard clamps on theta and tau.
    """
    rhok = 1.0 / ys
    hk = ykHkyk * rhok
    bk = -alpha * rhok * sg
    ak = bk * hk - 1.0
    denom = (1.0 + ak) or eps
    rad = max(abs(ak) / denom, 0.0)
    rhokm = min(1.0, hk * (1.0 - math.sqrt(rad)))

    thetakm = 0.0 if abs(ak) < eps else (rhokm - 1.0) / ak
    thetakp = (1.0 / rhokm) if abs(rhokm) > eps else (1.0 / eps)
    inner = thetakp if abs(bk) < eps else (1.0 - bk) / bk
    thetak = max(thetakm, min(thetakp, inner))

    rhokk = min(1.0, (1.0 / bk) if abs(bk) > eps else (1.0 / eps))
    sigmak = 1.0 + thetak * ak
    expo = 1.0 / (1.0 - float(P))
    sigmaknm1 = abs(sigmak) ** expo if abs(sigmak) > 0 else 0.0

    tauk = (min(rhokk * sigmaknm1, sigmak) if thetak <= 0.0
            else rhokk * min(sigmaknm1, 1.0 / thetak))

    denom3 = (1.0 + ak * thetak) or eps
    phik = (1.0 - thetak) / denom3
    tauk = tauk if abs(tauk) > eps else (eps if tauk >= 0 else -eps)
    return rhok, phik, tauk


# ---------------------------------------------------------------------------
# The store: H as generators
# ---------------------------------------------------------------------------
class LMStore:
    """H = S (sigma I + sum_j [u_j s_j] C_j [u_j s_j]'), two vectors per step."""

    def __init__(self, P, m, device, dtype=torch.float64):
        self.P, self.m = int(P), int(m)
        self.device, self.dtype = torch.device(device), dtype
        self.U = torch.zeros(self.m, self.P, device=self.device, dtype=dtype)
        self.Sv = torch.zeros(self.m, self.P, device=self.device, dtype=dtype)
        self.cuu = torch.zeros(self.m, device=self.device, dtype=dtype)
        self.cus = torch.zeros(self.m, device=self.device, dtype=dtype)
        self.css = torch.zeros(self.m, device=self.device, dtype=dtype)
        self.reset()

    def reset(self):
        self.n = 0              # pairs held
        self.ptr = 0            # next slot
        self.S = 1.0            # global scale
        self.sigma = 1.0        # coefficient of I
        self.n_renorm = 0
        self.dropped = 0        # > 0 means m was too small: results are wrong
        self.n_updates = 0

    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.U, self.Sv, self.cuu, self.cus, self.css))

    @staticmethod
    def bytes_for(P, m, itemsize=8):
        """Store size before allocating it: 2 m P doubles plus 3 m scalars."""
        return m * (2 * P + 3) * itemsize

    def matvec(self, v):
        """H v."""
        out = v * self.sigma
        if self.n:
            U, Sv = self.U[:self.n], self.Sv[:self.n]
            a, b = U @ v, Sv @ v
            out = out + (U.t() @ (self.cuu[:self.n] * a + self.cus[:self.n] * b)
                         + Sv.t() @ (self.cus[:self.n] * a
                                     + self.css[:self.n] * b))
        return out * self.S

    def update(self, sk, yk, gk, alpha, eps=1e-30):
        """One SSBroyden update from the step sk and gradient difference yk."""
        ys = torch.dot(yk, sk).item()
        if abs(ys) < eps:
            return False
        Hkyk = self.matvec(yk)
        ykHkyk = torch.dot(yk, Hkyk).item()
        if abs(ykHkyk) < eps:
            return False
        rho, phi, tau = ssb_scalars(self.P, ys, ykHkyk,
                                    torch.dot(sk, gk).item(), alpha, eps)
        self._push(Hkyk, sk, rho, ykHkyk, phi, tau)
        return True

    def _push(self, u, s, rho, q, phi, tau):
        """Write one (u, s) pair and its 2x2 block; fold tau into the scale."""
        S_old = self.S
        self.U[self.ptr] = u
        self.Sv[self.ptr] = s
        self.cuu[self.ptr] = ((phi - 1.0) / q) / S_old
        self.cus[self.ptr] = (-phi * rho) / S_old
        self.S = S_old / tau
        # phi q rho^2 s s' is added BEFORE the division by tau, rho s s' after,
        # so the two s s' contributions carry different scales.
        self.css[self.ptr] = (phi * q * rho * rho) / S_old + rho / self.S

        self.ptr = (self.ptr + 1) % self.m
        if self.n < self.m:
            self.n += 1
        else:
            self.dropped += 1       # see the module docstring: this is fatal
        s_abs = abs(self.S)
        if s_abs and (s_abs > 10.0 ** SAFE_DECADES
                      or s_abs < 10.0 ** -SAFE_DECADES):
            self._renorm()
        self.n_updates += 1

    def _renorm(self):
        """Fold the global scale back into the stored coefficients."""
        if self.n:
            for c in (self.cuu, self.cus, self.css):
                c[:self.n] *= self.S
        self.sigma *= self.S
        self.S = 1.0
        self.n_renorm += 1

    # -- persistence, so a run can be continued in a later job ---------------
    def state_dict(self):
        """Rows in ring order, oldest first.

        Note this materialises the buffers; at the default sizes the store is a
        few GB.  If you push m and P much higher, write it out in blocks.
        """
        if self.n < self.m:
            order = list(range(self.n))
        else:
            order = [(self.ptr + i) % self.m for i in range(self.m)]
        idx = torch.tensor(order, dtype=torch.long, device=self.device)
        return dict(P=self.P, n=self.n, S=self.S, sigma=self.sigma,
                    n_renorm=self.n_renorm, dropped=self.dropped,
                    n_updates=self.n_updates,
                    U=self.U[idx].cpu(), Sv=self.Sv[idx].cpu(),
                    cuu=self.cuu[idx].cpu(), cus=self.cus[idx].cpu(),
                    css=self.css[idx].cpu())

    def load_state_dict(self, sd):
        assert int(sd["P"]) == self.P, "checkpoint is for a different network"
        n = int(sd["n"])
        assert n <= self.m, "store too small for this checkpoint (raise m)"
        self.reset()
        self.n = n
        self.ptr = n % self.m
        self.S, self.sigma = float(sd["S"]), float(sd["sigma"])
        self.n_renorm, self.dropped = int(sd["n_renorm"]), int(sd["dropped"])
        self.n_updates = int(sd["n_updates"])
        self.U[:n] = sd["U"].to(self.device)
        self.Sv[:n] = sd["Sv"].to(self.device)
        self.cuu[:n] = sd["cuu"].to(self.device)
        self.cus[:n] = sd["cus"].to(self.device)
        self.css[:n] = sd["css"].to(self.device)


class SSBroydenLM(torch.optim.Optimizer):
    """SSBroyden over the limited-memory store.  One GPU, full-batch closure.

    Usage mirrors torch.optim.LBFGS: `step(closure)`, where the closure
    re-evaluates the loss and populates .grad.
    """

    def __init__(self, params, m, lr=1.0, gtol=1e-12, c1=1e-4, c2=0.9,
                 wolfe_maxiter=20, zoom_maxiter=40, amax=10, eps=1e-30,
                 dtype=torch.float64, device=None):
        super().__init__(params, defaults={})
        self.lr, self.gtol = lr, gtol
        self.c1, self.c2 = c1, c2
        self.wolfe_maxiter, self.zoom_maxiter = wolfe_maxiter, zoom_maxiter
        self.amax, self.eps, self.dtype = amax, eps, dtype
        self._params = list(self.param_groups[0]["params"])
        self.device = device if device is not None else self._params[0].device
        self._numels = [p.numel() for p in self._params]
        self._P = int(sum(self._numels))
        self.H = LMStore(self._P, m, self.device, dtype)
        self.state["k"] = 0
        self.n_ls_evals = 0

    def _gather_flat_params(self):
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    def _set_flat_params_(self, flat):
        with torch.no_grad():
            off = 0
            for p, n in zip(self._params, self._numels):
                p.copy_(flat[off:off + n].view_as(p))
                off += n

    def _gather_flat_grad(self):
        return torch.cat([
            p.grad.detach().reshape(-1) if p.grad is not None
            else torch.zeros(n, device=self.device, dtype=self.dtype)
            for p, n in zip(self._params, self._numels)])

    def _eval_loss_and_grad(self, closure):
        loss = closure()
        if not torch.is_tensor(loss):
            loss = torch.tensor(loss, device=self.device, dtype=self.dtype)
        return loss.detach(), self._gather_flat_grad()

    def h_bytes(self):
        return self.H.nbytes()

    def save_state(self, path):
        """Store + counters.  Model weights are the caller's responsibility."""
        tmp = path + ".tmp"
        torch.save(dict(store=self.H.state_dict(), k=self.state["k"],
                        n_ls_evals=self.n_ls_evals, m=self.H.m), tmp)
        os.replace(tmp, path)

    def load_state(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.H.load_state_dict(ck["store"])
        self.state["k"] = int(ck["k"])
        self.n_ls_evals = int(ck["n_ls_evals"])

    def step(self, closure):
        self.zero_grad(set_to_none=True)
        fk, gk = self._eval_loss_and_grad(closure)
        xk = self._gather_flat_params().detach()
        if torch.linalg.vector_norm(gk).item() <= self.gtol:
            return fk
        pk = -self.H.matvec(gk)

        def eval_fg_at(x_flat):
            self.n_ls_evals += 1
            self._set_flat_params_(x_flat)
            self.zero_grad(set_to_none=True)
            return self._eval_loss_and_grad(closure)

        alpha, xkp1, fkp1, gkp1 = strong_wolfe_line_search(
            eval_fg_at, xk, pk, gk, fk, c1=self.c1, c2=self.c2,
            amax=self.amax, maxiter=self.wolfe_maxiter,
            zoom_maxiter=self.zoom_maxiter, alpha1=float(self.lr))
        self._set_flat_params_(xkp1)
        self.H.update(xkp1 - xk, gkp1 - gk, gk, alpha, eps=self.eps)
        self.state["k"] += 1
        return fkp1
