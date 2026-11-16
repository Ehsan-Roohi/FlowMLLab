"""Deep lid-driven cavity: geometry, boundary conditions, and the PDE residual.

The cavity is the box

    x in [0, W],    y in [y0, 1],    y0 = 1 - D,

with the lid at y = 1 moving in +x.  D/W is the aspect ratio; D = 1 is the
classical square cavity, D = 2 the first deep case, and nothing here assumes
either.

FORMULATION (stream function + pressure, boundary conditions satisfied exactly)

    psi = psi_lid + B^2 E(y) N_psi(x, y)
    u   =  d(psi)/dy
    v   = -d(psi)/dx
    p   = E(y) N_p(x, y)

where (N_psi, N_p) are the two outputs of the network and

    B       = 16 xh (1 - xh) eta (1 - eta),   xh = x/W,  eta = (y - y0)/D
    psi_lid = (y - 1) eta^2 (1 - e^{-(x-W)^2/dx^2})(1 - e^{-x^2/dx^2})
              e^{-(1-y)^2/dy^2}

B vanishes on all four walls, so B^2 kills both psi and its normal derivative
there: no-slip holds EXACTLY for any network weights, and no boundary loss term
is needed.  psi_lid supplies the lid motion: d/dy[(y-1) eta^2] = 1 at y = 1, so
u = 1 on the lid away from the corners, for any depth D.

CONTINUITY IS EXACT: u and v are derivatives of a single scalar psi, so
div(u, v) = psi_yx - psi_xy = 0 identically.  The loss is therefore only the two
momentum residuals -- there is no continuity term to weight against anything.

THE ENVELOPE E(y) is what makes a DEEP cavity trainable; see `envelope` below.
For a square cavity use envelope=False, where E == 1 and every expression here
collapses to the plain formulation.
"""
import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Physical / numerical constants of the regularised lid.
#
# The exact lid-driven cavity has a velocity discontinuity where the moving lid
# meets each stationary side wall, and psi is singular there.  We smooth it,
# which is what makes a collocation method converge cleanly:
#
#   dx  width over which the lid accelerates from 0 at each top corner to its
#       full speed.  sqrt(1e-3) ~ 0.0316: u_lid = 0.63 at 0.032 from a corner,
#       0.98 at 0.063.  This is the one place our problem differs from the
#       standard benchmarks, which use a discontinuous lid.
#   dy  depth over which psi_lid decays into the interior (a lid layer ~0.1
#       deep).  It localises the analytic term; the network supplies the rest.
# ---------------------------------------------------------------------------
DX_LID = math.sqrt(1.0e-3)
DY_LID = math.sqrt(1.0e-2)

# Residual mask radius: the momentum residual is multiplied by a smooth factor
# that vanishes at the two TOP corners, so the (regularised but still stiff)
# corner region cannot dominate the loss.  At 0.02 -- the closest a collocation
# point ever gets -- the mask is already 1 - e^{-4} = 0.982, so it changes the
# loss by at most ~2% in a thin ring.  The corner is handled mainly by the lid
# smoothing above and by the exclusion radius below.
MASK_RADIUS = 1e-2

# Collocation points closer than this to a corner are dropped.  Motivated by
# the TOP corners (see above).  Note the consequence at the BOTTOM corners:
# the innermost Moffatt corner eddies live inside this radius and are therefore
# never sampled -- see README "Caveats".
EXCL_RADIUS = 2e-2

# Moffatt decay rate for a parallel-walled channel, kappa per unit width.
# Successive corner/stacked eddies weaken by a factor ~357.7 over a spacing of
# 1.3958 widths, so kappa = ln(357.7)/1.3958 = 4.212.  (Root of
# 2ka + sin 2ka = 0, ka = 2.10620 + 1.12536i; Moffatt 1964, JFM 18:1.)
KAPPA_MOFFATT = 4.212

__all__ = ["Geometry", "MLP", "CavityNet", "lid_terms", "envelope",
           "fields", "residual", "residual_sq_sum", "sample_points"]


class Geometry:
    """Cavity dimensions, Reynolds number, and the envelope parameters.

    envelope=False gives E == 1 and dE/dy == 0 identically, which is the plain
    square-cavity formulation; use it for D = 1 and as the control at D = 2.
    """

    def __init__(self, W=1.0, D=2.0, Re=100.0,
                 envelope=False, kappa=KAPPA_MOFFATT, y0_env=0.25, beta=10.0,
                 dx=DX_LID, dy=DY_LID,
                 mask_radius=MASK_RADIUS, excl_radius=EXCL_RADIUS):
        self.W, self.D, self.Re = float(W), float(D), float(Re)
        self.y0 = 1.0 - self.D                      # floor
        self.dx, self.dy = dx, dy
        self.mask_radius, self.excl_radius = mask_radius, excl_radius
        self.envelope = bool(envelope)
        self.kappa = float(kappa)
        self.y0_env = float(y0_env)
        self.beta = float(beta)

    def corners(self, device="cpu"):
        """The four corners, used for collocation exclusion."""
        return torch.tensor([[0.0, self.y0], [self.W, self.y0],
                             [0.0, 1.0], [self.W, 1.0]],
                            dtype=torch.float64, device=device)

    def to_dict(self):
        return dict(W=self.W, D=self.D, Re=self.Re, dx=self.dx, dy=self.dy,
                    mask_radius=self.mask_radius, excl_radius=self.excl_radius,
                    envelope=self.envelope, kappa=self.kappa,
                    y0_env=self.y0_env, beta=self.beta)


class MLP(nn.Module):
    """Plain tanh MLP, Xavier-normal weights, zero biases.  `depth` = hidden layers."""

    def __init__(self, width=64, depth=6, in_dim=2, out_dim=2):
        super().__init__()
        dims = [in_dim] + [width] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b)
                                    for a, b in zip(dims[:-1], dims[1:]))
        self.last = nn.Linear(width, out_dim)
        for m in list(self.hidden) + [self.last]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z):
        for lin in self.hidden:
            z = torch.tanh(lin(z))
        return self.last(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class CavityNet(nn.Module):
    """The MLP, taking (x, y) as raw coordinates."""

    def __init__(self, mlp, geom):
        super().__init__()
        self.mlp, self.geom = mlp, geom

    def forward(self, x, y):
        return self.mlp(torch.cat([x, y], dim=1))


def lid_terms(x, y, g):
    """B, its two derivatives, psi_lid and its two derivatives -- all analytic.

    Returned as a tuple so the caller can build psi and its derivatives without
    asking autograd for anything it can get in closed form; only the NETWORK
    output needs differentiating.  That keeps the momentum residual at three
    nested autograd levels rather than four.
    """
    W, D, dx, dy = g.W, g.D, g.dx, g.dy
    xh = x / W
    eta = (y - g.y0) / D
    bcv = 16 * xh * (1 - xh) * eta * (1 - eta)
    ExpB = torch.exp(-(1 - y)**2 / dy**2)
    e1 = torch.exp(-(x - W)**2 / dx**2)
    e0 = torch.exp(-x**2 / dx**2)

    psilid = (y - 1) * eta**2 * (1 - e1) * (1 - e0) * ExpB
    psilid_x = ((y - 1) * eta**2
                * (2 * (x - W) / dx**2 * e1)
                * (1 - e0) * ExpB
                + (y - 1) * eta**2
                * (1 - e1)
                * (2 * x / dx**2 * e0) * ExpB)
    psilid_y = ((eta**2 + 2 * eta * (y - 1) / D)
                * (1 - e1)
                * (1 - e0) * ExpB
                + psilid * 2 * (1 - y) / dy**2)

    dbcv_x = 16 * (1 - 2 * xh) * eta * (1 - eta) / W
    dbcv_y = 16 * xh * (1 - xh) * (1 - 2 * eta) / D
    return bcv, dbcv_x, dbcv_y, psilid, psilid_x, psilid_y


def _softplus(t, beta):
    """log1p(e^{beta t})/beta, written as logaddexp(beta t, 0)/beta.

    TWO OBVIOUS ALTERNATIVES ARE WRONG HERE, both silently:

      * torch.nn.functional.softplus switches to an exactly linear branch once
        beta*t > 20, where its derivative is 1 instead of sigmoid(beta t);
      * the max(t, 0) + log1p(e^{-beta|t|}) identity has a clamp/abs kink at
        t = 0, where autograd returns a subgradient (0) rather than 1/2 -- a
        50% derivative error on the line y = 1 - y0, which is exactly where a
        grid-based scorer puts sample points.

    logaddexp is smooth and stable everywhere and its derivative is exactly
    sigmoid(beta t), so the analytic dE/dy below agrees with autograd to
    round-off.  This is not cosmetic: the momentum residual differentiates
    through this expression three times, so a wrong derivative here corrupts
    the loss without ever raising an error.  selftest.py checks it.
    """
    return torch.logaddexp(beta * t, torch.zeros_like(t)) / beta


def envelope(y, g):
    """(E, dE/dy): the amplitude envelope that makes a deep cavity trainable.

    WHY IT EXISTS.  In a deep cavity the flow is a stack of counter-rotating
    eddies, each roughly 357.7x weaker than the one above it (Moffatt 1964).
    At D = 2 that is ~3 decades in psi; at D = 5, ~7 decades.  A plain
    mean-squared residual is dominated by the top eddy: the deep ones are far
    below the level the loss is fitting, and the network is free to leave them
    wrong.  Rescaling fixes both halves of the problem:

        psi = psi_lid + B^2 E(y) N,   loss = mean[(R_x/E)^2 + (R_y/E)^2]

    so the network's own output stays O(1) at every depth, and each eddy
    contributes to the gradient on an equal footing.  Below the lid region the
    viscous and pressure terms of R scale like E and the inertial term like
    E^2, so R/E is O(1) all the way down.

    THE FORM.  With d = 1 - y the depth below the lid,

        E(d) = exp(-kappa [s(d - y0_env) - s(-y0_env)]),   s = softplus_beta

    E = 1 near the lid and decays at the Moffatt rate kappa per width below
    the knee at d = y0_env.  The -s(-y0_env) shift makes E(d = 0) = 1 EXACTLY,
    so the top eddy is parameterised exactly as in the unscaled formulation; a
    raw softplus leaves a ~3% offset there.

    Since s'(t) = sigmoid(beta t) and dd/dy = -1,

        dE/dy = +kappa sigmoid(beta (d - y0_env)) E >= 0

    (E grows towards the lid).  E underflows float64 only at kappa*d > 708,
    i.e. ~168 widths deep, so the envelope itself does not limit depth.

    y0_env depends weakly on Re (the lid eddy reaches deeper as Re grows):
    ~0.25 at Re <= 100, ~0.6 at Re = 1000.  Fit it if you move far in Re.
    """
    if not g.envelope:
        one = torch.ones_like(y)
        return one, torch.zeros_like(y)
    t = (1.0 - y) - g.y0_env
    t0 = torch.full_like(y[:1], -g.y0_env)
    s = _softplus(t, g.beta) - _softplus(t0, g.beta)
    E = torch.exp(-g.kappa * s)
    E_y = g.kappa * torch.sigmoid(g.beta * t) * E
    return E, E_y


def _uvp(net, x, y, g):
    """(u, v, p, psi) WITH the autograd graph attached.  x, y need grad."""
    outputs = net(x, y)
    N = outputs[:, 0:1]
    bcv, dbcv_x, dbcv_y, psilid, psilid_x, psilid_y = lid_terms(x, y, g)
    N_x, N_y = torch.autograd.grad(N, (x, y), torch.ones_like(N),
                                   create_graph=True)
    E, E_y = envelope(y, g)

    # psi = psilid + bcv^2 E N, differentiated by the product rule
    psi_x = psilid_x + 2 * bcv * dbcv_x * E * N + bcv**2 * E * N_x
    psi_y = (psilid_y + 2 * bcv * dbcv_y * E * N + bcv**2 * E_y * N
             + bcv**2 * E * N_y)
    u = psi_y
    v = -psi_x
    p = E * outputs[:, 1:2]
    psi = psilid + bcv**2 * E * N
    return u, v, p, psi


def fields(net, X, g):
    """(u, v, p, psi) at the points X, detached.  For plotting and scoring."""
    x = X[:, 0:1].clone().detach().requires_grad_(True)
    y = X[:, 1:2].clone().detach().requires_grad_(True)
    with torch.enable_grad():
        u, v, p, psi = _uvp(net, x, y, g)
    return u.detach(), v.detach(), p.detach(), psi.detach()


def residual(net, X, g, Re=None):
    """The two steady incompressible momentum residuals at X.

        R_x = u u_x + v u_y - (1/Re) lap(u) + p_x
        R_y = u v_x + v v_y - (1/Re) lap(v) + p_y

    multiplied by the smooth top-corner mask, and divided by E when the
    envelope is on.  Continuity needs no residual: it is exact by construction.
    """
    Re = g.Re if Re is None else Re
    x = X[:, 0:1].clone().detach().requires_grad_(True)
    y = X[:, 1:2].clone().detach().requires_grad_(True)
    u, v, p, _psi = _uvp(net, x, y, g)

    du_x, du_y = torch.autograd.grad(u, (x, y), torch.ones_like(u),
                                     create_graph=True)
    dv_x, dv_y = torch.autograd.grad(v, (x, y), torch.ones_like(v),
                                     create_graph=True)
    du_xx = torch.autograd.grad(du_x, x, torch.ones_like(du_x),
                                create_graph=True)[0]
    du_yy = torch.autograd.grad(du_y, y, torch.ones_like(du_y),
                                create_graph=True)[0]
    dv_xx = torch.autograd.grad(dv_x, x, torch.ones_like(dv_x),
                                create_graph=True)[0]
    dv_yy = torch.autograd.grad(dv_y, y, torch.ones_like(dv_y),
                                create_graph=True)[0]
    dp_x, dp_y = torch.autograd.grad(p, (x, y), torch.ones_like(p),
                                     create_graph=True)

    res1 = u * du_x + v * du_y - (1 / Re) * (du_xx + du_yy) + dp_x
    res2 = u * dv_x + v * dv_y - (1 / Re) * (dv_xx + dv_yy) + dp_y

    radius = g.mask_radius
    d2_tl = x**2 + (1.0 - y)**2
    d2_tr = (g.W - x)**2 + (1.0 - y)**2
    mask = ((1.0 - torch.exp(-d2_tl / radius**2))
            * (1.0 - torch.exp(-d2_tr / radius**2)))
    if g.envelope:
        E, _ = envelope(y, g)
        mask = mask / E
    return mask * res1, mask * res2


def residual_sq_sum(net, g, Re=None):
    """Returns f(X_chunk) -> sum of squared residuals over the chunk.

    The chunk SUM (not mean) is what the chunked closure in train.py needs: it
    divides by the global point count itself, so chunking cannot change the
    loss.
    """
    def fn(Xc):
        r1, r2 = residual(net, Xc, g, Re)
        return (r1**2).sum() + (r2**2).sum()
    return fn


def sample_points(n, g, seed, device="cpu"):
    """n uniform random points in the cavity, corner neighbourhoods removed.

    Drawn on the CPU from a seeded generator so a run is reproducible and
    device-independent.  Points within excl_radius of any corner are rejected
    and redrawn, so the returned set has exactly n points.
    """
    gen = torch.Generator().manual_seed(int(seed))
    corners = g.corners("cpu")
    pts, total = [], 0
    while total < n:
        X = torch.rand(n, 2, generator=gen, dtype=torch.float64)
        X[:, 0] *= g.W
        X[:, 1] = g.y0 + g.D * X[:, 1]
        dmin = torch.cdist(X, corners).min(dim=1).values
        X = X[dmin > g.excl_radius]
        pts.append(X)
        total += X.shape[0]
    return torch.cat(pts, dim=0)[:n].to(device)
