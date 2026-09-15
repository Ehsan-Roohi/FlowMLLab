"""Small, independent audit utilities for the attributed Week 14 case.

These functions are FlowMLLab diagnostics, not a replacement for pyCALC-RANS.
"""
import numpy as np


def closure_features(profile, re_tau=5200.0):
    """Reconstruct the released c_k training inputs (u_tau=delta=1).

    Column order: y, U, k, omega, uv. Omega is outer-scaled, not omega+.
    First-order gradient endpoints and the 0.995 cap match the source script.
    No logarithm, scaling, training split or target is applied here.
    """
    a = np.asarray(profile, dtype=float)
    if a.ndim != 2 or a.shape[1] != 5 or len(a) < 3:
        raise ValueError('Expected at least three rows of y,U,k,omega,uv')
    if not np.isfinite(a).all() or not np.isfinite(re_tau) or re_tau <= 0:
        raise ValueError('Finite data and positive Re_tau are required')
    y,u,k,omega,_ = a.T
    if np.any(y <= 0) or np.any(np.diff(y) <= 0) or np.any(omega <= 0) or np.any(k < 0):
        raise ValueError('Require increasing positive y, positive omega and nonnegative k')
    nut = k / omega
    shear = np.minimum(np.abs((1/re_tau + nut)*np.gradient(u,y)), .995)
    return np.column_stack([nut/y, shear])


def relative_l2(prediction, reference, weights=None):
    """Relative L2 with explicit optional nonnegative quadrature weights."""
    p,r = np.asarray(prediction,float), np.asarray(reference,float)
    if p.shape != r.shape or p.size == 0 or not np.isfinite([p,r]).all():
        raise ValueError('Finite arrays of equal nonempty shape required')
    w = np.ones_like(r) if weights is None else np.asarray(weights,float)
    if w.shape != r.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Weights must be finite, nonnegative and match the data')
    denominator = np.sum(w*r*r)
    if denominator <= 0:
        raise ValueError('Reference norm must be positive')
    return float(np.sqrt(np.sum(w*(p-r)**2)/denominator))


def cell_widths(y, upper=1.0):
    """Midpoint diagnostic quadrature, not the original finite-volume mesh."""
    y = np.asarray(y,float)
    if y.ndim != 1 or len(y)<2 or not np.isfinite(y).all() or not (0<y[0]<y[-1]<upper) or np.any(np.diff(y)<=0):
        raise ValueError('Require ordered interior half-channel coordinates')
    return np.diff(np.r_[0, .5*(y[:-1]+y[1:]), upper])
