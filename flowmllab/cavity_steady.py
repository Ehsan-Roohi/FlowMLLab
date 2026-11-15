"""Steady square-cavity Navier--Stokes solver, independent of Ghia data.

Second-order central differences and Thom wall vorticity match the Week-4
spatial equations. Damped Newton solves those equations directly. A converged
root is a 2-D steady branch, not evidence that physical time evolution is stable.
All arrays use [y, x]; Re=U_lid L/nu, L=U_lid=1 in the production study.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.fft import dstn, idstn
from scipy.interpolate import RegularGridInterpolator


class SteadyCavity:
    def __init__(self, cells: int, lid: float = 1.0):
        if int(cells) != cells or cells < 8:
            raise ValueError('cells must be an integer >= 8')
        self.cells, self.lid = int(cells), float(lid)
        self.m = self.cells - 1
        self.n = self.m**2
        self.h = 1.0 / self.cells
        m, h = self.m, self.h
        eye = sparse.eye(m, format='csr')
        t = sparse.diags([np.ones(m-1), -2*np.ones(m), np.ones(m-1)],
                         [-1, 0, 1], format='csr') / h**2
        d = sparse.diags([-np.ones(m-1), np.ones(m-1)], [-1, 1], format='csr') / (2*h)
        self.L = sparse.kron(eye, t, format='csr') + sparse.kron(t, eye, format='csr')
        self.Dx = sparse.kron(eye, d, format='csr')
        self.Dy = sparse.kron(d, eye, format='csr')
        b_lap = np.zeros((m, m))
        b_x = np.zeros((m, m))
        b_y = np.zeros((m, m))
        b_lap[:, 0] -= 2/h**4
        b_lap[:, -1] -= 2/h**4
        b_lap[0, :] -= 2/h**4
        b_lap[-1, :] -= 2/h**4
        b_x[:, 0] += 1/h**3
        b_x[:, -1] -= 1/h**3
        b_y[0, :] += 1/h**3
        b_y[-1, :] -= 1/h**3
        self.B = sparse.diags(b_lap.ravel(), format='csr')
        self.Bx = sparse.diags(b_x.ravel(), format='csr')
        self.By = sparse.diags(b_y.ravel(), format='csr')
        forcing_lap = np.zeros((m, m))
        forcing_y = np.zeros((m, m))
        forcing_lap[-1, :] = -2*self.lid/h**3
        forcing_y[-1, :] = -self.lid/h**2
        self.forcing_lap = forcing_lap.ravel()
        self.forcing_y = forcing_y.ravel()
        self.eye = sparse.eye(self.n, format='csr')
        k = np.arange(1, m+1)
        lam = 2*(np.cos(np.pi*k/self.cells)-1)/h**2
        self.poisson_eigenvalues = lam[:, None] + lam[None, :]

    def residual(self, z, reynolds):
        if reynolds <= 0:
            raise ValueError('Re must be positive')
        p, w = np.asarray(z)[:self.n], np.asarray(z)[self.n:]
        u, v = self.Dy @ p, -(self.Dx @ p)
        wx = self.Dx @ w + self.Bx @ p
        wy = self.Dy @ w + self.By @ p + self.forcing_y
        lapw = self.L @ w + self.B @ p + self.forcing_lap
        return np.r_[self.L @ p + w, lapw/reynolds-u*wx-v*wy]

    def jacobian(self, z, reynolds):
        p, w = np.asarray(z)[:self.n], np.asarray(z)[self.n:]
        u, v = self.Dy @ p, -(self.Dx @ p)
        wx = self.Dx @ w + self.Bx @ p
        wy = self.Dy @ w + self.By @ p + self.forcing_y
        diag = lambda a: sparse.diags(a, format='csr')
        jp = self.B/reynolds - diag(wx) @ self.Dy + diag(wy) @ self.Dx
        jp = jp - diag(u) @ self.Bx - diag(v) @ self.By
        jw = self.L/reynolds - diag(u) @ self.Dx - diag(v) @ self.Dy
        return sparse.bmat([[self.L, self.eye], [jp, jw]], format='csc')

    def solve(self, initial, reynolds, tol=1e-8, max_newton=35, callback=None):
        z = np.array(initial, dtype=float, copy=True)
        if z.shape != (2*self.n,) or not np.isfinite(z).all():
            raise ValueError('Invalid initial state')
        history = []
        for iteration in range(max_newton+1):
            f = self.residual(z, reynolds)
            poisson = float(np.max(np.abs(f[:self.n])))
            steady = float(np.max(np.abs(f[self.n:])))
            record = dict(iteration=iteration, poisson_linf=poisson, steady_vorticity_linf=steady)
            history.append(record)
            if callback is not None:
                callback(z, record)
            if max(poisson, steady) < tol:
                return z, history
            if iteration == max_newton:
                break
            jac = self.jacobian(z, reynolds)
            step = spsolve(jac, -f, permc_spec='COLAMD')
            if not np.isfinite(step).all():
                raise FloatingPointError('Nonfinite Newton step')
            linear_defect = np.linalg.norm(jac @ step+f)/max(np.linalg.norm(f), 1e-30)
            record['linear_relative_residual'] = float(linear_defect)
            if linear_defect > 1e-5:
                raise ArithmeticError(f'Inaccurate Newton linear solve: {linear_defect}')
            merit = np.linalg.norm(f)
            for reduction in range(20):
                alpha = 0.5**reduction
                candidate = z+alpha*step
                trial = self.residual(candidate, reynolds)
                if np.isfinite(trial).all() and np.linalg.norm(trial) <= (1-1e-4*alpha)*merit:
                    z = candidate
                    record['accepted_step'] = alpha
                    break
            else:
                raise ArithmeticError(f'Newton line search failed: Re={reynolds}, residual={steady}')
        raise ArithmeticError(f'Newton limit: Re={reynolds}, residual={steady}')

    def pack(self, psi, omega):
        expected = (self.cells+1, self.cells+1)
        if np.shape(psi) != expected or np.shape(omega) != expected:
            raise ValueError('State grid mismatch')
        return np.r_[np.asarray(psi)[1:-1, 1:-1].ravel(),
                     np.asarray(omega)[1:-1, 1:-1].ravel()]

    def wall_vorticity(self, psi, omega):
        h = self.h
        omega[0, 1:-1] = -2*psi[1, 1:-1]/h**2
        omega[-1, 1:-1] = -2*psi[-2, 1:-1]/h**2-2*self.lid/h
        omega[1:-1, 0] = -2*psi[1:-1, 1]/h**2
        omega[1:-1, -1] = -2*psi[1:-1, -2]/h**2
        omega[0, 0] = .5*(omega[0, 1]+omega[1, 0])
        omega[0, -1] = .5*(omega[0, -2]+omega[1, -1])
        omega[-1, 0] = .5*(omega[-1, 1]+omega[-2, 0])
        omega[-1, -1] = .5*(omega[-1, -2]+omega[-2, -1])
        return omega

    def fields(self, z):
        shape = (self.cells+1, self.cells+1)
        psi, omega = np.zeros(shape), np.zeros(shape)
        psi[1:-1, 1:-1] = np.asarray(z)[:self.n].reshape(self.m, self.m)
        omega[1:-1, 1:-1] = np.asarray(z)[self.n:].reshape(self.m, self.m)
        self.wall_vorticity(psi, omega)
        u, v = np.zeros(shape), np.zeros(shape)
        u[1:-1, 1:-1] = (psi[2:, 1:-1]-psi[:-2, 1:-1])/(2*self.h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:]-psi[1:-1, :-2])/(2*self.h)
        u[-1, 1:-1] = self.lid
        axis = np.linspace(0, 1, self.cells+1)
        return dict(x=axis, y=axis.copy(), psi=psi, omega=omega, u=u, v=v)

    def interpolate_seed(self, source):
        axis = np.linspace(0, 1, self.cells+1)
        Y, X = np.meshgrid(axis, axis, indexing='ij')
        points = np.column_stack([Y.ravel(), X.ravel()])
        values = [RegularGridInterpolator((source['y'], source['x']), source[key])
                  (points).reshape(Y.shape) for key in ('psi', 'omega')]
        return self.pack(*values)

    def time_rhs(self, interior_omega, reynolds):
        """Physical-time RHS with a fresh Poisson and wall solve at every stage."""
        w = np.asarray(interior_omega).reshape(self.m, self.m)
        psi = idstn(dstn(-w, type=1, norm='ortho')/self.poisson_eigenvalues,
                    type=1, norm='ortho')
        z = np.r_[psi.ravel(), w.ravel()]
        return self.residual(z, reynolds)[self.n:]

    def rk3(self, interior_omega, reynolds, dt):
        w = np.asarray(interior_omega)
        a = w+dt*self.time_rhs(w, reynolds)
        b = .75*w+.25*(a+dt*self.time_rhs(a, reynolds))
        return (w+2*(b+dt*self.time_rhs(b, reynolds)))/3
