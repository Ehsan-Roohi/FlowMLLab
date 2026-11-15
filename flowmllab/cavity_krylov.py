"""Experimental Newton linear solvers; the cavity PDE and outer Newton are unchanged.

A coupled geometric Galerkin V-cycle uses bilinear transfer for both interior
fields, ILUT pre/post smoothing, and a sparse direct coarsest solve. GMRES sees
an explicitly row-equilibrated Jacobian; acceptance also checks the original
unscaled linear equation. This is a benchmark candidate, not a speedup claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, spilu, splu, spsolve


@dataclass(frozen=True)
class KrylovConfig:
    coarse_cells: int = 16
    ilu_drop_tol: float = 1e-3
    ilu_fill_factor: float = 4.0
    pre_sweeps: int = 1
    post_sweeps: int = 1
    gmres_restart: int = 50
    gmres_cycles: int = 12
    scaled_rtol: float = 1e-9
    original_relative_gate: float = 1e-7
    refinement_limit: int = 3


def prolongation(cells: int):
    """Bilinear interior correction transfer; homogeneous boundary correction."""
    if cells % 2 or cells < 4:
        raise ValueError('Nested grids require an even cell count >= 4')
    mc = cells // 2 - 1
    rows, cols, data = [], [], []
    for i in range(1, cells):
        if i % 2 == 0:
            rows.append(i-1); cols.append(i//2-1); data.append(1.)
        else:
            for j in (i//2, i//2+1):
                if 1 <= j <= mc:
                    rows.append(i-1); cols.append(j-1); data.append(.5)
    p1 = sparse.csr_matrix((data, (rows, cols)), shape=(cells-1, mc))
    p2 = sparse.kron(p1, p1, format='csr')
    return sparse.block_diag((p2, p2), format='csr')


class CoupledVcycle:
    def __init__(self, matrix, cells, config):
        self.levels = []
        a = sparse.csc_matrix(matrix)
        while cells > config.coarse_cells:
            smoother = spilu(a, drop_tol=config.ilu_drop_tol,
                             fill_factor=config.ilu_fill_factor,
                             permc_spec='COLAMD')
            p = prolongation(cells)
            r = (.25*p.T).tocsr()
            self.levels.append((a, p, r, smoother))
            a = (r @ a @ p).tocsc()
            cells //= 2
        self.coarse = splu(a, permc_spec='COLAMD')
        self.coarse_shape = a.shape
        self.config = config
        self.applications = 0
        self.apply_seconds = 0.

    def _cycle(self, level, b):
        if level == len(self.levels):
            return self.coarse.solve(b)
        a, p, r, ilu = self.levels[level]
        x = np.zeros_like(b)
        for _ in range(self.config.pre_sweeps):
            x += ilu.solve(b-a@x)
        x += p @ self._cycle(level+1, r@(b-a@x))
        for _ in range(self.config.post_sweeps):
            x += ilu.solve(b-a@x)
        return x

    def solve(self, b):
        start = perf_counter()
        value = self._cycle(0, b)
        self.applications += 1
        self.apply_seconds += perf_counter()-start
        return value


class LinearSolver:
    """Callable matching cavity_steady.spsolve, with per-Newton timing evidence."""
    def __init__(self, method, cells, config=None):
        if method not in ('direct', 'gmres_mg', 'gmres_ilu'):
            raise ValueError(method)
        self.method, self.cells = method, cells
        self.config = config or KrylovConfig()
        self.records = []

    def __call__(self, a, b, **kwargs):
        start = perf_counter()
        record = dict(method=self.method, matrix_shape=list(a.shape), nnz=int(a.nnz))
        self.records.append(record)
        try:
            if self.method == 'direct':
                x = spsolve(a, b, **kwargs)
                record.update(setup_seconds=None, inner_iterations=0)
            else:
                cfg = self.config
                # Balance the second-derivative relation L psi + omega = 0.
                # This invertible variable change does not alter the physical equations.
                columns = np.r_[np.ones(a.shape[0]//2), np.full(a.shape[0]//2, self.cells**2)]
                balanced = (a@sparse.diags(columns)).tocsc()
                scale = 1/np.maximum(np.asarray(abs(balanced).max(axis=1).toarray()).ravel(), 1e-30)
                scaled = (sparse.diags(scale)@balanced).tocsc()
                record['variable_scaling'] = 'omega-column scale=cells^2, followed by row max norm equilibration'
                setup_start = perf_counter()
                if self.method == 'gmres_mg':
                    preconditioner = CoupledVcycle(scaled, self.cells, cfg)
                    apply = preconditioner.solve
                    record['levels'] = len(preconditioner.levels)+1
                    record['coarse_unknowns'] = preconditioner.coarse_shape[0]
                else:
                    preconditioner = spilu(scaled, drop_tol=cfg.ilu_drop_tol,
                                          fill_factor=cfg.ilu_fill_factor, permc_spec='COLAMD')
                    apply = preconditioner.solve
                record['setup_seconds'] = perf_counter()-setup_start
                m = LinearOperator(a.shape, matvec=apply, dtype=float)
                x = np.zeros_like(b)
                records = []
                for refinement in range(cfg.refinement_limit):
                    rhs = b-a@x
                    inner_history = []
                    correction, info = gmres(scaled, scale*rhs, M=m,
                        rtol=cfg.scaled_rtol, atol=0., restart=cfg.gmres_restart,
                        maxiter=cfg.gmres_cycles, callback=inner_history.append,
                        callback_type='pr_norm')
                    x += columns*correction
                    defect = float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b), 1e-30))
                    records.append(dict(info=int(info), iterations=len(inner_history),
                        original_relative_defect=defect,
                        preconditioned_residuals=[float(v) for v in inner_history]))
                    if not np.isfinite(x).all():
                        raise FloatingPointError('Nonfinite Krylov step')
                    if info == 0 and defect <= cfg.original_relative_gate:
                        break
                else:
                    raise ArithmeticError('Krylov solve did not meet the original-equation gate')
                record['krylov'] = records
                record['inner_iterations'] = sum(v['iterations'] for v in records)
                if self.method == 'gmres_mg':
                    record['preconditioner_applications'] = preconditioner.applications
                    record['preconditioner_apply_seconds'] = preconditioner.apply_seconds
            record['original_relative_defect'] = float(np.linalg.norm(a@x-b)/max(np.linalg.norm(b), 1e-30))
            record['success'] = True
            return x
        except Exception as exc:
            record.update(success=False, error=f'{type(exc).__name__}: {exc}')
            if 'records' in locals():
                record['krylov'] = records
                record['inner_iterations'] = sum(v['iterations'] for v in records)
            raise
        finally:
            record['total_seconds'] = perf_counter()-start


def config_record():
    return asdict(KrylovConfig())
