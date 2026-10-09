"""Week 1.2: inspectable finite-volume SIMPLE, PISO and PIMPLE at Re=100.

Independent implementation, informed by the pressure-correction derivations in
OpenFOAM and the educational PySIMPLE/FiPy examples. No external CFD solver is
called. All three algorithms share conservative MAC-grid momentum operators.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.fft import dstn, idstn


@dataclass(frozen=True)
class CavityConfig:
    method: str = 'simple'
    cells: int = 32
    reynolds: float = 100.0
    convection: str = 'central'
    dt: float = 0.05
    velocity_relaxation: float = 0.7
    pressure_relaxation: float = 0.3
    pressure_correctors: int = 2
    outer_correctors: int = 3
    tolerance: float = 1e-7
    max_iterations: int = 4000
    check_every: int = 10
    end_time: float | None = None

    def __post_init__(self):
        if self.method not in ('simple', 'piso', 'pimple'):
            raise ValueError('method must be simple, piso or pimple')
        if self.cells < 4 or self.reynolds != 100:
            raise ValueError('This qualified teaching release requires cells >= 4 and Re=100')
        if self.convection not in ('central', 'upwind'):
            raise ValueError('convection must be central or upwind')
        if not (0 < self.velocity_relaxation <= 1 and 0 < self.pressure_relaxation <= 1):
            raise ValueError('Relaxation factors must lie in (0,1]')
        if self.dt <= 0 or self.tolerance <= 0 or min(self.pressure_correctors, self.outer_correctors, self.max_iterations, self.check_every) < 1:
            raise ValueError('Steps, tolerances and iteration counts must be positive')
        if self.end_time is not None and (self.end_time <= 0 or self.method == 'simple'):
            raise ValueError('A physical end_time is supported only for PISO/PIMPLE')


def divergence(u, v, h):
    """Finite-volume divergence of face velocities, on pressure cells."""
    return (u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / h


def pressure_gradient(p, h):
    return np.diff(p, axis=1)/h, np.diff(p, axis=0)/h


def correct_velocity(u, v, p, du, dv, h):
    gx, gy = pressure_gradient(p, h)
    out_u, out_v = u.copy(), v.copy()
    out_u[:, 1:-1] -= du*gx
    out_v[1:-1, :] -= dv*gy
    return out_u, out_v


def pressure_matrix(du, dv, h):
    """Positive operator -D d G with impermeable walls and constant nullspace."""
    n = du.shape[0]
    e = np.zeros((n, n)); e[:, :-1] = du/h**2
    w = np.zeros((n, n)); w[:, 1:] = du/h**2
    north = np.zeros((n, n)); north[:-1, :] = dv/h**2
    south = np.zeros((n, n)); south[1:, :] = dv/h**2
    return _stencil_matrix(e+w+north+south, e, w, north, south)


def _stencil_matrix(ap, east, west, north, south):
    ny, nx = ap.shape
    idx = np.arange(ny*nx).reshape(ny, nx)
    rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [ap.ravel()]
    for a, b, coefficient in [(idx[:, :-1], idx[:, 1:], east[:, :-1]),
                               (idx[:, 1:], idx[:, :-1], west[:, 1:]),
                               (idx[:-1, :], idx[1:, :], north[:-1, :]),
                               (idx[1:, :], idx[:-1, :], south[1:, :])]:
        rows.append(a.ravel()); cols.append(b.ravel()); vals.append(-coefficient.ravel())
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny*nx, ny*nx)).tocsc()


def _pressure_solver(du, dv, h):
    matrix = pressure_matrix(du, dv, h)
    # Pin cell 0 during the solve; shift to zero mean for reported pressure.
    lu = splu(matrix[1:, 1:].tocsc())
    def solve(rhs):
        if abs(float(rhs.sum())) > 1e-9*max(1., float(np.abs(rhs).sum())):
            raise ValueError('Closed-cavity pressure RHS violates compatibility')
        p = np.zeros(rhs.size)
        p[1:] = lu.solve(rhs.ravel()[1:])
        p = p.reshape(rhs.shape)
        return p - p.mean()
    return solve


def _momentum(u, v, config, old_u=None, old_v=None, dt=None, relaxation=1.):
    """Assemble normalized FV equations A q = wall/time source - G p.

    Diffusion and convection are implicit for the frozen advecting velocities.
    Tangential wall values use half-cell distances. Central interpolation is
    second-order on this uniform mesh; first-order upwind is an optional control.
    """
    n = config.cells; h = 1./n; diffusion = 1./config.reynolds/h**2
    ue = .5*(u[:, 1:-1]+u[:, 2:]); uw = .5*(u[:, :-2]+u[:, 1:-1])
    vn = .5*(v[1:, :-1]+v[1:, 1:]); vs = .5*(v[:-1, :-1]+v[:-1, 1:])
    uve = .5*(u[:-1, 1:]+u[1:, 1:]); uvw = .5*(u[:-1, :-1]+u[1:, :-1])
    vvn = .5*(v[1:-1, :]+v[2:, :]); vvs = .5*(v[:-2, :]+v[1:-1, :])
    blocks = []
    for component, (fe, fw, fn, fs) in enumerate([(ue, uw, vn, vs), (uve, uvw, vvn, vvs)]):
        if config.convection == 'central':
            east = diffusion-fe/(2*h); west = diffusion+fw/(2*h)
            north = diffusion-fn/(2*h); south = diffusion+fs/(2*h)
        else:
            east = diffusion+np.maximum(-fe/h, 0); west = diffusion+np.maximum(fw/h, 0)
            north = diffusion+np.maximum(-fn/h, 0); south = diffusion+np.maximum(fs/h, 0)
        ap = east+west+north+south+(fe-fw+fn-fs)/h
        b = np.zeros_like(ap)
        if component == 0:
            ap[[0, -1], :] += diffusion
            b[-1, :] = 2*diffusion  # moving lid speed = 1
            q = u[:, 1:-1]; old = None if old_u is None else old_u[:, 1:-1]
        else:
            ap[:, [0, -1]] += diffusion
            q = v[1:-1, :]; old = None if old_v is None else old_v[1:-1, :]
        if dt is not None:
            ap += 1/dt
            b += old/dt
        b += (1/relaxation-1)*ap*q
        ap /= relaxation
        blocks.append((_stencil_matrix(ap, east, west, north, south), b, ap))
    return blocks


def _residual(u, v, p, config, old_u=None, old_v=None, dt=None):
    blocks = _momentum(u, v, config, old_u, old_v, dt)
    gradients = pressure_gradient(p, 1/config.cells)
    values = [u[:, 1:-1], v[1:-1, :]]
    errors = [a@q.ravel() - (b-g).ravel() for (a,b,_), q,g in zip(blocks, values, gradients)]
    # Units U_lid^2/L, with U_lid=L=1. This is not update/Delta t.
    return float(max(np.max(np.abs(e)) for e in errors))


def coupling_step(u, v, p, config, dt=None):
    """One SIMPLE iteration or one physical PISO/PIMPLE time step.

    PISO recomputes off-diagonal momentum contributions between pressure
    corrections. PIMPLE additionally rebuilds and solves momentum in its outer
    loop, always keeping the preceding physical-time fields immutable.
    """
    n = config.cells; h = 1/n
    old_u, old_v = u.copy(), v.copy()
    transient = config.method != 'simple'
    step_dt = (config.dt if dt is None else dt) if transient else None
    outer = config.outer_correctors if config.method == 'pimple' else 1
    n_pressure_solves = 0
    for _ in range(outer):
        relaxation = config.velocity_relaxation if not transient else 1.
        blocks = _momentum(u, v, config, old_u, old_v, step_dt, relaxation)
        gx, gy = pressure_gradient(p, h)
        u[:, 1:-1] = splu(blocks[0][0]).solve((blocks[0][1]-gx).ravel()).reshape(n,n-1)
        v[1:-1, :] = splu(blocks[1][0]).solve((blocks[1][1]-gy).ravel()).reshape(n-1,n)
        du = 1/blocks[0][2]; dv = 1/blocks[1][2]
        solve_pressure = _pressure_solver(du, dv, h)
        if not transient:
            pc = solve_pressure(-divergence(u, v, h))
            u, v = correct_velocity(u, v, pc, du, dv, h)
            p = p + config.pressure_relaxation*pc
            p -= p.mean()
            n_pressure_solves += 1
        else:
            for _ in range(config.pressure_correctors):
                # H(q) = b - offdiag(A) q. Re-evaluation is the extra PISO
                # momentum correction, not repeated iterations of one Poisson solve.
                hu = blocks[0][1].ravel() - (blocks[0][0]-diags(blocks[0][2].ravel()))@u[:,1:-1].ravel()
                hv = blocks[1][1].ravel() - (blocks[1][0]-diags(blocks[1][2].ravel()))@v[1:-1,:].ravel()
                trial_u = np.zeros_like(u); trial_v = np.zeros_like(v)
                trial_u[:,1:-1] = du*hu.reshape(n,n-1)
                trial_v[1:-1,:] = dv*hv.reshape(n-1,n)
                p = solve_pressure(-divergence(trial_u, trial_v, h))
                u, v = correct_velocity(trial_u, trial_v, p, du, dv, h)
                n_pressure_solves += 1
    return u, v, p, n_pressure_solves


def run_cavity(config=CavityConfig(), progress=False):
    n = config.cells; h = 1/n
    u = np.zeros((n,n+1)); v = np.zeros((n+1,n)); p = np.zeros((n,n))
    history = []; start = perf_counter(); time = 0.; pressure_solves = 0
    converged = False; reached_end = False
    for iteration in range(1,config.max_iterations+1):
        old_u, old_v = u.copy(), v.copy()
        dt = config.dt
        if config.end_time is not None:
            dt = min(dt, config.end_time-time)
        u,v,p,count = coupling_step(u,v,p,config,dt)
        pressure_solves += count
        if config.method != 'simple': time += dt
        if not all(np.isfinite(a).all() for a in [u,v,p]):
            raise FloatingPointError('Non-finite field; no completed result may be reported')
        reached_end = config.end_time is not None and time >= config.end_time-1e-12
        if iteration % config.check_every == 0 or reached_end or iteration == 1:
            steady_residual = _residual(u,v,p,config)
            continuity = float(np.max(np.abs(divergence(u,v,h))))
            step_residual = _residual(u,v,p,config,old_u,old_v,dt) if config.method != 'simple' else steady_residual
            history.append([iteration,time,perf_counter()-start,steady_residual,continuity,step_residual])
            if progress: print(config.method,n,iteration,f'Rmom={steady_residual:.3e}',f'Div={continuity:.3e}',flush=True)
            converged = steady_residual < config.tolerance and continuity < config.tolerance
            if reached_end or (config.end_time is None and converged): break
    elapsed = perf_counter()-start
    result = {'config':asdict(config),'u_faces':u,'v_faces':v,'p':p,
        'u':.5*(u[:,:-1]+u[:,1:]),'v':.5*(v[:-1,:]+v[1:,:]),
        'x':(np.arange(n)+.5)*h,'y':(np.arange(n)+.5)*h,
        'history':np.asarray(history),'converged':bool(converged),
        'reached_end_time':bool(reached_end),'iterations':iteration,'physical_time':time if config.method!='simple' else None,
        'runtime_seconds':elapsed,'pressure_solves':pressure_solves,
        'momentum_residual_linf':_residual(u,v,p,config),
        'continuity_linf':float(np.max(np.abs(divergence(u,v,h))))}
    result.update(derived_fields(u,v,h))
    result.update(ghia_errors(result))
    return result


def derived_fields(u,v,h):
    n = u.shape[0]
    # Nodal streamfunction from integrating u = d psi/dy. The same divergence
    # free MAC field also satisfies v = -d psi/dx to pressure-solve precision.
    psi = np.zeros((n+1,n+1)); psi[1:,:] = np.cumsum(u*h,axis=0)
    uc = .5*(u[:,:-1]+u[:,1:]); vc = .5*(v[:-1,:]+v[1:,:])
    omega = np.gradient(vc,h,axis=1,edge_order=2)-np.gradient(uc,h,axis=0,edge_order=2)
    j,i = np.unravel_index(np.argmin(psi),psi.shape)
    return {'psi':psi,'omega':omega,'vortex_x':i*h,'vortex_y':j*h,'psi_min':float(psi[j,i]),
            'kinetic_energy':float(.5*h*h*np.sum(uc*uc+vc*vc))}


# Ghia, Ghia & Shin (1982), tables I and II, Re=100. Values are benchmark data,
# not an exact solution. Exclude no points from the declared velocity comparison.
GHIA_Y=np.array([1,.9766,.9688,.9609,.9531,.8516,.7344,.6172,.5,.4531,.2813,.1719,.1016,.0703,.0625,.0547,0])
GHIA_U=np.array([1,.84123,.78871,.73722,.68717,.23151,.00332,-.13641,-.20581,-.21090,-.15662,-.10150,-.06434,-.04775,-.04192,-.03717,0])
GHIA_X=np.array([1,.9688,.9609,.9531,.9453,.9063,.8594,.8047,.5,.2344,.2266,.1563,.0938,.0781,.0703,.0625,0])
GHIA_V=np.array([0,-.05906,-.07391,-.08864,-.10313,-.16914,-.22445,-.24533,.05454,.17527,.17507,.16077,.12317,.10890,.10091,.09233,0])


def centerlines(result):
    x,y = result['x'],result['y']
    if 'u_faces' in result:
        n = len(x)
        # Centerline passes exactly through MAC faces for even n; interpolate
        # the remaining tangential coordinate including actual wall values.
        u = np.array([np.interp(.5,np.linspace(0,1,n+1),row) for row in result['u_faces']])
        v = np.array([np.interp(.5,np.linspace(0,1,n+1),col) for col in result['v_faces'].T])
        return np.r_[0,y,1], np.r_[0,u,1], np.r_[0,x,1], np.r_[0,v,0]
    return y,result['u'][:,len(x)//2],x,result['v'][len(y)//2,:]


def ghia_errors(result):
    y,u,x,v = centerlines(result)
    eu = np.interp(GHIA_Y,y,u)-GHIA_U; ev = np.interp(GHIA_X,x,v)-GHIA_V
    return {'ghia_u_relative_l2':float(np.linalg.norm(eu)/np.linalg.norm(GHIA_U)),
            'ghia_v_relative_l2':float(np.linalg.norm(ev)/np.linalg.norm(GHIA_V)),
            'ghia_u_linf':float(np.max(np.abs(eu))),'ghia_v_linf':float(np.max(np.abs(ev)))}
