"""Small CPU experiments separating linear, coupling and time errors.

Original teaching implementations inspired by KTH SG2212 (2017), Q6, and
Barton (1998). They do not reproduce either source's full solver or cases.
"""
from dataclasses import asdict
import inspect
from time import perf_counter

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import gmres, splu

from .pressure_velocity import (CavityConfig, run_cavity, pressure_matrix,
                                divergence, correct_velocity)


def predictor_fields(n, count=3):
    """Analytic trial velocities, not cavity solutions; zero normal wall flux."""
    if n < 4 or count < 1:
        raise ValueError('Use at least four cells and one right-hand side')
    xf, yc = np.linspace(0, 1, n+1), (np.arange(n)+.5)/n
    xc, yf = yc, xf
    fields = []
    for k in range(count):
        u = (np.sin(np.pi*xf)[None, :]*yc[:, None]*(1-yc[:, None]) *
             (1+.3*np.cos((k+2)*np.pi*yc[:, None])+.1*np.sin(3*np.pi*xf)[None, :]))
        v = (.4*np.sin(np.pi*yf)[:, None]*xc[None, :]*(1-xc[None, :]) *
             (1+.2*np.cos((k+3)*np.pi*xc)[None, :]))
        u[:, [0, -1]] = 0; v[[0, -1], :] = 0
        fields.append((u, v))
    return fields


def pressure_work_study(cells=(16, 32), repeats=3, rhs_count=3,
                        tolerances=(1e-4, 1e-8), scale=.05):
    """Repeated fixed-operator projections with full residual/mass checks.

    Setup plus all RHS solves are timed. One untimed batch warms each method.
    Assembly and diagnostics are excluded. LU reuse is valid only because A
    is identical for every RHS. GMRES uses Jacobi, restart 50, zero starts.
    """
    if repeats < 1 or scale <= 0:
        raise ValueError('Positive repeats and projection scale required')
    rows, fields = [], {}
    for n in cells:
        h = 1/n; du = np.full((n,n-1), scale); dv = np.full((n-1,n), scale)
        full = pressure_matrix(du, dv, h); matrix = full[1:, 1:].tocsc()
        trials = predictor_fields(n, rhs_count)
        rhs = [-divergence(u, v, h).ravel() for u,v in trials]
        assert all(abs(b.sum()) < 1e-10 for b in rhs)
        reference = splu(matrix)
        reference_p = []
        for b in rhs:
            p = np.r_[0., reference.solve(b[1:])]; reference_p.append(p-p.mean())
        methods = [('LU each RHS', None), ('LU reused', None)]
        methods += [('GMRES Jacobi', float(t)) for t in tolerances]
        for method, tol in methods:
            label = method if tol is None else f'{method} rtol={tol:g}'
            timings, last = [], None
            for repeat in range(repeats+1):
                setup_time = 0.; iterations = 0; infos = []; solutions = []
                start = perf_counter()
                if method == 'LU reused':
                    tick = perf_counter(); factor = splu(matrix); setup_time += perf_counter()-tick
                elif method == 'GMRES Jacobi':
                    tick = perf_counter(); preconditioner = diags(1/matrix.diagonal()); setup_time += perf_counter()-tick
                for b in rhs:
                    if method == 'LU each RHS':
                        tick = perf_counter(); factor = splu(matrix); setup_time += perf_counter()-tick
                    if method.startswith('LU'):
                        q = factor.solve(b[1:]); info = 0
                    else:
                        counter = []
                        options = {'rtol': tol} if 'rtol' in inspect.signature(gmres).parameters else {'tol': tol}
                        q, info = gmres(matrix, b[1:], M=preconditioner, atol=0,
                            restart=50, maxiter=200, callback=counter.append,
                            callback_type='pr_norm', **options)
                        iterations += len(counter)
                    solutions.append(np.r_[0., q]); infos.append(int(info))
                total = perf_counter()-start
                if repeat:
                    timings.append({'setup_seconds':setup_time, 'solve_seconds':total-setup_time,
                                    'total_seconds':total})
                last = (solutions, infos, iterations)
            solutions, infos, iterations = last
            diagnostics = []
            for j, ((u,v), b, p, exact) in enumerate(zip(trials, rhs, solutions, reference_p)):
                p = p-p.mean(); uc,vc = correct_velocity(u,v,p.reshape(n,n),du,dv,h)
                mass = divergence(uc,vc,h); residual = b-full@p
                # Includes the omitted gauge row: this identity catches a bad pin/sign.
                np.testing.assert_allclose(mass.ravel(), -residual, atol=2e-11, rtol=1e-8)
                diagnostics.append({'full_residual_relative_l2':float(np.linalg.norm(residual)/np.linalg.norm(b)),
                    'divergence_linf':float(np.max(np.abs(mass))),
                    'pressure_relative_l2_vs_lu':float(np.linalg.norm(p-exact)/np.linalg.norm(exact))})
                prefix=f'n{n}_{label.replace(" ","_")}_rhs{j}'
                for name, value in [('p',p.reshape(n,n)),('div',mass),('u_star',u),('v_star',v),('u',uc),('v',vc)]:
                    fields[prefix+'_'+name] = value
            rows.append({'cells':n,'method':label,'rhs_count':rhs_count,'repeats':repeats,
                'total_seconds_median':float(np.median([t['total_seconds'] for t in timings])),
                'setup_seconds_median':float(np.median([t['setup_seconds'] for t in timings])),
                'solve_seconds_median':float(np.median([t['solve_seconds'] for t in timings])),
                'timing_samples':timings,'gmres_iterations_all_rhs':iterations,
                'solver_info':infos,'linear_solver_converged':all(i==0 for i in infos),
                'max_divergence_linf':max(d['divergence_linf'] for d in diagnostics),
                'max_pressure_relative_l2_vs_lu':max(d['pressure_relative_l2_vs_lu'] for d in diagnostics),
                'rhs_diagnostics':diagnostics})
    return rows, fields


def velocity_distance(a, b):
    """Relative face-velocity vector norm on one identical MAC grid."""
    return float(np.sqrt(sum(np.sum((a[k]-b[k])**2) for k in ('u_faces','v_faces')) /
                         sum(np.sum(b[k]**2) for k in ('u_faces','v_faces'))))


def startup_study(cells=16, end_time=.2, steps=(.05,.025,.0125),
                  reference_dt=.003125, repeats=3):
    """Fixed-end-time startup; reference has independent time/loop checks.

    Same Re100 MAC mesh, central convection and backward Euler for every run.
    Discrepancies are against a numerical comparator, not an exact solution.
    """
    if min(steps) <= 2*reference_dt or repeats < 1:
        raise ValueError('Test steps must exceed the coarse-reference step; repeats must be positive')
    specs=[('reference_coarse','pimple',2*reference_dt,6,4),
           ('reference_fine','pimple',reference_dt,6,4),
           ('reference_loops','pimple',reference_dt,12,8)]
    specs += [(f'{method}_dt{dt:g}',method,dt,3 if method=='pimple' else 1,2)
              for method in ('piso','pimple') for dt in steps]
    rows, fields, results = [], {}, {}
    # Untimed tiny warm-up; all recorded trajectories start again from rest.
    run_cavity(CavityConfig(cells=cells,method='piso',dt=.01,end_time=.01,check_every=1))
    for key, method, dt, outer, inner in specs:
        cfg=CavityConfig(cells=cells,method=method,dt=dt,end_time=end_time,
                        pressure_correctors=inner,outer_correctors=outer,
                        check_every=1,max_iterations=int(np.ceil(end_time/dt))+2)
        durations=[]; result=None
        for repeat in range(repeats):
            current=run_cavity(cfg)
            if not current['reached_end_time'] or abs(current['physical_time']-end_time)>1e-10:
                raise RuntimeError(f'{key} failed to reach the same physical time')
            if current['continuity_linf']>1e-9 or not np.isfinite(current['u_faces']).all() or not np.isfinite(current['v_faces']).all():
                raise RuntimeError(f'{key} failed the finite-field/mass gate')
            if result is not None and velocity_distance(current,result)>1e-12:
                raise RuntimeError('Repeated deterministic trajectories disagree')
            result=current; durations.append(result['runtime_seconds'])
        results[key]=result
        for name in ('u_faces','v_faces','p','x','y','u','v','history'):
            fields[key+'_'+name]=result[name]
        rows.append({'case':key,'config':asdict(cfg),'physical_time':result['physical_time'],
                     'steps':result['iterations'],'pressure_solves':result['pressure_solves'],
                     'runtime_seconds_median':float(np.median(durations)), 'runtime_samples':durations,
                     'continuity_linf':result['continuity_linf'],
                     'max_nonlinear_step_defect':float(np.max(result['history'][:,5])),
                     'last_nonlinear_step_defect':float(result['history'][-1,5])})
    fine=results['reference_fine']
    for row in rows:
        row['velocity_discrepancy_vs_fine']=velocity_distance(results[row['case']],fine)
    checks={'reference_dt_change':velocity_distance(results['reference_coarse'],fine),
            'reference_loop_change':velocity_distance(results['reference_loops'],fine),
            'meaning':'Observed reference sensitivity, not a rigorous error bound. No steady Ghia score is used for startup.'}
    return rows, fields, checks
