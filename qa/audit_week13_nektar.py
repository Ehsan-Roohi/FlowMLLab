"""Read-only, export-grid diagnostics; NOT spectral residuals or accuracy proof.

Requires ASCII VTU (:vtu:uncompress), numpy and scipy. Re=100, D=5.
Two path integrations use the same bottom-left gauge, but opposite paths.
Weak-vortex accuracy cannot be inferred from coarse visualization exports.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

from check_week13_nektar_vtu import read_vtu
from audit_week13_cfd import REF, extrema


def structured(pieces, decimals=12):
    points = np.concatenate([p[0] for p in pieces])
    if not np.allclose(points[:, 2], 0, atol=1e-12, rtol=0):
        raise ValueError('Only planar z=0 exports supported')
    rounded = np.round(points[:, :2], decimals)
    unique, inverse, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    x, y = np.unique(unique[:, 0]), np.unique(unique[:, 1])
    if len(unique) != len(x)*len(y) or min(len(x), len(y)) < 5:
        raise ValueError('Export is not a complete tensor-product grid; no silent interpolation')
    if not np.allclose([x[0], x[-1], y[0], y[-1]], [0, 1, 0, 5], atol=1e-10, rtol=0):
        raise ValueError('Expected width=1 depth=5 boundary-inclusive export')
    ix, iy = np.searchsorted(x, unique[:, 0]), np.searchsorted(y, unique[:, 1])
    fields, jumps = {}, {}
    for name in ('u', 'v', 'p'):
        values = np.concatenate([piece[1][name] for piece in pieces])
        low = np.full(len(unique), np.inf)
        high = np.full(len(unique), -np.inf)
        np.minimum.at(low, inverse, values)
        np.maximum.at(high, inverse, values)
        average = np.bincount(inverse, weights=values)/counts
        grid = np.empty((len(y), len(x)))
        grid[iy, ix] = average
        fields[name] = grid
        jumps[name] = float(np.max(high-low))
    return x, y, fields, dict(raw_points=len(points), unique_points=len(unique),
        coordinate_rounding_decimals=decimals, maximum_duplicate_field_jump=jumps,
        consolidation='duplicate values averaged; jumps retained for review')


def integrate_paths(x, y, u, v):
    # psi_y=u, psi_x=-v. Gauge psi(bottom-left)=0. Neither path fitted to other.
    left = cumulative_trapezoid(u[:, 0], y, initial=0)
    bottom = -cumulative_trapezoid(v[0, :], x, initial=0)
    horizontal = left[:, None]-cumulative_trapezoid(v, x, axis=1, initial=0)
    vertical = bottom[None, :]+cumulative_trapezoid(u, y, axis=0, initial=0)
    return horizontal, vertical


def diagnostics(x, y, fields):
    u, v = fields['u'], fields['v']
    horizontal, vertical = integrate_paths(x, y, u, v)
    disagreement = horizontal-vertical
    divergence = np.gradient(u, x, axis=1, edge_order=2)+np.gradient(v, y, axis=0, edge_order=2)
    omega = np.gradient(u, y, axis=0, edge_order=2)-np.gradient(v, x, axis=1, edge_order=2)
    interior = divergence[1:-1, 1:-1]
    vortices = extrema(horizontal, x, y)
    interp_omega = RegularGridInterpolator((y, x), omega)
    interp_disagreement = RegularGridInterpolator((y, x), disagreement)
    interp_vertical = RegularGridInterpolator((y, x), vertical)
    comparisons = []
    for index, row in enumerate(REF[(100, 5)]):
        rx, ry, rp, rw = row
        candidates = [q for q in vortices if q['psi']*rp > 0 and abs(q['y']-ry) < .45]
        record = dict(vortex=index+1, reference=row, status='not_resolved_in_reference_neighborhood')
        if candidates:
            q = min(candidates, key=lambda z: np.hypot(z['x']-rx, z['y']-ry)).copy()
            location = [[q['y'], q['x']]]
            q['omega_clockwise'] = float(interp_omega(location)[0])
            q['psi_vertical_path'] = float(interp_vertical(location)[0])
            local_disagreement = abs(float(interp_disagreement(location)[0]))
            record.update(computed=q, centre_distance_W=float(np.hypot(q['x']-rx, q['y']-ry)),
                psi_difference_percent=100*abs(q['psi']/rp-1),
                omega_difference_percent=100*abs(q['omega_clockwise']/rw-1),
                local_path_disagreement=local_disagreement,
                local_path_disagreement_over_reference_psi=local_disagreement/abs(rp),
                status='compared_not_certified')
        comparisons.append(record)
    report = dict(
        streamfunction=dict(gauge='bottom-left zero; psi_x=-v psi_y=u',
            horizontal_then_vertical_are_independent_paths=True,
            max_path_disagreement=float(np.max(abs(disagreement))),
            sample_rms_path_disagreement=float(np.sqrt(np.mean(disagreement**2))),
            right_wall_horizontal_closure=float(np.max(abs(horizontal[:, -1]))),
            top_wall_horizontal_closure=float(np.max(abs(horizontal[-1, :])))),
        divergence=dict(method='finite differences on visualization grid, not spectral derivative',
            max_abs_interior=float(np.max(abs(interior))),
            sample_rms_interior=float(np.sqrt(np.mean(interior**2)))),
        boundaries=dict(lid_max_speed_error_excluding_corners=float(np.max(np.hypot(u[-1, 1:-1]-1, v[-1, 1:-1]))),
            bottom_max_speed=float(np.max(np.hypot(u[0, :], v[0, :]))),
            side_max_speed_excluding_lid_corners=float(max(np.max(np.hypot(u[:-1, 0], v[:-1, 0])), np.max(np.hypot(u[:-1, -1], v[:-1, -1])))),
            top_corner_u=[float(u[-1, 0]), float(u[-1, -1])]),
        vortex_candidates=vortices, paper_comparison=comparisons)
    return report


def audit(path):
    x, y, fields, continuity = structured(read_vtu(path))
    result = diagnostics(x, y, fields)
    result.update(field_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        Re=100, depth_over_width=5, nx_points=len(x), ny_points=len(y),
        continuity=continuity, reference='Cheng & Hung 2006 Table 2; DOI 10.1016/j.compfluid.2005.08.006',
        limitations=['Transient time and steady convergence must be checked separately.',
            'Trapezoidal path disagreement mixes export/integration error and field inconsistency.',
            'Duplicate-coordinate averaging is not proof of continuity; inspect reported jumps.',
            'No claim of tiny lower/corner vortex accuracy; extraction excludes x<0.1 and x>0.9.',
            'Grid independence must include solver mesh/order and export resolution.',
            'RMS values are unweighted sample norms, not physical volume L2 norms.'])
    return result


def self_test():
    errors = []
    for n in (41, 81):
        x, y = np.linspace(0, 1, n), np.linspace(0, 5, 5*n)
        xx, yy = np.meshgrid(x, y)
        exact = np.sin(np.pi*xx)**2*np.sin(np.pi*yy/5)**2
        u = np.sin(np.pi*xx)**2*(np.pi/5)*np.sin(2*np.pi*yy/5)
        v = -np.pi*np.sin(2*np.pi*xx)*np.sin(np.pi*yy/5)**2
        a, b = integrate_paths(x, y, u, v)
        errors.append(float(max(np.max(abs(a-exact)), np.max(abs(b-exact)))))
        assert np.max(abs(a[:, 0])) < 1e-12
        assert np.max(abs(b[0, :])) < 1e-12
    assert 3.8 < errors[0]/errors[1] < 4.2, errors
    # Duplicate consolidation reports discontinuity, rather than hiding it.
    xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    fields = dict(u=u.ravel(), v=v.ravel(), p=np.zeros(xx.size))
    altered = {k: value.copy() for k, value in fields.items()}
    altered['u'][5] += .01
    _, _, _, report = structured([(xyz, fields, {}), (xyz, altered, {})])
    assert abs(report['maximum_duplicate_field_jump']['u']-.01) < 1e-12
    return dict(manufactured_divergence_free_path_test='passed',
                trapezoidal_errors=errors, refinement_ratio=errors[0]/errors[1],
                duplicate_jump_test='passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('vtu', nargs='?')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if not a.self_test and not a.vtu:
        p.error('VTU path or --self-test required')
    print(json.dumps(self_test() if a.self_test else audit(a.vtu), indent=2, allow_nan=False))
