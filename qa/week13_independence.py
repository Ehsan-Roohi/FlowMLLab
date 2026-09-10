"""Re500 D/W=5 verification design; never uses paper values as stop targets."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.polynomial import legendre as L
from scipy.optimize import root

from prepare_week13_nektar import make_case
from audit_week13_nektar import structured, integrate_paths
from audit_week13_cfd import extrema
from check_week13_nektar_vtu import read_vtu


CASES = [
    dict(label='base', nx=16, ny=80, order=6, dt=.00025),
    dict(label='h32', nx=32, ny=160, order=6, dt=.00025),
    dict(label='h64', nx=64, ny=320, order=6, dt=.00025),
    dict(label='p8', nx=16, ny=80, order=8, dt=.00025),
    dict(label='p10', nx=16, ny=80, order=10, dt=.00025),
    dict(label='dt2', nx=16, ny=80, order=6, dt=.000125),
    dict(label='dt4', nx=16, ny=80, order=6, dt=.0000625),
    dict(label='corner_h4', nx=22, ny=80, order=6, dt=.00025),
]


def x_vertices(case):
    if case['label'] == 'corner_h4':
        # Split only the first/last base x elements four ways. All old vertices
        # remain; no altered lid velocity or smoothed physical boundary condition.
        return np.unique(np.r_[np.linspace(0, 1, 17),
                               np.linspace(0, 1/16, 5),
                               np.linspace(15/16, 1, 5)])
    return np.linspace(0, 1, case['nx']+1)


def session(directory, case, *, restart=None, interval=.25):
    steps = round(interval/case['dt'])
    if steps < 1 or abs(steps*case['dt']-interval) > 1e-12:
        raise ValueError('Time step must divide checkpoint interval')
    make_case(directory, re=500, order=case['order'], nx=case['nx'], ny=case['ny'],
              dt=case['dt'], steps=steps, check_steps=steps, restart=restart,
              corner_convention='stationary_endpoints', purpose='independence_not_certified')
    if case['label'] == 'corner_h4':
        path = Path(directory)/'cavity.xml'
        tree = ET.parse(path)
        xs = x_vertices(case)
        for vertex in tree.findall('.//VERTEX/V'):
            i = int(vertex.attrib['ID']) % (case['nx']+1)
            xyz = vertex.text.split()
            xyz[0] = format(xs[i], '.17g')
            vertex.text = ' '.join(xyz)
        tree.write(path, encoding='utf-8', xml_declaration=True)


class PolynomialField:
    """Element-local tensor Legendre reconstruction, no cross-element smoothing.

    Uses at least p+1 distinct export points per element axis. Integration is
    analytic within each element. Two independent paths expose field divergence.
    This reconstruction is checked by manufactured tests; it is not the solver's
    quadrature-weighted residual. Clockwise omega = u_y-v_x.
    """
    def __init__(self, x, y, fields, ex, ey, order):
        self.x, self.y, self.fields = x, y, fields
        self.ex, self.ey, self.p = np.asarray(ex), np.asarray(ey), order
        self.cache = {}

    def coeff(self, name, i, j):
        key = (name, i, j)
        if key not in self.cache:
            ix = np.flatnonzero((self.x >= self.ex[i]-1e-11) & (self.x <= self.ex[i+1]+1e-11))
            iy = np.flatnonzero((self.y >= self.ey[j]-1e-11) & (self.y <= self.ey[j+1]+1e-11))
            if min(len(ix), len(iy)) < self.p+1:
                raise ValueError('Export undersamples element polynomial')
            sx = 2*(self.x[ix]-self.ex[i])/(self.ex[i+1]-self.ex[i])-1
            sy = 2*(self.y[iy]-self.ey[j])/(self.ey[j+1]-self.ey[j])-1
            vx, vy = L.legvander(sx, self.p), L.legvander(sy, self.p)
            values = self.fields[name][np.ix_(iy, ix)]
            c = np.linalg.lstsq(vy, values, rcond=None)[0]
            c = np.linalg.lstsq(vx, c.T, rcond=None)[0].T
            error = np.max(abs(vy@c@vx.T-values))
            if error > 1e-11+1e-8*np.max(abs(values)):
                raise ValueError(f'Nonpolynomial/insufficient export: {name} {error}')
            self.cache[key] = c
        return self.cache[key]

    def locate(self, x, y):
        if not (0 <= x <= 1 and 0 <= y <= 5):
            raise ValueError('Point outside cavity')
        i = min(np.searchsorted(self.ex, x, side='right')-1, len(self.ex)-2)
        j = min(np.searchsorted(self.ey, y, side='right')-1, len(self.ey)-2)
        sx = 2*(x-self.ex[i])/(self.ex[i+1]-self.ex[i])-1
        sy = 2*(y-self.ey[j])/(self.ey[j+1]-self.ey[j])-1
        return i, j, sx, sy

    def value(self, name, x, y, dx=0, dy=0):
        i, j, sx, sy = self.locate(x, y)
        c = self.coeff(name, i, j)
        if dy:
            c = L.legder(c, m=dy, axis=0)*(2/(self.ey[j+1]-self.ey[j]))**dy
        if dx:
            c = L.legder(c, m=dx, axis=1)*(2/(self.ex[i+1]-self.ex[i]))**dx
        return float(L.legval2d(sy, sx, c))

    def psi(self, x, y, vertical=False):
        i, j, sx, sy = self.locate(x, y)
        total = 0.
        # Closed-cavity stationary left/bottom walls make their gauge legs zero.
        if vertical:
            for k in range(j+1):
                c = L.legint(self.coeff('u', i, k), axis=0)
                top = sy if k == j else 1.
                total += (L.legval2d(top, sx, c)-L.legval2d(-1., sx, c))*(self.ey[k+1]-self.ey[k])/2
        else:
            for k in range(i+1):
                c = L.legint(self.coeff('v', k, j), axis=1)
                right = sx if k == i else 1.
                total -= (L.legval2d(sy, right, c)-L.legval2d(sy, -1., c))*(self.ex[k+1]-self.ex[k])/2
        return float(total)


def audit(vtu, case):
    x, y, fields, continuity = structured(read_vtu(vtu))
    poly = PolynomialField(x, y, fields, x_vertices(case),
                           np.linspace(0, 5, case['ny']+1), case['order'])
    psi, _ = integrate_paths(x, y, fields['u'], fields['v'])
    seeds = extrema(psi, x, y)
    found = []
    for seed in seeds:
        ix, iy = np.argmin(abs(x-seed['x'])), np.argmin(abs(y-seed['y']))
        scale = max(np.max(np.hypot(fields['u'][max(0, iy-2):iy+3, max(0, ix-2):ix+3],
                                      fields['v'][max(0, iy-2):iy+3, max(0, ix-2):ix+3])), 1e-20)
        def velocity(z):
            if not (0 < z[0] < 1 and 0 < z[1] < 5):
                return [1e6, 1e6]
            return [poly.value('u', *z)/scale, poly.value('v', *z)/scale]
        sol = root(velocity, [seed['x'], seed['y']], tol=1e-10)
        if np.linalg.norm(velocity(sol.x)) > 1e-6:
            raise ValueError('Vortex centre solve failed')
        cx, cy = map(float, sol.x)
        if not (.1 < cx < .9 and .1 < cy < 4.95):
            continue
        strength = poly.psi(cx, cy)
        jac = np.array([[poly.value('u', cx, cy, dx=1), poly.value('u', cx, cy, dy=1)],
                        [poly.value('v', cx, cy, dx=1), poly.value('v', cx, cy, dy=1)]])
        if np.linalg.det(jac) <= 0:  # Reject saddles, not just all velocity zeros.
            continue
        if any(np.hypot(cx-q['x'], cy-q['y']) < .01 for q in found):
            continue
        found.append(dict(x=cx, y=cy, psi=strength,
                          psi_vertical=poly.psi(cx, cy, vertical=True),
                          omega_clockwise=poly.value('u', cx, cy, dy=1)-poly.value('v', cx, cy, dx=1)))
    found.sort(key=lambda q: -q['y'])
    if len(found) != 4 or [np.sign(q['psi']) for q in found] != [-1, 1, -1, 1]:
        raise ValueError(f'Expected four alternating main vortices, got {found}')
    return dict(vortices=found, continuity=continuity,
                method='element-local polynomial integration and u=v=0 centres',
                scope='four main vortices; small corner eddies not certified')


def temporal_gate(records):
    if len(records) < 3:
        return dict(passed=False, reason='Need three snapshots, separated by ten time units')
    selected = records[-3:]
    windows = []
    for a, b in zip(selected, selected[1:]):
        if abs(b['time']-a['time']-10) > 1e-7:
            raise ValueError('Temporal audit times must differ by ten')
        if len(a['vortices']) != 4 or len(b['vortices']) != 4:
            raise ValueError('Four vortices required')
        rows = []
        for old, new in zip(a['vortices'], b['vortices']):
            strength = abs(new['psi']/old['psi']-1)
            omega = abs(new['omega_clockwise']/old['omega_clockwise']-1)
            shift = float(np.hypot(new['x']-old['x'], new['y']-old['y']))
            rows.append(dict(strength_relative_change=strength, omega_relative_change=omega,
                             centre_shift_W=shift,
                             passed=bool(strength < .001 and omega < .001 and shift < .001)))
        windows.append(dict(start=a['time'], end=b['time'], vortices=rows))
    return dict(passed=all(q['passed'] for w in windows for q in w['vortices']),
                windows=windows, scope='temporal only, not h/p/dt or paper validation')


def atomic_json(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    temp.replace(path)
