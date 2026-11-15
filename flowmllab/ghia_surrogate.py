"""Small CPU MLP experiment on Ghia centreline samples, not a full-field PINN.

Whole Reynolds numbers are withheld. Optimizer, architecture, weighting and
seed are explicit. NumPy backprop makes the exercise independent of GPU or
framework version; SciPy supplies L-BFGS. No CFD pressure or 2D field is inferred.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from .ghia import GHIA, REYNOLDS, case_roles, valid_mask


@dataclass(frozen=True)
class ModelConfig:
    name: str = 'hybrid_2'
    depth: int = 2
    width: int = 24
    optimizer: str = 'hybrid'
    weighting: str = 'uniform'
    evaluations: int = 1200
    learning_rate: float = .004


CONFIGS = [ModelConfig('adam_2', optimizer='adam'),
           ModelConfig('lbfgs_2', optimizer='lbfgs'),
           ModelConfig('hybrid_2'), ModelConfig('hybrid_5', depth=5),
           ModelConfig('hybrid_5_wall', depth=5, weighting='wall')]


def inputs(reynolds, positions, component):
    t = np.asarray(positions, dtype=float)
    if component not in ('u', 'v') or reynolds <= 0:
        raise ValueError('Require positive Re and u/v component')
    return np.column_stack([2*t-1, np.full(t.shape, np.log10(reynolds)-3),
                            np.full(t.shape, -1. if component == 'u' else 1.)])


def training_data(weighting='uniform'):
    xs, ys, ws = [], [], []
    for re, role in case_roles().items():
        if role != 'train':
            continue
        for component in ('u', 'v'):
            ref = GHIA[re]; t = ref['y' if component == 'u' else 'x']
            keep = valid_mask(re, component) & (t > 0) & (t < 1)
            t = t[keep]; weight = np.ones_like(t)
            if weighting == 'wall':
                weight[(t < .12) | (t > .88)] = 3.
            elif weighting != 'uniform':
                raise ValueError('weighting must be uniform or wall')
            xs.append(inputs(re, t, component)); ys.append(ref[component][keep])
            ws.append(weight/weight.sum())  # equal total weight per case/component
    return np.vstack(xs), np.concatenate(ys), np.concatenate(ws)/len(ws)


def initialize(config, seed):
    rng = np.random.default_rng(seed)
    sizes = [3] + [config.width]*config.depth + [1]
    shapes = [(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
    flat = []
    for a, b in shapes:
        flat.extend([rng.normal(size=(a, b)).ravel()*np.sqrt(2/(a+b)), np.zeros(b)])
    return np.concatenate(flat), shapes


def unpack(theta, shapes):
    pos = 0; layers = []
    for a, b in shapes:
        w = theta[pos:pos+a*b].reshape(a, b); pos += a*b
        bias = theta[pos:pos+b]; pos += b
        layers.append((w, bias))
    if pos != len(theta):
        raise ValueError('Parameter shape mismatch')
    return layers


def forward(theta, shapes, x):
    layers = unpack(theta, shapes); acts = [x]
    for i, (w, b) in enumerate(layers):
        z = acts[-1]@w+b
        acts.append(np.tanh(z) if i < len(layers)-1 else z)
    t = .5*(x[:, 0]+1)
    base = np.where(x[:, 2] < 0, t, 0.)
    factor = t*(1-t)
    return base+factor*acts[-1][:, 0], acts, factor, layers


def objective(theta, shapes, x, target, weights):
    prediction, acts, factor, layers = forward(theta, shapes, x)
    err = prediction-target
    value = float(np.sum(weights*err**2))
    delta = (2*weights*err*factor)[:, None]
    grads = []
    for i in range(len(layers)-1, -1, -1):
        grads.append((acts[i].T@delta, delta.sum(axis=0)))
        if i:
            delta = (delta@layers[i][0].T)*(1-acts[i]**2)
    grad = np.concatenate([g.ravel() for pair in reversed(grads) for g in pair])
    return value, grad


def fit(config, seed):
    if config.optimizer not in ('adam', 'lbfgs', 'hybrid'):
        raise ValueError('Unknown optimizer')
    x, y, weights = training_data(config.weighting)
    theta, shapes = initialize(config, seed)
    count = 0; best = theta.copy(); best_loss = np.inf; history = []
    start = perf_counter(); phase = 'Adam'

    class BudgetExhausted(Exception):
        pass

    def fun(t):
        nonlocal count, best, best_loss
        if count >= config.evaluations:
            raise BudgetExhausted
        value, grad = objective(t, shapes, x, y, weights); count += 1
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('Nonfinite objective/gradient')
        if value < best_loss:
            best_loss, best = value, t.copy()
        if count == 1 or count % 10 == 0:
            history.append({'evaluation': count, 'loss': value,
                            'seconds': perf_counter()-start, 'phase': phase})
        return value, grad

    steps = config.evaluations if config.optimizer == 'adam' else config.evaluations//2 if config.optimizer == 'hybrid' else 0
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    for k in range(1, steps+1):
        _, g = fun(theta); m = .9*m+.1*g; v = .999*v+.001*g*g
        theta -= config.learning_rate*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
    status = 'fixed Adam evaluation budget'
    if config.optimizer != 'adam':
        phase = 'L-BFGS'
        try:
            result = minimize(fun, best.copy(), method='L-BFGS-B', jac=True,
                              options={'maxiter': config.evaluations, 'maxfun': config.evaluations,
                                       'maxls': 30, 'ftol': 1e-15, 'gtol': 1e-10, 'maxcor': 20})
            status = str(result.message)
        except BudgetExhausted:
            status = 'fixed joint function/gradient evaluation budget'
    return {'theta': best, 'shapes': shapes, 'config': asdict(config), 'seed': seed,
            'training_loss': best_loss, 'evaluations': count,
            'seconds': perf_counter()-start, 'status': status, 'history': history}


def predict(model, reynolds, positions, component):
    return forward(model['theta'], model['shapes'], inputs(reynolds, positions, component))[0]


def linear_baseline(reynolds, positions, component):
    """Same four training cases; linear in log10(Re), linear in position."""
    train = [re for re, role in case_roles().items() if role == 'train']
    if not train[0] <= reynolds <= train[-1]:
        raise ValueError('No extrapolation in the baseline')
    columns = []
    for re in train:
        ref = GHIA[re]; axis = ref['y' if component == 'u' else 'x']
        keep = valid_mask(re, component)
        columns.append(np.interp(positions, axis[keep][::-1], ref[component][keep][::-1]))
    return np.array([np.interp(np.log10(reynolds), np.log10(train), col)
                     for col in np.asarray(columns).T])


def scores(predictor):
    rows = []
    for re, role in case_roles().items():
        for component in ('u', 'v'):
            ref = GHIA[re]; t = ref['y' if component == 'u' else 'x']
            p = predictor(re, t, component); keep = valid_mask(re, component)
            e = p-ref[component]
            rows.append({'Re': re, 'role': role, 'component': component,
                         'rmse_U': float(np.sqrt(np.mean(e[keep]**2))),
                         'relative_l2': float(np.linalg.norm(e[keep])/np.linalg.norm(ref[component][keep])),
                         'published_relative_l2': float(np.linalg.norm(e)/np.linalg.norm(ref[component])),
                         'wall_rmse_U': float(np.sqrt(np.mean(e[keep & ((t < .12) | (t > .88))]**2)))})
    return rows
