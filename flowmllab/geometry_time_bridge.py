"""Small NumPy contracts for the optional geometry–time evaluation bridge.

The fields are manufactured arrays, not solutions of Navier–Stokes. No training,
network, download, or file writes occur here. Positive SDF means fluid.
"""
from __future__ import annotations

import numpy as np


def manufactured_cases():
    """Return 12 deterministic cases: six geometries, two parameter values each.

    values: [time, y, x, component]; sdf: [y,x]; coordinates and time are
    dimensionless. The parameter is an oscillator rate, NOT Reynolds number.
    Masking these velocity-like fields does not enforce wall conditions.
    """
    x, y = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-1, 1, 17))
    t = np.arange(25, dtype=float) * .25
    cases = {}
    for k in range(6):
        if k < 3:
            sdf = np.hypot(x, y) - (.25 + .1*k)
            family = 'circle'
        else:
            qx, qy = np.abs(x) - (.25 + .1*(k-3)), np.abs(y) - .3
            sdf = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
            family = 'box'
        for rate in (.2, .3):
            phase = 2*np.pi*rate*t[:, None, None]
            u = 1 + .2*np.cos(phase)*np.exp(-x*x-y*y)
            v = .3*np.sin(phase)*np.exp(-x*x-y*y)
            values = np.stack((u, v), axis=-1)
            key = f'g{k}_rate{rate:g}'
            cases[key] = dict(geometry_id=f'g{k}', family=family, rate=rate,
                              time=t.copy(), x=x.copy(), y=y.copy(),
                              sdf=sdf.copy(), values=values)
    return cases


def split_before_windows(cases, geometry_partitions, history=4):
    """Build lightweight window indices only after an exhaustive group split.

    partitions maps train/validation/test (or other labels) to geometry IDs.
    Each output row is (case_id, inclusive_history_start, target_index).
    Histories are half-open [start:target]; no window crosses a case boundary.
    Family holdout is a separate property and is not implied by this function.
    """
    if isinstance(history, bool) or not isinstance(history, (int, np.integer)) or history < 1:
        raise ValueError('history must be a positive integer')
    if not cases or not geometry_partitions:
        raise ValueError('cases and partitions must be nonempty')
    owner = {}
    for name, ids in geometry_partitions.items():
        if not ids:
            raise ValueError('each partition must contain a geometry')
        for gid in ids:
            if gid in owner:
                raise ValueError(f'duplicate geometry assignment: {gid}')
            owner[gid] = name
    if set(owner) != {case['geometry_id'] for case in cases.values()}:
        raise ValueError('partitions must cover exactly the available geometries')
    records = {name: [] for name in geometry_partitions}
    for key, case in cases.items():
        values = np.asarray(case['values'])
        if values.ndim < 2 or len(values) <= history:
            raise ValueError('each case needs a time axis and more frames than history')
        records[owner[case['geometry_id']]].extend((key, t-history, t) for t in range(history, len(values)))
    return records


def masked_mse(prediction, truth, fluid):
    """Per-case and equal-case/pooled MSE for explicit [B,H,W,C] arrays.

    The Boolean [B,H,W] fluid mask is applied to every component. The pooled
    statistic weights cases by their fluid-cell count; equal-case MSE does not.
    No implicit broadcasting between different batch entries is permitted.
    """
    p, q, m = np.asarray(prediction, dtype=float), np.asarray(truth, dtype=float), np.asarray(fluid)
    if p.ndim != 4 or p.shape != q.shape or m.shape != p.shape[:3] or m.dtype != np.bool_:
        raise ValueError('expected identical [B,H,W,C] fields and Boolean [B,H,W] mask')
    if any(d == 0 for d in p.shape) or not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError('fields must be nonempty and finite, including masked cells')
    count = m.sum(axis=(1, 2)) * p.shape[-1]
    if np.any(count == 0):
        raise ValueError('each case needs at least one fluid cell')
    sums = (((p-q)**2) * m[..., None]).sum(axis=(1, 2, 3))
    per_case = sums/count
    return dict(per_case=per_case, equal_case=float(per_case.mean()), pooled=float(sums.sum()/count.sum()))


def _checked_next(predict_next, history):
    nxt = np.asarray(predict_next(history.copy()), dtype=float)
    if nxt.shape != history.shape[1:] or not np.isfinite(nxt).all():
        raise ValueError('predictor must return one finite state of the expected shape')
    return nxt


def autonomous(predict_next, initialization, steps):
    """Forecast without accepting any future truth argument."""
    past = np.asarray(initialization, dtype=float).copy()
    if past.ndim < 2 or len(past) < 1 or any(d == 0 for d in past.shape) or not np.isfinite(past).all():
        raise ValueError('initialization must be finite [history,...state]')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer')
    result = []
    for _ in range(steps):
        nxt = _checked_next(predict_next, past)
        result.append(nxt.copy())
        past = np.concatenate((past[1:], nxt[None]), axis=0)
    return np.stack(result)


def teacher_forced(predict_next, observed, history):
    """One-step predictions with real history at every target (a distinct task)."""
    values = np.asarray(observed, dtype=float)
    if (values.ndim < 2 or not np.isfinite(values).all() or isinstance(history, bool)
            or not isinstance(history, (int, np.integer)) or not 1 <= history < len(values)):
        raise ValueError('finite sequence and 1 <= history < frame count required')
    return np.stack([_checked_next(predict_next, values[t-history:t]) for t in range(history, len(values))])


def harmonic_summary(time, signal, frequency, amplitude_floor=1e-10):
    """Least-squares offset+sine+cosine at a SUPPLIED positive frequency.

    Phase is relative to time[0]. Frequency is not estimated here. Vanishing
    fitted amplitude makes phase undefined (None). Residuals must accompany
    phase: a fitted sinusoid need not describe the complete signal well.
    amplitude_floor is absolute, in the signal's units, and must be declared.
    """
    t, y = np.asarray(time, dtype=float), np.asarray(signal, dtype=float)
    if (t.ndim != 1 or y.shape != t.shape or len(t) < 4 or not np.isfinite(t).all()
            or not np.isfinite(y).all() or np.any(np.diff(t) <= 0)
            or not np.isfinite(frequency) or frequency <= 0
            or not np.isfinite(amplitude_floor) or amplitude_floor <= 0):
        raise ValueError('finite increasing times, matching signal and positive frequency/floor required')
    angle = 2*np.pi*frequency*(t-t[0])
    design = np.column_stack((np.ones(len(t)), np.sin(angle), np.cos(angle)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 3:
        raise ValueError('sample times cannot identify this sinusoid')
    amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    residual = float(np.mean((design@coefficients-y)**2))
    variance = float(np.var(y))
    return dict(offset=float(coefficients[0]), amplitude=amplitude,
                phase=None if amplitude <= amplitude_floor else float(np.arctan2(coefficients[2], coefficients[1])),
                frequency=float(frequency), phase_origin=float(t[0]), residual_mse=residual,
                r_squared=None if variance <= amplitude_floor**2 else 1-residual/variance)


def phase_metrics(predicted, reference):
    """Circular errors for paired phases; undefined pairs remain counted.

    None is the only accepted missing-value marker. NaN/inf are rejected.
    Signed mean is included as a cancellation diagnostic, never as accuracy.
    """
    p, q = list(predicted), list(reference)
    if len(p) != len(q) or not p:
        raise ValueError('phase lists must have equal, nonzero length')
    errors = []
    for a, b in zip(p, q):
        for value in (a, b):
            if value is not None and not np.isfinite(value):
                raise ValueError('phase must be finite or None')
        if a is not None and b is not None:
            errors.append(float(np.arctan2(np.sin(a-b), np.cos(a-b))))
    return dict(valid_pairs=len(errors), undefined_pairs=len(p)-len(errors),
                signed_mean=None if not errors else float(np.mean(errors)),
                mean_absolute=None if not errors else float(np.mean(np.abs(errors))),
                rms=None if not errors else float(np.sqrt(np.mean(np.square(errors)))))
