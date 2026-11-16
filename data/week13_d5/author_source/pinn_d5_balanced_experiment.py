"""Separate, unvalidated Re1000 D5 collocation experiment.

Preserves Chris's PDE, boundary representation and amplitude envelope.
Changes the equal-point loss distribution, NOT an unbiased uniform integral.
Never resume an optimizer trained on a different collocation objective.
No HPC submission is performed by this module.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np

SAMPLER_ID = 'd5-upper-floor-balanced-v1'
TEST_SEED = 12345 + 7919


def balanced_points(n, seed, W=1., D=5., exclude=.02):
    """50% full box, 30% upper width, 15% floor width, 5% floor corners.

    Both upper corners are excluded as in the existing refined run. Upper
    sampling spans the whole width: it does not force a desired vortex sign
    or fit CFD data. Deep cells retain full-domain coverage and E scaling.
    """
    if n < 0 or W != 1. or D != 5. or not 0 < exclude < .1:
        raise ValueError('This experiment requires n>=0, W=1, D=5, 0<exclude<.1')
    if n == 0:
        return np.empty((0, 2))
    rng = np.random.default_rng(seed)
    counts = [n // 2, 3*n // 10, 15*n // 100]
    counts.append(n - sum(counts))
    blocks = []
    for kind, count in enumerate(counts):
        accepted, left = [], count
        while left:
            z = rng.random((left + 32, 2))
            x = W*z[:, 0]
            if kind == 0:
                y = 1-D+D*z[:, 1]
            elif kind == 1:
                y = 1-W+W*z[:, 1]
            elif kind == 2:
                y = 1-D+W*z[:, 1]
            else:
                r = W*np.exp(np.log(1e-4)+z[:, 0]*np.log(2500.))
                theta = .01+z[:, 1]*(np.pi/2-.02)
                x, y = r*np.cos(theta), 1-D+r*np.sin(theta)
                side = rng.random(len(x)) < .5
                x[side] = W-x[side]
            keep = (x*x+(1-y)**2 > exclude**2) & ((W-x)**2+(1-y)**2 > exclude**2)
            block = np.column_stack((x[keep], y[keep]))[:left]
            accepted.append(block)
            left -= len(block)
        blocks.append(np.concatenate(accepted) if count else np.empty((0, 2)))
    points = np.concatenate(blocks)
    return points[rng.permutation(n)]


def main():
    # Validate before importing/starting the GPU trainer. This deliberately
    # offers no resume path until objective/configuration provenance is audited.
    if '--resume' in sys.argv:
        raise ValueError('New objective: do not reuse baseline/refined curvature; resume unsupported')
    out = Path(sys.argv[sys.argv.index('--out')+1])
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory; existing results are protected')
    import torch
    import train
    original = train.sample_points

    def sample(n, g, seed, device='cpu'):
        if (g.W, g.D, g.Re) != (1., 5., 1000.) or not g.envelope:
            raise ValueError('Requires Re1000 D5 W1 and the unchanged envelope')
        if (g.kappa, g.y0_env, g.beta) != (4.212, .25, 10.):
            raise ValueError('Keep the existing envelope pinned; knee changes require a separate experiment')
        if seed == TEST_SEED:
            return original(n, g, seed, device)
        return torch.as_tensor(balanced_points(n, seed, g.W, g.D, g.excl_radius),
                               dtype=torch.float64, device=device)

    # Restrict seed so the trainer's independent held-out stream stays untouched.
    if '--seed' not in sys.argv or sys.argv[sys.argv.index('--seed')+1] != '12345':
        raise ValueError('Requires seed 12345 for held-out stream separation')
    out.mkdir(parents=True, exist_ok=True)
    (out/'sampling_experiment.json').write_text(json.dumps(dict(
        sampler=SAMPLER_ID, fractions=[.5, .3, .15, .05],
        regions=['full', 'upper-width', 'floor-width', 'floor-corners'],
        independent_test='original sampler; seed 20264',
        numerical_validation=False, physics_changed=False,
        objective_changed=True), indent=2), encoding='utf-8')
    os.environ['PYTHONUNBUFFERED'] = '1'
    train.sample_points = sample
    train.main()


if __name__ == '__main__':
    main()
