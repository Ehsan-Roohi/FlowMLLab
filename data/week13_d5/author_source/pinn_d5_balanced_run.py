"""Requeue-safe experiment runner. Original PDE/trainer are imported unchanged."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

from pinn_d5_balanced_experiment import balanced_points, SAMPLER_ID, TEST_SEED


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(argv, code_dir):
    # Everything affecting the objective/optimizer is immutable across requeues.
    args = [x for x in argv if x != '--resume']
    i = args.index('--max-hours')
    args = args[:i] + args[i+2:]
    init = Path(args[args.index('--init')+1])
    return dict(sampler=SAMPLER_ID, argv=args, init_sha256=digest(init),
                code={p.name: digest(p) for p in sorted(Path(code_dir).glob('*.py'))})


def check_restart(out, expected, resume):
    out = Path(out)
    marker = out/'experiment-contract.json'
    histpath = out/'history.json'
    if marker.exists():
        if json.loads(marker.read_text()) != expected:
            raise ValueError('Resume contract mismatch: code, objective, initialization or configuration changed')
        if not resume:
            raise ValueError('Existing experiment requires explicit --resume')
        if histpath.exists():
            h = json.loads(histpath.read_text())
            if h['step']:
                required = ['optimizer.pth', 'weights_opt.pth']
            elif h['soap_done']:
                required = ['soap.pth']
            else:
                required = []
            if any(not (out/name).is_file() for name in required):
                raise ValueError('Missing full checkpoint; refusing optimizer reset')
            if h.get('stop_reason') in ('nonfinite', 'nonfinite_soap', 'store_overflow'):
                raise ValueError('Failed checkpoint requires diagnosis')
            return h.get('stop_reason') == 'completed'
        if any(p.name != marker.name for p in out.iterdir()):
            raise ValueError('Partial outputs without history require inspection')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Refusing foreign output directory or old curvature')
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    return False


def regional_diagnostics(net, g, out, label):
    import numpy as np
    import torch
    import cavity
    rng = np.random.default_rng(907316)
    report = dict(label=label, physical_time=None, numerical_validation=False, regions={})
    dev = next(net.parameters()).device
    for region, (x0, x1, y0, y1) in {
        'full': (0., 1., -4., 1.), 'upper': (0., 1., 0., 1.),
        'upper_left': (0., .4, .4, 1.), 'floor': (0., 1., -4., -3.)}.items():
        z = rng.random((8192, 2))
        z = z * [x1-x0, y1-y0] + [x0, y0]
        z = z[(z[:,0]**2+(1-z[:,1])**2 > .02**2) &
              ((1-z[:,0])**2+(1-z[:,1])**2 > .02**2)]
        sums = np.zeros(3)
        for raw in np.array_split(z, 16):
            X = torch.as_tensor(raw, dtype=torch.float64, device=dev)
            u, v, _, _ = cavity.fields(net, X, g)
            r1, r2 = cavity.residual(net, X, g)
            E, _ = cavity.envelope(X[:,1:2], g)
            mask = (1-torch.exp(-(X[:,0:1]**2+(1-X[:,1:2])**2)/g.mask_radius**2)) * (1-torch.exp(-((1-X[:,0:1])**2+(1-X[:,1:2])**2)/g.mask_radius**2))
            scaled = r1.detach()**2+r2.detach()**2
            physical = scaled*(E/mask)**2
            sums += [float((u*u+v*v).sum()), float(physical.sum()), float(scaled.sum())]
        if not np.isfinite(sums).all():
            raise ValueError('Nonfinite independent regional diagnostics')
        report['regions'][region] = dict(n=len(z), velocity_rms=float(np.sqrt(sums[0]/len(z))),
            physical_momentum_rms=float(np.sqrt(sums[1]/len(z))), scaled_momentum_rms=float(np.sqrt(sums[2]/len(z))))
    folder = Path(out)/'diagnostics'
    folder.mkdir(exist_ok=True)
    (folder/(label+'-'+str(time.time_ns())+'.json')).write_text(json.dumps(report, indent=2))
    print('INDEPENDENT_REGIONAL_DIAGNOSTICS', label, report['regions'], flush=True)


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index('--out')+1])
    if argv[argv.index('--seed')+1] != '12345':
        raise ValueError('Held-out seed separation requires seed12345')
    expected = contract(argv, Path(__file__).parent)
    if check_restart(out, expected, '--resume' in argv):
        print('ALREADY_COMPLETED_MATCHED_EXPERIMENT', flush=True)
        return
    import torch
    import train
    import cavity
    torch.set_default_dtype(torch.float64)
    original_sample = train.sample_points
    original_save = train.atomic_save

    def sample(n, g, seed, device='cpu'):
        if (g.W, g.D, g.Re, g.envelope, g.kappa, g.y0_env, g.beta) != (1., 5., 1000., True, 4.212, .25, 10.):
            raise ValueError('Unexpected scientific configuration')
        if seed == TEST_SEED:
            return original_sample(n, g, seed, device)
        return torch.as_tensor(balanced_points(n, seed, g.W, g.D, g.excl_radius), dtype=torch.float64, device=device)

    def save(obj, path):
        original_save(obj, path)
        if Path(path).name != 'weights.pth':
            return
        frame = inspect.currentframe().f_back
        while frame and 'hist' not in frame.f_locals:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError('Cannot associate snapshot with iteration')
        h = frame.f_locals['hist']
        step = h['step'][-1] if h['step'] else h['soap_done']
        stage = 'ssb' if h['step'] else 'soap'
        label = f'{stage}-{step:07d}'
        folder = out/'snapshots'
        folder.mkdir(exist_ok=True)
        stem = label+'-'+str(time.time_ns())
        original_save(obj, str(folder/(stem+'.pth')))
        (folder/(stem+'.json')).write_text(json.dumps(dict(stage=stage, training_step=step,
            physical_time=None, geometry=h['geometry'], args=h['args'], sampler=SAMPLER_ID)))
        if step % 500 == 0 or h.get('stop_reason') == 'completed':
            regional_diagnostics(frame.f_locals['net'], frame.f_locals['g'], out, label)

    train.sample_points = sample
    train.atomic_save = save
    print('BALANCED_CONTRACT_OK_ORIGINAL_PHYSICS', expected['init_sha256'], flush=True)
    train.main()
    h = json.loads((out/'history.json').read_text())
    if h['stop_reason'] not in ('completed', 'wall_guard', 'wall_guard_soap'):
        raise RuntimeError('Training failure: '+str(h['stop_reason']))
    print('BALANCED_SEGMENT_EXIT', h['stop_reason'], h['soap_done'], h['step'][-1] if h['step'] else 0, flush=True)


if __name__ == '__main__':
    main()
