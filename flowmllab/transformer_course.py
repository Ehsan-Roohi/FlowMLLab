"""Active Transformer laboratories, with explicit information and compute budgets.

All fitting functions receive development data only. Retained tests are scored
after checkpoint selection. No claims of blind discovery or grid convergence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import hashlib
import json
import platform
import time
import importlib.metadata

import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize_scalar
import torch
from torch import nn


@dataclass(frozen=True)
class Protocol:
    rank: int = 8
    context: int = 4
    width: int = 24
    branch_width: int = 208
    history_mlp_width: int = 250
    seeds: tuple = (17, 29, 43)
    sensor_count: int = 16
    noise_fraction: float = .01
    train_re: tuple = (90, 110)
    validation_re: int = 100
    evaluation_re: int = 105
    train_end: int = 160
    validation_end: int = 210
    test_end: int = 281
    max_steps: int = 1200
    eval_every: int = 30
    patience: int = 240
    learning_rate: float = .003
    ridge: float = 1e-6
    source_re: int = 90
    target_res: tuple = (100, 105, 110)
    transfer_blocks: tuple = (4, 8, 15)
    block_length: int = 10
    pretrain_steps: int = 300
    adaptation_steps: int = 300
    threads: int = 2


def seed_all(seed=17, threads=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(path):
    """Normalize only text line endings; binary evidence remains byte-exact."""
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix in {'.py', '.md', '.json', '.ipynb', '.txt', '.toml'}:
        raw = raw.replace(b'\r\n', b'\n')
    return hashlib.sha256(raw).hexdigest()


def load_data(root):
    folder = Path(root) / 'data/modal_labs'
    manifest = json.loads((folder / 'manifest.json').read_text())
    cases = {}
    for row in manifest['cases']:
        p = folder / row['file']
        if digest(p) != row['sha256']:
            raise ValueError(f'Input hash mismatch: {p.name}')
        with np.load(p, allow_pickle=False) as data:
            cases[int(row['reynolds'])] = {k: data[k].copy() for k in data.files}
    return cases, manifest


@dataclass
class Representation:
    mean: np.ndarray
    modes: np.ndarray
    scale: np.ndarray
    retained_energy: float

    @classmethod
    def fit(cls, fields, rank=8):
        a = np.asarray(fields, dtype=np.float64).reshape(len(fields), -1)
        mean = a.mean(0)
        # A small temporal Gram matrix avoids an expensive full spatial SVD.
        c = a - mean
        eig, vectors = np.linalg.eigh(c @ c.T)
        order = np.argsort(eig)[::-1]
        eig = np.maximum(eig[order], 0)
        keep = min(rank, int(np.sum(eig > max(eig[0], 1.) * 1e-12)))
        if keep != rank:
            raise ValueError('Training snapshots have insufficient numerical rank')
        modes = c.T @ vectors[:, order[:rank]] / np.sqrt(eig[:rank])
        # Resolve isolated-vector sign ambiguity. Near-degenerate rotations still
        # require saving the exact basis; canonical signs are not a substitute.
        pivots = np.argmax(np.abs(modes), axis=0)
        modes *= np.where(modes[pivots, np.arange(rank)] < 0, -1., 1.)
        scale = np.maximum((c @ modes).std(0), 1e-8)
        return cls(mean, modes, scale, float(eig[:rank].sum() / eig.sum()))

    def encode(self, fields):
        a = np.asarray(fields).reshape(len(fields), -1)
        return (a - self.mean) @ self.modes / self.scale

    def decode(self, coefficients):
        return (np.asarray(coefficients) * self.scale) @ self.modes.T + self.mean

    def tensors(self):
        return {k: torch.as_tensor(v) for k, v in asdict(self).items()}

    @classmethod
    def from_tensors(cls, values):
        return cls(values['mean'].numpy(), values['modes'].numpy(), values['scale'].numpy(), float(values['retained_energy']))


def attention(q, k, v, causal=False):
    scores = q @ k.transpose(-2, -1) / np.sqrt(q.shape[-1])
    if causal:
        if q.shape[-2] != k.shape[-2]:
            raise ValueError('Causal self-attention requires equal sequence lengths')
        scores = scores.masked_fill(torch.ones_like(scores, dtype=torch.bool).triu(1), float('-inf'))
    weights = scores.softmax(-1)
    return weights @ v, weights


class Block(nn.Module):
    def __init__(self, width=24, heads=4, causal=False):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a, _ = attention(q, k, v, self.causal)
        z = x + self.out(a.transpose(1, 2).reshape(b, n, d))
        return z + self.ff(z)


class CausalDecoder(nn.Module):
    """Supervise EVERY position: output[:,t] predicts input sequence position t+1."""
    def __init__(self, rank=8, context=4, width=24):
        super().__init__()
        self.embed = nn.Linear(rank, width)
        self.position = nn.Parameter(torch.zeros(context, width))
        self.blocks = nn.ModuleList([Block(width, causal=True), Block(width, causal=True)])
        self.out = nn.Linear(width, rank)

    def forward(self, x):
        z = self.embed(x) + self.position[:x.shape[1]]
        for block in self.blocks:
            z = block(z)
        return self.out(z)

    def next(self, x):
        return self(x)[:, -1]


class HistoryMLP(nn.Module):
    def __init__(self, rank=8, context=4, width=250):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(rank * context, width), nn.Tanh(), nn.Linear(width, rank))

    def forward(self, x):
        return self.net(x)

    def next(self, x):
        return self(x)


class SensorSet(nn.Module):
    """Variable-length (normalized value, x/D, y/D) tokens; no sensor index ID."""
    def __init__(self, rank=8, width=24):
        super().__init__()
        self.embed = nn.Linear(3, width)
        self.block = Block(width)
        self.query = nn.Parameter(torch.zeros(width))
        self.out = nn.Linear(width, rank)

    def forward(self, tokens):
        z = self.block(self.embed(tokens))
        weight = (z @ self.query / np.sqrt(z.shape[-1])).softmax(1)
        return self.out((weight[..., None] * z).sum(1))


class TinyLM(nn.Module):
    def __init__(self, vocab, context=24, width=24):
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.zeros(context, width))
        self.block = Block(width, causal=True)
        self.out = nn.Linear(width, vocab)

    def forward(self, ids):
        return self.out(self.block(self.embed(ids) + self.position[:ids.shape[1]]))


def sequence_windows(z, context=4):
    if len(z) <= context:
        raise ValueError('Not enough frames for a shifted sequence')
    return (np.stack([z[i:i+context] for i in range(len(z)-context)]),
            np.stack([z[i+1:i+context+1] for i in range(len(z)-context)]))


def infer(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()


def rollout(model, history, steps):
    past = torch.tensor(history[None], dtype=torch.float32)
    result = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            nxt = model.next(past)
            if not torch.isfinite(nxt).all() or float(nxt.abs().max()) > 1e8:
                raise FloatingPointError('Unstable autonomous trajectory')
            result.append(nxt[0].numpy())
            past = torch.cat([past[:, 1:], nxt[:, None]], 1)
    return np.asarray(result)


def safe_rollout_loss(model, history, truth):
    try:
        return float(np.mean((rollout(model, history, len(truth)) - truth)**2))
    except (FloatingPointError, ValueError):
        return float('inf')


def fit(model, x, y, validation, protocol, steps=None, early_stop=True, augment=None):
    """Only a validation callback selects state. Infinity rejects a checkpoint."""
    x, y = [torch.as_tensor(a, dtype=torch.float32) for a in (x, y)]
    opt = torch.optim.Adam(model.parameters(), lr=protocol.learning_rate)
    ceiling = protocol.max_steps if steps is None else steps
    best, selected, state, trace = float('inf'), 0, copy.deepcopy(model.state_dict()), []
    started = time.perf_counter()
    last = 0
    for step in range(ceiling + 1):
        last = step
        if step % protocol.eval_every == 0 or step == ceiling:
            model.eval()
            with torch.no_grad():
                val = float(validation(model))
                train_loss = float(((model(x) - y)**2).mean())
            trace.append({'step': step, 'train_mse': train_loss, 'validation_mse': val if np.isfinite(val) else None})
            if np.isfinite(val) and val < best - 1e-10:
                best, selected, state = val, step, copy.deepcopy(model.state_dict())
            if early_stop and step - selected >= protocol.patience:
                break
        if step < ceiling:
            model.train()
            opt.zero_grad()
            xx = augment(x, step) if augment else x
            loss = ((model(xx) - y)**2).mean()
            if not torch.isfinite(loss):
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
    model.load_state_dict(state)
    model.eval()
    return {'selected_step': selected, 'executed_steps': last, 'ceiling': ceiling,
            'at_budget_boundary': selected == ceiling, 'validation_mse': best if np.isfinite(best) else None,
            'history': trace, 'seconds': time.perf_counter() - started,
            'parameters': sum(p.numel() for p in model.parameters())}


def error_components(prediction, truth, representation):
    # Promote before subtraction and norm accumulation on large CFD arrays.
    p = np.asarray(prediction, dtype=np.float64).reshape(len(truth), -1)
    y = np.asarray(truth, dtype=np.float64).reshape(len(truth), -1)
    oracle = representation.decode(representation.encode(y))
    norm = max(float(np.linalg.norm(y)), 1e-14)
    total = float(np.linalg.norm(p-y) / norm)
    floor = float(np.linalg.norm(oracle-y) / norm)
    dynamics = float(np.linalg.norm(p-oracle) / norm)
    return {'field_relative_l2': total, 'representation_floor': floor,
            'in_subspace_relative_l2': dynamics,
            'pythagorean_residual': total**2-floor**2-dynamics**2,
            'per_frame_relative_l2': (np.linalg.norm(p-y, axis=1)/np.maximum(np.linalg.norm(y, axis=1), 1e-14)).tolist()}


def runtime_environment():
    """Current scoring/inference environment, distinct from training provenance."""
    return {'python':platform.python_version(),'platform':platform.platform(),
            'packages':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch')},
            'torch_threads':torch.get_num_threads(),'torch_build':torch.__config__.show()}


def checkpoint_agreement(actual, expected, truth):
    """Separate historical elementwise fidelity from field-level scientific drift.

    The scientific threshold was introduced after external review, not before
    the original experiment: 1e-5 of truth field norm and 1e-5 absolute drift in
    relative-L2 error (0.001 percentage point). It is much smaller than a 1%
    reconstruction error. It is not evidence that an untested platform passes.
    """
    actual, expected, truth = [np.asarray(a,dtype=np.float64).reshape(len(truth),-1)
                               for a in (actual,expected,truth)]
    norm=max(float(np.linalg.norm(truth)),1e-14)
    difference=float(np.linalg.norm(actual-expected)/norm)
    original_error=float(np.linalg.norm(expected-truth)/norm)
    restored_error=float(np.linalg.norm(actual-truth)/norm)
    strict=bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-6))
    drift=abs(restored_error-original_error)
    return {'passed':strict,'strict_passed':strict,
            'max_absolute_error':float(np.max(abs(actual-expected))),
            'truth_normalized_prediction_difference':difference,
            'reference_field_relative_l2':original_error,
            'reloaded_field_relative_l2':restored_error,
            'field_relative_l2_absolute_drift':drift,
            'scientific_passed':bool(difference<=1e-5 and drift<=1e-5)}


def spectral_fit(t, signal, bounds=(.1, .3)):
    """Full-rollout frequency and phase at the first sample, not absolute zero.

    This probe fit is not force-derived Strouhal.
    """
    t, y = np.asarray(t, dtype=np.float64), np.asarray(signal, dtype=np.float64)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 3:
        raise ValueError('Spectrum requires matching one-dimensional samples')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Spectrum requires finite samples and increasing time')
    elapsed = t - t[0]
    # A float32 mean can differ from a repeated constant through rounding.
    if float(np.ptp(y)) == 0. or float(np.std(y)) < 1e-12:
        return {'frequency':None,'amplitude':0.,'phase':None,'residual_mse':0.,'r_squared':None,'duration':float(t[-1]-t[0])}
    def solve(f):
        design = np.c_[np.ones(len(t)), np.sin(2*np.pi*f*elapsed), np.cos(2*np.pi*f*elapsed)]
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        return coef, float(np.mean((design @ coef-y)**2))
    grid = np.linspace(*bounds, 201)
    j = int(np.argmin([solve(f)[1] for f in grid]))
    lo, hi = grid[max(0,j-1)], grid[min(len(grid)-1,j+1)]
    fit_ = minimize_scalar(lambda f: solve(f)[1], bounds=(lo,hi), method='bounded', options={'xatol':1e-10})
    coef, mse = solve(fit_.x)
    return {'frequency': float(fit_.x), 'amplitude': float(np.hypot(coef[1],coef[2])),
            'phase': float(np.arctan2(coef[2],coef[1])), 'residual_mse': mse,
            'r_squared': float(1. - mse / np.mean((y-y.mean())**2)),
            'duration': float(t[-1]-t[0])}


def qr_sensors(representation, count=16):
    _, _, pivots = qr(representation.modes.T, pivoting=True, mode='economic')
    selected=list(pivots[:min(count,representation.modes.shape[1])])
    while len(selected)<count:
        basis=representation.modes
        information=basis[selected].T@basis[selected]+np.eye(basis.shape[1])*1e-10
        scores=np.sum((basis@np.linalg.inv(information))*basis,axis=1)
        scores[selected]=-np.inf
        selected.append(int(np.argmax(scores)))
    return np.asarray(selected)


def sensor_tokens(fields, ids, case, mean, scale):
    yy, xx = np.meshgrid(case['y'], case['x'], indexing='ij')
    coords = np.c_[xx.ravel(), yy.ravel()][ids]
    # Coordinates use (coordinate - grid mean) / full grid span.
    # Uniform symmetric grid endpoints are -0.5 and +0.5, not -1 and +1.
    coords = (coords - np.array([case['x'].mean(), case['y'].mean()])) / np.array([np.ptp(case['x']), np.ptp(case['y'])])
    values = (np.asarray(fields).reshape(len(fields), -1)[:, ids] - mean) / scale
    return np.concatenate([values[...,None], np.broadcast_to(coords, (len(values),len(ids),2))], axis=-1)


def linear_fit(x, y, penalty=1e-6):
    x = np.c_[np.ones(len(x)), x]
    regularizer = np.eye(x.shape[1])*penalty
    regularizer[0,0] = 0
    return np.linalg.solve(x.T@x+regularizer, x.T@y)


def linear_predict(weights, x):
    return np.c_[np.ones(len(x)),x]@weights


def sensor_experiment(cases, p):
    training = np.concatenate([cases[r]['v'][::2] for r in p.train_re])
    val, test = cases[p.validation_re]['v'], cases[p.evaluation_re]['v']
    rep = Representation.fit(training,p.rank)
    ids = qr_sensors(rep,p.sensor_count)
    mean, scale = float(training.mean()), float(training.std())
    arrays = [sensor_tokens(a,ids,cases[p.source_re],mean,scale) for a in (training,val,test)]
    rng = np.random.default_rng(1901)
    for a in arrays:
        a[:,:,0] += rng.normal(0,p.noise_fraction,a[:,:,0].shape)
    x,vx,tx = arrays
    y,vy = rep.encode(training),rep.encode(val)
    rows, fields, states = [],{},{}
    # Fixed sensor models get a declared mean-imputation control under dropout.
    for seed in p.seeds:
        for name in ('Sensor-set','POD-DeepONet'):
            seed_all(seed,p.threads)
            model = SensorSet(p.rank,p.width) if name=='Sensor-set' else nn.Sequential(nn.Linear(len(ids),p.branch_width),nn.Tanh(),nn.Linear(p.branch_width,p.rank))
            xx,vxx = (x,vx) if name=='Sensor-set' else (x[:,:,0],vx[:,:,0])
            def augment(a,step):
                # Training sees variable cardinalities. Coordinates follow values.
                count = len(ids) if step%3==0 else max(p.rank, len(ids)//2)
                return a[:,torch.randperm(len(ids))[:count]]
            record=fit(model,xx,y,lambda m:np.mean((infer(m,vxx)-vy)**2),p,
                       augment=augment if name=='Sensor-set' else None)
            key=f'{name}-{seed}'
            states[key]={'state':copy.deepcopy(model.state_dict()),'architecture':name,
                         'representation':rep.tensors(),'sensor_ids':torch.tensor(ids),
                         'value_mean':mean,'value_scale':scale,'protocol':asdict(p)}
            for condition,keep in [('all',np.arange(len(ids))),('drop-half',np.arange(0,len(ids),2))]:
                if name=='Sensor-set':
                    inputs=tx[:,keep]
                else:
                    inputs=tx[:,:,0].copy()
                    inputs[:,np.setdiff1d(np.arange(len(ids)),keep)]=0.
                prediction=rep.decode(infer(model,inputs))
                fieldkey=f'{key}-{condition}'
                fields[fieldkey]=prediction
                rows.append({'key':fieldkey,'method':name,'seed':seed,'condition':condition,'training':record,
                             'metrics':error_components(prediction,test,rep)})
    weights=linear_fit(x[:,:,0],y,p.ridge)
    for condition,keep in [('all',np.arange(len(ids))),('drop-half',np.arange(0,len(ids),2))]:
        inp=tx[:,:,0].copy();inp[:,np.setdiff1d(np.arange(len(ids)),keep)]=0
        ridge=rep.decode(linear_predict(weights,inp))
        observations=tx[:,keep,0]*scale+mean
        coef=np.linalg.lstsq(rep.modes[ids[keep]],(observations-rep.mean[ids[keep]]).T,rcond=None)[0].T
        gappy=coef@rep.modes.T+rep.mean
        for name,prediction in [('Ridge-imputation',ridge),('Gappy-POD-variable',gappy)]:
            key=f'{name}-{condition}';fields[key]=prediction
            rows.append({'key':key,'method':name,'seed':None,'condition':condition,'metrics':error_components(prediction,test,rep)})
    return {'rows':rows,'sensor_ids':ids.tolist(),'retained_energy':rep.retained_energy,
            'noise_definition':'Gaussian sigma=0.01 of global training-field standard deviation',
            'scope':'All/drop-half only; moving-location generalization is not claimed.'},fields,states


def temporal_model(name,p):
    return CausalDecoder(p.rank,p.context,p.width) if name=='Causal-Transformer' else HistoryMLP(p.rank,p.context,p.history_mlp_width)


def forecast_experiment(cases,p):
    case=cases[110]; field=case['omega']; end,ve,te=p.train_end,p.validation_end,p.test_end
    rep=Representation.fit(field[:end],p.rank);z=rep.encode(field)
    x,y=sequence_windows(z[:end],p.context)
    rows,fields,states=[],{},{}
    for seed in p.seeds:
        for name in ('Causal-Transformer','History-MLP'):
            seed_all(seed,p.threads); model=temporal_model(name,p)
            targets=y if name=='Causal-Transformer' else y[:,-1]
            record=fit(model,x,targets,lambda m:safe_rollout_loss(m,z[end-p.context:end],z[end:ve]),p)
            pred=rep.decode(rollout(model,z[end-p.context:end],te-end))
            key=f'{name}-{seed}';fields[key]=pred
            # Additional start times use observed histories: separate reset protocol.
            starts={str(start):error_components(rep.decode(rollout(model,z[start-p.context:start],te-start)),field[start:te],rep)
                    for start in (ve,ve+20) if start<te}
            rows.append({'key':key,'method':name,'seed':seed,'training':record,
                         'metrics':error_components(pred[ve-end:],field[ve:te],rep),
                         'additional_observed_initializations':starts})
            states[key]={'state':copy.deepcopy(model.state_dict()),'architecture':name,
                         'representation':rep.tensors(),'protocol':asdict(p)}
    operator=np.linalg.lstsq(z[:end-1],z[1:end],rcond=None)[0]
    a=z[end-1].copy();pred=[]
    for _ in range(te-end):
        a=a@operator;pred.append(a.copy())
    controls={'DMD-r8':rep.decode(pred),'Persistence':np.repeat(field[end-1:end].reshape(1,-1),te-end,axis=0),
              'POD-oracle':rep.decode(z[end:te])}
    for name,pred in controls.items():
        fields[name]=pred
        rows.append({'key':name,'method':name,'seed':None,'metrics':error_components(pred[ve-end:],field[ve:te],rep)})
    iy=int(np.argmin(abs(case['y']-.5)));ix=int(np.argmin(abs(case['x']-4)))
    probe=np.ravel_multi_index((iy,ix),field.shape[1:]);t=case['t'][end:te]
    spectra={'reference':spectral_fit(t,field[end:te].reshape(te-end,-1)[:,probe])}
    for key,values in fields.items():
        spectra[key]=spectral_fit(t,values[:,probe])
        if spectra[key]['frequency'] is None:
            spectra[key].update(frequency_relative_error=None,initial_phase_error=None,phase_drift_over_horizon=None)
            continue
        spectra[key]['frequency_relative_error']=abs(spectra[key]['frequency']/spectra['reference']['frequency']-1)
        delta=spectra[key]['phase']-spectra['reference']['phase']
        spectra[key]['initial_phase_error']=float(np.arctan2(np.sin(delta),np.cos(delta)))
        spectra[key]['phase_drift_over_horizon']=float(2*np.pi*(spectra[key]['frequency']-spectra['reference']['frequency'])*(t[-1]-t[0]))
    return {'rows':rows,'spectral_fit':spectra,'probe_index':int(probe),
            'legacy_MLP_reference':{'relative_l2':.05537,'source':'results/modal_labs/metrics.json',
                                    'status':'Published Week 7 reference; use the new matched History-MLP for direct comparisons.'}},fields,states


def transfer_indices(block_count, context=4):
    if block_count not in (4,8,15):
        raise ValueError('Declared block budgets are 4, 8, 15')
    nval={4:1,8:2,15:3}[block_count]
    ntrain=block_count-nval
    trainblocks=np.unique(np.linspace(0,11,ntrain,dtype=int))*10
    valblocks=np.arange(15-nval,15)*10
    train=np.concatenate([np.arange(s,s+10) for s in trainblocks])
    val=np.concatenate([np.arange(s,s+10) for s in valblocks])
    initial=np.arange(160-context,160)
    assert not set(train)&set(val) and not set(initial)&set(train)|set(initial)&set(val)
    return trainblocks,valblocks,{'training_frames':train.tolist(),'validation_frames':val.tolist(),
                                  'initialization_frames':initial.tolist(),'total_target_labels':len(set(train)|set(val)|set(initial))}


def transfer_experiment(cases,p):
    source=cases[p.source_re]['omega']; rep=Representation.fit(source[:p.train_end],p.rank)
    zs=rep.encode(source);sx,sy=sequence_windows(zs[:p.train_end],p.context)
    rows,fields,states=[],{},{}
    for seed in p.seeds:
        seed_all(seed,p.threads);pre=CausalDecoder(p.rank,p.context,p.width)
        pre_record=fit(pre,sx,sy,lambda m:safe_rollout_loss(m,zs[p.train_end-p.context:p.train_end],zs[p.train_end:p.validation_end]),p,
                       steps=p.pretrain_steps,early_stop=False)
        for target_re in p.target_res:
            field=cases[target_re]['omega']
            for budget in p.transfer_blocks:
                trainblocks,valblocks,labels=transfer_indices(budget,p.context)
                targetrep=Representation.fit(field[labels['training_frames']],p.rank)
                for arm in ('scratch','pretrained','matched-steps','target-POD-MLP'):
                    representation=targetrep if arm=='target-POD-MLP' else rep
                    z=representation.encode(field)
                    windows=[sequence_windows(z[s:s+p.block_length],p.context) for s in trainblocks]
                    x=np.concatenate([v[0] for v in windows]); y=np.concatenate([v[1] for v in windows])
                    seed_all(seed,p.threads)
                    model=copy.deepcopy(pre) if arm=='pretrained' else (HistoryMLP(p.rank,p.context,p.history_mlp_width) if arm=='target-POD-MLP' else CausalDecoder(p.rank,p.context,p.width))
                    steps=p.adaptation_steps+(p.pretrain_steps if arm=='matched-steps' else 0)
                    def validation(m):
                        return np.mean([safe_rollout_loss(m,z[s:s+p.context],z[s+p.context:s+p.block_length]) for s in valblocks])
                    record=fit(model,x,y[:,-1] if arm=='target-POD-MLP' else y,validation,p,steps=steps,early_stop=False)
                    prediction=representation.decode(rollout(model,z[160-p.context:160],p.test_end-160))
                    key=f'Re{target_re}-B{budget}-{arm}-{seed}';fields[key]=prediction
                    rows.append({'key':key,'target_re':target_re,'block_budget':budget,'arm':arm,'seed':seed,
                                 'labels':labels,'training':record,'pretraining':pre_record if arm=='pretrained' else None,
                                 'total_optimizer_steps':record['executed_steps']+(pre_record['executed_steps'] if arm=='pretrained' else 0),
                                 'metrics':error_components(prediction[p.validation_end-160:],field[p.validation_end:p.test_end],representation)})
                    states[key]={'state':copy.deepcopy(model.state_dict()),'architecture':'History-MLP' if arm=='target-POD-MLP' else 'Causal-Transformer',
                                 'representation':representation.tensors(),'protocol':asdict(p),'labels':labels}
                # Classical dynamics use only adjacent labelled training pairs.
                z=rep.encode(field)
                xx=np.concatenate([z[s:s+9] for s in trainblocks]);yy=np.concatenate([z[s+1:s+10] for s in trainblocks])
                operator=np.linalg.lstsq(xx,yy,rcond=None)[0];a=z[159].copy();pred=[]
                for _ in range(p.test_end-160):
                    a=a@operator;pred.append(a.copy())
                prediction=rep.decode(pred);key=f'Re{target_re}-B{budget}-DMD-{seed}';fields[key]=prediction
                rows.append({'key':key,'target_re':target_re,'block_budget':budget,'arm':'DMD','seed':seed,'labels':labels,
                             'metrics':error_components(prediction[p.validation_end-160:],field[p.validation_end:p.test_end],rep)})
    return {'rows':rows,'source_re':p.source_re,'initialization':f'All arms observe frames {160-p.context}:160; predict 160:{p.test_end} without resets.',
            'scope':'Retained-case audit extending Week 7.4. Equal optimizer steps are not equal FLOPs or source-data access.'},fields,states


def checkpoint_model(bundle):
    p=Protocol(**bundle['protocol']);name=bundle['architecture']
    model=(SensorSet(p.rank,p.width) if name=='Sensor-set' else
           nn.Sequential(nn.Linear(len(bundle['sensor_ids']),p.branch_width),nn.Tanh(),nn.Linear(p.branch_width,p.rank)) if name=='POD-DeepONet' else temporal_model(name,p))
    model.load_state_dict(bundle['state']);model.eval()
    return model,Representation.from_tensors(bundle['representation'])


def pack_prediction(values):
    """Near-lossless low-rank storage; these storage factors are NOT fitted POD."""
    a=np.asarray(values,dtype=np.float64);mean=a.mean(0);c=a-mean
    eig,u=np.linalg.eigh(c@c.T);order=np.argsort(eig)[::-1]
    eig=np.maximum(eig[order],0)
    keep=min(8,int(np.sum(eig>max(eig[0],1e-20)*1e-12)))
    if keep:
        coefficients=u[:,order[:keep]]*np.sqrt(eig[:keep])
        modes=c.T@u[:,order[:keep]]/np.sqrt(eig[:keep])
    else:
        coefficients=np.zeros((len(a),0));modes=np.zeros((a.shape[1],0))
    reconstructed=coefficients@modes.T+mean
    if not np.allclose(reconstructed,a,rtol=1e-8,atol=1e-8):
        raise ValueError('Prediction storage compression exceeded declared tolerance')
    return {'mean':mean,'coefficients':coefficients,'storage_modes':modes}


def prediction(archive,key):
    return archive[key+'__coefficients']@archive[key+'__storage_modes'].T+archive[key+'__mean']


def run(root, output, protocol=None):
    root,output=Path(root),Path(output);p=protocol or Protocol()
    if (p.train_end,p.validation_end,p.test_end,p.block_length)!=(160,210,281,10):
        raise ValueError('This retained-data protocol fixes boundaries 160/210/281 and ten-frame label blocks')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty experiment directory')
    output.mkdir(parents=True,exist_ok=True)
    seed_all(17,p.threads);json_write(output/'plan.json',asdict(p))
    cases,inputs=load_data(root);allfields={};allstates={}
    for name,experiment in [('sensors',sensor_experiment),('forecast',forecast_experiment),('transfer',transfer_experiment)]:
        print('Running',name,flush=True)
        record,fields,states=experiment(cases,p)
        json_write(output/f'{name}.json',record)
        for key,values in fields.items():
            allfields.update({name+'__'+key+'__'+suffix:a for suffix,a in pack_prediction(values).items()})
        allstates.update({name+'__'+k:v for k,v in states.items()})
    np.savez_compressed(output/'predictions.npz',**allfields)
    torch.save(allstates,output/'checkpoints.pt')
    modules=['flowmllab/transformer_course.py','qa/run_transformer_course.py','qa/plot_transformer_course.py']
    manifest={'input_manifest':inputs,'source_hashes':{x:canonical_digest(root/x) for x in modules},
              'hash_rule':'Text LF normalization, binary byte-exact SHA-256',
              'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
                             'packages':{x:importlib.metadata.version(x) for x in ('numpy','scipy','torch','pandas','matplotlib','nbformat','nbclient','reportlab')},
                             'torch_build':torch.__config__.show(),'threads':p.threads},
              'files':{x.name:digest(x) for x in output.iterdir() if x.is_file()}}
    json_write(output/'manifest.json',manifest)
    print('Saved complete predictions, model weights, representations, protocol and environment.',flush=True)
