"""One preregistered, area-unbiased sampling variant of the frozen Ghia PINN.

The original model, corner lifting, physical derivatives and audits are imported
unchanged. No reference values enter fitting. Training ends at the paired
baseline's recorded residual-evaluation count, including rejected line searches.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import numpy as np
import torch

BASE_SOURCE_SHA256 = '507a6e0a6d632a173c8bdc8748564beffee77ce3664aeffcee25edc5ca305b9e'
BASELINE_ARCHIVE_SHA256 = 'd041091ee84b5e1150fc186b2f15087852f0e1d2929be78ce96d81558c953235'
BASE_PATH = Path(__file__).with_name('cavity_pinn_campaign.py')
if hashlib.sha256(BASE_PATH.read_text(encoding='utf-8-sig').encode()).hexdigest() != BASE_SOURCE_SHA256:
    raise RuntimeError('The immutable baseline physics source changed; explicit review required')
_spec = importlib.util.spec_from_file_location('_stratified_original_cavity_physics', BASE_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)
DTYPE = base.DTYPE
SCHEMA = 1
STRATA = ('top_corner_annuli', 'remaining_wall_band', 'interior')
RADIUS = 0.05
CavityPINN = base.CavityPINN
predict = base.predict
final_audit = base.final_audit
numerical_admission = base.numerical_admission
tensor_hash = base.tensor_hash
residuals = base.residuals


def source_hash():
    return hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_hash(value):
    """Portable hash for nested checkpoint state, including optimizer and RNG."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b'tensor'); visit(item.detach().cpu().contiguous().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(b'array'); visit(str(item.dtype)); visit(item.shape)
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode()); visit(len(item))
            for child in item: visit(child)
        elif isinstance(item, (str, bytes, bool, int, float)) or item is None:
            data = repr(item).encode()
            digest.update(type(item).__name__.encode() + str(len(data)).encode() + b':' + data)
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            raise TypeError(f'Unsupported checkpoint hash type: {type(item)}')
    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class Config(base.Config):
    chunk_size: int = 2048
    closure_budget: int = 2500
    study_kind: str = 'controlled_variant'
    sampling: str = 'fixed_conditional_uniform_area_strata_v1'

    def __post_init__(self):
        super().__post_init__()
        if self.collar not in (.01, .02) or self.collocation % 8:
            raise ValueError('Two declared collars and a point count divisible by eight required')
        if self.closure_budget < max(1, self.adam_steps):
            raise ValueError('Closure budget must cover the complete Adam stage')
        if self.study_kind not in ('controlled_variant', 'cpu_admission'):
            raise ValueError('Unknown study kind')
        if self.sampling != 'fixed_conditional_uniform_area_strata_v1':
            raise ValueError('Sampling design is frozen')
        if self.study_kind == 'controlled_variant':
            if (self.reynolds not in (100, 7500) or self.seed not in (7, 23, 61)
                    or (self.width, self.depth, self.collocation, self.adam_steps, self.lbfgs_steps,
                        self.adam_lr, self.lbfgs_history, self.audit_points, self.audit_seed,
                        self.export_grid) != (64, 4, 2048, 2000, 500, .001, 50, 2048, 983451, 129)):
                raise ValueError('Production is the predeclared 12-case sampling-only intervention')


def stratum_areas(collar):
    if collar not in (.01, .02): raise ValueError('Undeclared collar')
    return np.array([math.pi / 2 * (RADIUS ** 2 - collar ** 2),
                     1 - .9 ** 2 - math.pi * RADIUS ** 2 / 2, .9 ** 2])


def stratum_membership(points, collar):
    radius = base.corner_radius(points)
    inside = (points > 0).all(1) & (points < 1).all(1)
    central = ((points > RADIUS) & (points < 1 - RADIUS)).all(1)
    result = torch.full((len(points),), -1, dtype=torch.int64)
    result[inside & (radius > collar) & (radius < RADIUS)] = 0
    result[inside & ~central & (radius >= RADIUS)] = 1
    result[inside & central] = 2
    return result


def stratified_points(n, generator, collar):
    """Uniform in area, fixed quotas, no residual-driven selection or redraw."""
    if n < 8 or n % 8: raise ValueError('Point count must be divisible by eight')
    counts = (n // 4, n // 4, n // 2)
    rows = []
    for side in (0, 1):
        accepted = []; count = 0
        while count < counts[0] // 2:
            z = torch.rand((counts[0] // 2 - count, 2), generator=generator, dtype=DTYPE)
            radius = torch.sqrt(collar ** 2 + (RADIUS ** 2 - collar ** 2) * z[:, 0])
            angle = z[:, 1] * math.pi / 2
            x = radius * torch.cos(angle); y = 1 - radius * torch.sin(angle)
            xy = torch.stack((x if side == 0 else 1 - x, y), 1)
            xy = xy[stratum_membership(xy, collar) == 0]
            accepted.append(xy); count += len(xy)
        rows.append(torch.cat(accepted))
    for stratum in (1, 2):
        accepted = []; count = 0
        while count < counts[stratum]:
            size = max(2 * (counts[stratum] - count), 32)
            xy = torch.rand((size, 2), generator=generator, dtype=DTYPE)
            if stratum == 2: xy = RADIUS + .9 * xy
            xy = xy[stratum_membership(xy, collar) == stratum][:counts[stratum] - count]
            accepted.append(xy); count += len(xy)
        rows.append(torch.cat(accepted))
    points = torch.cat(rows)
    ids = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    areas = stratum_areas(collar)
    weights = torch.cat([torch.full((count,), float(area / areas.sum() / count), dtype=DTYPE)
                         for count, area in zip(counts, areas)])
    if not torch.equal(ids, stratum_membership(points, collar)):
        raise RuntimeError('Strata are not disjoint or a point lies outside its support')
    return points, weights, ids


def read_baseline_contract(directory):
    """Read only hash-verified training metadata; never read CFD/Ghia labels."""
    directory = Path(directory)
    provenance = json.loads((directory / 'baseline_provenance.json').read_text(encoding='utf-8'))
    if provenance['archive_sha256'] != BASELINE_ARCHIVE_SHA256:
        raise ValueError('Budget evidence is not the preregistered baseline archive')
    for name in ('config.json', 'current_receipt.json', 'frozen_manifest.json'):
        if file_sha256(directory / name) != provenance['members'][name]['sha256']:
            raise ValueError(f'Baseline metadata changed: {name}')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    receipt = json.loads((directory / 'current_receipt.json').read_text(encoding='utf-8'))
    frozen = json.loads((directory / 'frozen_manifest.json').read_text(encoding='utf-8'))
    if not receipt['finished'] or frozen['config'] != config:
        raise ValueError('Baseline is incomplete or mismatched')
    if frozen['source_sha256'] != BASE_SOURCE_SHA256:
        raise ValueError('Baseline used different physical source')
    if receipt['checkpoint_sha256'] != provenance['checkpoint_sha256']:
        raise ValueError('Baseline recovery identity differs from the verified archive')
    if receipt['residual_point_evaluations'] != receipt['closure_calls'] * config['collocation']:
        raise ValueError('Baseline did not complete all counted residual evaluations')
    if receipt['adam_completed'] != config['adam_steps'] or receipt['lbfgs_completed'] != config['lbfgs_steps']:
        raise ValueError('Baseline stage counts are incomplete')
    return dict(schema=1, baseline_config=config,
                baseline_config_sha256=base.config_hash(base.Config(**config)),
                baseline_checkpoint_sha256=receipt['checkpoint_sha256'],
                baseline_frozen_model_sha256=frozen['frozen_model']['model_sha256'],
                baseline_source_sha256=frozen['source_sha256'],
                baseline_closure_calls=receipt['closure_calls'],
                baseline_residual_point_evaluations=receipt['residual_point_evaluations'],
                archive_sha256=provenance['archive_sha256'],
                metadata=provenance['members'],
                baseline_provenance_sha256=file_sha256(directory / 'baseline_provenance.json'))


class WorkBudgetExhausted(RuntimeError):
    pass


class Trainer:
    def __init__(self, config: Config, device='cpu', baseline_contract=None):
        self.config = config; self.device = torch.device(device)
        self.baseline_contract = copy.deepcopy(baseline_contract)
        if config.study_kind == 'controlled_variant':
            if baseline_contract is None: raise ValueError('A completed paired-baseline budget is required')
            baseline_config = baseline_contract['baseline_config']
            if {k: asdict(config)[k] for k in baseline_config} != baseline_config:
                raise ValueError('Paired baseline physics/network/optimizer settings differ')
            if config.closure_budget != baseline_contract['baseline_closure_calls']:
                raise ValueError('Residual-work cap differs from the paired baseline')
            if baseline_contract['baseline_source_sha256'] != BASE_SOURCE_SHA256:
                raise ValueError('Paired baseline source differs')
        random.seed(config.seed); np.random.seed(config.seed % (2 ** 32)); torch.manual_seed(config.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        self.model = CavityPINN(config.width, config.depth, config.reynolds).to(self.device)
        self.initial_model_sha256 = tensor_hash(self.model.state_dict())
        self.generator = torch.Generator().manual_seed(config.seed + 104729)
        self.points, self.weights, self.stratum_ids = stratified_points(config.collocation, self.generator, config.collar)
        self.audit_points = base.audit_locations(config.audit_points, config.audit_seed)
        self.objective_hash = tensor_hash(dict(points=self.points, weights=self.weights, stratum_ids=self.stratum_ids))
        self.audit_points_sha256 = tensor_hash(dict(points=self.audit_points))
        self.declared_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.adam_completed = 0; self.lbfgs_completed = 0; self.lbfgs_aborted_steps = 0
        self.closure_calls = 0; self.residual_point_evaluations = 0
        self.audit_closure_calls = 0; self.audit_residual_point_evaluations = 0
        self.elapsed_seconds = 0.; self.history = []; self.evaluation_history = []
        self.frozen = None; self.endpoint_audit = None
        self.stage = 'adam' if config.adam_steps else 'lbfgs'
        self.optimizer = self.make_optimizer(self.stage)
        self.runtime_history = [self.runtime()]

    def runtime(self):
        cuda = self.device.type == 'cuda'
        return dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=str(self.device),
                    gpu_name=torch.cuda.get_device_name(self.device) if cuda else None,
                    gpu_capability=list(torch.cuda.get_device_capability(self.device)) if cuda else None,
                    hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    slurm_restart_count=os.environ.get('SLURM_RESTART_COUNT', '0'), phase=self.stage)

    def make_optimizer(self, stage):
        if stage == 'adam': return torch.optim.Adam(self.model.parameters(), lr=self.config.adam_lr)
        return torch.optim.LBFGS(self.model.parameters(), lr=1., max_iter=1, max_eval=20,
                history_size=self.config.lbfgs_history, tolerance_grad=1e-12, tolerance_change=1e-14,
                line_search_fn='strong_wolfe')

    @property
    def completed_steps(self): return self.adam_completed + self.lbfgs_completed

    @property
    def finished(self): return self.closure_calls == self.config.closure_budget

    def coverage(self):
        areas = stratum_areas(self.config.collar)
        return dict(objective_sha256=self.objective_hash, total_area=float(areas.sum()),
                    collar=self.config.collar, top_corner_radius=RADIUS,
                    counts={name: int((self.stratum_ids == i).sum()) for i, name in enumerate(STRATA)},
                    areas=dict(zip(STRATA, areas.tolist())),
                    weight_sums={name: float(self.weights[self.stratum_ids == i].sum()) for i, name in enumerate(STRATA)},
                    minimum_corner_radius=float(base.corner_radius(self.points).min()),
                    measure='normalized area on the square excluding only two upper quarter-disks',
                    adaptive_sampling=False)

    def objective(self, backward=True, work=None):
        if backward: self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        for start in range(0, len(self.points), self.config.chunk_size):
            stop = start + self.config.chunk_size
            rx, ry, _ = residuals(self.model, self.points[start:stop].to(self.device))
            weights = self.weights[start:stop].to(self.device)
            loss = ((rx.square() + ry.square()).ravel() * weights).sum()
            if work == 'training': self.residual_point_evaluations += len(weights)
            if work == 'audit': self.audit_residual_point_evaluations += len(weights)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite weighted objective')
            if backward: loss.backward()
            total += float(loss.detach().cpu())
        if backward and not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()):
            raise FloatingPointError('Nonfinite weighted gradient')
        return total

    def step(self):
        if self.finished: return None
        if tensor_hash(dict(points=self.points, weights=self.weights, stratum_ids=self.stratum_ids)) != self.objective_hash:
            raise RuntimeError('Frozen weighted objective changed')
        started = time.perf_counter(); stage = self.stage; values = []
        def closure():
            if self.closure_calls >= self.config.closure_budget: raise WorkBudgetExhausted()
            self.closure_calls += 1
            value = self.objective(work='training'); values.append(value)
            self.evaluation_history.append(dict(closure_calls=self.closure_calls, training_loss=value,
                stage=stage, evaluation_kind='pre_update' if stage == 'adam' else 'line_search_or_initial_closure',
                accepted_update=self.completed_steps, elapsed_seconds=self.elapsed_seconds + time.perf_counter() - started))
            return torch.tensor(value, device=self.device, dtype=DTYPE)
        aborted = False
        if stage == 'adam':
            closure(); self.optimizer.step(); self.adam_completed += 1
            if self.adam_completed == self.config.adam_steps and not self.finished:
                self.stage = 'lbfgs'; self.optimizer = self.make_optimizer('lbfgs')
                self.runtime_history.append(self.runtime())
        else:
            parameters = [p.detach().clone() for p in self.model.parameters()]
            gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
            optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            try:
                self.optimizer.step(closure); self.lbfgs_completed += 1
            except Exception as exc:
                with torch.no_grad():
                    for p, before, grad in zip(self.model.parameters(), parameters, gradients):
                        p.copy_(before); p.grad = grad
                self.optimizer.load_state_dict(optimizer_state)
                if not isinstance(exc, WorkBudgetExhausted): raise
                self.lbfgs_aborted_steps += 1; aborted = True
        self.elapsed_seconds += time.perf_counter() - started
        row = dict(step=self.completed_steps, stage=stage, training_loss=values[-1],
                   loss_location='last consumed closure; not an accepted-endpoint recomputation',
                   lbfgs_budget_rollback=aborted, closure_calls=self.closure_calls,
                   residual_point_evaluations=self.residual_point_evaluations, elapsed_seconds=self.elapsed_seconds)
        self.history.append(row)
        if self.finished:
            self.frozen = dict(model_sha256=tensor_hash(self.model.state_dict()), config_sha256=config_hash(self.config),
                objective_sha256=self.objective_hash, initial_model_sha256=self.initial_model_sha256,
                baseline_contract_sha256=state_hash(self.baseline_contract), reference_labels_used=False,
                endpoint_rule='accepted state at paired-baseline closure cap; unfinished line search rolled back',
                closure_calls=self.closure_calls, lbfgs_budget_rollback=aborted,
                frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        return row

    def audit_endpoint(self):
        if not self.finished or self.frozen is None: raise RuntimeError('Only frozen endpoints may be audited')
        if self.endpoint_audit is None:
            self.audit_closure_calls += 1
            value = self.objective(backward=False, work='audit')
            self.endpoint_audit = dict(weighted_momentum_loss=value, weighted_momentum_rms=math.sqrt(value),
                model_sha256=tensor_hash(self.model.state_dict()), included_in_training_budget=False,
                objective_evaluations=1, residual_point_evaluations=len(self.points),
                purpose='accepted endpoint diagnostic only; no stopping or selection')
        if self.endpoint_audit['model_sha256'] != self.frozen['model_sha256']:
            raise ValueError('Frozen endpoint changed')
        return self.endpoint_audit

    def state_dict(self):
        result = dict(schema=SCHEMA, source_sha256=source_hash(), baseline_source_sha256=BASE_SOURCE_SHA256,
            source_raw_sha256=file_sha256(__file__), baseline_source_raw_sha256=file_sha256(BASE_PATH),
            config=asdict(self.config), config_sha256=config_hash(self.config), model=self.model.state_dict(),
            model_sha256=tensor_hash(self.model.state_dict()), optimizer=self.optimizer.state_dict(),
            baseline_contract=self.baseline_contract, initial_model_sha256=self.initial_model_sha256,
            declared_utc=self.declared_utc, stage=self.stage,
            sampler_rng=self.generator.get_state(), rng_python=random.getstate(), rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            adaptive_state=dict(enabled=False), audit_points_sha256=self.audit_points_sha256)
        for key in ('adam_completed', 'lbfgs_completed', 'lbfgs_aborted_steps', 'closure_calls',
                    'residual_point_evaluations', 'audit_closure_calls', 'audit_residual_point_evaluations',
                    'elapsed_seconds', 'points', 'weights', 'stratum_ids', 'audit_points', 'objective_hash',
                    'frozen', 'endpoint_audit', 'history', 'evaluation_history', 'runtime_history'):
            result[key] = getattr(self, key)
        result['optimizer_sha256'] = state_hash(result['optimizer'])
        result['checkpoint_state_sha256'] = state_hash(result)
        return result

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('wb') as stream:
            torch.save(self.state_dict(), stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        return file_sha256(path)

    @classmethod
    def load(cls, path, device='cpu', config=None):
        state = torch.load(path, map_location='cpu', weights_only=False)
        retained_digest = state.pop('checkpoint_state_sha256')
        if state_hash(state) != retained_digest: raise ValueError('Complete checkpoint state hash failed')
        if state['schema'] != SCHEMA or state['source_sha256'] != source_hash() or state['baseline_source_sha256'] != BASE_SOURCE_SHA256:
            raise ValueError('Checkpoint source/schema changed')
        if state['source_raw_sha256'] != file_sha256(__file__) or state['baseline_source_raw_sha256'] != file_sha256(BASE_PATH):
            raise ValueError('Checkpoint raw source bytes changed; preserve the original staged files')
        restored = Config(**state['config'])
        if config_hash(restored) != state['config_sha256'] or (config is not None and config != restored):
            raise ValueError('Checkpoint configuration mismatch')
        if tensor_hash(state['model']) != state['model_sha256'] or state_hash(state['optimizer']) != state['optimizer_sha256']:
            raise ValueError('Model/optimizer state hash failed')
        result = cls(restored, device, state['baseline_contract'])
        if result.initial_model_sha256 != state['initial_model_sha256'] or result.objective_hash != state['objective_hash']:
            raise ValueError('Fresh initial model or deterministic weighted sampler changed')
        result.model.load_state_dict(state['model']); result.stage = state['stage']
        result.optimizer = result.make_optimizer(result.stage); result.optimizer.load_state_dict(state['optimizer'])
        for key in ('adam_completed', 'lbfgs_completed', 'lbfgs_aborted_steps', 'closure_calls',
                    'residual_point_evaluations', 'audit_closure_calls', 'audit_residual_point_evaluations',
                    'elapsed_seconds', 'points', 'weights', 'stratum_ids', 'audit_points', 'objective_hash',
                    'frozen', 'endpoint_audit', 'history', 'evaluation_history', 'runtime_history', 'declared_utc'):
            setattr(result, key, state[key])
        if tensor_hash(dict(points=result.points, weights=result.weights, stratum_ids=result.stratum_ids)) != result.objective_hash:
            raise ValueError('Weighted sample identity changed')
        if tensor_hash(dict(points=result.audit_points)) != state['audit_points_sha256'] or not torch.equal(result.audit_points, base.audit_locations(restored.audit_points, restored.audit_seed)):
            raise ValueError('Independent audit points changed')
        if result.closure_calls > restored.closure_budget or result.residual_point_evaluations != result.closure_calls * restored.collocation:
            raise ValueError('Residual-evaluation accounting is incomplete')
        if result.frozen and (not result.finished or result.frozen['model_sha256'] != state['model_sha256']):
            raise ValueError('Frozen endpoint mismatch')
        if result.runtime() != result.runtime_history[-1]: result.runtime_history.append(result.runtime())
        result.generator.set_state(state['sampler_rng']); random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_torch'])
        if state['rng_cuda'] and torch.cuda.is_available():
            if len(state['rng_cuda']) != torch.cuda.device_count(): raise ValueError('CUDA RNG device-count mismatch')
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return result
