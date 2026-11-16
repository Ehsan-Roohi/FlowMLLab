"""Scientific CPU gates for the single area-stratified sampling intervention."""
import copy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cavity_pinn_stratified', ROOT/'flowmllab/cavity_pinn_stratified.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)


@pytest.fixture(scope='module', autouse=True)
def preserve_process_torch_settings():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32; cudnn_tf32 = torch.backends.cudnn.allow_tf32
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    torch.set_num_threads(1)
    try: yield
    finally:
        torch.set_default_dtype(dtype); torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32; torch.backends.cudnn.allow_tf32 = cudnn_tf32
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)


def config(**overrides):
    values = dict(width=4, depth=2, collocation=32, adam_steps=2, lbfgs_steps=3,
                  closure_budget=12, audit_points=17, export_grid=9, chunk_size=16, study_kind='cpu_admission')
    values.update(overrides)
    return c.Config(**values)


@pytest.mark.parametrize('collar', [.01, .02])
def test_partition_counts_positive_area_weights_and_top_only_exclusion(collar):
    p, w, ids = c.stratified_points(2048, torch.Generator().manual_seed(19), collar)
    assert torch.equal(torch.bincount(ids), torch.tensor([512, 512, 1024]))
    assert torch.equal(c.stratum_membership(p, collar), ids)
    assert bool((w > 0).all()) and abs(float(w.sum()) - 1) < 2e-15
    areas = c.stratum_areas(collar)
    assert abs(areas.sum() - (1 - math.pi * collar**2 / 2)) < 1e-15
    assert np.allclose([float(w[ids == i].sum()) for i in range(3)], areas/areas.sum(), atol=1e-15)
    annuli = p[ids == 0]
    assert int((annuli[:, 0] < .5).sum()) == 256
    assert bool((c.base.corner_radius(p) > collar).all())
    # Lower corner neighborhoods belong to the wall stratum; none are excluded.
    lower = torch.tensor([[.001, .001], [.999, .001]], dtype=c.DTYPE)
    assert torch.equal(c.stratum_membership(lower, collar), torch.tensor([1, 1]))


@pytest.mark.parametrize('collar', [.01, .02])
def test_area_uniform_radial_sampling_and_analytic_weighted_integrals(collar):
    p, w, ids = c.stratified_points(65536, torch.Generator().manual_seed(121), collar)
    r = c.base.corner_radius(p[ids == 0])
    scaled_r2 = (r*r - collar**2) / (.05**2 - collar**2)
    # Radius squared, rather than radius, is uniform on the annuli.
    assert abs(float(scaled_r2.mean()) - .5) < .007
    assert abs(float((scaled_r2**2).mean()) - 1/3) < .007
    excluded_area = math.pi * collar**2 / 2
    expected_y = (.5 - excluded_area + 2*collar**3/3) / (1 - excluded_area)
    assert abs(float((w*p[:, 0]).sum()) - .5) < .003
    assert abs(float((w*p[:, 1]).sum()) - expected_y) < .003


@pytest.mark.parametrize('seed', [7, 23, 61])
def test_initial_network_and_audit_are_exact_original_baseline(seed):
    values = dict(reynolds=7500, seed=seed, width=64, depth=4, collocation=2048,
                  adam_steps=2000, lbfgs_steps=500, chunk_size=2048, audit_points=2048)
    original = c.base.Trainer(c.base.Config(**values))
    variant = c.Trainer(c.Config(**values, closure_budget=3000, study_kind='cpu_admission'))
    assert c.tensor_hash(original.model.state_dict()) == variant.initial_model_sha256
    assert all(torch.equal(v, variant.model.state_dict()[k]) for k, v in original.model.state_dict().items())
    assert torch.equal(original.audit_points, variant.audit_points)
    assert c.CavityPINN is c.base.CavityPINN and c.residuals is c.base.residuals
    assert not torch.equal(original.points, variant.points)


def test_original_manufactured_derivatives_and_boundary_conditions():
    assert c.numerical_admission()['passed']


def test_weighted_full_chunk_loss_gradient_and_directional_difference():
    a = c.Trainer(config(chunk_size=5)); b = c.Trainer(config(chunk_size=32))
    va, vb = a.objective(), b.objective()
    assert va == pytest.approx(vb, rel=2e-13, abs=2e-12)
    for p, q in zip(a.model.parameters(), b.model.parameters()):
        assert torch.allclose(p.grad, q.grad, rtol=2e-12, atol=2e-12)
    params = list(a.model.parameters()); before = [p.detach().clone() for p in params]
    direction = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((d*d).sum() for d in direction)); direction = [d/norm for d in direction]
    exact = float(sum((p.grad*d).sum() for p, d in zip(params, direction)))
    def value(step):
        with torch.no_grad():
            for p, initial, d in zip(params, before, direction): p.copy_(initial + step*d)
        return a.objective(backward=False)
    h = 1e-6; difference = (value(h)-value(-h))/(2*h); value(0)
    assert abs(exact-difference)/max(1, abs(exact), abs(difference)) < 2e-5
    assert a.closure_calls == a.residual_point_evaluations == 0


@pytest.mark.parametrize('cut', [1, 3])
def test_complete_rng_adam_and_lbfgs_restart(tmp_path, cut):
    cfg = config()
    uninterrupted = c.Trainer(cfg)
    while not uninterrupted.finished: uninterrupted.step()
    interrupted = c.Trainer(cfg)
    for _ in range(cut): interrupted.step()
    checkpoint = tmp_path/'checkpoint.pt'; interrupted.save(checkpoint)
    expected_rng = (random.random(), np.random.random(), torch.rand(4))
    resumed = c.Trainer.load(checkpoint, config=cfg)
    actual_rng = (random.random(), np.random.random(), torch.rand(4))
    assert expected_rng[:2] == actual_rng[:2] and torch.equal(expected_rng[2], actual_rng[2])
    while not resumed.finished: resumed.step()
    assert c.tensor_hash(resumed.model.state_dict()) == c.tensor_hash(uninterrupted.model.state_dict())
    assert c.state_hash(resumed.optimizer.state_dict()) == c.state_hash(uninterrupted.optimizer.state_dict())
    assert resumed.closure_calls == uninterrupted.closure_calls == cfg.closure_budget
    assert resumed.residual_point_evaluations == cfg.closure_budget * cfg.collocation
    assert resumed.lbfgs_aborted_steps == uninterrupted.lbfgs_aborted_steps
    assert [r['training_loss'] for r in resumed.evaluation_history] == [r['training_loss'] for r in uninterrupted.evaluation_history]


def test_budget_abort_restores_populated_lbfgs_state_and_records_spent_work():
    probe = c.Trainer(config(adam_steps=0, closure_budget=100))
    probe.step(); probe.step(); first_two_calls = probe.closure_calls
    bounded = c.Trainer(config(adam_steps=0, closure_budget=first_two_calls+1))
    bounded.step(); bounded.step()
    assert any('old_dirs' in state and len(state['old_dirs']) > 0 for state in bounded.optimizer.state.values())
    before_model = c.tensor_hash(bounded.model.state_dict())
    before_optimizer = c.state_hash(bounded.optimizer.state_dict())
    before_updates = bounded.lbfgs_completed
    row = bounded.step()
    assert row['lbfgs_budget_rollback'] and bounded.finished
    assert bounded.closure_calls == first_two_calls+1
    assert bounded.residual_point_evaluations == bounded.closure_calls*bounded.config.collocation
    assert bounded.lbfgs_completed == before_updates
    assert c.tensor_hash(bounded.model.state_dict()) == before_model
    assert c.state_hash(bounded.optimizer.state_dict()) == before_optimizer
    calls = bounded.closure_calls; work = bounded.residual_point_evaluations
    endpoint = bounded.audit_endpoint()
    assert endpoint['included_in_training_budget'] is False
    assert bounded.closure_calls == calls and bounded.residual_point_evaluations == work
    assert bounded.audit_closure_calls == 1 and bounded.audit_residual_point_evaluations == bounded.config.collocation
    assert bounded.audit_endpoint() == endpoint and bounded.audit_closure_calls == 1


@pytest.mark.parametrize('field', ['points', 'weights', 'stratum_ids', 'optimizer', 'rng_torch', 'config'])
def test_checkpoint_rejects_objective_optimizer_rng_or_config_tampering(tmp_path, field):
    trainer = c.Trainer(config()); trainer.step()
    checkpoint = tmp_path/'checkpoint.pt'; trainer.save(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if field in ('points', 'weights', 'stratum_ids', 'rng_torch'): state[field].reshape(-1)[0] += 1
    elif field == 'config': state[field]['seed'] += 1
    else: next(iter(state['optimizer']['state'].values()))['exp_avg'].reshape(-1)[0] += 1
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='checkpoint state hash'): c.Trainer.load(checkpoint)


def test_production_requires_bound_counterpart_and_frozen_settings():
    with pytest.raises(ValueError, match='paired-baseline'): c.Trainer(c.Config())
    with pytest.raises(ValueError, match='12-case'): c.Config(width=32)
    with pytest.raises(ValueError, match='12-case'): c.Config(reynolds=1000)
    cfg = c.Config(closure_budget=3017)
    baseline = {key: value for key, value in asdict(cfg).items() if key in c.base.Config.__dataclass_fields__}
    contract = dict(baseline_config=baseline, baseline_closure_calls=3017, baseline_source_sha256=c.BASE_SOURCE_SHA256)
    c.Trainer(cfg, baseline_contract=contract)
    contract['baseline_closure_calls'] += 1
    with pytest.raises(ValueError, match='cap'): c.Trainer(cfg, baseline_contract=contract)


def test_freeze_export_is_immutable_and_interrupted_history_reconstructs(tmp_path):
    spec = importlib.util.spec_from_file_location('run_cavity_pinn_stratified', ROOT/'qa/run_cavity_pinn_stratified.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    trainer = c.Trainer(config(adam_steps=1, closure_budget=1))
    trainer.step(); trainer.audit_endpoint(); trainer.save(tmp_path/'checkpoint.pt')
    first = runner.export_frozen(trainer, tmp_path)
    before = {p.name: c.file_sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    resumed = c.Trainer.load(tmp_path/'checkpoint.pt')
    assert runner.export_frozen(resumed, tmp_path) == first
    assert before == {p.name: c.file_sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    (tmp_path/'frozen_manifest.json').unlink(); (tmp_path/'history.json').unlink()
    restored = runner.export_frozen(resumed, tmp_path)
    assert restored['frozen_utc'] == first['frozen_utc']
    assert restored['frozen_model'] == first['frozen_model']
    assert c.file_sha256(tmp_path/'history.json') == first['artifact_sha256']['history.json']
    (tmp_path/'history.json').write_text('[]')
    with pytest.raises(ValueError, match='artifact hash'): runner.export_frozen(resumed, tmp_path)
