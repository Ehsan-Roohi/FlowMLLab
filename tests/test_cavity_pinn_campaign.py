"""Scientific CPU admission and exact same-runtime restart tests."""
import importlib.util
from pathlib import Path
import sys
import random
import numpy as np
import pytest
torch = pytest.importorskip('torch')

MODULE=Path(__file__).resolve().parents[1]/'flowmllab'/'cavity_pinn_campaign.py'
spec=importlib.util.spec_from_file_location('cavity_pinn_campaign',MODULE)
c=importlib.util.module_from_spec(spec); sys.modules[spec.name]=c; spec.loader.exec_module(c)
@pytest.fixture(scope="module", autouse=True)
def preserve_process_torch_settings():
    """Admission settings must not leak into the rest of the package suite."""
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    python_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Do not initialize CUDA merely to run CPU admission tests.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def test_manufactured_fields_boundaries_and_directional_gradient():
    report=c.numerical_admission('cpu')
    assert report['passed'],report


def test_collar_and_final_audit_are_independent_of_train_seed():
    config=c.Config(width=4,depth=2,collocation=20,adam_steps=1,lbfgs_steps=0,audit_points=23)
    a=c.Trainer(config)
    b=c.Trainer(c.Config(**{**config.__dict__,'seed':91,'collar':.01}))
    assert bool((c.corner_radius(a.points)>=.02).all())
    assert torch.equal(a.audit_points,b.audit_points)
    assert not torch.equal(a.points,b.points)
    # Common initial weights for the paired epsilon sensitivity use the SAME seed.
    paired=c.Trainer(c.Config(**{**config.__dict__,'collar':.01}))
    assert c.tensor_hash(a.model.state_dict())==c.tensor_hash(paired.model.state_dict())
    assert bool(torch.isfinite(a.points).all())


def test_exact_corners_rejected_and_open_lid_constant():
    model=c.CavityPINN(4,2,7500)
    with pytest.raises(ValueError,match='singular'):
        model(torch.tensor([[0.,1.]],dtype=torch.float64))
    with pytest.raises(TypeError,match='float64'):
        model(torch.tensor([[.3,.6]],dtype=torch.float32))
    assert max(v for row in c.boundary_audit(model).values() for v in row.values())<1e-10


@pytest.mark.parametrize('split',[1,3,4])
def test_resume_exact_during_adam_at_phase_boundary_and_during_lbfgs(tmp_path,split):
    config=c.Config(width=4,depth=2,collocation=16,chunk_size=8,adam_steps=3,lbfgs_steps=2,
                    lbfgs_history=3,audit_points=11)
    full=c.Trainer(config)
    while not full.finished: full.step()
    partial=c.Trainer(config)
    for _ in range(split): partial.step()
    path=tmp_path/'checkpoint.pt'; partial.save(path)
    resumed=c.Trainer.load(path,config=config)
    while not resumed.finished: resumed.step()
    assert c.tensor_hash(full.model.state_dict())==c.tensor_hash(resumed.model.state_dict())
    assert full.closure_calls==resumed.closure_calls
    assert full.residual_point_evaluations==resumed.residual_point_evaluations
    assert full.stage==resumed.stage
    assert {k:v for k,v in full.frozen.items() if k!='frozen_utc'}=={k:v for k,v in resumed.frozen.items() if k!='frozen_utc'}
    assert torch.equal(full.points,resumed.points)
    assert torch.equal(full.generator.get_state(),resumed.generator.get_state())
    assert [row['training_loss'] for row in full.history]==[row['training_loss'] for row in resumed.history]
    assert resumed.optimizer.state_dict()['state']


def test_objective_gradient_matches_chunks_and_restart_rejects_mismatch(tmp_path):
    config=c.Config(width=4,depth=2,collocation=16,chunk_size=5,adam_steps=1,lbfgs_steps=0,audit_points=8)
    a=c.Trainer(config); loss=a.objective()
    gradients=[p.grad.detach().clone() for p in a.model.parameters()]
    b=c.Trainer(c.Config(**{**config.__dict__,'chunk_size':16}))
    loss_b=b.objective()
    assert np.isclose(loss,loss_b,rtol=1e-13)
    for ga,p in zip(gradients,b.model.parameters()):
        torch.testing.assert_close(ga,p.grad,rtol=1e-11,atol=1e-11)
    path=tmp_path/'checkpoint.pt'; a.save(path)
    with pytest.raises(ValueError,match='configuration mismatch'):
        c.Trainer.load(path,config=c.Config(**{**config.__dict__,'collar':.01}))
    a.points[0,0]+=.01
    with pytest.raises(RuntimeError,match='fixed objective'):
        a.step()


def test_freeze_only_after_declared_budget_and_signed_field_export():
    config=c.Config(width=4,depth=2,collocation=8,adam_steps=1,lbfgs_steps=1,chunk_size=8,audit_points=8)
    trainer=c.Trainer(config)
    assert trainer.frozen is None
    trainer.step(); assert trainer.frozen is None
    trainer.step(); assert trainer.finished and trainer.frozen['reference_labels_used'] is False
    xy=np.array([[.5,.5],[.25,.75],[.75,.25]])
    result=c.predict(trainer.model,xy)
    assert set(result)=={'psi','p','u','v','omega'}
    assert all(np.isfinite(value).all() for value in result.values())
    assert abs(result['p'][0])<1e-12


def test_frozen_export_is_immutable_and_recovers_incomplete_manifest(tmp_path,monkeypatch):
    runner_path=MODULE.parents[1]/'qa'/'run_cavity_pinn_campaign.py'
    runner_spec=importlib.util.spec_from_file_location('run_cavity_pinn_campaign',runner_path)
    runner=importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
    config=c.Config(width=4,depth=2,collocation=8,adam_steps=1,lbfgs_steps=0,
                    chunk_size=8,audit_points=8,export_grid=5)
    trainer=c.Trainer(config); trainer.step()
    monkeypatch.setattr(runner,'final_audit',lambda *a,**kw:{'tested_elsewhere':True})
    manifest=runner.export_frozen(trainer,tmp_path,5)
    assert manifest['frozen_utc']==trainer.frozen['frozen_utc']
    frozen_checkpoint=tmp_path/'frozen.pt'; trainer.save(frozen_checkpoint)
    reloaded=c.Trainer.load(frozen_checkpoint)
    assert reloaded.frozen==trainer.frozen
    original={name:(tmp_path/name).read_bytes() for name in ('frozen_predictions.npz','frozen_manifest.json','final_physics_audit.json')}
    assert runner.export_frozen(trainer,tmp_path,5)==manifest
    assert all((tmp_path/name).read_bytes()==data for name,data in original.items())
    # Reconstruct an interrupted export from the checkpoint's original frozen identity.
    (tmp_path/'frozen_manifest.json').unlink()
    recovered=runner.export_frozen(trainer,tmp_path,5)
    assert recovered==manifest
    with pytest.raises(ValueError,match='export grid'):
        runner.export_frozen(trainer,tmp_path,7)
    (tmp_path/'frozen_predictions.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact hash'):
        runner.export_frozen(trainer,tmp_path,5)
