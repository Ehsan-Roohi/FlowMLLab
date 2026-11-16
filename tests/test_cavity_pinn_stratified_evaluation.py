"""Frozen inference repair and complete-batch reference barrier admission."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stratified_inference_repair', ROOT/'qa/evaluate_cavity_pinn_stratified.py')
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


@pytest.fixture(scope='module')
def engine():
    core, result = w.dependencies()
    return result


@pytest.fixture(autouse=True)
def preserve_torch_globals():
    before = torch.get_rng_state(); threads = torch.get_num_threads()
    dtype = torch.get_default_dtype()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_num_threads(1)
    try: yield
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng)
        torch.set_rng_state(before); torch.set_num_threads(threads); torch.set_default_dtype(dtype)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def frozen_toy(engine, tmp_path):
    cfg = engine.Config(width=4, depth=2, collocation=32, adam_steps=1, lbfgs_steps=3,
        closure_budget=1, audit_points=17, export_grid=9, chunk_size=16, study_kind='cpu_admission')
    trainer = engine.Trainer(cfg); trainer.step(); trainer.audit_endpoint()
    path = tmp_path/'checkpoint.pt'; trainer.save(path)
    return trainer,path


def test_saved_objective_restores_exactly_despite_regenerated_last_bit(engine,tmp_path,monkeypatch):
    trainer,path = frozen_toy(engine,tmp_path); digest=w.sha(path)
    sampler=engine.stratified_points
    def changed(n,generator,collar):
        points,weights,ids=sampler(n,generator,collar)
        points[0,0]=torch.nextafter(points[0,0],torch.tensor(float('inf'),dtype=points.dtype))
        return points,weights,ids
    monkeypatch.setattr(engine,'stratified_points',changed)
    with pytest.raises(ValueError,match='Fresh initial model or deterministic weighted sampler changed'):
        engine.Trainer.load(path)
    rng=torch.get_rng_state().clone()
    restored=w.load_frozen(path,engine)
    assert torch.equal(rng,torch.get_rng_state())
    assert engine.tensor_hash(restored.model.state_dict()) == trainer.frozen['model_sha256']
    assert restored.restoration['serialized_objective_sha256'] == trainer.objective_hash
    assert restored.restoration['training_continuation_certified'] is False
    assert not hasattr(restored,'optimizer') and not hasattr(restored,'step') and not hasattr(restored,'save')
    assert w.sha(path)==digest
    points=engine.torch.tensor([[.3,.4],[.6,.7]],dtype=engine.DTYPE).numpy()
    original=engine.predict(trainer.model,points)
    actual=engine.predict(restored.model,points)
    for key in original: assert (original[key]==actual[key]).all()


@pytest.mark.parametrize('field',['points','weights','model','rng_torch'])
def test_saved_state_tampering_is_never_tolerated(engine,tmp_path,field):
    trainer,path=frozen_toy(engine,tmp_path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if field=='model':
        next(iter(state['model'].values())).reshape(-1)[0]+=1
    else: state[field].reshape(-1)[0]+=1
    torch.save(state,path)
    with pytest.raises(ValueError,match='Complete serialized checkpoint hash failed'):
        w.load_frozen(path,engine)


def batch(tmp_path,monkeypatch,fail_index=None):
    runs=[]; events=[]
    for re,seed,collar in sorted(w.EXPECTED):
        run=tmp_path/f're{re}s{seed}e{collar}';run.mkdir()
        (run/'config.json').write_text(json.dumps(dict(reynolds=re,seed=seed,collar=collar)))
        runs.append(run)
    output=tmp_path/'output';output.mkdir()
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True)+'\n')
    def prepare(run,out,core,engine,device):
        index=len([e for e in events if e=='prepare'])
        events.append('prepare')
        if index==fail_index: raise ValueError('Synthetic restoration failure')
        cfg=json.loads((run/'config.json').read_text());identifier=run.name
        return dict(config=cfg,run_id=identifier,
            projection=dict(at_utc='2026-11-15T00:00:00Z',prediction_sha256=identifier,identity=dict(model_sha256=identifier)))
    def targets(*args):
        events.append('target_read');return object()
    def score(item,*args):
        events.append('score')
        return dict(run_id=item['run_id'],Re=item['config']['reynolds'],seed=item['config']['seed'],
                    collar=item['config']['collar'],outcome='scored')
    core=SimpleNamespace(read_json=lambda path:json.loads(path.read_text()),
        utc=lambda:'2026-11-15T01:00:00Z',write_json=write,module_at=targets,score_prepared=score,
        collar_comparisons=lambda rows:[],file_entry=lambda path:dict(path=str(path),sha256=w.sha(path)),
        checked_entry=lambda entry:entry,__file__=str(ROOT/'qa/evaluate_cavity_pinn_campaign.py'))
    monkeypatch.setattr(w,'prepare_run',prepare)
    monkeypatch.setattr(w,'preserve_prior_attempt',lambda *args:None)
    return runs,output,core,events


def test_any_native_projection_failure_prevents_every_target_read(tmp_path,monkeypatch):
    runs,output,core,events=batch(tmp_path,monkeypatch,fail_index=7)
    with pytest.raises(ValueError,match='Synthetic restoration'):
        w.completed_pass(runs,output,core,None,'cpu')
    assert 'target_read' not in events and 'score' not in events
    assert not (output/'evaluation_manifest.json').exists()


def test_complete_barrier_and_repeat_preserve_manifest_and_timestamp(tmp_path,monkeypatch):
    runs,output,core,events=batch(tmp_path,monkeypatch)
    result=w.completed_pass(runs,output,core,None,'cpu')
    assert events[:12]==['prepare']*12 and events[12]=='target_read'
    before={p.name:w.sha(p) for p in (result,output/'completed_pass/projection_barrier.json')}
    core.utc=lambda:'2026-12-01T01:00:00Z'
    events.clear()
    assert w.completed_pass(runs,output,core,None,'cpu')==result
    assert before=={p.name:w.sha(p) for p in (result,output/'completed_pass/projection_barrier.json')}


def test_prior_partial_target_access_and_all_original_endpoints_remain_recorded(tmp_path,monkeypatch):
    preserve=w.preserve_prior_attempt
    runs,output,core,events=batch(tmp_path,monkeypatch)
    monkeypatch.setattr(w,'preserve_prior_attempt',preserve)
    original_records=[]
    for i,run in enumerate(runs):
        cfg=core.read_json(run/'config.json')
        core.write_json(run/'frozen_manifest.json',dict(config=cfg,frozen_model=dict(model_sha256=run.name),
            frozen_utc='2026-11-15T00:00:00Z'))
        record=dict(run_id=run.name,outcome='evaluation_rejected',reason='Synthetic platform regeneration failure')
        if i<4:
            record.update(outcome='scored',score=dict(reference_accessed_at_utc='2026-11-15T00:30:00Z'))
            for name in ('prediction','reference','history','ghia_profiles'): record[name]=dict(path=name,sha256='synthetic')
        original_records.append(record)
    original=output/'evaluation_manifest.json';core.write_json(original,dict(records=original_records))
    original_hash=w.sha(original)
    evidence=w.preserve_prior_attempt(output,runs,core,dict(path='forensic',sha256='synthetic'))
    receipt=core.read_json(output/'prior_evaluation_attempt.json')
    assert receipt['original_manifest_sha256']==original_hash
    assert receipt['original_manifest']['records']==original_records
    assert receipt['prior_scored_case_count']==4 and receipt['prior_reference_values_read'] is True
    assert receipt['training_endpoints_frozen_before_prior_target_access'] is True
    assert len(receipt['all_training_endpoints'])==12
    assert w.preserve_prior_attempt(output,runs,core,dict(path='forensic',sha256='synthetic'))==evidence
    (runs[-1]/'frozen_manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='Original trained endpoint changed'):
        w.preserve_prior_attempt(output,runs,core,dict(path='forensic',sha256='synthetic'))
