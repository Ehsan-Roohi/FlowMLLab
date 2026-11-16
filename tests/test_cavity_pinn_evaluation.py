"""Bounded evidence-integrity tests; all generated fields are synthetic fixtures."""
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip('torch')

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cavity_pinn_evaluation',REPO/'qa/evaluate_cavity_pinn_campaign.py')
e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)

def toy_predict(points):
    x,y=points.T
    return dict(u=np.ones_like(x),v=np.zeros_like(x),psi=y,p=x*0,omega=x*0)

def test_coordinate_only_reader_does_not_access_targets(tmp_path,monkeypatch):
    path=tmp_path/'reference.npz';x=np.linspace(0,1,5)
    np.savez(path,x=x,y=x,u=np.ones((5,5)))
    real=e.np.load;access=[]
    class Guard:
        def __enter__(self):self.f=real(path,allow_pickle=False);return self
        def __exit__(self,*args):self.f.close()
        def __getitem__(self,key):
            access.append(key)
            assert key in ('x','y'),'target field accessed before freeze'
            return self.f[key]
    monkeypatch.setattr(e.np,'load',lambda *args,**kwargs:Guard())
    coords=e.native_coordinates(path,expected_cells=4)
    assert access==['x','y']
    np.testing.assert_array_equal(coords['x'],[.25,.5,.75])
    with pytest.raises(ValueError,match='shape'):e.native_coordinates(path,expected_cells=8)

def test_coordinate_ast_ignores_target_code(tmp_path):
    path=tmp_path/'ghia.py'
    path.write_text("X=np.array([0.,.5,1.])\nY=np.array([1.,.5,0.])\nraise RuntimeError('target execution prohibited')\n")
    a=e.coordinate_literals(path)
    np.testing.assert_array_equal(a['y'],[1.,.5,0.])

def test_projection_is_immutable_and_bound_to_native_coordinates(tmp_path):
    ref=tmp_path/'reference.npz';axis=np.linspace(0,1,5)
    np.savez(ref,x=axis,y=axis,u=np.ones((5,5)))
    coords=e.native_coordinates(ref,4);g=dict(x=np.array([1.,.5,0.]),y=np.array([1.,.5,0.]))
    events=[]
    def predictor(points):
        assert not (tmp_path/'out/projection_freeze.json').exists()
        events.append(points.copy())
        return toy_predict(points)
    frozen,path=e.freeze_projection(tmp_path/'out',{'model_sha256':'toy'},coords,g,predictor)
    assert len(events)==3 and events[0].shape==(9,2)
    assert frozen['reference_values_read'] is False and frozen['prediction_sha256']==e.sha(path)
    with np.load(path) as f:
        np.testing.assert_array_equal(f['x'],coords['x'])
        np.testing.assert_array_equal(f['ghia_y'],g['y'])
        assert f['u'].shape==(3,3)
    before=path.read_bytes()
    retained,_=e.freeze_projection(tmp_path/'out',{'model_sha256':'toy'},coords,g,
        lambda _:(_ for _ in ()).throw(AssertionError('repeat projection executed')))
    assert retained==frozen and path.read_bytes()==before
    with pytest.raises(ValueError,match='identity'):e.freeze_projection(tmp_path/'out',{'model_sha256':'different'},coords,g,toy_predict)
    path.write_bytes(before+b'corruption')
    with pytest.raises(ValueError,match='bytes'):e.freeze_projection(tmp_path/'out',{'model_sha256':'toy'},coords,g,toy_predict)

def test_native_error_signs_and_exact_coordinates():
    x=np.array([.1,.4,.9]);y=np.array([.1,.5,.9]);shape=(3,3)
    ref=dict(x=x,y=y,u=np.ones(shape),v=np.zeros(shape),omega=np.full(shape,3.))
    pred={k:v.copy() for k,v in ref.items()};pred['u']+=.1;pred['omega']-=2.
    score=e.native_scores(pred,ref)['regions']['interior']
    assert score['velocity_relative_l2']==pytest.approx(.1)
    assert score['omega_error_signed_mean_U_over_W']==pytest.approx(-2.)
    assert score['omega_relative_l2']==pytest.approx(2/3)
    pred['x'][1]+=.001
    with pytest.raises(ValueError,match='coordinates'):e.native_scores(pred,ref)

def test_direct_ghia_queries_retain_printed_entry_policy():
    ghia=e.module_at(REPO/'flowmllab/ghia.py','test_post_freeze_ghia')
    ref=ghia.reference(3200)
    pred=dict(ghia_x=ref['x'],ghia_y=ref['y'],ghia_u_prediction=ref['u'].copy(),ghia_v_prediction=ref['v'].copy())
    bad=~ghia.valid_mask(3200,'u');pred['ghia_u_prediction'][bad]+=.1
    scores,rows=e.ghia_scores(3200,pred,ghia)
    assert scores['u']['screened_relative_l2']==0 and scores['u']['published_relative_l2']>0
    assert sum(not r['screened_included'] for r in rows)==1
    pred['ghia_y']=pred['ghia_y'][::-1]
    with pytest.raises(ValueError,match='query coordinates'):e.ghia_scores(3200,pred,ghia)

def test_training_curve_is_never_labelled_independent_audit(tmp_path):
    history=tmp_path/'history.json'
    e.write_json(history,[dict(training_loss=.25,closure_calls=3,residual_point_evaluations=48,
        elapsed_seconds=2.,stage='lbfgs',loss_location='last closure')])
    entry=e.history_csv(history,tmp_path)
    assert entry['independent_audit'] is False
    assert entry['metric_name']=='training_collar_momentum_rms'
    data=np.genfromtxt(entry['path'],delimiter=',',names=True,dtype=None,encoding='utf-8')
    assert float(data['metric_value'])==.5 and 'audit_rms' not in data.dtype.names

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_end_to_end_fixture_freezes_before_reference_values(tmp_path,monkeypatch,device):
    """Real file/hash/scoring path; a fake frozen predictor avoids costly training."""
    monkeypatch.setattr(e,'ROOT',tmp_path)
    run=tmp_path/'private_run';out=tmp_path/'results/cavity_pinn_campaign';run.mkdir()
    cfd=tmp_path/'results/cavity_highre_ghia/n512/re100.npz';cfd.parent.mkdir(parents=True)
    axis=np.linspace(0,1,513);xx,yy=np.meshgrid(axis,axis)
    np.savez_compressed(cfd,x=axis,y=axis,u=np.ones_like(xx),v=xx*0,psi=yy,omega=xx)
    @dataclass
    class Config:
        reynolds:int=100
        seed:int=7
        collar:float=.02
        chunk_size:int=256
    cfg=Config();config=asdict(cfg);model=SimpleNamespace(state_dict=lambda:{})
    freeze=dict(model_sha256='toy-model',config_sha256='toy-config-1234',frozen_utc='2026-11-15T00:00:00Z')
    devices=[]
    trainer=SimpleNamespace(config=cfg,model=model,finished=True,frozen=freeze,runtime=lambda:dict(device=device,fixture=True))
    def load(*args,**kwargs):
        devices.append(('load',kwargs['device']));return trainer
    def predict(model,points,device,chunk_size):
        devices.append(('predict',device));return toy_predict(points)
    engine=SimpleNamespace(Trainer=SimpleNamespace(load=load),
        tensor_hash=lambda *a:'toy-model',source_hash=lambda:'toy-source',predict=predict)
    e.write_json(run/'config.json',config);(run/'checkpoint.pt').write_bytes(b'synthetic fixture, not a model')
    np.savez(run/'frozen_predictions.npz',fixture=np.array([1.]))
    physical=dict(outer_uniform=dict(momentum_rms=.1,divergence_rms=0.))
    e.write_json(run/'final_physics_audit.json',physical)
    e.write_json(run/'frozen_manifest.json',dict(config=config,frozen_model=freeze,frozen_utc=freeze['frozen_utc'],source_sha256='toy-source',
        prediction_file='frozen_predictions.npz',prediction_sha256=e.sha(run/'frozen_predictions.npz'),
        physics_audit_file='final_physics_audit.json',physics_audit_sha256=e.sha(run/'final_physics_audit.json')))
    e.write_json(run/'current_receipt.json',dict(finished=True,checkpoint_sha256=e.sha(run/'checkpoint.pt')))
    e.write_json(run/'history.json',[dict(training_loss=.25,closure_calls=1,residual_point_evaluations=8,
        elapsed_seconds=1.,stage='adam',loss_location='last closure')])
    actual_load=e.np.load;accesses=[]
    class Guard:
        def __init__(self,path,*args,**kwargs):self.path=Path(path);self.f=actual_load(path,*args,**kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):self.f.close()
        @property
        def files(self):return self.f.files
        def __getitem__(self,key):
            if self.path==cfd:
                accesses.append(key)
                if key not in ('x','y'):
                    assert list((out/'runs').glob('*/projection_freeze.json')),'target field read before projection freeze'
            return self.f[key]
    monkeypatch.setattr(e.np,'load',Guard)
    if device=='cpu':item=e.prepare_run(run,out,engine,REPO/'flowmllab/ghia.py')
    else:item=e.prepare_run(run,out,engine,REPO/'flowmllab/ghia.py',device=device)
    assert devices==[('load',device)]+[('predict',device)]*3
    assert item['projection']['identity']['evaluation_device']==device
    assert accesses==['x','y']
    targets_at=e.utc()
    ghia=e.module_at(REPO/'flowmllab/ghia.py','fixture_post_freeze_ghia')
    record=e.score_prepared(item,out,ghia,targets_at)
    assert record['outcome']=='scored' and record['physical_review_status']=='pending'
    assert record['scores']['regions']['interior']['velocity_relative_l2']==0.
    assert not record['scores']['numeric_checks']['outer_momentum_rms_le_1e_2']
    assert record['pressure_reference_available'] is False
    assert record['history']['independent_audit'] is False
    assert e.checked_entry(record['prediction']).exists()
    assert e.checked_entry(record['ghia_profiles']).exists()
    assert record['freeze']['at_utc']<=record['score']['reference_accessed_at_utc']
    assert e.score_prepared(item,out,ghia,targets_at)==record



def test_collar_pair_keeps_seed_and_configuration_matching(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    axis=np.array([.1,.5,.9]);shape=(3,3);records=[]
    for collar,amplitude in [(.01,1.),(.02,1.01)]:
        path=tmp_path/f'{collar}.npz'
        np.savez(path,x=axis,y=axis,u=np.full(shape,amplitude),v=np.zeros(shape))
        records.append(dict(run_id=str(collar),Re=100,seed=7,collar=collar,outcome='scored',
            config=dict(reynolds=100,seed=7,collar=collar,adam_steps=10),
            source=dict(source_canonical_sha256='same-source'),prediction=e.file_entry(path)))
    pair=e.collar_comparisons(records)
    assert len(pair)==1 and pair[0]['relative_velocity_difference']==pytest.approx(.01)
    assert pair[0]['numeric_target_le_0_02']
    records[1]['config']['seed']=23
    assert not e.collar_comparisons(records)


def test_cuda_request_fails_before_evaluation_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert e.validate_evaluation_device('cpu')=='cpu'
    with pytest.raises(RuntimeError,match='CUDA is unavailable'):e.validate_evaluation_device('cuda')
    with pytest.raises(ValueError,match='cpu or cuda'):e.validate_evaluation_device('other')
