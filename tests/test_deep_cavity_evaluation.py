"""Synthetic unit fixtures test scoring logic; these are not scientific results."""
import csv
from datetime import datetime,timezone,timedelta
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('d5_evaluation_test',ROOT/'qa/evaluate_deep_cavity_ablation.py')
evalmod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evalmod)


def fixture_fields():
    x=np.linspace(.002,.998,17);h=np.linspace(.002,4.998,101)
    X,H=np.meshgrid(x,h)
    return {'x':x,'h':h,'u':X*H,'v':-.5*H**2,'p':X**2+H**3,'psi':np.zeros_like(X),'omega':np.full_like(X,777.)}


def fixture_run(path,method,purpose='controlled_pilot',seed=12345):
    path.mkdir()
    fields=fixture_fields()
    fields['u']+=.1;fields['v']+=.2;fields['p']+=.125
    np.savez_compressed(path/'prediction.npz',**fields)
    (path/'checkpoint.pt').write_bytes(b'synthetic unit-test checkpoint, not a trained model')
    (path/'final_audit.json').write_text('{}')
    rows=[{'function_evaluations':i,'gradient_evaluations':i,'loss':1/i,'elapsed_seconds':i*.1,
           'phase':'warm' if i==1 else 'qn','evaluation_kind':'pre_update_closure' if i==1 else 'quasi_Newton_trial','accepted':False}
          for i in (1,2,3)]
    with (path/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config={'Re':1000.,'W':1.,'D':5.,'dtype':'float64','method':method,'seed':seed,
            'purpose':purpose,'warm_evaluations':1,'qn_evaluations':2}
    (path/'contract.json').write_text(json.dumps(config))
    manifest={'schema':'flowmllab-d5-controlled-v1','status':'frozen','configuration':config,
              'method':method,'seed':seed,'purpose':purpose,'CFD_used_for_training_or_selection':False,
              'frozen_at_utc':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
              'frozen_model_sha256':'unit_model','initial_model_sha256':'unit_initial','sample_sha256':'unit_samples',
              'counters':{'warm_evaluations':1,'qn_evaluations':2,'function_evaluations':3,'gradient_evaluations':3},
              'training_seconds':.2,'active_seconds':.4,'runtime':[{'hostname':'unit','gpu_name':'unitGPU','gpu_capability':[8,0],'torch':'unit','slurm_job_id':'unit1'}],
              'final_audit':{}}
    for file,key in [('prediction.npz','prediction_sha256'),('checkpoint.pt','checkpoint_sha256'),
                     ('final_audit.json','final_audit_sha256'),('history.csv','history_sha256')]:
        manifest[key]=evalmod.sha(path/file)
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path


@pytest.fixture
def pair(tmp_path):
    return [fixture_run(tmp_path/method,method) for method in evalmod.METHODS]


def reference_fixture():
    f=fixture_fields()
    data={'x':f['x'],'h':f['h'],**{f're1000_cfd_{q}':f[q] for q in ('u','v','p')}}
    return data,{'data_sha256':'synthetic_reference_fixture','CFD_times':{'1000':317.}}


def test_scoring_pressure_gauge_shared_fd_and_bridge(pair,tmp_path,monkeypatch):
    accesses=[]
    def load(_):
        accesses.append(datetime.now(timezone.utc))
        return reference_fixture()
    monkeypatch.setattr(evalmod.audit,'load_bundle',load)
    monkeypatch.setattr(evalmod,'convergence_figures',lambda *a:[])
    monkeypatch.setattr(evalmod,'physical_figures',lambda *a:[])
    (tmp_path/'reference').mkdir()
    (tmp_path/'reference'/'manifest.json').write_text(json.dumps(reference_fixture()[1]))
    result=evalmod.evaluate(pair,tmp_path/'evaluation',tmp_path/'reference')
    assert len(accesses)==1 and result['status']=='controlled_pilot_scored'
    assert result['path_base']=='manifest_directory'
    for row in result['metrics']:
        assert row['velocity_RMS_U']==pytest.approx(np.sqrt(.05),rel=1e-12)
        assert row['pressure_RMS_rhoU2']<2e-14
        assert row['vorticity_RMS_UoverW']<2e-13
    for record in result['records']:
        assert record['prediction']['sha256']==record['freeze']['prediction_sha256']==record['score']['prediction_sha256']
        assert record['comparison_prediction']['source_prediction_sha256']==record['prediction']['sha256']
        assert record['comparison_prediction']['pressure_shift_subtracted']==pytest.approx(.125)
        assert datetime.fromisoformat(record['freeze']['at_utc']) < datetime.fromisoformat(record['score']['reference_accessed_at_utc'])
        for key in ('prediction','comparison_prediction','reference','history'):
            file=tmp_path/'evaluation'/record[key]['path']
            assert file.exists() and evalmod.sha(file)==record[key]['sha256']


def test_cost_smoke_never_opens_reference(tmp_path,monkeypatch):
    pair=[fixture_run(tmp_path/m,m,'cost_smoke') for m in evalmod.METHODS]
    def forbid(_):raise AssertionError('Cost smoke must not read CFD values')
    monkeypatch.setattr(evalmod.audit,'load_bundle',forbid)
    monkeypatch.setattr(evalmod,'convergence_figures',lambda *a:[])
    result=evalmod.evaluate(pair,tmp_path/'evaluation',tmp_path/'nonexistent')
    assert result['reference_values_accessed'] is False
    assert result['metrics']==[]
    assert all('reference' not in r and 'scores' not in r for r in result['records'])


def test_tampered_frozen_data_rejected_before_reference(pair,tmp_path,monkeypatch):
    with (pair[0]/'prediction.npz').open('ab') as f:f.write(b'tamper')
    monkeypatch.setattr(evalmod.audit,'load_bundle',lambda _:pytest.fail('Reference accessed before frozen-byte admission'))
    with pytest.raises(ValueError,match='SHA256'):
        evalmod.evaluate(pair,tmp_path/'evaluation',tmp_path/'reference')


def test_unmatched_pair_rejected_before_reference(pair,tmp_path,monkeypatch):
    p=pair[1]/'manifest.json';data=json.loads(p.read_text());data['initial_model_sha256']='different';p.write_text(json.dumps(data))
    monkeypatch.setattr(evalmod.audit,'load_bundle',lambda _:pytest.fail('Reference accessed for unmatched pair'))
    with pytest.raises(ValueError,match='initial_model'):
        evalmod.evaluate(pair,tmp_path/'evaluation',tmp_path/'reference')


def test_missing_peer_rejected(pair):
    with pytest.raises(ValueError,match='Incomplete'):
        evalmod.validate_pairs([evalmod.verify_frozen_run(pair[0])])
