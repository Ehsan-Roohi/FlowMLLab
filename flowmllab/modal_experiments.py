"""Frozen teaching protocols on author-generated LBM, never a blind CFD claim."""
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from .modal_tools import (fit_pod,project_pod,reconstruct_pod,fit_dmd,rollout_dmd,
    dmd_modes,select_sensors,reconstruct_sensors,fit_integral_sindy,rollout_sindy)
from .field_metrics import field_metrics,relative_l2,temporal_spectrum

PLAN={
 'forecast':{'case':110,'train':[0,160],'validation':[160,210],'test':[210,281],
  'dmd_ranks':[2,4,6,8],'sindy_rank':2,'sindy_thresholds':[.01,.05,.1],
  'sindy_degree':3,'sindy_window':4,'mlp_rank':8,'mlp_history':4,
  'mlp_hidden':[32,32],'mlp_seed':17,'mlp_max_iter':1500,
  'selection':'Minimum validation field relative L2; no reset at test boundary'},
 'sensors':{'train':[90,110],'validation':100,'test':105,'rank':8,'budgets':[8,16,32],
  'noise_fraction':.01,'seeds':[10,11,12,13,14],
  'selection':'Smallest budget with mean optimized validation L2 <= 0.05; otherwise minimum validation L2'},
 'scope':'Previously inspected coarse author LBM; educational ROI, not grid-independent CFD or blind research.'}


def load_cases(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    cases={}
    for row in manifest['cases']:
        path=root/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Data checksum mismatch: '+str(path))
        with np.load(path,allow_pickle=False) as d:
            cases[int(row['reynolds'])]={k:d[k].copy() for k in d.files}
    if set(cases)!={90,100,105,110}:
        raise ValueError('Unexpected cases')
    for d in cases.values():
        if d['v'].shape!=(281,32,78) or not all(np.isfinite(d[k]).all() for k in ('v','omega','t')):
            raise ValueError('Invalid field subset')
    return cases


def score(pred,truth,case):
    edge=np.zeros(truth.shape[1:],bool);edge[[0,-1],:]=True;edge[:,[0,-1]]=True
    # Equal sample-cell area; not a wall-boundary or conservative finite-volume flux.
    area=float(np.diff(case['x'])[0]*np.diff(case['y'])[0])
    return field_metrics(pred,truth,np.full(edge.shape,area),edge)


def forecast_experiment(cases):
    c=cases[110];field=c['omega'];flat=field.reshape(len(field),-1)
    train=flat[:160];dt=float(np.diff(c['t'])[0]);steps=len(flat)-160
    if not np.allclose(np.diff(c['t']),dt):raise ValueError('Nonuniform time step')
    predictions={};records={};pods={};coeffs={}
    for rank in PLAN['forecast']['dmd_ranks']:
        pod=fit_pod(train,rank=rank);z=project_pod(pod,train)
        pods[rank]=pod;coeffs[rank]=z
        a=fit_dmd(z);name=f'DMD-r{rank}'
        p=reconstruct_pod(pod,rollout_dmd(a,z[-1],steps)).reshape((-1,*field.shape[1:]))
        predictions[name]=p
        records[name]={'validation_l2':relative_l2(p[:50],field[160:210]),'modes':dmd_modes(a,dt)}
    chosen=min(records,key=lambda k:records[k]['validation_l2'])
    # Fresh nonlinear baseline; every input/history and scaling comes from training.
    pod=pods[8];z=coeffs[8];scale=np.maximum(z.std(axis=0),1e-12);zs=z/scale
    history=4;x=np.array([zs[i-history:i].ravel() for i in range(history,len(zs))]);y=zs[history:]
    mlp=MLPRegressor(hidden_layer_sizes=(32,32),activation='tanh',solver='lbfgs',
        alpha=1e-4,max_iter=1500,random_state=17,tol=1e-8,max_fun=30000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning);mlp.fit(x,y)
    past=list(zs[-history:]);out=[]
    for _ in range(steps):
        nxt=mlp.predict(np.array(past[-history:]).reshape(1,-1))[0]
        if not np.isfinite(nxt).all():raise ValueError('MLP diverged')
        past.append(nxt);out.append(nxt*scale)
    predictions['MLP-r8']=reconstruct_pod(pod,np.array(out)).reshape((-1,*field.shape[1:]))
    records['MLP-r8']={'warnings':[str(w.message) for w in caught],
        'training_one_step_coefficient_mse':float(np.mean((mlp.predict(x)-y)**2))}
    predictions['persistence']=np.repeat(field[159:160],steps,axis=0)
    predictions['training-mean']=np.repeat(train.mean(axis=0)[None],steps,axis=0).reshape((-1,*field.shape[1:]))
    for rank in (2,8):
        predictions[f'POD-r{rank}-oracle']=reconstruct_pod(pods[rank],project_pod(pods[rank],flat[160:])).reshape((-1,*field.shape[1:]))
    # SINDy fitted only to rank-two standardized training coordinates.
    z=coeffs[2];s=np.maximum(z.std(axis=0),1e-12);zs=z/s;sindy_records={};models={}
    for threshold in PLAN['forecast']['sindy_thresholds']:
        name=f'SINDy-{threshold:g}'
        model=fit_integral_sindy(zs,dt,threshold=threshold,window=4,degree=3)
        rec={'threshold':threshold,'equations':model.equations(),'active_terms':int(np.count_nonzero(model.coefficients))}
        try:
            roll=rollout_sindy(model,zs[-1],dt,50)
            pred=reconstruct_pod(pods[2],roll*s).reshape((-1,*field.shape[1:]))
            rec.update(status='ok',validation_l2=relative_l2(pred[:50],field[160:210]))
            models[name]=model
        except ValueError as exc:
            rec.update(status='failed',reason=str(exc),validation_l2=None)
        sindy_records[name]=rec
    valid=[k for k,v in sindy_records.items() if v['status']=='ok']
    selected_sindy=min(valid,key=lambda k:sindy_records[k]['validation_l2']) if valid else None
    # Freeze the winner BEFORE any candidate's test horizon is attempted.
    for name in valid:
        try:
            roll=rollout_sindy(models[name],zs[-1],dt,steps)
            predictions[name]=reconstruct_pod(pods[2],roll*s).reshape((-1,*field.shape[1:]))
            sindy_records[name]['test_status']='ok'
        except ValueError as exc:
            sindy_records[name].update(test_status='failed',test_reason=str(exc))
    # Test scores are computed only after model selection above.
    for name,p in predictions.items():
        records.setdefault(name,{})
        records[name]['validation']=score(p[:50],field[160:210],c)
        records[name]['test']=score(p[50:],field[210:],c)
    # Probe location selected geometrically, not by test performance.
    iy=int(np.argmin(abs(c['y']-.5)));ix=int(np.argmin(abs(c['x']-4)))
    spectra={name:temporal_spectrum(p[50:,iy,ix],field[210:,iy,ix],dt) for name,p in predictions.items()}
    return {'selected_dmd':chosen,'selected_sindy':selected_sindy,'methods':records,
        'sindy_candidates':sindy_records,'spectra':spectra,'probe_xy':[float(c['x'][ix]),float(c['y'][iy])],
        'source_full_history_strouhal_diagnostic_only':float(c['original_strouhal']),
        'test_duration':float(len(field[210:])*dt),'dt':dt},predictions


def sensor_experiment(cases):
    train=np.concatenate([cases[r]['v'] for r in (90,110)]);shape=train.shape[1:]
    flat=train.reshape(len(train),-1);pod=fit_pod(flat,rank=8)
    sigma=.01*float(np.sqrt(np.mean(flat**2)));records=[];positions={}
    # All locations frozen before validation/test fields are evaluated.
    for budget in (8,16,32):
        positions[f'QR-{budget}']=select_sensors(pod.modes,budget)
        for seed in range(10,15):
            positions[f'random-{budget}-{seed}']=np.random.default_rng(seed).choice(flat.shape[1],budget,replace=False)
    examples={}
    for re,split in ((100,'validation'),(105,'test')):
        truth=cases[re]['v'];f=truth.reshape(len(truth),-1)
        # Paired full-grid artificial measurement noise, not additional CFD noise.
        for seed in range(10,15):
            noisy=f+np.random.default_rng(seed+1000*re).normal(0,sigma,f.shape)
            for budget in (8,16,32):
                for method in ('QR','random'):
                    key=f'QR-{budget}' if method=='QR' else f'random-{budget}-{seed}'
                    ids=positions[key];pred=reconstruct_sensors(pod,ids,noisy[:,ids]).reshape(truth.shape)
                    records.append({'split':split,'case':re,'method':method,'budget':budget,'seed':seed,
                        'condition_number':float(np.linalg.cond(pod.modes[ids])),**score(pred,truth,cases[re])})
                    if re==105 and method=='QR' and seed==10:examples[budget]=pred
        oracle=reconstruct_pod(pod,project_pod(pod,f)).reshape(truth.shape)
        mean=np.broadcast_to(pod.mean.reshape(shape),truth.shape)
        for name,pred in (('POD-oracle',oracle),('training-mean',mean)):
            records.append({'split':split,'case':re,'method':name,'budget':None,'seed':None,**score(pred,truth,cases[re])})
        if split=='validation':
            errors={b:float(np.mean([r['relative_l2'] for r in records if r['split']==split and r['method']=='QR' and r['budget']==b])) for b in (8,16,32)}
            meets=[b for b in errors if errors[b]<=.05]
            selected=min(meets) if meets else min(errors,key=errors.get)
    return {'selected_budget':selected,'noise_sigma':sigma,'validation_means':errors,'records':records,
        'positions':{k:v.tolist() for k,v in positions.items()},
        'replicate_scope':'Five artificial measurement-noise/layout seeds on one held Reynolds case; not independent CFD realizations.'},examples
