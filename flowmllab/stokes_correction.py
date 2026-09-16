"""Stokes field -> Navier-Stokes field, with matched direct-learning controls."""
from __future__ import annotations
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, diags, eye, kron
from scipy.sparse.linalg import splu
from flowmllab.cavity_diversity import (lid_profile, vorticity_bc, fields,
    tensor_features, build_model, derivative_loss, physics_loss, evaluate, MODELS)


def stokes_operator(n):
    """Block FD Stokes operator; no convection, same Thom lid closure as NS.

    A psi + omega = 0; B psi + A omega = 2*g/h^3 at top-adjacent nodes.
    Stokes velocities, for prescribed wall motion, are viscosity-independent;
    pressure scales with viscosity. Re remains essential for the NS target.
    """
    if n<9:raise ValueError('n must be >=9')
    t=time.perf_counter();m=n-2;h=1/(n-1)
    d=diags([np.ones(m-1),-2*np.ones(m),np.ones(m-1)],[-1,0,1],format='csc')/h**2
    a=kron(eye(m),d)+kron(d,eye(m))
    wall=np.zeros((m,m));wall[0]+=1;wall[-1]+=1;wall[:,0]+=1;wall[:,-1]+=1
    b=diags(-2*wall.ravel()/h**4)
    matrix=bmat([[a,eye(m*m)],[b,a]],format='csc')
    lu=splu(matrix)
    return lu,matrix,time.perf_counter()-t


def solve_stokes(cases,n=25):
    lu,matrix,setup=stokes_operator(n);h=1/(n-1);m=n-2
    x=np.linspace(0,1,n)
    lids=np.array([lid_profile(x,c['coefficients']) for c in cases])
    psi=np.zeros((len(cases),n,n));omega=np.zeros_like(psi);times=[];res=[]
    for i,g in enumerate(lids):
        rhs=np.zeros(2*m*m);rhs[m*m:].reshape(m,m)[-1,:]=2*g[1:-1]/h**3
        t=time.perf_counter();sol=lu.solve(rhs);times.append(time.perf_counter()-t)
        res.append(np.linalg.norm(matrix@sol-rhs)/max(np.linalg.norm(rhs),1))
        psi[i,1:-1,1:-1]=sol[:m*m].reshape(m,m)
        omega[i,1:-1,1:-1]=sol[m*m:].reshape(m,m)
    vorticity_bc(omega,psi,lids,h)
    u,v=fields(psi,lids)
    return dict(psi=psi,omega=omega,u=u,v=v,lid=lids,linear_relative_residual=np.array(res),
                factorization_seconds=setup,solve_seconds=np.array(times))


def correction_features(cases,stokes,mode):
    n=stokes['psi'].shape[-1];desc,img=tensor_features(cases,n)
    use=mode=='stokes_corrected'
    extra=np.stack([stokes['psi']/.1,stokes['omega']/10],1)
    if not use:extra=np.zeros_like(extra)
    img=np.concatenate([img,extra],1).astype(np.float32)
    # Full lid sensor branch plus local Stokes centerline sensors; equal dimensions
    # in direct and corrected controls. No NS target enters these features.
    sensor=np.linspace(0,n-1,9).astype(int)
    low=stokes['psi'][:,n//2,sensor]/.1 if use else np.zeros((len(cases),9))
    desc=np.concatenate([desc[:,:1],stokes['lid'][:,sensor],low],1).astype(np.float32)
    return desc,img


def run(source='results/cavity_diversity_pilot',out='results/stokes_correction',epochs=300,seeds=(7,17,27)):
    import torch
    import pandas as pd
    from scipy.interpolate import RBFInterpolator
    torch.set_num_threads(2)
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text());cases=manifest['cases'];data=dict(np.load(source/'dataset.npz'));n=data['psi'].shape[-1];h=1/(n-1)
    low=solve_stokes(cases,n)
    assert np.max(low['linear_relative_residual'])<1e-9
    np.savez_compressed(out/'stokes_fields.npz',**low)
    protocol=dict(epochs=epochs,seeds=list(seeds),cases=cases,n=n,source_sha256=hashlib.sha256((source/'dataset.npz').read_bytes()).hexdigest(),task='same-BC Stokes field + Re -> NS correction',target_status='previously inspected same-grid NS regression targets, not fresh blind evidence',models=MODELS)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];history=[];predictions={};target=torch.tensor(data['psi'][:,None]/.1,dtype=torch.float32)
    for family in ('constant','diverse'):
        tr=np.array([i for i,c in enumerate(cases) if c['family']==family and c['split']=='train'])
        va=np.array([i for i,c in enumerate(cases) if c['family']==family and c['split']=='val'])
        for mode in ('direct','stokes_corrected'):
            des,im=correction_features(cases,low,mode);des=torch.tensor(des);im=torch.tensor(im)
            baseline=torch.tensor(low['psi'][:,None]/.1,dtype=torch.float32) if mode=='stokes_corrected' else torch.zeros_like(target)
            for seed in seeds:
                for name in MODELS:
                    torch.manual_seed(seed)
                    model=build_model(name,n,(target-baseline)[tr,0].numpy()*.1,descriptor_dim=des.shape[1],input_channels=6)
                    opt=torch.optim.Adam(model.parameters(),lr=.003);best=float('inf');t=time.perf_counter()
                    for epoch in range(epochs):
                        model.train();opt.zero_grad();p=baseline[tr]+model(des[tr],im[tr])
                        loss=((p-target[tr])**2).mean()+derivative_loss(p,target[tr],h)
                        if name=='fno_physics':loss=loss+physics_loss(p,des[tr],im[tr],h)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                        model.eval()
                        with torch.no_grad():
                            v=baseline[va]+model(des[va],im[va]);vl=((v-target[va])**2).mean()+derivative_loss(v,target[va],h)
                        if float(vl)<best:best=float(vl);state=copy.deepcopy(model.state_dict());chosen=epoch+1
                        history.append(dict(family=family,mode=mode,model=name,seed=seed,epoch=epoch+1,train=float(loss.detach()),validation=float(vl)))
                    elapsed=time.perf_counter()-t;model.load_state_dict(state);model.eval()
                    path=out/'checkpoints'/f'{family}_{mode}_{name}_{seed}.pt'
                    torch.save(dict(state_dict=state,model=name,mode=mode,family=family,seed=seed,epoch=chosen,descriptor_dim=des.shape[1],n=n),path)
                    sha=hashlib.sha256(path.read_bytes()).hexdigest();params=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
                    for tf in ('constant','diverse','ood_shape','ood_re'):
                        ids=np.array([i for i,c in enumerate(cases) if c['family']==tf and c['split']=='test'])
                        with torch.no_grad():
                            model(des[ids],im[ids]);tick=time.perf_counter();p=.1*(baseline[ids]+model(des[ids],im[ids]))[:,0].numpy();inference=(time.perf_counter()-tick)/len(ids)
                        predictions[f'{family}_{mode}_{name}_{seed}_{tf}']=p
                        met=evaluate(p,data['psi'][ids],data['lid'][ids])
                        for j,i in enumerate(ids):rows.append(dict(train_family=family,test_family=tf,mode=mode,model=name,seed=seed,case=cases[i]['id'],Re=cases[i]['Re'],parameters=params,epoch=chosen,training_seconds=elapsed,inference_seconds=inference,checkpoint_sha256=sha,**{k:float(v[j]) for k,v in met.items()}))
                    print(f'{family} {mode} {name} seed={seed} val={best:.5g} {elapsed:.1f}s',flush=True)
                    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(history).to_csv(out/'training.csv',index=False)
    desc,_=tensor_features(cases,n)
    for family in ('constant','diverse'):
        tr=np.array([i for i,c in enumerate(cases) if c['family']==family and c['split']=='train']);active=np.ptp(desc[tr],axis=0)>1e-7
        rbf=RBFInterpolator(desc[tr][:,active],data['psi'][tr].reshape(len(tr),-1),smoothing=1e-7)
        for tf in ('constant','diverse','ood_shape','ood_re'):
            ids=np.array([i for i,c in enumerate(cases) if c['family']==tf and c['split']=='test'])
            for name,p in [('stokes',low['psi'][ids]),('rbf_direct',rbf(desc[ids][:,active]).reshape(-1,n,n))]:
                met=evaluate(p,data['psi'][ids],data['lid'][ids]);predictions[f'{family}_baseline_{name}_0_{tf}']=p
                for j,i in enumerate(ids):rows.append(dict(train_family=family,test_family=tf,mode='baseline',model=name,seed=0,case=cases[i]['id'],Re=cases[i]['Re'],**{k:float(v[j]) for k,v in met.items()}))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);np.savez_compressed(out/'predictions.npz',**predictions)
    summary=pd.DataFrame(rows).groupby(['train_family','test_family','mode','model']).velocity_rel_l2.agg(['mean','std']).reset_index();summary.to_csv(out/'summary.csv',index=False)
    (out/'status.json').write_text(json.dumps(dict(status='complete',trained_checkpoints=2*2*len(seeds)*len(MODELS),paired_cases=len(cases),epochs=epochs,seeds=list(seeds)),indent=2))
    return summary


if __name__=='__main__':run()
