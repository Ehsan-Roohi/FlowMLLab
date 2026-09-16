"""Validation-selected POD neural correction, with pressure recovered afterwards.

Two stages separate optimization/representation improvements from added data.
The original pilot test cases are retained as regression cases, never fit.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from flowmllab.cavity_diversity import case_manifest,solve_cases,evaluate,fields
from flowmllab.stokes_correction import solve_stokes
from common.w4utils import recover_pressure


def refined_data(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    old=Path('results/cavity_diversity_pilot')
    manifest=json.loads((old/'manifest.json').read_text());cases=manifest['cases'];data=dict(np.load(old/'dataset.npz'))
    if (out/'expanded.npz').exists():
        return json.loads((out/'expanded_cases.json').read_text()),dict(np.load(out/'expanded.npz'))
    extra=case_manifest(dict(train=48,val=12,test=0),seed=984512)
    extra=[dict(c,id='added_'+c['id']) for c in extra]
    def fingerprint(c):return (c['Re'],*c['coefficients'])
    assert not set(map(fingerprint,extra))&set(map(fingerprint,cases))
    print('Generating',len(extra),'additional training/validation cases; original tests unchanged',flush=True)
    generated=solve_cases(extra,n=data['psi'].shape[-1])
    full={k:np.concatenate([data[k],generated[k]]) for k in ('psi','omega','u','v','lid','residual')}
    np.savez_compressed(out/'expanded.npz',**full)
    (out/'expanded_cases.json').write_text(json.dumps(cases+extra,indent=2))
    return cases+extra,full


def gradient_metrics(basis,n):
    h=1/(n-1);p=basis.reshape(-1,n,n)
    u=(p[:,2:,1:-1]-p[:,:-2,1:-1])/(2*h)
    v=-(p[:,1:-1,2:]-p[:,1:-1,:-2])/(2*h)
    lap=(p[:,1:-1,2:]+p[:,1:-1,:-2]+p[:,2:,1:-1]+p[:,:-2,1:-1]-4*p[:,1:-1,1:-1])/h**2
    def gram(z):
        q=z.reshape(len(p),-1);return q@q.T/q.shape[1]
    return gram(u)+gram(v),gram(p),gram(lap)


class PODCorrection(nn.Module):
    def __init__(self,low_train,target_train,re_train,rank=20,width=64):
        super().__init__();n=low_train.shape[-1];self.n=n
        st=np.asarray(low_train).reshape(len(low_train),-1)
        lowmean=st.mean(0);_,s,vt=np.linalg.svd(st-lowmean,full_matrices=False)
        lowrank=max(1,int(np.sum(s>s[0]*1e-7)))
        lowbasis=vt[:lowrank];lowfeatures=(st-lowmean)@lowbasis.T
        lowstd=np.maximum(lowfeatures.std(0),1e-6)
        delta=(np.asarray(target_train)-np.asarray(low_train)).reshape(len(st),-1)
        mean=delta.mean(0);_,sv,b=np.linalg.svd(delta-mean,full_matrices=False)
        r=min(rank,len(st)-1,max(1,int(np.sum(sv>sv[0]*1e-7))))
        basis=b[:r];coeff=(delta-mean)@basis.T;scale=np.maximum(coeff.std(0),1e-6)
        for name,value in [('lowmean',lowmean),('lowbasis',lowbasis),('lowstd',lowstd),('mean',mean),('basis',basis),('scale',scale)]:
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))
        self.net=nn.Sequential(nn.Linear(lowrank+2,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,r))
        self.double()
    def coefficients(self,low,re):
        c=(low.flatten(1)-self.lowmean)@self.lowbasis.T/self.lowstd
        r=re[:,None]/400
        features=torch.cat([r,torch.log(r),c],1)
        return self.net(features)*self.scale
    def forward(self,low,re):
        return low+(self.mean+self.coefficients(low,re)@self.basis).reshape(-1,self.n,self.n)


def train_one(low,target,re,tr,va,rank,width,seed,omega_weight,epochs=1800):
    torch.manual_seed(seed);t=time.perf_counter()
    model=PODCorrection(low[tr],target[tr],re[tr],rank,width)
    lo=torch.tensor(low,dtype=torch.float64);ta=torch.tensor(target,dtype=torch.float64);rs=torch.tensor(re,dtype=torch.float64)
    target_coeff=(ta.flatten(1)-lo.flatten(1)-model.mean)@model.basis.T
    gv,gp,gw=gradient_metrics(model.basis.numpy(),model.n)
    # Fixed quadratic Sobolev metric avoids reconstructing full fields each step.
    gram=torch.tensor(gv+.1*gp+omega_weight*gw/100,dtype=torch.float64)
    validation_velocity=torch.tensor(gv,dtype=torch.float64)
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=2e-5)
    best=float('inf');history=[]
    def objective(ids):
        error=model.coefficients(lo[ids],rs[ids])-target_coeff[ids]
        return (error*(error@gram)).sum(1).mean()
    def validate(step,phase):
        nonlocal best,state,chosen
        with torch.no_grad():
            err=model.coefficients(lo[va],rs[va])-target_coeff[va]
            # Full-field validation error also includes POD projection error.
            pred=model(lo[va],rs[va]).numpy()
            val=float(np.mean(evaluate(pred,target[va],np.zeros((len(va),model.n)))['velocity_rel_l2']))
            training=float(objective(tr))
        history.append(dict(step=step,phase=phase,train=training,validation_velocity_rel_l2=val))
        if val<best:best=val;state=copy.deepcopy(model.state_dict());chosen=step
    state=None;chosen=0
    for epoch in range(epochs):
        opt.zero_grad();loss=objective(tr);loss.backward();opt.step();schedule.step()
        if epoch%20==0 or epoch==epochs-1:validate(epoch+1,'Adam')
    # Start L-BFGS from the validation-selected Adam checkpoint, not its last epoch.
    model.load_state_dict(state)
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=10,history_size=30,line_search_fn='strong_wolfe')
    for step in range(25):
        def closure():opt.zero_grad();loss=objective(tr);loss.backward();return loss
        opt.step(closure);validate(epochs+(step+1)*10,'LBFGS')
    model.load_state_dict(state);model.eval()
    return model,dict(rank=rank,width=width,omega_weight=omega_weight,seed=seed,validation=best,chosen_step=chosen,seconds=time.perf_counter()-t),history


def pressure_fields(psi,cases,lid):
    u,v=fields(psi,lid);n=psi.shape[-1];x=np.linspace(0,1,n);p=[];audit=[]
    for i,c in enumerate(cases):
        pressure,diag=recover_pressure(u[i],v[i],c['Re'],x,x);p.append(pressure);audit.append(diag)
    return u,v,np.array(p),audit


def run(stage='same_data'):
    torch.set_num_threads(2)
    out=Path('results/stokes_refined')/stage;out.mkdir(parents=True,exist_ok=True)
    old=Path('results/cavity_diversity_pilot');cases=json.loads((old/'manifest.json').read_text())['cases'];data=dict(np.load(old/'dataset.npz'))
    if stage=='expanded':cases,data=refined_data(out)
    low=solve_stokes(cases,data['psi'].shape[-1]);re=np.array([c['Re'] for c in cases]);rows=[];configs=[];hist=[];preds={}
    for family in ('constant','diverse'):
        tr=np.array([i for i,c in enumerate(cases) if c['family']==family and c['split']=='train'])
        va=np.array([i for i,c in enumerate(cases) if c['family']==family and c['split']=='val'])
        candidates=[(12,64,0.),(24,64,.02),(24,96,.02)]
        fitted=[]
        # Architecture choice uses validation only; no test prediction in this loop.
        for rank,width,weight in candidates:
            model,record,history=train_one(low['psi'],data['psi'],re,tr,va,rank,width,7,weight)
            configs.append(dict(family=family,**record));fitted.append((model,record))
            hist.extend(dict(family=family,candidate=f'{rank}_{width}_{weight}',**v) for v in history)
            print(stage,family,record,flush=True)
        bestmodel,best=min(fitted,key=lambda p:p[1]['validation'])
        ensemble=[(bestmodel,best)]
        for seed in (17,27):
            model,record,history=train_one(low['psi'],data['psi'],re,tr,va,best['rank'],best['width'],seed,best['omega_weight'])
            ensemble.append((model,record));configs.append(dict(family=family,**record))
            hist.extend(dict(family=family,candidate='selected_seed_'+str(seed),**v) for v in history)
        lo=torch.tensor(low['psi'],dtype=torch.float64);rr=torch.tensor(re,dtype=torch.float64)
        for model,record in ensemble:
            torch.save(dict(state_dict=model.state_dict(),config=record,n=model.n,family=family,train_indices=tr.tolist()),out/f'{family}_seed{record["seed"]}.pt')
        for tf in ('constant','diverse','ood_shape','ood_re'):
            ids=np.array([i for i,c in enumerate(cases) if c['family']==tf and c['split']=='test'])
            outputs=[]
            for model,record in ensemble:
                with torch.no_grad():p=model(lo[ids],rr[ids]).numpy()
                outputs.append(p);preds[f'{family}_{tf}_{record["seed"]}']=p
                metric=evaluate(p,data['psi'][ids],data['lid'][ids])
                for j,i in enumerate(ids):rows.append(dict(train_family=family,test_family=tf,seed=record['seed'],case=cases[i]['id'],**{k:float(v[j]) for k,v in metric.items()}))
            mean=np.mean(outputs,axis=0);preds[f'{family}_{tf}_ensemble']=mean
            metric=evaluate(mean,data['psi'][ids],data['lid'][ids])
            for j,i in enumerate(ids):rows.append(dict(train_family=family,test_family=tf,seed='ensemble',case=cases[i]['id'],**{k:float(v[j]) for k,v in metric.items()}))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(configs).to_csv(out/'selection.csv',index=False)
        pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
    np.savez_compressed(out/'predictions.npz',**preds)
    (out/'status.json').write_text(json.dumps(dict(stage=stage,complete=True,train_per_family=len(tr),validation_per_family=len(va),test_status='same original pilot regression tests',selection='validation only; mean ensemble predetermined'),indent=2))
    print(pd.DataFrame(rows).groupby(['train_family','test_family']).velocity_rel_l2.mean(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['same_data','expanded'],default='same_data');run(p.parse_args().stage)
