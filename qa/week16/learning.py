"""POD surrogates, untouched geometry tests, constrained near-field design."""
from pathlib import Path
import json,time,warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'results/week16_lowboom'

class Surrogate:
    def __init__(self,kind):self.kind=kind
    def fit(self,x,wave,cd):
        self.sx=StandardScaler().fit(x);self.pca=PCA(n_components=min(12,len(x)-1)).fit(wave)
        z=np.c_[self.pca.transform(wave),np.log(cd)]
        self.sy=StandardScaler().fit(z)
        if self.kind=='ridge':self.model=Ridge(alpha=.01)
        elif self.kind=='mlp':self.model=MLPRegressor(hidden_layer_sizes=(32,32),activation='tanh',solver='lbfgs',alpha=.01,max_iter=3000,random_state=16,tol=1e-7)
        elif self.kind=='gp':self.model=GaussianProcessRegressor(kernel=ConstantKernel(1,(.01,100))*Matern([1,1],(.05,20),nu=2.5)+WhiteKernel(1e-6,(1e-8,1e-2)),alpha=1e-10,normalize_y=False,n_restarts_optimizer=2,random_state=16)
        t=time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');self.model.fit(self.sx.transform(x),self.sy.transform(z))
        self.fit_warnings=[str(w.message) for w in caught];self.seconds=time.time()-t
        return self
    def predict(self,x):
        z=self.sy.inverse_transform(self.model.predict(self.sx.transform(np.atleast_2d(x))))
        return self.pca.inverse_transform(z[:,:-1]),np.exp(z[:,-1])

def assemble():
    rows=json.loads((E/'design_plan.json').read_text())['rows'];params=[];wave=[];cd=[];splits=[];names=[]
    for row in rows:
        p=E/'runs'/row['name'];m=json.loads((p/'metrics.json').read_text())
        if not m['converged']:raise ValueError(f'Unconverged {row["name"]}')
        d=np.load(p/'extracted.npz');params.append([row['a'],row['b']]);wave.append(d['cp'][1]);cd.append(m['cd_pressure']);splits.append(row['split']);names.append(row['name'])
    np.savez_compressed(E/'dataset.npz',parameters=params,waveforms=wave,cd=cd,splits=splits,names=names,x=d['x'])
    return np.load(E/'dataset.npz')

def train(data=None):
    if data is None:data=np.load(E/'dataset.npz')
    tr=data['splits']=='train';records=[];models={}
    for kind in ['ridge','mlp','gp']:
        m=Surrogate(kind).fit(data['parameters'][tr],data['waveforms'][tr],data['cd'][tr]);models[kind]=m
        for split in ['validation','test','extrapolation']:
            mask=data['splits']==split;truth=data['waveforms'][mask];w,cd=m.predict(data['parameters'][mask]);pk=truth.max(axis=1)
            records.append(dict(model=kind,split=split,wave_relative_l2=float(np.linalg.norm(w-truth)/np.linalg.norm(truth)),peak_mean_relative_error=float(np.mean(abs(w.max(axis=1)-pk)/pk)),cd_mean_relative_error=float(np.mean(abs(cd/data['cd'][mask]-1))),training_seconds=m.seconds,pod_retained_variance=float(sum(m.pca.explained_variance_ratio_)),warnings=m.fit_warnings))
    # Model selection uses validation only; test results never choose the optimizer.
    val=[r for r in records if r['split']=='validation'];best=min(val,key=lambda r:r['peak_mean_relative_error']+r['cd_mean_relative_error'])['model']
    report={'comparison':records,'selected_model':best,'selection_rule':'minimum validation peak MARE + drag MARE; fixed architectures'}
    (E/'learning_metrics.json').write_text(json.dumps(report,indent=2));return models[best],report

def optimize(model,write=True):
    baseline=json.loads((E/'runs/baseline_fine_stable/metrics.json').read_text());limit=1.02*baseline['cd_pressure'];pk0=baseline['peak_cp']
    def objective(x):
        w,cd=model.predict(x);return float(w.max()/pk0+1000*max(cd[0]/(.98*limit)-1,0)**2)
    opt=differential_evolution(objective,[(-.45,.45),(-.25,.25)],seed=1616,popsize=12,maxiter=120,tol=1e-8,polish=True)
    refined=minimize(lambda x: float(model.predict(x)[0].max()/pk0), opt.x, method='SLSQP', bounds=[(-.45,.45),(-.25,.25)], constraints=[{'type':'ineq','fun':lambda x:float(.98*limit-model.predict(x)[1][0])}], options={'ftol':1e-10,'maxiter':200})
    if refined.success: opt.x=refined.x
    w,cd=model.predict(opt.x)
    candidates=[dict(name='optimized',a=float(opt.x[0]),b=float(opt.x[1]),predicted_peak_cp=float(w.max()),predicted_cd=float(cd[0]))]
    # A separate matching problem: fit a feasible target obtained from this surrogate.
    # Retain actual feasible alternatives only; diversity is measured in parameter space.
    rng=np.random.default_rng(16);pool=rng.uniform([-.45,-.25],[.45,.25],(5000,2));pw,pc=model.predict(pool)
    good=np.where((pw.max(axis=1)<1.08*w.max())&(pc<=limit))[0]
    for k in range(2):
        if not len(good):break
        prev=np.array([[c['a'],c['b']] for c in candidates]);dist=np.linalg.norm((pool[good,None,:]-prev[None,:,:])/np.array([.9,.5]),axis=2).min(axis=1)
        j=good[np.argmax(dist)]
        if max(dist)<.04:break
        candidates.append(dict(name=f'alternative_{k+1}',a=float(pool[j,0]),b=float(pool[j,1]),predicted_peak_cp=float(pw[j].max()),predicted_cd=float(pc[j])))
        good=good[good!=j]
    result={'objective':'minimize max Cp at r/L=0.5, pressure drag <= 1.02 baseline','baseline_peak_cp':pk0,'drag_limit':limit,'candidates':candidates,'de_success':bool(opt.success),'surrogate_evaluations':opt.nfev}
    if write: (E/'design_candidates.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    d=assemble();m,r=train(d);print(json.dumps(r,indent=2));print(json.dumps(optimize(m),indent=2))
