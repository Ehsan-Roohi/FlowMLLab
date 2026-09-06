"""Reproduce original modal teaching comparisons into a fresh output directory."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import platform
import importlib.metadata
import hashlib
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from flowmllab.modal_experiments import PLAN,load_cases,forecast_experiment,sensor_experiment


def figures(out,cases,forecast,pred,sensors,examples):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
    c=cases[110];truth=c['omega'];extent=[c['x'][0],c['x'][-1],c['y'][0],c['y'][-1]]
    chosen=forecast['selected_dmd'];sindy=forecast['selected_sindy']
    methods=[chosen,'MLP-r8']+([sindy] if sindy in pred else [])
    fig,axes=plt.subplots(len(methods)+1,1,figsize=(11,2.4*(len(methods)+1)),layout='compressed')
    vmax=float(np.max(abs(truth[-1])))
    for ax,name,field in zip(axes,['LBM reference']+methods,[truth[-1]]+[pred[n][-1] for n in methods]):
        im=ax.contourf(c['x'],c['y'],field,levels=np.linspace(-vmax,vmax,33),cmap='RdBu_r',extend='both')
        ax.set_aspect('equal')
        ax.set(title=name,ylabel='y/D',xlabel='x/D')
    fig.colorbar(im,ax=axes.tolist(),label='Vorticity, omega D/U',shrink=.8)
    fig.suptitle('Real LBM wake: final autonomous forecast (no test reset)')
    fig.savefig(out/'forecast_fields.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in methods+['persistence','POD-r2-oracle']:
        error=np.linalg.norm((pred[name]-truth[160:]).reshape(121,-1),axis=1)/np.linalg.norm(truth[160:].reshape(121,-1),axis=1)
        axes[0].plot(c['t'][160:]-c['t'][159],100*error,label=name)
    axes[0].axvline(c['t'][210]-c['t'][159],color='.4',ls='--',label='Test begins')
    axes[0].set(xlabel='Forecast horizon tU/D',ylabel='Frame relative L2 (%)',title='Validation then untouched test continuation')
    axes[0].legend(fontsize=8)
    theta=np.linspace(0,2*np.pi,200);axes[1].plot(np.cos(theta),np.sin(theta),'--',color='.6')
    modes=forecast['methods'][chosen]['modes'];axes[1].scatter([m['real'] for m in modes],[m['imag'] for m in modes])
    axes[1].set(xlabel='Real(lambda)',ylabel='Imag(lambda)',title=chosen+' eigenvalues',aspect='equal')
    fig.savefig(out/'forecast_audit.png');plt.close(fig)
    c=cases[105];budget=sensors['selected_budget'];truth=c['v'];estimate=examples[budget];ids=np.array(sensors['positions'][f'QR-{budget}']);iy,ix=np.unravel_index(ids,truth.shape[1:])
    vmax=float(np.max(abs(truth[-1])));err=abs(estimate[-1]-truth[-1])
    fig,axes=plt.subplots(3,1,figsize=(11,7),layout='compressed')
    for ax,name,field in zip(axes[:2],['Unseen-Re105 LBM reference',f'QR + D-optimal reconstruction: {budget} sensors'],[truth[-1],estimate[-1]]):
        im=ax.contourf(c['x'],c['y'],field,levels=np.linspace(-vmax,vmax,33),cmap='RdBu_r',extend='both')
        ax.set_aspect('equal')
        ax.scatter(c['x'][ix],c['y'][iy],s=18,facecolors='none',edgecolors='black',linewidths=.7)
        ax.set(title=name,ylabel='y/D',xlabel='x/D')
    fig.colorbar(im,ax=axes[:2].tolist(),label='v/U')
    im=axes[2].contourf(c['x'],c['y'],err,levels=np.linspace(0,max(float(err.max()),1e-12),25),cmap='magma')
    axes[2].set_aspect('equal')
    axes[2].set(title='Absolute reconstruction error (seed 10, declared example)',xlabel='x/D',ylabel='y/D')
    fig.colorbar(im,ax=axes[2],label='Absolute error in v/U')
    fig.savefig(out/'sensor_fields.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for method,color in [('QR','tab:blue'),('random','tab:orange')]:
        means=[]
        for b in (8,16,32):
            values=[100*r['relative_l2'] for r in sensors['records'] if r['split']=='test' and r['method']==method and r['budget']==b]
            ax.scatter([b]*len(values),values,color=color,alpha=.4,s=20);means.append(np.mean(values))
        ax.plot([8,16,32],means,'o-',color=color,label=method+' (mean of 5 seeds)')
    oracle=next(r['relative_l2'] for r in sensors['records'] if r['split']=='test' and r['method']=='POD-oracle')
    ax.axhline(100*oracle,ls='--',color='black',label='Full-field POD oracle, not a sensor method')
    ax.set(xlabel='Point velocity sensor count',ylabel='Relative L2 (%)',title='Re105 test; sensor budget selected on Re100',yscale='log',xticks=[8,16,32]);ax.legend(fontsize=8)
    fig.savefig(out/'sensor_audit.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output directory; retained evidence is read-only')
    a.output.mkdir(parents=True)
    (a.output/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n',newline='\n')
    start=time.perf_counter();cases=load_cases(ROOT/'data/modal_labs')
    with threadpool_limits(limits=1):
        forecast,pred=forecast_experiment(cases);sensors,examples=sensor_experiment(cases)
    result={'forecast':forecast,'sensors':sensors,'elapsed_seconds':time.perf_counter()-start,
        'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','matplotlib')}},
        'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('flowmllab/modal_tools.py','flowmllab/field_metrics.py','flowmllab/modal_experiments.py','data/modal_labs/manifest.json')}}
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',newline='\n')
    figures(a.output,cases,forecast,pred,sensors,examples)
    print(json.dumps({'seconds':result['elapsed_seconds'],'dmd':forecast['selected_dmd'],'sindy':forecast['selected_sindy'],'sensor_budget':sensors['selected_budget'],'test':{k:v['test']['relative_l2'] for k,v in forecast['methods'].items()}},indent=2))


if __name__=='__main__':main()
