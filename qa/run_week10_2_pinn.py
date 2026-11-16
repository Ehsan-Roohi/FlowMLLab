"""Freeze and run the CPU teaching protocol; DVM scores never select a model."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from flowmllab import kinetic_pinn as kp, kinetic_dvm as kd, kinetic_shock as ks

OUT=ROOT/'results/week10_2_kinetic_shock'


def protocol():
    base=kp.PINNConfig(seed=1729,depth=2,width=24,optimizer='adam_lbfgs',
        adam_steps=100,lbfgs_steps=20,collocation_points=64,velocity_order=32,
        time_order=8,vmax=7.,weights=(1.,1.,1.),max_seconds=120.,threads=1)
    return [('default',base),
            ('adam',replace(base,optimizer='adam',adam_steps=120,lbfgs_steps=0)),
            ('lbfgs',replace(base,optimizer='lbfgs',adam_steps=0,lbfgs_steps=40)),
            ('depth1',replace(base,depth=1)),('depth3',replace(base,depth=3)),
            ('energy0p1',replace(base,weights=(1.,1.,.1))),
            ('energy10',replace(base,weights=(1.,1.,10.)))]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(x,pred,ref):
    return {f:ks.profile_metrics(x,pred[:,i],ref[:,i])['relative_l2_percent']
            for i,f in enumerate(kp.FIELDS)}


def run(default_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    cases=protocol()[:1] if default_only else protocol()
    record={'created_utc':datetime.now(timezone.utc).isoformat(),
        'case':'BGK smoothed tube; tau=1; t_final=.02; x=[-.5,.5]',
        'frozen_before_training_and_DVM_scoring':True,
        'configurations':{name:asdict(cfg) for name,cfg in cases},
        'evaluation_quadrature':asdict(kp.Quadrature(48,12,7.)),
        'DVM_configurations':[asdict(kd.DVMConfig(cells=n,velocity_order=48)) for n in (400,800)],
        'velocity_sensitivity':asdict(kd.DVMConfig(cells=800,velocity_order=64)),
        'selection':'final completed update in every declared run; no best-DVM selection',
        'optimizer_budget_note':'Illustrative predetermined update budgets; compare actual gradient/function evaluations and seconds, not outer steps alone.',
        'source_hashes':{p:digest(ROOT/p) for p in ('flowmllab/kinetic_pinn.py','flowmllab/kinetic_dvm.py','qa/run_week10_2_pinn.py')}}
    (OUT/'pinn_training_protocol.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    bundles={}; reports={}; states={}
    for name,cfg in cases:
        bundle=kp.train(cfg)
        bundles[name]=bundle; reports[name]=kp.training_report(bundle)
        for role,model in [('initial',bundle.initial_model),('trained',bundle.model)]:
            for key,value in model.state_dict().items(): states[name+'__'+role+'__'+key]=value.detach().numpy()
        print(json.dumps({'run':name,'stop':bundle.stop_reason,'work':bundle.work,
                          'initial_loss':bundle.history[0]['loss'],'final_loss':bundle.history[-1]['loss']}),flush=True)
    # Architecture varies capacity. Optimizer/weight comparisons retain identical initial parameters.
    base=reports['default']['provenance']
    for name,report in reports.items():
        assert report['provenance']['collocation_sha256']==base['collocation_sha256']
        if name not in ('depth1','depth3'):
            assert report['provenance']['initial_state_sha256']==base['initial_state_sha256']
    # Freeze learned states before producing or reading any DVM target fields.
    np.savez_compressed(OUT/'pinn_cpu_states.npz',**states)
    refs={str(n):kd.solve(kd.DVMConfig(cells=n,velocity_order=48)) for n in (400,800)}
    refs['800_v64']=kd.solve(kd.DVMConfig(cells=800,velocity_order=64))
    fine,coarse=refs['800'],refs['400']
    x=fine['x']; q=kp.Quadrature(48,12,7.)
    arrays={'x':x,'DVM':fine['fields'],'x_coarse':coarse['x'],'DVM_coarse':coarse['fields'],
            'DVM_v64':refs['800_v64']['fields'],'columns':np.asarray(kp.FIELDS)}
    rows=[]
    for name,bundle in bundles.items():
        pred=kp.evaluate(bundle,x,quadrature=q)
        initial=kp.evaluate(bundle,x,quadrature=q,state='initial')
        arrays[name]=pred['fields']; arrays[name+'__initial']=initial['fields']
        arrays[name+'__source_macros']=pred['source_macros']
        for state,fields in [('initial',initial['fields']),('trained',pred['fields'])]:
            for field,value in score(x,fields,fine['fields']).items():
                rows.append({'run':name,'state':state,'field':field,'relative_l2_percent':value})
        reports[name]['posttraining_consistency_rms_at_DVM_centres']=pred['consistency_rms'].tolist()
    refined=kp.evaluate(bundles['default'],x,quadrature=kp.Quadrature(64,16,7.))
    arrays['default__quadrature64']=refined['fields']
    # Conservative spatial restriction of g/h/k, then form moments on the coarse cells.
    restricted=fine['state'].reshape(coarse['config']['cells'],2,3,len(fine['v'])).mean(axis=1)
    reduced=kd.moments(restricted,fine['v'],fine['weights'])
    restricted_fields=np.column_stack([reduced[f] for f in kp.FIELDS])
    sensitivity={'space_time_400_to_800':score(coarse['x'],coarse['fields'],restricted_fields),
                 'velocity_48_to_64':score(x,fine['fields'],refs['800_v64']['fields']),
                 'frozen_PINN_quadrature_48x12_to_64x16':score(x,arrays['default'],refined['fields']),
                 'scope':'two space/time levels at fixed CFL; no convergence order or GCI claim; velocity and PINN quadratures checked separately'}
    # The paper reference uses a different finite transverse discretization; secondary only.
    paper=ks.load_case('tube_bgk')
    paper_prediction=kp.evaluate(bundles['default'],paper.x,quadrature=q)
    arrays.update(paper_x=paper.x,paper_DVM=paper.profiles['reference'],default_at_paper_nodes=paper_prediction['fields'])
    np.savez_compressed(OUT/'pinn_cpu_profiles.npz',**arrays)
    pd.DataFrame(rows).to_csv(OUT/'pinn_cpu_metrics.csv',index=False,float_format='%.12g')
    result={'protocol':record,'runs':reports,'DVM':{key:value['diagnostics'] for key,value in refs.items()},
            'sensitivity':sensitivity,'secondary_paper_DVM_error_percent':score(paper.x,paper_prediction['fields'],paper.profiles['reference']),
            'paper_reference_role':'secondary; finite transverse quadrature differs from this analytic-transverse teaching discretization',
            'artifact_sha256':{p:digest(OUT/p) for p in ('pinn_training_protocol.json','pinn_cpu_states.npz','pinn_cpu_profiles.npz','pinn_cpu_metrics.csv')},
            'fresh_CPU_training':True,'fresh_DVM':True,'DVM_data_used_for_training_or_model_selection':False}
    (OUT/'pinn_cpu_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'completed_runs':len(reports),'sensitivity':sensitivity}),flush=True)


def figures():
    import matplotlib.pyplot as plt
    from qa.lecture_typography import configure_matplotlib
    configure_matplotlib()
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    from qa.build_week10_2_materials import LABELS
    with np.load(OUT/'pinn_cpu_profiles.npz',allow_pickle=False) as data:
        x=data['x']; fields=('rho','u','T','q','sigma','Delta')
        fig,axes=plt.subplots(2,3,figsize=(9,5.6))
        for ax,f in zip(axes.flat,fields):
            i=kp.FIELDS.index(f)
            ax.plot(x,data['DVM'][:,i],'k-',lw=2,label='Independent BGK DVM')
            ax.plot(x,data['default__initial'][:,i],color='#2355b5',ls='--',lw=1.5,label='Random-start PINN')
            ax.plot(x,data['default'][:,i],color='#c52a32',ls='-.',lw=1.5,label='Trained CPU PINN')
            ax.set(xlabel='$x$',ylabel=LABELS[f],xlim=(-.5,.5)); ax.grid(alpha=.15)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,3),useMathText=True)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.86));fig.savefig(OUT/'pinn_from_scratch_profiles.png',dpi=200);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for key,color,style,label in [('DVM','black','-','DVM'),('adam','#2355b5','--','Adam'),
                                      ('lbfgs','#16834b',':','L-BFGS'),('default','#c52a32','-.','Adam then L-BFGS')]:
            if key not in data:continue
            for ax,f in zip(axes,('q','sigma')):
                ax.plot(x,data[key][:,kp.FIELDS.index(f)],color=color,ls=style,lw=1.6,label=label)
                ax.set(xlabel='$x$',ylabel=LABELS[f],xlim=(-.25,.25));ax.grid(alpha=.15)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.79));fig.savefig(OUT/'pinn_optimizer_profiles.png',dpi=200);plt.close(fig)
    report=json.loads((OUT/'pinn_cpu_report.json').read_text())
    history=report['runs']['default']['history']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for component,color in zip(kp.COMPONENTS,('#2355b5','#c52a32','#16834b')):
        values=[h['scaled_mse'][component] for h in history]
        axes[0].semilogy([h['gradient_evaluations'] for h in history],values,color=color,label=component)
        axes[1].semilogy([h['elapsed_seconds'] for h in history],values,color=color,label=component)
    axes[0].set_xlabel('Gradient evaluations');axes[1].set_xlabel('Elapsed training time (s)')
    for ax in axes:ax.set_ylabel('Unweighted scaled consistency MSE');ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(OUT/'pinn_training_history.png',dpi=200);plt.close(fig)
    print('Wrote 3 fresh-training figures.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--default-only',action='store_true');p.add_argument('--figures',action='store_true')
    args=p.parse_args()
    if args.run:run(args.default_only)
    if args.figures:figures()
