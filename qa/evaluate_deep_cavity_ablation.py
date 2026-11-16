#!/usr/bin/env python3
"""Post-freeze scoring of a controlled D5 study; cost smoke is never CFD scored."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('d5_field_audit', ROOT/'flowmllab/deep_cavity_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
METHODS = ('adam_lbfgs','soap_ssb')
LABELS = {'adam_lbfgs':'Adam → L-BFGS', 'soap_ssb':'SOAP → self-scaled Broyden'}
COLORS = {'adam_lbfgs':'#0072B2', 'soap_ssb':'#D55E00'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(data,path):
    tmp=Path(path).with_name(Path(path).name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def verify_frozen_run(directory):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('schema')!='flowmllab-d5-controlled-v1' or manifest.get('status')!='frozen':
        raise ValueError(f'Run is not a completed frozen D5 study: {directory}')
    config=manifest['configuration']
    if json.loads((directory/'contract.json').read_text(encoding='utf-8-sig'))!=config:
        raise ValueError('Frozen contract differs from manifest')
    if (config['Re'],config['W'],config['D'],config['dtype'])!=(1000.,1.,5.,'float64'):
        raise ValueError('Unexpected physical problem')
    for filename,key in [('checkpoint.pt','checkpoint_sha256'),('prediction.npz','prediction_sha256'),
                         ('final_audit.json','final_audit_sha256'),('history.csv','history_sha256')]:
        if sha(directory/filename)!=manifest[key]:
            raise ValueError(f'Frozen output SHA256 mismatch: {directory/filename}')
    counts=manifest['counters']
    if (counts['warm_evaluations'],counts['qn_evaluations'])!=(config['warm_evaluations'],config['qn_evaluations']):
        raise ValueError('Finite evaluation budget was not completed')
    if counts['function_evaluations']!=counts['warm_evaluations']+counts['qn_evaluations']:
        raise ValueError('Total evaluation count disagrees with phase counts')
    if counts['function_evaluations']!=counts['gradient_evaluations']:
        raise ValueError('Unequal completed function/gradient counts')
    if manifest.get('CFD_used_for_training_or_selection') is not False:
        raise ValueError('Training-data role is not explicitly data free')
    stamp=datetime.fromisoformat(manifest['frozen_at_utc'])
    if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):
        raise ValueError('Invalid freeze timestamp')
    with (directory/'history.csv').open(newline='',encoding='utf-8-sig') as stream:
        history=list(csv.DictReader(stream))
    if len(history)!=counts['function_evaluations']:
        raise ValueError('History does not account for every loss/gradient evaluation')
    if [int(r['function_evaluations']) for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('History counters are not contiguous')
    return {'directory':directory,'manifest':manifest,'config':config,'history':history}


def validate_pairs(runs):
    pairs={}
    for run in runs:
        m=run['manifest'];key=(m['seed'],m['method'])
        if m['method'] not in METHODS or key in pairs:
            raise ValueError('Duplicate seed/method or unknown optimizer pipeline')
        pairs[key]=run
    seeds=sorted({seed for seed,_ in pairs})
    checks=[]
    common=None
    for run in runs:
        shared={k:v for k,v in run['config'].items() if k not in ('method','seed','sample_seed')}
        if common is None:
            common=shared
        elif shared!=common:
            raise ValueError('Scientific schedule differs across paired seeds')
    for seed in seeds:
        if any((seed,method) not in pairs for method in METHODS):
            raise ValueError(f'Incomplete paired comparison at seed {seed}')
        a,b=(pairs[(seed,method)] for method in METHODS)
        ca,cb=(dict(run['config']) for run in (a,b))
        ca.pop('method');cb.pop('method')
        if ca!=cb:
            raise ValueError(f'Unmatched scientific/numerical contracts at seed {seed}')
        for key in ('initial_model_sha256','sample_sha256'):
            if a['manifest'][key]!=b['manifest'][key]:
                raise ValueError(f'Unmatched {key} at seed {seed}')
        def hardware(run):
            return {(r['hostname'],r['gpu_name'],str(r['gpu_capability']),r['torch'],r['slurm_job_id'])
                    for r in run['manifest']['runtime']}
        checks.append({'seed':seed,'identical_initial_model':True,'identical_samples':True,
                       'identical_numerical_contract_except_method':True,
                       'same_single_allocation_and_hardware':hardware(a)==hardware(b) and len(hardware(a))==1})
    purposes={run['config']['purpose'] for run in runs}
    if len(purposes)!=1:
        raise ValueError('Do not mix cost smoke and scientific pilot outputs')
    return pairs,seeds,checks,purposes.pop()


def normalize_history(run,output):
    path=output/'histories'/f"{run['manifest']['method']}_s{run['manifest']['seed']}.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=['objective_evaluations','metric_value','elapsed_seconds','phase','evaluation_kind','accepted']
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in run['history']:
            writer.writerow({'objective_evaluations':row['function_evaluations'],
                             'metric_value':row['loss'],'elapsed_seconds':row['elapsed_seconds'],
                             'phase':row['phase'],'evaluation_kind':row['evaluation_kind'],
                             'accepted':row['accepted']})
    raw_path=output/'frozen_histories'/path.name
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run['directory']/'history.csv',raw_path)
    return {'path':str(path.relative_to(output).as_posix()),'sha256':sha(path),
            'source_path':str(raw_path.relative_to(output).as_posix()),
            'source_sha256':run['manifest']['history_sha256'],
            'metric_name':'weighted_momentum_MSE','metric_label':'Training objective J',
            'independent_audit':False,'time_scope':'active wall time including audit/checkpoint I/O, excluding downtime',
            'loss_location':'warm pre-update closure; QN evaluated trial; accepted flag identifies endpoint when evaluated'}


def sampled_vorticity(u,v,x,h):
    # The same actual coordinate spacings and operator are used for both methods.
    return np.gradient(v,x,axis=1,edge_order=2)-np.gradient(u,h,axis=0,edge_order=2)


def load_prediction(run):
    with np.load(run['directory']/'prediction.npz',allow_pickle=False) as z:
        fields={name:z[name].copy() for name in ('x','h','u','v','p','psi','omega')}
    x,h=fields['x'],fields['h'];shape=(len(h),len(x))
    if min(len(x),len(h))<3 or not np.all(np.diff(x)>0) or not np.all(np.diff(h)>0):
        raise ValueError('Invalid frozen prediction axes')
    if any(fields[q].shape!=shape or not np.isfinite(fields[q]).all() for q in ('u','v','p','psi','omega')):
        raise ValueError('Invalid prediction fields')
    return fields


def profile(field,h,height):
    if not h[0]<=height<=h[-1]:
        raise ValueError('Requested profile exceeds retained physical coverage')
    j=int(np.searchsorted(h,height))
    if j==0:return field[0]
    if j==len(h):return field[-1]
    w=(height-h[j-1])/(h[j]-h[j-1])
    return (1-w)*field[j-1]+w*field[j]


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,
                         'legend.frameon':False,'figure.dpi':120})
    return plt


def save_figure(fig,path):
    path=Path(path)
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    return [{'path':path.with_suffix(ext).name,'sha256':sha(path.with_suffix(ext))}
            for ext in ('.png','.pdf')]


def convergence_figures(runs,output):
    plt=plotting();fig,axes=plt.subplots(1,2,figsize=(10.8,3.6),layout='constrained')
    handles=[]
    seed_styles={seed:('-', '--', ':', '-.')[i%4] for i,seed in enumerate(sorted({r['manifest']['seed'] for r in runs}))}
    for run in sorted(runs,key=lambda r:(r['manifest']['seed'],r['manifest']['method'])):
        m=run['manifest'];rows=run['history']
        loss=np.array([float(r['loss']) for r in rows]);count=np.arange(1,len(loss)+1)
        elapsed=np.array([float(r['elapsed_seconds']) for r in rows])
        if not np.isfinite(loss).all() or np.any(loss<0):
            raise ValueError('Invalid logged objective')
        label=f"{LABELS[m['method']]} | seed {m['seed']}"
        line,=axes[0].semilogy(count,loss,color=COLORS[m['method']],lw=1.1,ls=seed_styles[m['seed']],label=label)
        axes[1].semilogy(elapsed,loss,color=COLORS[m['method']],lw=1.1,ls=seed_styles[m['seed']])
        handles.append(line)
    axes[0].set(xlabel='Completed loss/gradient evaluations',ylabel='Weighted momentum objective J')
    axes[1].set(xlabel='Active wall time (s), including I/O and audits',ylabel='Weighted momentum objective J')
    for ax in axes:ax.grid(alpha=.2)
    fig.legend(handles=handles,loc='outside upper center',ncol=2,fontsize=9)
    result=save_figure(fig,output/'optimizer_objective_history');plt.close(fig)
    return result


def physical_figures(seed,reference,candidates,output):
    plt=plotting();x,h=reference['x'],reference['h'];result=[]
    for region,heights in [('upper',(4.90,4.95,4.98)),('floor',(.05,.10,.30))]:
        fig,axes=plt.subplots(3,3,figsize=(10.8,7.9),layout='constrained')
        for col,height in enumerate(heights):
            for row,q in enumerate(('u','p','omega')):
                ax=axes[row,col]
                z=profile(reference[q],h,height)
                if region=='floor' and q=='p':z=z-z.mean()
                ax.plot(x,z,color='black',lw=1.6,label='Matched CFD')
                for method in METHODS:
                    z=profile(candidates[method][q],h,height)
                    if region=='floor' and q=='p':z=z-z.mean()
                    ax.plot(x,z,color=COLORS[method],lw=1.2,ls='--' if method==METHODS[0] else '-.',label=LABELS[method])
                if row==0:ax.set_title(f'h/W = {height:g}',fontsize=10)
                ax.set_xlabel('x/W');ax.grid(alpha=.18)
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
                ax.set_ylabel({'u':'u/U','p':'(p − cut mean)/(ρU²)' if region=='floor' else 'p/(ρU²), interior gauge',
                               'omega':'ωW/U, common-grid derivative'}[q])
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
        result.extend(save_figure(fig,output/f'profiles_{region}_s{seed}'));plt.close(fig)
    fig=plt.figure(figsize=(8.8,8.5),layout='constrained')
    gs=fig.add_gridspec(2,2,width_ratios=(.8,1.3))
    axes=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
    from matplotlib.lines import Line2D
    for ax,(low,high) in zip(axes,((h[0],h[-1]),(4.35,h[-1]),(h[0],.65))):
        keep=(h>=low)&(h<=high)
        for method,field,color,style in [('CFD',reference,'black','-')]+[(m,candidates[m],COLORS[m],'--' if m==METHODS[0] else '-.') for m in METHODS]:
            streams=ax.streamplot(x,h[keep],field['u'][keep],field['v'][keep],color=color,
                                  density=.8,linewidth=.65,arrowsize=.55,broken_streamlines=True)
            streams.lines.set_linestyle(style)
        ax.set(xlabel='x/W',ylabel='h/W',xlim=(0,1),ylim=(low,high),aspect='equal')
    handles=[Line2D([0],[0],color='black',label='Matched CFD')]+[Line2D([0],[0],color=COLORS[m],ls='--' if m==METHODS[0] else '-.',label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,fontsize=9)
    result.extend(save_figure(fig,output/f'streamlines_s{seed}'));plt.close(fig)
    return result


def evaluate(run_paths,output,reference_directory):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    # No reference values are opened before ALL submitted runs are verified frozen.
    runs=[verify_frozen_run(p) for p in run_paths]
    pairs,seeds,pair_checks,purpose=validate_pairs(runs)
    records=[]
    for run in runs:
        m=run['manifest'];run_id=f"{m['method']}_s{m['seed']}"
        frozen_path=output/'frozen_predictions'/f'{run_id}.npz'
        frozen_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(run['directory']/'prediction.npz',frozen_path)
        if sha(frozen_path)!=m['prediction_sha256']:
            raise ValueError('Frozen prediction export verification failed')
        provenance=output/'run_provenance'
        provenance.mkdir(parents=True,exist_ok=True)
        exported={}
        for filename in ('manifest.json','contract.json'):
            path=provenance/f'{run_id}_{filename}'
            shutil.copy2(run['directory']/filename,path)
            exported[filename]={'path':str(path.relative_to(output).as_posix()),'sha256':sha(path)}
        records.append({'run_id':run_id,'Re':1000.,'seed':m['seed'],'method':m['method'],
                        'outcome':'cost_smoke_unscored' if purpose=='cost_smoke' else 'controlled_pilot_scored',
                        'run_directory':str(run['directory']),
                        'run_manifest_sha256':sha(run['directory']/'manifest.json'),
                        'training_manifest':exported['manifest.json'],'training_contract':exported['contract.json'],
                        'initial_model_sha256':m['initial_model_sha256'],'sample_sha256':m['sample_sha256'],
                        'prediction':{'path':str(frozen_path.relative_to(output).as_posix()),
                                      'sha256':m['prediction_sha256'],'coordinates':['x','h'],
                                      'layout':'h,x','omega_kind':'native_AD'},
                        'freeze':{'prediction_sha256':m['prediction_sha256'],'at_utc':m['frozen_at_utc'],
                                  'model_sha256':m['frozen_model_sha256'],'checkpoint_sha256':m['checkpoint_sha256']},
                        'history':normalize_history(run,output),
                        'training_seconds':m['training_seconds'],'active_seconds':m['active_seconds'],
                        'function_evaluations':m['counters']['function_evaluations'],
                        'gradient_evaluations':m['counters']['gradient_evaluations'],
                        'runtime':m['runtime'],'physical_residual_audit':m['final_audit'],
                        'configuration':run['config']})
    manifest={'schema_version':1,'campaign_kind':'d5_gaussian_lid',
              'study':'Fresh paired D5 optimizer-pipeline experiment',
              'path_base':'manifest_directory',
              'scorer_sha256':sha(__file__),
              'metric_module_sha256':sha(ROOT/'flowmllab/deep_cavity_audit.py'),
              'status':'cost_smoke_complete' if purpose=='cost_smoke' else 'controlled_pilot_scored',
              'purpose':purpose,'created_at_utc':utcnow(),'records':records,
              'paired_seeds':seeds,'pair_checks':pair_checks,
              'all_runs_frozen_before_reference_read':True,
              'historical_workflow_reproduction':False,'CFD_training_labels':False,
              'winner_claim':False,'metrics':[],'figures':[],
              'limitations':['Fixed balanced samples replace historical SOAP redraws; starts are fresh, not transferred.',
                             'A two-pipeline comparison cannot isolate each optimizer stage separately.',
                             'No convergence claim follows from finite budget completion.']}
    manifest['figures'].extend(convergence_figures(runs,output))
    if purpose=='cost_smoke':
        manifest['reference_values_accessed']=False
        manifest['limitations'].append('Cost smoke tests admission and expense only; no CFD accuracy comparison is made.')
        atomic_json(manifest,output/'evaluation_manifest.json')
        return manifest
    accessed=utcnow()
    data,reference_manifest=audit.load_bundle(reference_directory)
    x,h=data['x'],data['h'];regions=audit.masks(x,h)
    reference={'x':x,'h':h,**{q:data[f're1000_cfd_{q}'].copy() for q in ('u','v','p')}}
    reference['omega']=sampled_vorticity(reference['u'],reference['v'],x,h)
    reference_path=output/'comparison_reference.npz'
    np.savez_compressed(reference_path,**reference)
    ref_info={'path':reference_path.name,'sha256':sha(reference_path),
              'source_path':str((Path(reference_directory)/'reynolds_common_fields.npz').resolve()),
              'source_sha256':reference_manifest['data_sha256'],
              'source_manifest_sha256':sha(Path(reference_directory)/'manifest.json'),'Re':1000.,'D_over_W':5.,
              'lid':'Gaussian delta_x=sqrt(.001), W=1','omega_kind':'common_grid_second_order_FD',
              'physical_time_Ut_over_W':reference_manifest['CFD_times']['1000']}
    candidates={}
    for run,record in zip(runs,records):
        pred=load_prediction(run)
        if not np.array_equal(pred['x'],x) or not np.array_equal(pred['h'],h):
            # Source generators may differ only in floating-point endpoint association.
            if pred['x'].shape!=x.shape or pred['h'].shape!=h.shape or not np.allclose(pred['x'],x,rtol=0,atol=2e-14) or not np.allclose(pred['h'],h,rtol=0,atol=2e-14):
                raise ValueError('CFD/prediction grids differ; no silent resampling is permitted')
        pshift=float((pred['p']-reference['p'])[regions['Interior']].mean())
        comparison={q:pred[q].copy() for q in ('u','v','p','psi')}
        comparison['p']-=pshift
        comparison['omega']=sampled_vorticity(pred['u'],pred['v'],x,h)
        path=output/f"comparison_{record['run_id']}.npz"
        np.savez_compressed(path,x=x,h=h,**comparison)
        region_results={name:audit.metrics(comparison,reference,mask) for name,mask in regions.items()}
        record['reference']=ref_info
        record['score']={'prediction_sha256':record['prediction']['sha256'],
                         'reference_accessed_at_utc':accessed,'reference_source_sha256':reference_manifest['data_sha256']}
        record['comparison_prediction']={'path':path.name,'sha256':sha(path),
                                         'source_prediction_sha256':record['prediction']['sha256'],
                                         'pressure_shift_subtracted':pshift,
                                         'pressure_gauge':'single common Interior difference mean',
                                         'omega_kind':'common_grid_second_order_FD','coordinates':['x','h'],'layout':'h,x'}
        record['scores']={'regions':region_results}
        for name,result in region_results.items():
            manifest['metrics'].append({'seed':record['seed'],'method':record['method'],'region':name,**result})
        candidates[(record['seed'],record['method'])]=comparison
    manifest['paired_differences']=[]
    metric_names=('velocity_RMS_U','pressure_RMS_rhoU2','vorticity_RMS_UoverW')
    indexed={(r['seed'],r['method']):r['scores']['regions'] for r in records}
    for seed in seeds:
        for region in regions:
            for metric in metric_names:
                baseline=indexed[(seed,METHODS[0])][region][metric]
                candidate=indexed[(seed,METHODS[1])][region][metric]
                manifest['paired_differences'].append({
                    'seed':seed,'region':region,'metric':metric,
                    'baseline_method':METHODS[0],'candidate_method':METHODS[1],
                    'baseline':baseline,'candidate':candidate,
                    'candidate_minus_baseline':candidate-baseline,
                    'relative_change_percent':100*(candidate-baseline)/baseline if baseline>0 else None})
    manifest['descriptive_seed_summary']=[]
    for region in regions:
        for metric in metric_names:
            rows=[r for r in manifest['paired_differences'] if r['region']==region and r['metric']==metric]
            differences=np.array([r['candidate_minus_baseline'] for r in rows])
            manifest['descriptive_seed_summary'].append({
                'region':region,'metric':metric,'paired_seed_count':len(rows),
                'baseline_median':float(np.median([r['baseline'] for r in rows])),
                'candidate_median':float(np.median([r['candidate'] for r in rows])),
                'median_paired_difference':float(np.median(differences)),
                'candidate_lower_error_seeds':int(np.sum(differences<0)),
                'candidate_higher_error_seeds':int(np.sum(differences>0)),
                'interpretation':'descriptive finite paired seeds; no significance or universal-winner claim'})
    for seed in seeds:
        manifest['figures'].extend(physical_figures(seed,reference,{method:candidates[(seed,method)] for method in METHODS},output))
    manifest['reference_values_accessed']=True
    manifest['reference_accessed_at_utc']=accessed
    manifest['reference']=ref_info
    manifest['normalization']='vector RMS includes both components; relative velocity uses matched regional CFD RMS, pressure retains one interior gauge'
    manifest['vorticity']='Both fields differentiated with the same second-order sampled operator; native AD omega is not used for CFD error'
    if len(seeds)<3:
        manifest['limitations'].append('Fewer than three paired seeds: no seed-robust optimizer conclusion.')
    metric_path=output/'regional_metrics.csv'
    with metric_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest['metrics'][0]));writer.writeheader();writer.writerows(manifest['metrics'])
    manifest['metrics_file']={'path':metric_path.name,'sha256':sha(metric_path)}
    atomic_json(manifest,output/'evaluation_manifest.json')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path,default=ROOT/'data/week13_d5')
    args=parser.parse_args()
    result=evaluate(args.runs,args.output,args.reference_directory)
    print(json.dumps({'status':result['status'],'seeds':result['paired_seeds'],
                      'reference_values_accessed':result['reference_values_accessed'],
                      'manifest':str(args.output/'evaluation_manifest.json')},indent=2))


if __name__=='__main__':
    main()
