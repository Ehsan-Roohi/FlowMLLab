"""Native post-freeze evaluation for the case-specific, label-free Ghia PINNs.

No training or model selection occurs here. All supplied projections are frozen
before CFD field values or Ghia target arrays are loaded in this process.
"""
from __future__ import annotations
import argparse
import ast
import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REYNOLDS=(100,400,1000,3200,5000,7500,10000)

def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(arrays):
    h=hashlib.sha256()
    for key,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a)
        h.update(key.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def file_entry(path):
    return {'path':Path(path).resolve().relative_to(ROOT.resolve()).as_posix(),'sha256':sha(path)}
def checked_entry(entry):
    path=(ROOT/entry['path']).resolve()
    if not path.is_relative_to(ROOT.resolve()) or sha(path)!=entry['sha256']:
        raise ValueError('Evidence path or SHA256 mismatch')
    return path
@contextmanager
def writer_lock(path):
    stream=Path(path).open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:
        stream.close()

def run_id(config,fallback='unknown'):
    if not all(k in config for k in ('reynolds','seed','collar')):return fallback
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return f"re{int(config['reynolds'])}_s{config['seed']}_eps{config['collar']:g}_{digest[:10]}"

def npz_once(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in arrays.items()):
                raise ValueError('Refusing to replace different immutable arrays: '+str(path))
        return sha(path)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return sha(path)

def coordinate_literals(source):
    """Parse X/Y literals only; do not execute reference target assignments."""
    result={}
    for node in ast.parse(Path(source).read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('X','Y'):
                result[name.lower()]=np.asarray(ast.literal_eval(node.value.args[0]),dtype=float)
    if set(result)!={'x','y'}:raise ValueError('Ghia coordinate declarations changed; explicit review required')
    for a in result.values():
        if a.ndim!=1 or len(np.unique(a))!=len(a) or not np.isfinite(a).all():raise ValueError('Invalid published coordinates')
    return result

def native_coordinates(reference,expected_cells=512):
    """Access only coordinate NPZ members, not u/v/omega reference values."""
    reference=Path(reference);before=sha(reference)
    with np.load(reference,allow_pickle=False) as f:
        x=f['x'].copy();y=f['y'].copy()
    for a in (x,y):
        if a.ndim!=1 or len(a)!=expected_cells+1 or not np.isfinite(a).all():raise ValueError('Unexpected native CFD coordinate shape')
        if not np.all(np.diff(a)>0) or a[0]!=0 or a[-1]!=1:raise ValueError('CFD coordinates must span the unit square monotonically')
    return dict(x=x[1:-1],y=y[1:-1],full_x=x,full_y=y,reference_sha256=before,
        coordinates_read_at_utc=utc(),coordinate_sha256=array_sha({'x':x,'y':y}),
        subset='exact native interior x[1:-1], y[1:-1]; no interpolation; boundaries audited separately')

def freeze_projection(directory,identity,coords,ghia_coords,predictor):
    """Predict all locations without reference values; verify and reuse a freeze."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ledger=directory/'projection_freeze.json';path=directory/'prediction_native_interior.npz'
    bound={**identity,'coordinate_sha256':coords['coordinate_sha256'],
        'ghia_coordinate_sha256':array_sha(ghia_coords),'reference_file_sha256':coords['reference_sha256']}
    if ledger.exists():
        old=read_json(ledger)
        if old['identity']!=bound or sha(path)!=old['prediction_sha256']:
            raise ValueError('Frozen projection identity or bytes changed')
        return old,path
    xx,yy=np.meshgrid(coords['x'],coords['y'])
    values=predictor(np.column_stack((xx.ravel(),yy.ravel())))
    arrays={'x':coords['x'],'y':coords['y']}
    for name in ('u','v','psi','p','omega'):
        a=np.asarray(values[name]).reshape(xx.shape)
        if not np.isfinite(a).all():raise ValueError('Nonfinite native interior prediction: '+name)
        arrays[name]=a
    v=predictor(np.column_stack((np.full_like(ghia_coords['y'],.5),ghia_coords['y'])))
    h=predictor(np.column_stack((ghia_coords['x'],np.full_like(ghia_coords['x'],.5))))
    arrays.update(ghia_y=ghia_coords['y'],ghia_u_prediction=np.asarray(v['u']),
                  ghia_x=ghia_coords['x'],ghia_v_prediction=np.asarray(h['v']))
    if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite projection')
    result=dict(schema_version=1,identity=bound,prediction_sha256=npz_once(path,arrays),at_utc=utc(),
        coordinates_read_at_utc=coords['coordinates_read_at_utc'],subset=coords['subset'],reference_values_read=False,
        pressure='PINN centre gauge; this CFD bundle contains no pressure reference')
    write_json(ledger,result)
    return result,path

def weights(x,y):
    def axis(a):
        d=np.diff(a)
        if len(d)<1 or not np.all(d>0):raise ValueError('At least two increasing native coordinates required')
        return np.r_[d[0]/2,(d[:-1]+d[1:])/2,d[-1]/2]
    return np.outer(axis(y),axis(x))

def native_scores(pred,ref):
    for key in ('x','y'):
        if not np.array_equal(pred[key],ref[key]):raise ValueError('Native coordinates differ; interpolation is forbidden')
    shape=(len(ref['y']),len(ref['x']))
    for field in (pred,ref):
        for key in ('u','v','omega'):
            if field[key].shape!=shape or not np.isfinite(field[key]).all():raise ValueError('Invalid native field '+key)
    w=weights(ref['x'],ref['y']);yy=np.broadcast_to(ref['y'][:,None],shape)
    out={}
    for name,mask in {'interior':np.ones(shape,bool),'lower_quarter':yy<=.25,'upper_quarter':yy>=.75}.items():
        wm=w*mask;total=wm.sum()
        if total<=0:raise ValueError('Region has no samples: '+name)
        mean=lambda a:float(np.sum(wm*a)/total)
        rms=lambda a:float(np.sqrt(mean(a*a)))
        truth=np.sqrt(mean(ref['u']**2+ref['v']**2))
        error=np.sqrt(mean((pred['u']-ref['u'])**2+(pred['v']-ref['v'])**2))
        de=pred['omega']-ref['omega'];om=rms(ref['omega'])
        out[name]=dict(sample_count=int(mask.sum()),reference_velocity_rms_U=float(truth),
            velocity_error_rms_U=float(error),velocity_relative_l2=float(error/truth) if truth>0 else None,
            u_error_signed_mean_U=mean(pred['u']-ref['u']),v_error_signed_mean_U=mean(pred['v']-ref['v']),
            omega_error_rms_U_over_W=rms(de),omega_error_signed_mean_U_over_W=mean(de),
            omega_relative_l2=rms(de)/om if om>0 else None)
    return dict(regions=out,quadrature='tensor trapezoid over exact sampled interior rectangle',
        sampled_rectangle=[float(ref['x'][0]),float(ref['x'][-1]),float(ref['y'][0]),float(ref['y'][-1])],
        pressure_comparison='unavailable: retained CFD bundle has no pressure field')

def ghia_scores(reynolds,pred,ghia):
    truth=ghia.reference(reynolds);result={};rows=[]
    for component,coord in (('u','y'),('v','x')):
        positions=pred['ghia_'+coord];values=pred['ghia_'+component+'_prediction']
        if not np.array_equal(positions,truth[coord]):raise ValueError('Ghia query coordinates differ; no prediction interpolation permitted')
        if values.shape!=truth[component].shape or not np.isfinite(values).all():raise ValueError('Invalid Ghia query values')
        err=values-truth[component];keep=ghia.valid_mask(reynolds,component,positions)
        result[component]=dict(screened_relative_l2=float(np.linalg.norm(err[keep])/np.linalg.norm(truth[component][keep])),
            published_relative_l2=float(np.linalg.norm(err)/np.linalg.norm(truth[component])),
            screened_rmse_U=float(np.sqrt(np.mean(err[keep]**2))),screened_linf_U=float(np.max(np.abs(err[keep]))),
            n_screened=int(keep.sum()),n_published=len(keep),
            prediction_evaluation='direct frozen network at original tabulated coordinates; no interpolation')
        for pos,nn,t,e,k in zip(positions,values,truth[component],err,keep):
            rows.append(dict(component=component,position=float(pos),published=float(t),prediction=float(nn),
                signed_error_U=float(e),screened_included=bool(k)))
    return result,rows

def history_csv(history_path,output):
    history=read_json(history_path);path=Path(output)/'training_history.csv'
    columns=['objective_evaluations','residual_point_evaluations','elapsed_seconds','metric_value','stage','loss_location']
    rows=[]
    for row in history:
        loss=row['training_loss']
        if not np.isfinite(loss) or loss<0:raise ValueError('Invalid recorded training loss')
        rows.append(dict(objective_evaluations=row['closure_calls'],residual_point_evaluations=row['residual_point_evaluations'],
            elapsed_seconds=row['elapsed_seconds'],metric_value=float(np.sqrt(loss)),stage=row['stage'],loss_location=row['loss_location']))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    return dict(path=path,metric_name='training_collar_momentum_rms',metric_label='Training momentum RMS outside collar',
        independent_audit=False,time_scope='optimizer-step time; excludes admission, checkpoint I/O and final reference evaluation',
        work_scope='training closure residual points only; final audits reported separately',
        loss_location='last closure; not a recomputed accepted endpoint')

def validate_evaluation_device(device):
    if device not in ('cpu','cuda'):raise ValueError('Evaluation device must be cpu or cuda')
    if device=='cuda':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA evaluation requested but CUDA is unavailable')
    return device


def prepare_run(run,output,engine,ghia_source,device='cpu'):
    run=Path(run);config=read_json(run/'config.json');re=int(config['reynolds'])
    if re not in REYNOLDS:raise ValueError('Unsupported Reynolds number')
    freeze_path=run/'frozen_manifest.json'
    if not freeze_path.exists():
        return dict(pending=True,record=dict(run_id=run_id(config,run.name),Re=re,seed=config['seed'],
            method='Adam-to-L-BFGS, fixed collocation',collar=config['collar'],
            outcome='training_incomplete',reason='No completed frozen endpoint'))
    frozen=read_json(freeze_path)
    if sha(run/frozen['prediction_file'])!=frozen['prediction_sha256']:raise ValueError('Original frozen prediction hash mismatch')
    if sha(run/frozen['physics_audit_file'])!=frozen['physics_audit_sha256']:raise ValueError('Original physical audit hash mismatch')
    receipt=read_json(run/'current_receipt.json')
    if not receipt['finished'] or sha(run/'checkpoint.pt')!=receipt['checkpoint_sha256']:raise ValueError('Completed checkpoint identity failed')
    trainer=engine.Trainer.load(run/'checkpoint.pt',device=device)
    if not trainer.finished or trainer.frozen!=frozen['frozen_model']:raise ValueError('Model differs from frozen endpoint')
    if frozen['source_sha256']!=engine.source_hash():raise ValueError('Original frozen source identity failed')
    if engine.tensor_hash(trainer.model.state_dict())!=trainer.frozen['model_sha256']:raise ValueError('Frozen model tensor hash failed')
    if config!=frozen['config'] or config!=asdict(trainer.config):raise ValueError('Frozen configuration mismatch')
    identifier=run_id(config)
    directory=Path(output)/'runs'/identifier
    reference=ROOT/'results/cavity_highre_ghia/n512'/f're{re}.npz'
    coords=native_coordinates(reference)
    identity=dict(run_id=identifier,Re=re,seed=config['seed'],config=config,
        model_sha256=trainer.frozen['model_sha256'],config_sha256=trainer.frozen['config_sha256'],
        source_canonical_sha256=engine.source_hash(),projection_runtime=trainer.runtime(),evaluation_device=device,
        ghia_source_sha256=sha(ghia_source),evaluator_source_sha256=sha(__file__),checkpoint_sha256=receipt['checkpoint_sha256'],
        original_freeze_manifest_sha256=sha(freeze_path),original_prediction_sha256=frozen['prediction_sha256'],
        original_frozen_at_utc=frozen['frozen_utc'],training_runtime_history=receipt.get('runtime_history',[]))
    projection,prediction=freeze_projection(directory,identity,coords,coordinate_literals(ghia_source),
        lambda xy:engine.predict(trainer.model,xy,device,trainer.config.chunk_size))
    return dict(pending=False,run=run,run_id=identifier,directory=directory,reference=reference,coords=coords,
        projection=projection,prediction=prediction,config=config,physics=read_json(run/'final_physics_audit.json'))

def score_prepared(item,output,ghia,targets_accessed_at_utc=None):
    directory=item['directory'];config=item['config'];re=config['reynolds']
    scored_path=directory/'evaluation_record.json'
    if scored_path.exists():
        existing=read_json(scored_path)
        for key in ('prediction','reference','history','ghia_profiles'):checked_entry(existing[key])
        if existing['prediction']['sha256']!=item['projection']['prediction_sha256']:raise ValueError('Scored projection changed')
        return existing
    if sha(item['reference'])!=item['coords']['reference_sha256']:raise ValueError('CFD source changed after coordinate query')
    cfd_accessed=utc()
    accessed=targets_accessed_at_utc or cfd_accessed
    with np.load(item['reference'],allow_pickle=False) as f:
        ref={key:f[key][1:-1,1:-1].copy() for key in ('u','v','psi','omega')}
    ref.update(x=item['coords']['x'],y=item['coords']['y'])
    truth=ghia.reference(re)
    ref.update(ghia_x=truth['x'],ghia_y=truth['y'],ghia_u=truth['u'],ghia_v=truth['v'])
    if sha(item['reference'])!=item['coords']['reference_sha256']:raise ValueError('CFD source changed while scoring')
    refpath=Path(output)/'references'/f're{re}_n512_interior.npz';npz_once(refpath,ref)
    with np.load(item['prediction'],allow_pickle=False) as f:pred={key:f[key].copy() for key in f.files}
    scores=native_scores(pred,ref);scores['ghia'],rows=ghia_scores(re,pred,ghia)
    with (directory/'ghia_signed_profiles.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    physical=item['physics'];outer=physical['outer_uniform'];g=scores['ghia']
    scores['numeric_checks']=dict(outer_momentum_rms_le_1e_2=outer['momentum_rms']<=1e-2,
        divergence_rms_le_1e_10=outer['divergence_rms']<=1e-10,
        both_screened_ghia_relative_l2_le_0_02=all(g[k]['screened_relative_l2']<=.02 for k in ('u','v')),
        interior_velocity_relative_l2_le_0_05=scores['regions']['interior']['velocity_relative_l2']<=.05)
    mesh_path=ROOT/'results/cavity_highre_ghia/mesh_sensitivity.csv'
    if mesh_path.exists():
        with mesh_path.open(encoding='utf-8-sig',newline='') as f:
            mesh=[r for r in csv.DictReader(f) if int(r['Re'])==re and int(r['cells'])==256 and int(r['reference_cells'])==512]
        if len(mesh)==1:
            scores['CFD_grid_sensitivity']=dict(cells=256,reference_cells=512,velocity_relative_l2=float(mesh[0]['velocity_relative_l2']),
                source_sha256=sha(mesh_path),interpretation='retained comparison protocol; sensitivity, not an exact-error bound')
    scores['acceptance']='Not automatically accepted: paired collar sensitivity and physical-vortex review remain required'
    write_json(directory/'scores.json',scores)
    hist=history_csv(item['run']/'history.json',directory);histentry=file_entry(hist.pop('path'));histentry.update(hist)
    record=dict(run_id=item['run_id'],Re=re,seed=config['seed'],collar=config['collar'],
        method='Adam-to-L-BFGS, fixed collocation',outcome='scored',config=config,
        prediction=file_entry(item['prediction']),reference=file_entry(refpath),history=histentry,
        ghia_profiles=file_entry(directory/'ghia_signed_profiles.csv'),
        freeze=dict(prediction_sha256=item['projection']['prediction_sha256'],at_utc=item['projection']['at_utc'],
            model_sha256=item['projection']['identity']['model_sha256']),
        score=dict(prediction_sha256=item['projection']['prediction_sha256'],reference_accessed_at_utc=accessed,cfd_values_accessed_at_utc=cfd_accessed),
        source=item['projection']['identity'],native_subset=item['projection']['subset'],
        reference_source=dict(path=item['reference'].relative_to(ROOT).as_posix(),sha256=item['coords']['reference_sha256']),
        scores=scores,physics_audit=physical,pressure_reference_available=False,physical_review_status='pending',
        history_metric=hist['metric_name'])
    write_json(scored_path,record)
    return record

def collar_comparisons(records):
    groups={}
    for row in records:
        if row['outcome']!='scored' or row.get('collar') not in (.01,.02):continue
        cfg={k:v for k,v in row['config'].items() if k!='collar'}
        identity=json.dumps([cfg,row['source']['source_canonical_sha256']],sort_keys=True)
        group=groups.setdefault(identity,{})
        if row['collar'] in group:raise ValueError('Ambiguous repeated scientific trajectory in collar pair')
        group[row['collar']]=row
    result=[]
    for group in groups.values():
        if set(group)!={.01,.02}:continue
        fine,coarse=group[.01],group[.02]
        with np.load(checked_entry(fine['prediction']),allow_pickle=False) as f:
            a={k:f[k] for k in ('x','y','u','v')}
        with np.load(checked_entry(coarse['prediction']),allow_pickle=False) as f:
            b={k:f[k] for k in ('x','y','u','v')}
        if not np.array_equal(a['x'],b['x']) or not np.array_equal(a['y'],b['y']):raise ValueError('Collar predictions have different native coordinates')
        w=weights(a['x'],a['y'])
        delta=float(np.sqrt(np.sum(w*((a['u']-b['u'])**2+(a['v']-b['v'])**2))/w.sum()))
        amplitude=float(np.sqrt(np.sum(w*(a['u']**2+a['v']**2))/w.sum()))
        relative=delta/amplitude if amplitude>0 else None
        result.append(dict(Re=fine['Re'],seed=fine['seed'],fine_run_id=fine['run_id'],coarse_run_id=coarse['run_id'],
            fine_prediction_sha256=fine['prediction']['sha256'],coarse_prediction_sha256=coarse['prediction']['sha256'],
            absolute_velocity_difference_rms_U=delta,relative_velocity_difference=relative,
            denominator='epsilon=0.01 PINN vector RMS on exactly matched interior nodes',
            numeric_target_le_0_02=relative<=.02 if relative is not None else False,
            interpretation='paired collar sensitivity only; not full-domain convergence or automatic acceptance'))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/cavity_pinn_campaign')
    parser.add_argument('--engine-source',type=Path,default=ROOT/'flowmllab/cavity_pinn_campaign.py')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu',help='Frozen-model evaluation device; fitting provenance is unchanged')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Compact published evidence must stay inside the checkout')
    args.output.mkdir(parents=True,exist_ok=True)
    with writer_lock(args.output/'evaluation.lock'):
        execute(args)

def execute(args):
    engine=module_at(args.engine_source,'frozen_cavity_engine')
    import torch
    torch.set_num_threads(1)
    device=validate_evaluation_device(args.device)
    ghia_source=ROOT/'flowmllab/ghia.py';prepared=[];failures=[]
    # Finish every new projection before importing the module containing targets.
    for run in args.run:
        try:prepared.append(prepare_run(run,args.output,engine,ghia_source,device=device))
        except Exception as exc:
            config=read_json(run/'config.json') if (run/'config.json').exists() else {}
            failures.append(dict(run_id=run_id(config,run.name),Re=config.get('reynolds'),seed=config.get('seed'),
                method='Adam-to-L-BFGS, fixed collocation',outcome='evaluation_rejected',reason=str(exc)))
    targets_accessed=utc()
    ghia=module_at(ghia_source,'post_freeze_ghia_targets')
    records=[]
    for item in prepared:
        if item['pending']:records.append(item['record']);continue
        try:records.append(score_prepared(item,args.output,ghia,targets_accessed))
        except Exception as exc:
            records.append(dict(run_id=item['run_id'],Re=item['config']['reynolds'],seed=item['config']['seed'],
                method='Adam-to-L-BFGS, fixed collocation',outcome='evaluation_rejected',reason=str(exc)))
    records+=failures;path=args.output/'evaluation_manifest.json'
    old=read_json(path)['records'] if path.exists() else []
    merged={r['run_id']:r for r in old}
    for row in records:
        if row['run_id'] in merged and merged[row['run_id']]['outcome']=='scored' and row['outcome']!='scored':
            raise ValueError('A previously scored run no longer verifies; refusing to hide its failure')
        merged[row['run_id']]=row
    ordered=sorted(merged.values(),key=lambda r:(r.get('Re') or 0,r.get('seed') or 0,r['run_id']))
    write_json(path,dict(schema_version=1,campaign_kind='ghia_constant_lid',
        status='scored available frozen runs; campaign and physical acceptance not implied',
        updated_at_utc=utc(),records=ordered,collar_comparisons=collar_comparisons(ordered),expected_reynolds=list(REYNOLDS),
        information_policy='case-specific label-free training; no reference-based endpoint selection',
        projection_policy='all new predictions frozen before target arrays loaded in this process',
        limitations='Boundary rows/columns excluded; CFD is a discrete reference; no CFD pressure truth; collar and physical review separate'))
    print(json.dumps(dict(manifest=str(path),new_outcomes={r['run_id']:r['outcome'] for r in records}),indent=2))
    if any(r['outcome']=='evaluation_rejected' for r in records):raise SystemExit(2)

if __name__=='__main__':main()

