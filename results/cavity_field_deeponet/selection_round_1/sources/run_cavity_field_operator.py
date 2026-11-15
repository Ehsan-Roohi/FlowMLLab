"""Train/freeze first; evaluate held-out CFD only in a separate score command."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab import cavity_field_operator as op
from flowmllab.cavity_highre import audit_field,file_hash
OUT=ROOT/'results/cavity_field_deeponet'
DATA=ROOT/'results/cavity_highre_ghia'


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_field(re,allow):
    if re not in allow:raise ValueError('Re not admitted by this experiment stage')
    path=DATA/'n512'/f're{re}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if file_hash(path)!=meta['field_sha256'] or not meta['converged']:raise ValueError('Unverified CFD field')
    f=dict(np.load(path,allow_pickle=False));audit=audit_field(f,re)
    if max(audit.values())>=1e-6:raise ValueError('CFD acceptance gate failed')
    return f,meta['field_sha256']


def state(f):return np.r_[f['u'].ravel(),f['v'].ravel()]


def train():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'test_scores.csv').exists():raise RuntimeError('Tests are already disclosed; do not refit this experiment')
    gate=json.loads((DATA/'verification.json').read_text(encoding='utf-8'))
    required=set(op.TRAIN+op.VALIDATION+op.TEST)
    if not gate['all_fields_pass'] or {(c['Re'],c['cells']) for c in gate['cases']}!={(re,n) for re in required for n in (128,256,512)}:
        raise ValueError('All 39 three-grid CFD fields must be qualified first')
    fields={};identities={}
    for re in op.TRAIN+op.VALIDATION:fields[re],identities[str(re)]=load_field(re,op.TRAIN+op.VALIDATION)
    x=np.vstack([state(fields[re]) for re in op.TRAIN]);trunk=op.make_trunk(x)
    np.savez_compressed(OUT/'trunk.npz',**trunk)
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in ('flowmllab/cavity_field_operator.py','qa/run_cavity_field_operator.py','docs/CAVITY_FIELD_NEURAL_PROTOCOL.md')}
    signature=hashlib.sha256(json.dumps(dict(sources=source_hashes,data=identities),sort_keys=True).encode()).hexdigest()
    rows=[];models=[];predicted=[];model_hashes={}
    for config in op.CONFIGS:
        for seed in op.SEEDS:
            name=f'{config.name}-seed{seed}'
            path=OUT/'models'/f'{name}.npz';meta_path=path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                info=json.loads(meta_path.read_text(encoding='utf-8'))
                if info['training_signature']!=signature or info['model_sha256']!=file_hash(path):raise ValueError('Model identity changed')
                z=np.load(path,allow_pickle=False)
                model=dict(theta=z['theta'],shapes=z['shapes'].tolist(),scale=z['scale'],**info)
            else:
                model=op.fit(config,seed,trunk['train_coefficients'])
                np.savez_compressed(path,theta=model['theta'],shapes=np.asarray(model['shapes']),scale=model['scale'],
                                    adam_last_theta=model['adam_last_theta'],adam_m=model['adam_m'],adam_v=model['adam_v'])
                info={k:v for k,v in model.items() if k not in ('theta','shapes','scale','adam_last_theta','adam_m','adam_v')}
                info.update(training_signature=signature,model_sha256=file_hash(path),parameters=len(model['theta']))
                write_json(meta_path,info)
            model_hashes[str(path.relative_to(ROOT))]=file_hash(path)
            for re in op.VALIDATION:
                pred=op.reconstruct(trunk,op.predict_coefficients(model,[re]))[0]
                rows.append(dict(method=config.name,seed=seed,Re=re,**op.velocity_error(pred,state(fields[re])),
                                 parameters=len(model['theta']),evaluations=info['evaluations']))
            models.append(dict(name=name,configuration=config.name,seed=seed))
            predicted.append(op.predict_coefficients(model,op.TEST))
            print(json.dumps(dict(model=name,validation_mean_relative_l2=float(np.mean([r['relative_l2'] for r in rows if r['method']==config.name and r['seed']==seed])))),flush=True)
    baseline_rows=[]
    for coordinate in ('linear','log'):
        pred=op.reconstruct(trunk,op.baseline_coefficients(trunk,op.VALIDATION,coordinate))
        for j,re in enumerate(op.VALIDATION):baseline_rows.append(dict(method=f'linear_in_{coordinate}_Re',Re=re,**op.velocity_error(pred[j],state(fields[re]))))
    ranking={cfg.name:float(np.mean([r['relative_l2'] for r in rows if r['method']==cfg.name])) for cfg in op.CONFIGS}
    selected=min(ranking,key=lambda name:(ranking[name],name))
    baseline_ranking={coordinate:float(np.mean([r['relative_l2'] for r in baseline_rows if r['method']==f'linear_in_{coordinate}_Re'])) for coordinate in ('linear','log')}
    selected_baseline=min(baseline_ranking,key=baseline_ranking.get)
    selected_indices=[j for j,m in enumerate(models) if m['configuration']==selected]
    np.savez_compressed(OUT/'frozen_predictions.npz',test_re=np.array(op.TEST),candidate_coefficients=np.array(predicted),
                        selected_indices=np.array(selected_indices),linear_Re_coefficients=op.baseline_coefficients(trunk,op.TEST,'linear'),
                        log_Re_coefficients=op.baseline_coefficients(trunk,op.TEST,'log'))
    write_csv(OUT/'validation_scores.csv',rows);write_csv(OUT/'validation_baselines.csv',baseline_rows)
    selection=dict(training_signature=signature,train_re=op.TRAIN,validation_re=op.VALIDATION,test_re=op.TEST,
                   selected_configuration=selected,validation_ranking=ranking,selected_baseline_coordinate=selected_baseline,
                   baseline_validation_ranking=baseline_ranking,candidate_models=models,
                   selected_indices=selected_indices,model_sha256=model_hashes,trunk_sha256=file_hash(OUT/'trunk.npz'),
                   frozen_predictions_sha256=file_hash(OUT/'frozen_predictions.npz'),data_read_during_training=identities,
                   source_sha256_lf=source_hashes,rank=trunk['rank'],seeds=op.SEEDS,
                   selection_rule='mean full-field velocity relative L2 over both validation Re and three seeds',
                   target_read_during_training=False,prospectively_unknown_physics_claim=False)
    write_json(OUT/'frozen_selection.json',selection)
    print(json.dumps(selection['validation_ranking'],indent=2),flush=True)


def score():
    selection=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    if set(map(int,selection['data_read_during_training']))&set(op.TEST):raise ValueError('Held-out target leakage')
    for name,expected in selection['model_sha256'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Frozen model changed')
    for name,key in [('trunk.npz','trunk_sha256'),('frozen_predictions.npz','frozen_predictions_sha256')]:
        if file_hash(OUT/name)!=selection[key]:raise ValueError('Frozen predictions/trunk changed')
    trunk=dict(np.load(OUT/'trunk.npz',allow_pickle=False));frozen=dict(np.load(OUT/'frozen_predictions.npz',allow_pickle=False))
    rows=[];identities={};projections=[]
    for i,re in enumerate(op.TEST):
        field,identities[str(re)]=load_field(re,op.TEST);truth=state(field)
        for j,model in enumerate(selection['candidate_models']):
            pred=op.reconstruct(trunk,frozen['candidate_coefficients'][j,i])
            row=dict(method=model['configuration'],seed=model['seed'],Re=re,**op.velocity_error(pred,truth))
            size=field['u'].size
            row.update(u_relative_l2=float(np.linalg.norm(pred[:size]-truth[:size])/np.linalg.norm(truth[:size])),
                       v_relative_l2=float(np.linalg.norm(pred[size:]-truth[size:])/np.linalg.norm(truth[size:])))
            rows.append(row)
        coefficients=frozen['candidate_coefficients'][selection['selected_indices'],i].mean(axis=0)
        named={'selected_three_seed_mean':coefficients,'linear_in_Re':frozen['linear_Re_coefficients'][i],
               'linear_in_log_Re':frozen['log_Re_coefficients'][i]}
        for name,coef in named.items():
            pred=op.reconstruct(trunk,coef)
            rows.append(dict(method=name,seed='ensemble' if name.startswith('selected') else 'deterministic',Re=re,
                **op.velocity_error(pred,truth),u_relative_l2=float(np.linalg.norm(pred[:size]-truth[:size])/np.linalg.norm(truth[:size])),
                v_relative_l2=float(np.linalg.norm(pred[size:]-truth[size:])/np.linalg.norm(truth[size:]))))
        projection=(truth-trunk['mean'])@trunk['modes'].T
        projections.append(dict(Re=re,**op.velocity_error(op.reconstruct(trunk,projection),truth)))
    write_csv(OUT/'test_scores.csv',rows);write_csv(OUT/'projection_lower_bound.csv',projections)
    write_json(OUT/'scoring_receipt.json',dict(frozen_selection_sha256=file_hash(OUT/'frozen_selection.json'),
        heldout_field_sha256=identities,test_scores_sha256=file_hash(OUT/'test_scores.csv'),
        test_targets_loaded_only_after_freeze=True,split_scope='held out of this model fitting/selection; benchmark physics already disclosed'))
    print(json.dumps([r for r in rows if r['method'] in ('selected_three_seed_mean','linear_in_Re','linear_in_log_Re')],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('train','score'))
    args=parser.parse_args();train() if args.stage=='train' else score()
