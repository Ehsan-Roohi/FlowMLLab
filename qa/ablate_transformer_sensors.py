"""Paired SensorSet cardinality-augmentation ablation; retained-case audit only.

Same initialization, noisy inputs, optimizer, validation objective and maximum
budget for both arms. Selection uses all-sensor validation, not missing sensors.
This tests one training choice within one architecture, not architecture ranking.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import copy
import hashlib
import json
import numpy as np
import torch
from flowmllab import transformer_course as lab
ROOT=Path(__file__).resolve().parents[1]


def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def arrays(cases,p,ids,mean,scale):
    training=np.concatenate([cases[r]['v'][::2] for r in p.train_re])
    fields=(training,cases[p.validation_re]['v'],cases[p.evaluation_re]['v'])
    values=[lab.sensor_tokens(a,ids,cases[p.source_re],mean,scale) for a in fields]
    rng=np.random.default_rng(1901)
    for value in values:value[:,:,0]+=rng.normal(0,p.noise_fraction,value[:,:,0].shape)
    return fields,values


def run(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty ablation output directory')
    out.mkdir(parents=True,exist_ok=True)
    p=lab.Protocol()
    lab.seed_all(17,p.threads)
    plan={'protocol':asdict(p),'arms':['cardinality-augmented','all-sensors-only'],
          'matching':'Same seed and exact initialization, data, POD basis, sensor IDs, noise, optimizer and maximum steps',
          'validation':'All-sensor Re100 coefficient MSE only; validation-selected checkpoint; missing-sensor accuracy not used for selection',
          'augmentation':'At each step retain all sensors if step%3==0; otherwise max(rank, sensor_count//2); random permutation and subset',
          'test_conditions':['all','fixed even-index half of sensors'],
          'scope':'Post-review retained-case descriptive ablation; three paired seeds, not statistical significance or architecture superiority'}
    lab.json_write(out/'plan.json',plan)
    cases,inputs=lab.load_data(ROOT)
    training=np.concatenate([cases[r]['v'][::2] for r in p.train_re])
    rep=lab.Representation.fit(training,p.rank);ids=lab.qr_sensors(rep,p.sensor_count)
    mean,scale=float(training.mean()),float(training.std())
    fields,(x,vx,tx)=arrays(cases,p,ids,mean,scale)
    y,vy=rep.encode(fields[0]),rep.encode(fields[1])
    rows=[];states={};stored={};initializations={}
    for seed in p.seeds:
        pair=[]
        for arm in plan['arms']:
            lab.seed_all(seed,p.threads)
            model=lab.SensorSet(p.rank,p.width)
            initial=state_hash(model.state_dict());pair.append(initial)
            def augment(a,step):
                count=len(ids) if step%3==0 else max(p.rank,len(ids)//2)
                return a[:,torch.randperm(len(ids))[:count]]
            record=lab.fit(model,x,y,lambda m:np.mean((lab.infer(m,vx)-vy)**2),p,
                           augment=augment if arm=='cardinality-augmented' else None)
            key=f'{arm}-{seed}'
            states[key]={'state':copy.deepcopy(model.state_dict()),'architecture':'Sensor-set',
                         'representation':rep.tensors(),'sensor_ids':torch.tensor(ids),
                         'value_mean':mean,'value_scale':scale,'protocol':asdict(p),
                         'initial_state_sha256':initial}
            for condition,keep in [('all',np.arange(len(ids))),('drop-half',np.arange(0,len(ids),2))]:
                predicted=rep.decode(lab.infer(model,tx[:,keep]));fieldkey=f'{key}-{condition}'
                stored.update({fieldkey+'__'+suffix:a for suffix,a in lab.pack_prediction(predicted).items()})
                rows.append({'key':fieldkey,'checkpoint':key,'arm':arm,'seed':seed,'condition':condition,
                             'training':record,'initial_state_sha256':initial,
                             'metrics':lab.error_components(predicted,fields[2],rep)})
            print(seed,arm,'selected',record['selected_step'],'executed',record['executed_steps'],flush=True)
        assert pair[0]==pair[1],'Paired initialization mismatch'
        initializations[str(seed)]=pair[0]
    paired=[]
    for seed in p.seeds:
        for condition in ('all','drop-half'):
            matching={r['arm']:r for r in rows if r['seed']==seed and r['condition']==condition}
            augmented=matching['cardinality-augmented'];plain=matching['all-sensors-only']
            ea=augmented['metrics']['field_relative_l2'];ep=plain['metrics']['field_relative_l2']
            paired.append({'seed':seed,'condition':condition,'augmented_error':ea,'unaugmented_error':ep,
                           'augmentation_relative_gain':1-ea/ep,
                           'augmented_selected_step':augmented['training']['selected_step'],
                           'unaugmented_selected_step':plain['training']['selected_step'],
                           'augmented_at_boundary':augmented['training']['at_budget_boundary'],
                           'unaugmented_at_boundary':plain['training']['at_budget_boundary']})
    lab.json_write(out/'metrics.json',{'rows':rows,'paired_comparisons':paired,
                    'gain_definition':'1 - augmented_error/unaugmented_error; positive favors augmentation',
                    'interpretation':plan['scope'],'validation':plan['validation'],'paired_initializations':initializations})
    np.savez_compressed(out/'predictions.npz',**stored)
    torch.save(states,out/'checkpoints.pt')
    source=['qa/ablate_transformer_sensors.py','flowmllab/transformer_course.py']
    lab.json_write(out/'manifest.json',{'input_manifest':inputs,'environment':lab.runtime_environment(),
                   'source_hashes':{name:lab.canonical_digest(ROOT/name) for name in source},
                   'paired_initializations':initializations,
                   'files':{f.name:lab.digest(f) for f in out.iterdir() if f.is_file()}})
    print('Saved paired ablation:',len(rows),'rows',flush=True)


def verify(out, criterion="strict", report_path=None):
    out=Path(out)
    if report_path is not None and Path(report_path).exists():
        raise FileExistsError('Report must name a new file; previous verification evidence is preserved')
    manifest=json.loads((out/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        assert lab.digest(out/name)==expected,('artifact',name)
    for name,expected in manifest.get('verification_source_hashes',manifest['source_hashes']).items():
        assert lab.canonical_digest(ROOT/name)==expected,('source',name)
    cases,_=lab.load_data(ROOT)
    states=torch.load(out/'checkpoints.pt',map_location='cpu',weights_only=True)
    archive=np.load(out/'predictions.npz',allow_pickle=False)
    record=json.loads((out/'metrics.json').read_text());checks=[]
    plan=json.loads((out/'plan.json').read_text())
    current_environment=lab.runtime_environment()
    initialization_checks=[]
    # Matching is a within-training-run contract, independently attested by
    # the checksummed rows, both saved bundles, and the manifest metadata.
    for seed in plan['protocol']['seeds']:
        rows=[r for r in record['rows'] if r['seed']==seed]
        expected_arms=set(plan['arms'])
        assert expected_arms=={'cardinality-augmented','all-sensors-only'}
        assert {r['arm'] for r in rows}==expected_arms,'Missing paired training arm'
        recorded=record['paired_initializations'][str(seed)]
        assert manifest['paired_initializations'][str(seed)]==recorded,'Manifest initialization differs'
        for arm in expected_arms:
            arm_rows=[r for r in rows if r['arm']==arm]
            assert len(arm_rows)==2 and {r['condition'] for r in arm_rows}=={'all','drop-half'},'Missing or duplicated condition'
            assert len({r['checkpoint'] for r in arm_rows})==1,'Ambiguous arm checkpoint'
            for row in arm_rows:
                assert row['initial_state_sha256']==recorded,'Row initialization differs'
                bundle=states[row['checkpoint']]
                assert bundle['initial_state_sha256']==recorded,'Checkpoint initialization differs'
        bundle=states[rows[0]['checkpoint']]
        p=lab.Protocol(**bundle['protocol']);lab.seed_all(seed,p.threads)
        recreated=state_hash(lab.SensorSet(p.rank,p.width).state_dict())
        initialization_checks.append({'seed':seed,'recorded_initial_state_sha256':recorded,
                                      'current_environment_initial_state_sha256':recreated,
                                      'initialization_recreated_byte_exact':recreated==recorded,
                                      'within_training_run_pair_matched':True})
    for row in record['rows']:
        bundle=states[row['checkpoint']];model,rep=lab.checkpoint_model(bundle)
        p=lab.Protocol(**bundle['protocol']);lab.seed_all(row['seed'],p.threads)
        ids=bundle['sensor_ids'].numpy()
        fields,(_,_,tx)=arrays(cases,p,ids,bundle['value_mean'],bundle['value_scale'])
        keep=np.arange(len(ids)) if row['condition']=='all' else np.arange(0,len(ids),2)
        restored=rep.decode(lab.infer(model,tx[:,keep]));expected=lab.prediction(archive,row['key'])
        comparison=lab.checkpoint_agreement(restored,expected,fields[2])
        metrics=lab.error_components(expected,fields[2],rep)
        delta=abs(metrics['field_relative_l2']-row['metrics']['field_relative_l2'])
        # Always retain both fidelity outcomes; the requested criterion controls exit.
        comparison['score_passed']=bool(delta<=1e-8)
        checks.append({'key':row['key'],'checkpoint':comparison,'score_absolute_drift':delta})
    strict=all(c['checkpoint']['strict_passed'] and c['checkpoint']['score_passed'] for c in checks)
    scientific=all(c['checkpoint']['scientific_passed'] and c['checkpoint']['score_passed'] for c in checks)
    report={'passed':strict,'strict_passed':strict,'scientific_passed':scientific,
            'exit_criterion':criterion,'rows':len(checks),'basis_refitted':False,'environment':current_environment,
            'training_environment':manifest['environment'],
            'same_recorded_environment':current_environment==manifest['environment'],
            'initialization_checks':initialization_checks,
            'initialization_recreation_policy':'Diagnostic only: byte identity of a newly initialized model is not a cross-platform requirement. Matching original paired initialization is verified from recorded bundle, row and manifest metadata.',
            'criteria':{'strict':'Original elementwise allclose rtol1e-5 atol1e-6',
                        'scientific':'Post-review criterion: truth-normalized field difference <=1e-5 and absolute relative-L2 drift <=1e-5',
                        'rescored_error_absolute_drift_max':1e-8},'checks':checks}
    if report_path is not None:lab.json_write(report_path,report)
    print('Strict:', 'PASS' if strict else 'FAIL', 'Scientific:', 'PASS' if scientific else 'FAIL',
          len(checks),'checkpoint reconstructions, stored prediction scores and paired initializations',flush=True)
    print('Current-environment initialization byte-exact:',sum(c['initialization_recreated_byte_exact'] for c in initialization_checks),'/',len(initialization_checks),'(diagnostic, not cross-platform acceptance)',flush=True)
    return 0 if report[criterion+'_passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/transformer_sensor_ablation')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--criterion',choices=('strict','scientific'),default='strict',help='Verification exit criterion; both are always reported')
    parser.add_argument('--report',type=Path,help='Optional new verification report; default verification is read-only')
    args=parser.parse_args()
    lab.seed_all()
    if args.verify:raise SystemExit(verify(args.output,args.criterion,args.report))
    if args.report is not None:parser.error('--report requires --verify')
    run(args.output)
