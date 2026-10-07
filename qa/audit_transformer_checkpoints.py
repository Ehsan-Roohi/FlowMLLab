"""Reload every retained neural checkpoint without recomputing its POD basis."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from flowmllab import transformer_course as lab

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'results/transformer_course_v3',help='Experiment directory')
parser.add_argument('--report',type=Path,help='Optional new report; existing files are never overwritten')
parser.add_argument('--criterion',choices=('strict','scientific'),default='strict',help='Exit criterion; both are always reported')
args=parser.parse_args()
if args.report is not None and args.report.exists():
    parser.error('--report must name a new file to preserve previous evidence')
out=args.output
cases,_=lab.load_data(root)
lab.seed_all()
archive=np.load(out/'predictions.npz',allow_pickle=False)
bundles=torch.load(out/'checkpoints.pt',map_location='cpu',weights_only=True)
checks=[]
for key,bundle in bundles.items():
    model,rep=lab.checkpoint_model(bundle)
    p=lab.Protocol(**bundle['protocol'])
    if key.startswith('sensors__'):
        ids=bundle['sensor_ids'].numpy()
        arrays=[lab.sensor_tokens(a,ids,cases[p.source_re],bundle['value_mean'],bundle['value_scale']) for a in
                (np.concatenate([cases[r]['v'][::2] for r in p.train_re]),cases[p.validation_re]['v'],cases[p.evaluation_re]['v'])]
        rng=np.random.default_rng(1901)
        for a in arrays:a[:,:,0]+=rng.normal(0,p.noise_fraction,a[:,:,0].shape)
        for condition,keep in [('all',np.arange(len(ids))),('drop-half',np.arange(0,len(ids),2))]:
            if bundle['architecture']=='Sensor-set':inputs=arrays[2][:,keep]
            else:
                inputs=arrays[2][:,:,0].copy();inputs[:,np.setdiff1d(np.arange(len(ids)),keep)]=0
            actual=rep.decode(lab.infer(model,inputs));expected=lab.prediction(archive,key+'-'+condition)
            checks.append({'key':key+'-'+condition,**lab.checkpoint_agreement(actual,expected,cases[p.evaluation_re]['v'])})
    else:
        target=110 if key.startswith('forecast__') else int(key.split('__Re')[1].split('-')[0])
        history=rep.encode(cases[target]['omega'][p.train_end-p.context:p.train_end])
        actual=rep.decode(lab.rollout(model,history,p.test_end-p.train_end));expected=lab.prediction(archive,key)
        checks.append({'key':key,**lab.checkpoint_agreement(actual,expected,cases[target]['omega'][p.train_end:p.test_end])})
report={'passed':all(c['strict_passed'] for c in checks),
        'strict_passed':all(c['strict_passed'] for c in checks),
        'scientific_passed':all(c['scientific_passed'] for c in checks),
        'strict_failure_count':sum(not c['strict_passed'] for c in checks),
        'scientific_failure_count':sum(not c['scientific_passed'] for c in checks),
        'exit_criterion':args.criterion,'basis_refitted':False,'environment':lab.runtime_environment(),
        'criteria':{'strict':{'rtol':1e-5,'atol':1e-6,'status':'original unchanged elementwise criterion'},
                    'scientific':{'truth_normalized_prediction_difference_max':1e-5,
                                  'field_relative_l2_absolute_drift_max':1e-5,
                                  'status':'Introduced after external review; not retrospectively preregistered',
                                  'rationale':'Limits field drift to 0.001% of reference norm and error drift to 0.001 percentage point; does not erase strict failures or establish untested-platform results.'}},
        'checks':checks}
if args.report is not None:
    lab.json_write(args.report,report)
print('Strict:', 'PASS' if report['strict_passed'] else 'FAIL', 'Scientific:', 'PASS' if report['scientific_passed'] else 'FAIL', len(checks),'checkpoint predictions')
raise SystemExit(0 if report[args.criterion+'_passed'] else 1)
