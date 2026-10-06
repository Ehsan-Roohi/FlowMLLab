"""Reload every retained neural checkpoint without recomputing its POD basis."""
from pathlib import Path
import json
import numpy as np
import torch
from flowmllab import transformer_course as lab

root=Path(__file__).resolve().parents[1]
out=root/'results/transformer_course_v3'
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
            checks.append({'key':key+'-'+condition,'max_absolute_error':float(np.max(abs(actual-expected))),
                           'passed':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-6))})
    else:
        target=110 if key.startswith('forecast__') else int(key.split('__Re')[1].split('-')[0])
        history=rep.encode(cases[target]['omega'][156:160])
        actual=rep.decode(lab.rollout(model,history,121));expected=lab.prediction(archive,key)
        checks.append({'key':key,'max_absolute_error':float(np.max(abs(actual-expected))),
                       'passed':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-6))})
report={'passed':all(c['passed'] for c in checks),'basis_refitted':False,'checks':checks}
lab.json_write(root/'qa/transformer-checkpoint-audit.json',report)
print('PASS' if report['passed'] else 'FAIL',len(checks),'checkpoint predictions')
raise SystemExit(0 if report['passed'] else 1)
