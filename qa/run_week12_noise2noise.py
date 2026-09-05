"""Fit the real-data teaching experiment and retain a fresh evidence bundle."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.noise2noise import (split_indices,fit_experiment,predict_experiment,
    audit_errors,contour_figure,diagnostic_figure,noisy_validation_score)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('tmp/week12_noise2noise_run1'))
    a=p.parse_args();out=a.output
    if out.exists():
        raise FileExistsError('Choose a fresh scratch folder')
    data=ROOT/'data/week12_noise2noise'
    manifest=json.loads((data/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==sha
    with np.load(data/'observations.npz',allow_pickle=False) as source:
        seeds=source['seeds'];observations=source['raw3']
    split=split_indices(seeds,manifest['split'])
    plan={'split':manifest['split'],'architecture':[64,32],'patch_width':5,
          'max_iter':160,'samples_per_seed':3000,'initialization':12,
          'width_candidates':[.5,1,1.5,2],
          'reference_used_for_training_or_selection':False,
          'test_status':'Previously inspected archive, held out from this teaching fit; not new blind evidence.',
          'data_files':manifest['files']}
    out.mkdir(parents=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),newline='\n')
    start=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted=fit_experiment(observations[split['train']],observations[split['validation']])
    elapsed=time.monotonic()-start
    report={**plan,'fit_seconds':elapsed,'iterations':int(fitted['mlp'].model[-1].n_iter_),
            'last_training_loss':float(fitted['mlp'].model[-1].loss_),
            'warnings':[str(w.message) for w in caught],
            'gaussian_width':fitted['gaussian_width'],
            'gaussian_validation_scores':fitted['gaussian_validation']}
    val=np.stack([fitted['mlp'].predict(o) for o in observations[split['validation']]])
    report['mlp_noisy_validation_score']=noisy_validation_score(val,observations[split['validation']],fitted['mlp'].scale)
    # Fitting and width selection have ended before opening evaluation arrays.
    with np.load(data/'evaluation_only.npz',allow_pickle=False) as source:
        reference=source['reference'];raw10=source['raw10']
    rows=[];saved={};first=None
    for i in split['test']:
        predictions=predict_experiment(fitted,observations[i])
        predictions['Raw(10), larger budget']=raw10[i]
        if first is None:
            first=predictions
        for method,estimate in predictions.items():
            for field,metrics in zip(['qx','qy'],audit_errors(estimate,reference,observations[i])):
                rows.append({'seed':int(seeds[i]),'field':field,'method':method,**metrics})
            saved[str(seeds[i])+'_'+method]=estimate
        for field in ['qx','qy']:
            fig=contour_figure(observations[i],predictions['Noise2Noise MLP'],reference,component=field,seed=int(seeds[i]))
            fig.savefig(out/f'{field}_seed{seeds[i]}.png',dpi=240);plt.close(fig)
    fig=diagnostic_figure(first,reference,rows)
    fig.savefig(out/'profiles_and_errors.png',dpi=240);plt.close(fig)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'test_predictions.npz',**saved)
    report['mean_reference_nrmse']={field:{m:float(np.mean([r['reference_nrmse'] for r in rows if r['field']==field and r['method']==m]))
                                        for m in first} for field in ['qx','qy']}
    report['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()}
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
