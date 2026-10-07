"""Re-score saved main predictions and re-infer additional reset diagnostics.

The auxiliary observed-initialization trajectories are freshly inferred from
fixed checkpoints in the recorded current environment. No model is retrained.

Historical training source hashes and pre-rescore file hashes remain in the
manifest. The exact checkpoint POD basis is reused, never refitted.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from flowmllab import transformer_course as lab

ROOT = Path(__file__).resolve().parents[1]

def rescore(out):
    out = Path(out)
    manifest = json.loads((out/'manifest.json').read_text())
    original_files = dict(manifest['files'])
    for name in ('predictions.npz', 'checkpoints.pt'):
        if lab.digest(out/name) != original_files[name]:
            raise ValueError(f'Cannot rescore changed evidence: {name}')
    cases, _ = lab.load_data(ROOT)
    bundles = torch.load(out/'checkpoints.pt', map_location='cpu', weights_only=True)
    archive = np.load(out/'predictions.npz', allow_pickle=False)
    prefixed = any(k.startswith('forecast__') for k in bundles)
    experiments = ('sensors', 'forecast', 'transfer') if prefixed else ('forecast',)
    count = 0
    reset_inferences = []
    for experiment in experiments:
        record = json.loads((out/f'{experiment}.json').read_text())
        prefix = experiment+'__' if prefixed else ''
        candidates = {k: b for k,b in bundles.items() if k.startswith(prefix)}
        default = next(iter(candidates.values()))
        for row in record['rows']:
            key = prefix+row['key']
            lookup = key.rsplit('-',1)[0] if experiment=='sensors' and key.endswith('-all') else key
            if experiment=='sensors' and key.endswith('-drop-half'):
                lookup = key[:-len('-drop-half')]
            bundle = candidates.get(lookup, default)
            if experiment=='transfer' and row['arm']=='DMD':
                bundle = bundles[prefix+row['key'].replace('-DMD-', '-scratch-')]
            rep = lab.Representation.from_tensors(bundle['representation'])
            p = lab.Protocol(**bundle['protocol'])
            prediction = lab.prediction(archive, key)
            truth = cases[p.evaluation_re]['v'] if experiment=='sensors' else cases[row.get('target_re',110)]['omega'][p.validation_end:p.test_end]
            scored = prediction if experiment=='sensors' else prediction[p.validation_end-p.train_end:]
            row['metrics'] = lab.error_components(scored,truth,rep)
            if 'additional_observed_initializations' in row:
                model,_ = lab.checkpoint_model(bundle)
                field = cases[110]['omega']
                z = rep.encode(field)
                for start in row['additional_observed_initializations']:
                    i = int(start)
                    pred = rep.decode(lab.rollout(model,z[i-p.context:i],p.test_end-i))
                    row['additional_observed_initializations'][start] = lab.error_components(pred,field[i:p.test_end],rep)
                    reset_inferences.append({'checkpoint':key,'start_frame':i,'steps':p.test_end-i})
            count += 1
        if experiment=='forecast':
            probe = record['probe_index']
            field = cases[110]['omega'][p.train_end:p.test_end].reshape(p.test_end-p.train_end,-1)
            t = cases[110]['t'][p.train_end:p.test_end]
            spectra = {'reference':lab.spectral_fit(t,field[:,probe])}
            reference = spectra['reference']
            for row in record['rows']:
                key = row['key']
                fitted = lab.spectral_fit(t,lab.prediction(archive,prefix+key)[:,probe])
                if fitted['frequency'] is None or reference['frequency'] is None:
                    fitted.update(frequency_relative_error=None,initial_phase_error=None,phase_drift_over_horizon=None)
                else:
                    delta = fitted['phase']-reference['phase']
                    fitted.update(frequency_relative_error=abs(fitted['frequency']/reference['frequency']-1),
                                  initial_phase_error=float(np.arctan2(np.sin(delta),np.cos(delta))),
                                  phase_drift_over_horizon=float(2*np.pi*(fitted['frequency']-reference['frequency'])*(t[-1]-t[0])))
                spectra[key] = fitted
            record['spectral_fit'] = spectra
            record['phase_reference'] = 'First sample of rollout; sinusoid uses t - t[0].'
        lab.json_write(out/f'{experiment}.json',record)
    if (out/'summary.json').exists():
        summary = json.loads((out/'summary.json').read_text())
        current = {r['key']:r for r in json.loads((out/'forecast.json').read_text())['rows']}
        base = {r['key']:r for r in json.loads((ROOT/'results/transformer_course_v3/forecast.json').read_text())['rows']}
        for row in summary['rows']:
            row['base_field_relative_l2'] = base[row['key']]['metrics']['field_relative_l2']
            row['audit_field_relative_l2'] = current[row['key']]['metrics']['field_relative_l2']
        lab.json_write(out/'summary.json',summary)
    manifest.setdefault('training_source_hashes',dict(manifest['source_hashes']))
    manifest.setdefault('pre_rescoring_files',original_files)
    manifest['source_hashes'] = {name:lab.canonical_digest(ROOT/name) for name in manifest['source_hashes']}
    manifest['rescoring'] = {'script':'qa/rescore_transformer_evidence.py',
                            'script_sha256':lab.canonical_digest(Path(__file__)),
                            'prediction_sha256':lab.digest(out/'predictions.npz'),
                            'checkpoint_sha256':lab.digest(out/'checkpoints.pt'),
                            'basis_refitted':False,'training_repeated':False,'rows':count,
                            'environment':lab.runtime_environment(),
                            'main_prediction_source':'Unchanged stored predictions.npz',
                            'additional_observed_initializations_source':'Fresh fixed-checkpoint inference in this rescoring environment',
                            'fresh_reset_inferences':reset_inferences,
                            'changes':['float64 metric reductions','phase at first rollout sample','constant probe identification','single-sinusoid R squared; constant signals undefined']}
    manifest['files'] = {name:lab.digest(out/name) for name in manifest['files']}
    lab.json_write(out/'manifest.json',manifest)
    print(f'Rescored {count} saved prediction rows in {out.name}; no training or POD refit.')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    lab.seed_all()
    rescore(args.output)
