"""Independently reload checkpoints and reproduce every saved test prediction."""
import json
from pathlib import Path
import numpy as np
import torch
from train_family_transolver import ROOT, OUT, Model, load_data, make_features
from train_week15_vanilla_deeponet import sha256, case_metrics

def main():
    torch.set_num_threads(1)
    d,gids,groups,_=load_data();features=make_features(d)
    records=[]
    for seed in (17,29,43):
        dest=OUT/f'seed_{seed}'
        manifest=json.loads((dest/'manifest.json').read_text())
        assert sha256(dest/'best.pt')==manifest['checkpoint_sha256']
        for p,h in manifest['source_sha256'].items():assert sha256(ROOT/p)==h,p
        with np.load(dest/'scaling.npz') as s:
            x=torch.from_numpy((features-s['input_mean'])/s['input_std'])
            ym,ys=s['target_mean'].copy(),s['target_std'].copy()
        model=Model(**manifest['model_config'])
        model.load_state_dict(torch.load(dest/'best.pt',weights_only=True));model.eval()
        for ci in groups['test']:
            ids=np.flatnonzero(d['masks'][ci]);pred=np.zeros_like(d['raw'][ci])
            with torch.no_grad():pred[ids]=model(x[ci,ids,:2][None],x[ci,ids,2:][None])[0].numpy()*ys+ym
            case=f'g{gids[ci]:03d}_Re{int(d["Re"][ci])}_medium'
            path=dest/f'{case}_prediction.npz'
            with np.load(path) as z:np.testing.assert_array_equal(pred,z['prediction'])
            records.append(dict(seed=seed,case=case,bitwise_reproduced=True,prediction_sha256=sha256(path),
                                **case_metrics(d['raw'][ci],pred,d['masks'][ci])))
    result={'checks':'source/checkpoint hashes, frozen split, exact 57-field inference reproduction',
            'records':records,'verifier_sha256':sha256(Path(__file__))}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: all 57 predictions reproduced bit-for-bit from saved checkpoints')

if __name__=='__main__':main()
