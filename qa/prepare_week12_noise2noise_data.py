"""Extract raw observations only from the author-supplied JCP2 archive.

No archived neural predictions enter the executable Noise2Noise experiment.
The finite-budget reference is saved separately from training observations.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/week12_noise2noise'))
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve existing evidence; choose a fresh output')
    with zipfile.ZipFile(a.archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP checksum failure')
        raw=z.read('predictions.npz'); lock=json.loads(z.read('prediction_lock.json'))
        if hashlib.sha256(raw).hexdigest()!=lock['prediction_sha256']:
            raise ValueError('Prediction lock mismatch')
        pred=np.load(io.BytesIO(raw),allow_pickle=False)
        stats=np.load(io.BytesIO(z.read('reference_stats.npz')),allow_pickle=False)
        source=json.loads(z.read('summary.json'))
        fields=['qx','qy']; indices=[list(pred['field_names']).index(f) for f in fields]
        seeds=pred['seeds'].astype(np.int64)
        assert seeds.tolist()==source['selected_evaluation_seeds']
        assert len(seeds)==8 and len(set(seeds.tolist()))==8
        assert set(seeds).isdisjoint(source['selected_reference_seeds'])
        obs=pred['method_raw_b3'][:,indices].copy()
        raw10=pred['method_raw_b10'][:,indices].copy()
        reference=stats['reference'][indices].copy()
        assert obs.shape==(8,2,100,100) and np.isfinite(obs).all()
        assert np.isfinite(raw10).all() and np.isfinite(reference).all()
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'observations.npz',seeds=seeds,fields=fields,raw3=obs)
    np.savez_compressed(a.output/'evaluation_only.npz',seeds=seeds,fields=fields,
                        raw10=raw10,reference=reference)
    manifest={'source_archive':a.archive.name,
      'source_archive_sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),
      'source_prediction_sha256':lock['prediction_sha256'],
      'source_summary':source,
      'condition':'S2_kn0p085_u350','fields':fields,
      'split':{'train':seeds[:4].tolist(),'validation':seeds[4:6].tolist(),'test':seeds[6:].tolist()},
      'split_status':'Declared teaching split of previously inspected archive; not a new blind research test.',
      'independence':'Distinct seeds and observation/reference disjointness are documented by the source archive; RNG streams and block ancestry were not re-audited from solver logs.',
      'scope':'Same-condition denoising only; no new DSMC, no research-checkpoint use. Two held-out noise realizations do not establish new-condition generalization.',
      'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.glob('*.npz')}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(manifest['split']))


if __name__=='__main__':
    main()
