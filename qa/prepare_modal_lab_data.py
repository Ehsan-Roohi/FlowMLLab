"""Fetch checksummed author CFD and make an unfiltered wake-ROI teaching subset."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request
import numpy as np

BASE='https://github.com/Ehsan-Roohi/FlowMLLab/releases/download/cylinder-cfd-v1/'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=Path('tmp/modal_source'))
    p.add_argument('--output',type=Path,default=Path('data/modal_labs'))
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    a.cache.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(urllib.request.urlopen(BASE+'manifest.json',timeout=60).read())
    wanted=[r for r in manifest['cases'] if int(r['reynolds']) in (90,100,105,110)]
    assert len(wanted)==4
    def fetch(row):
        path=a.cache/row['file']
        if path.exists():
            if sha(path)!=row['sha256']:
                raise ValueError('Existing cache checksum mismatch: '+str(path))
        else:
            with urllib.request.urlopen(BASE+row['file'],timeout=120) as response, path.open('xb') as out:
                while chunk:=response.read(1024*1024):
                    out.write(chunk)
            if sha(path)!=row['sha256']:
                raise ValueError('Downloaded checksum mismatch')
        print('VERIFIED',row['file'],flush=True)
        return row,path
    with ThreadPoolExecutor(max_workers=4) as executor:
        files=list(executor.map(fetch,wanted))
    a.output.mkdir(parents=True)
    records=[]
    for row,path in files:
        with np.load(path,allow_pickle=False) as d:
            u=d['u'].astype(float);v=d['v'].astype(float)
            assert u.shape==v.shape==(281,96,240)
            # D/U-normalized vorticity from native-grid velocity derivatives.
            omega=12*(np.gradient(v,axis=2)-np.gradient(u,axis=1))
            roi=(slice(None),slice(16,80,2),slice(72,228,2))
            assert not d['solid'][15:81,71:229].any()
            t=d['snapshot_time'].astype(float)*.05/12
            x=(np.arange(72,228,2)-60)/12
            y=(np.arange(16,80,2)-47.5)/12
            name=f"re{int(row['reynolds']):03d}.npz"
            np.savez_compressed(a.output/name,omega=omega[roi].astype(np.float32),
                u=u[roi].astype(np.float32),v=v[roi].astype(np.float32),
                x=x,y=y,t=t,reynolds=int(row['reynolds']),
                original_strouhal=float(d['strouhal']))
            records.append({'file':name,'sha256':sha(a.output/name),
                'source_file':row['file'],'source_sha256':row['sha256'],
                'source_role':row['split'],'reynolds':int(row['reynolds'])})
    protocol={'source_release':BASE,'cases':records,
      'transform':'Native central differences for omega*D/U, then every second point of a fluid-only rectangular wake ROI. No filtering or interpolation.',
      'spatial_scope':'Wake ROI, x/D=1..13.833, y/D=-2.625..2.542, D=12 native lattice cells; not full-domain/wall CFD.',
      'forecast_protocol':{'case':110,'train':[0,160],'validation':[160,210],'test':[210,281],
        'rollout':'Initialize only at frame159; advance autonomously through validation AND test without resetting.'},
      'sensor_protocol':{'train':[90,110],'validation':[100],'test':[105],
        'rank':8,'budgets':[8,16,32],'noise_rms_fraction':.01,'random_seeds':[10,11,12,13,14]},
      'status':'New teaching experiments on previously inspected CFD; no new blind research claim. Source remains coarse educational LBM, not grid-independent CFD.'}
    (a.output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n',newline='\n')
    print('SUBSET_COMPLETE',flush=True)


if __name__=='__main__':
    main()
