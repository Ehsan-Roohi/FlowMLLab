"""Archive reference sources, executable scripts and retained numerical evidence."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/week16_lowboom/reference'

def package(raw=None):
    paths=[]
    for base in ['cases/week16_lowboom/reference','qa/week16','notebooks/week16']:
        paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.ipynb')
    paths.extend(p for p in R.glob('*') if p.is_file() and p.suffix in ['.json','.npz','.png'] and p.name!='raw_archive.json')
    for d in R.glob('seeb_level_*'):
        paths.extend(p for p in d.iterdir() if p.name in ['flow.cfg','metadata.json','history.csv','solver.log','signature.npz','cad_meridian.npz'])
    for d in (ROOT/'results/week16_lowboom/runs').glob('checkpoint_test_*'):
        paths.extend(p for p in d.iterdir() if p.name in ['flow.cfg','metadata.json','metrics.json','history.csv','solver.log','extracted.npz','prediction.json','run_evidence.json'])
    target=R/'reference_evidence.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,6) as z:
        for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(target,target.stat().st_size)
    if raw:
        target=Path(raw).resolve();target.parent.mkdir(parents=True,exist_ok=True)
        rawpaths=set(paths)
        for pattern in ['seeb_level_*','pilot_*']:
            for d in R.glob(pattern):rawpaths.update(p for p in d.rglob('*') if p.is_file())
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,6) as z:
            for p in sorted(rawpaths):z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        info={'filename':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'scope':'Reconstructed NASA CFD runs and interrupted numerical pilots plus compact recovered neural audit. Does not contain unavailable original finer-mesh neural audit raw fields/logs.'}
        (R/'raw_archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw');a=p.parse_args();package(a.raw)
