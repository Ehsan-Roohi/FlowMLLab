"""Archive reference sources, executable scripts and retained numerical evidence."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/week16_lowboom/reference'

def package(raw=None):
    paths=[]
    for base in ['cases/week16_lowboom/reference','qa/week16','notebooks/week16']:
        paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.ipynb')
    paths.extend(p for p in R.glob('*') if p.is_file() and p.suffix in ['.json','.npz','.png','.svg'] and p.name!='raw_archive.json')
    for d in [p for pattern in ['seeb_level_*','seeb_resolved_level_*','seeb_resolved_v801_level_*'] for p in R.glob(pattern)]:
        paths.extend(p for p in d.iterdir() if p.name in ['flow.cfg','metadata.json','history.csv','solver.log','signature.npz','cad_meridian.npz','run_evidence.json','physical_audit.json'])
    for d in [p for pattern in ['clean_v801_*','checkpoint_test_*','checkpoint_v801_*','design_v801_*','cone_v801_level_*','cone_benchmark'] for p in (ROOT/'results/week16_lowboom/runs').glob(pattern)]:
        paths.extend(p for p in d.iterdir() if p.name in ['flow.cfg','metadata.json','metrics.json','history.csv','solver.log','extracted.npz','prediction.json','run_evidence.json','planned.json','clean_evidence.json','design_frozen.json','design_evidence.json','benchmark.json','reference_run_evidence.json','reference_metrics.json','reference_extracted.npz'])
    target=R/'reference_evidence.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,6) as z:
        for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(target,target.stat().st_size)
    if raw:
        target=Path(raw).resolve();target.parent.mkdir(parents=True,exist_ok=True)
        rawpaths=set(paths)
        for pattern in ['seeb_level_*','seeb_resolved_level_*','seeb_resolved_v801_level_*','pilot_*']:
            for d in R.glob(pattern):rawpaths.update(p for p in d.rglob('*') if p.is_file())
        for pattern in ['clean_v801_*','checkpoint_test_*','checkpoint_v801_*','design_v801_*','cone_v801_level_*','cone_benchmark']:
            for d in (ROOT/'results/week16_lowboom/runs').glob(pattern):rawpaths.update(p for p in d.rglob('*') if p.is_file())
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,6) as z:
            for p in sorted(rawpaths):z.write(p,p.relative_to(ROOT))
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        info={'filename':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'scope':'Actual retained NASA and teaching-body CFD, cone refinement, design checks and rejected pilots present in this build, plus compact historical audits. Failed pilots remain labeled as failed. This archive is not the unavailable Beihang aircraft dataset.'}
        (R/'raw_archive.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw');a=p.parse_args();package(a.raw)
