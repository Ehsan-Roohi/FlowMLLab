"""Recompute surrogate proposals with SU2; no ML-only acceptance."""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from cfd import ROOT,run
from analyze import analyze
E=ROOT/'results/week16_lowboom'

def evaluate(row):
    name,a,b,level,mach=row
    p=E/'runs'/name
    if not (p/'metrics.json').exists():
        run(name,a,b,level=level,mach=mach,iterations=2200);analyze(p)
    return json.loads((p/'metrics.json').read_text())

if __name__=='__main__':
    d=json.loads((E/'design_candidates.json').read_text());rows=[]
    for c in d['candidates']:
        rows.append((c['name']+'_fine',c['a'],c['b'],1.5,1.8))
    c=d['candidates'][0];rows.append(('optimized_finer',c['a'],c['b'],2.,1.8))
    for m in [1.7,1.9]:
        rows.extend([(f'baseline_M{m}',0,0,1.5,m),(f'optimized_M{m}',c['a'],c['b'],1.5,m)])
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(evaluate,r) for r in rows]):
            r=f.result();print(r['name'],r['converged'],r['peak_cp'],r['cd_pressure'],flush=True)
