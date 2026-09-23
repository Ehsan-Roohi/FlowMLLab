"""Frozen geometry split and finite CFD campaign. Resume only completed cases."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
from cfd import run,ROOT
from analyze import analyze

def job(row):
    p=ROOT/'results/week16_lowboom/runs'/row['name']
    if (p/'metrics.json').exists():
        old=json.loads((p/'metrics.json').read_text())
        if old.get('converged'): return old
    run(row['name'],row['a'],row['b'],level=1.5,iterations=2200)
    return analyze(p)

def main():
    root=ROOT/'results/week16_lowboom';plan=root/'design_plan.json'
    if not plan.exists():
        rows=[]
        for split,n,seed in [('train',24,1601),('validation',6,1602),('test',8,1603),('extrapolation',6,1604)]:
            samples=qmc.LatinHypercube(2,seed=seed).random(n)
            samples=qmc.scale(samples,[-.45,-.25],[.45,.25])
            if split=='extrapolation': samples[:,1]=np.linspace(.30,.40,n)
            for i,(a,b) in enumerate(samples): rows.append(dict(name=f'{split}_{i:03}',a=float(a),b=float(b),split=split))
        plan.write_text(json.dumps({'seed_policy':'Frozen before CFD or ML; no sample-level leakage','rows':rows},indent=2))
    rows=json.loads(plan.read_text())['rows']
    errors=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs={pool.submit(job,row):row for row in rows}
        for f in as_completed(fs):
            row=fs[f]
            try:
                r=f.result();print(row['name'], 'PASS' if r['converged'] else 'NOT_CONVERGED',r['peak_cp'],flush=True)
            except Exception as ex:errors.append({'row':row,'error':str(ex)});print('FAILED',row['name'],str(ex)[-200:],flush=True)
    (root/'campaign_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Campaign has failures')
if __name__=='__main__':main()
