"""Check the imposed volume on actual exported surface nodes, not analytic shapes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from cfd import VOLUME
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'results/week16_lowboom';R=E/'reference'
def report(write=True):
 names=['clean_v801_'+r['name'] for r in json.loads((E/'design_plan.json').read_text())['rows']]
 from weakwall_designs import specifications
 names += [s['name'] for s in specifications()]
 rows=[]
 for name in names:
  path=E/'runs'/name/'extracted.npz'
  with np.load(path) as d:x=d['surface_x'];r=d['surface_r']
  assert np.all(np.diff(x)>0) and np.isfinite(r).all() and (r>=0).all()
  assert abs(x[0])<1e-12 and abs(x[-1]-1)<1e-12 and abs(r[0])<1e-12 and abs(r[-1])<1e-12
  volume=float(np.pi/3*np.sum(np.diff(x)*(r[:-1]**2+r[:-1]*r[1:]+r[1:]**2)))
  error=abs(volume/VOLUME-1)
  rows.append({'name':name,'frustum_volume_m3':volume,'relative_error':error,'passed':error<.002,'extracted_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 result={'target_volume_m3':VOLUME,'threshold_relative_error':.002,'method':'Sum exact frustum volumes between actual exported wall nodes; all 44 clean labels and ten retained-design runs.','runs':rows,'passed':all(r['passed'] for r in rows),'maximum_relative_error':max(r['relative_error'] for r in rows)}
 path=R/'geometry_volume_audit.json'
 if write:path.write_text(json.dumps(result,indent=2)+'\n')
 else:assert json.loads(path.read_text())==result
 assert result['passed'];return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');a=p.parse_args();r=report(not a.check_only);print(json.dumps({'passed':r['passed'],'cases':len(r['runs']),'maximum_relative_error':r['maximum_relative_error']},indent=2))
