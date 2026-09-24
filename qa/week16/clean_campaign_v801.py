"""Regenerate every frozen teaching geometry with physically checked SU2 8.0.1.

The design split and original data remain unchanged. New data use unique names.
"""
from pathlib import Path
import argparse, json, os, hashlib
import numpy as np
import cfd
from analyze import analyze
from design_physics import verify
from install_su2_801 import BINARY_SHA256, ASSET_SHA256
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'results/week16_lowboom'; R=E/'reference'
SOURCES=['clean_campaign_v801.py','cfd.py','analyze.py','design_physics.py','seeb_reference.py','install_su2_801.py']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity():
 return {'design_plan_sha256':sha(E/'design_plan.json'),'solver_binary_sha256':BINARY_SHA256,'source_sha256':{n:sha(ROOT/'qa/week16'/n) for n in SOURCES}}
def rows():return json.loads((E/'design_plan.json').read_text())['rows']
def run(index):
 row=rows()[index]; name='clean_v801_'+row['name']; folder=E/'runs'/name
 assert not folder.exists(),f'Refusing to overwrite {folder}'
 exe=Path(os.environ.get('SU2_CFD',str(ROOT/'.tools/week16_su2_801/bin/SU2_CFD'))).resolve()
 assert sha(exe)==BINARY_SHA256
 os.environ.update(SU2_CFD=str(exe),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 frozen=identity();folder.mkdir(parents=True)
 (folder/'planned.json').write_text(json.dumps({'row':row,'identity':frozen},indent=2)+'\n')
 try:
  cfd.run(name,row['a'],row['b'],level=1.5,mach=1.8,iterations=4000)
  analyze(folder);checks=verify(folder)
  report={'row':row,'identity':identity(),'solver_version':'8.0.1','asset_sha256':ASSET_SHA256,'checks':checks,'raw_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
  assert frozen==report['identity'];(folder/'clean_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
  assert checks['passed'],f'Physical or numerical failure: {name}'
  print(name,'PASS',flush=True)
 except Exception as e:
  (folder/'failure.json').write_text(json.dumps({'error':repr(e),'identity':frozen},indent=2)+'\n');raise

def assemble(write=True):
 wave=[];cd=[];params=[];names=[];splits=[];evidence=[]
 for row in rows():
  folder=E/'runs'/('clean_v801_'+row['name']);d=json.loads((folder/'clean_evidence.json').read_text())
  assert d['row']==row and d['identity']==identity()
  for n,s in d['raw_sha256'].items():assert sha(folder/n)==s,f'Changed {folder/n}'
  checks=verify(folder);assert checks==d['checks'] and checks['passed']
  meta=json.loads((folder/'metadata.json').read_text());m=json.loads((folder/'metrics.json').read_text())
  assert meta['a']==row['a'] and meta['b']==row['b'] and meta['level']==1.5 and meta['mach']==1.8
  for n in ['mesh.su2','flow.cfg']:assert sha(folder/n)==meta[n+'_sha256']
  with np.load(folder/'extracted.npz') as a:
   assert a['radii'][1]==.5
   if wave:np.testing.assert_array_equal(a['x'],x)
   x=a['x'].copy();wave.append(a['cp'][1].copy())
  assert np.isclose(wave[-1].max(),m['peak_cp'])
  cd.append(m['cd_pressure']);params.append([row['a'],row['b']]);names.append(row['name']);splits.append(row['split'])
  evidence.append({'name':row['name'],'checks':checks,'evidence_sha256':sha(folder/'clean_evidence.json')})
 arrays=dict(parameters=params,waveforms=wave,cd=cd,splits=splits,names=names,x=x)
 target=R/'clean_dataset_v801.npz'
 if write:np.savez_compressed(target,**arrays)
 else:
  with np.load(target) as a:
   for k,v in arrays.items():np.testing.assert_array_equal(a[k],v)
 counts={s:splits.count(s) for s in set(splits)};assert counts=={'train':24,'validation':6,'test':8,'extrapolation':6}
 report={'solver_version':'8.0.1','identity':identity(),'cases':len(rows()),'split_counts':counts,'mesh_level':1.5,'dataset_sha256':sha(target),'runs':evidence,'passed':True,'scope':'All 44 original geometries recomputed with full-field physical and numerical gates. Geometries are previously known; splits are preserved, not a new blind study. Historical SU2 8.5.0 data are not overwritten.'}
 if write:(R/'clean_campaign_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--batch',type=int,choices=range(11));g.add_argument('--assemble',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.assemble: print(json.dumps(assemble(not a.check_only),indent=2))
 else:
  for i in range(a.batch*4,a.batch*4+4):run(i)
