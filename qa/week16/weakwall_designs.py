"""Recompute ten fixed historical design checks using official SU2 8.0.1.

No optimization or training takes place; the retained candidate file is frozen.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os
import numpy as np
import cfd
from analyze import analyze
from freeze_model import FrozenSurrogate
from install_su2_801 import BINARY_SHA256,ASSET_SHA256,URL
from design_physics import verify
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'results/week16_lowboom'
SOURCES=['weakwall_designs.py','design_physics.py','install_su2_801.py','cfd.py','analyze.py','freeze_model.py','independent_audit_report.py','seeb_reference.py']
RAW=['mesh.su2','flow.cfg','metadata.json','history.csv','solver.log','restart_flow.csv','surface_flow.csv','metrics.json','extracted.npz','design_frozen.json']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity():
 return {'design_candidates_sha256':sha(E/'design_candidates.json'),'checkpoint_sha256':sha(E/'reference/model_checkpoint.npz'),'source_sha256':{n:sha(ROOT/'qa/week16'/n) for n in SOURCES},'solver_binary_sha256':BINARY_SHA256}
def specifications():
 definitions=json.loads((E/'design_candidates.json').read_text());candidates={r['name']:r for r in definitions['candidates']};candidates['baseline']={'name':'baseline','a':0.,'b':0.}
 matrix=[('baseline',1.5,1.8),('optimized',1.5,1.8),('baseline',2,1.8),('optimized',2,1.8),('alternative_1',1.5,1.8),('alternative_2',1.5,1.8),('baseline',1.5,1.7),('optimized',1.5,1.7),('baseline',1.5,1.9),('optimized',1.5,1.9)]
 return [dict(index=i,name=f'design_v801_{i:03d}_{name}',design=name,level=level,mach=mach,a=float(candidates[name]['a']),b=float(candidates[name]['b']),historical_candidate=candidates[name]) for i,(name,level,mach) in enumerate(matrix)]
def run(index):
 spec=specifications()[index];exe=Path(os.environ.get('SU2_CFD',str(ROOT/'.tools/week16_su2_801/bin/SU2_CFD'))).resolve();assert sha(exe)==BINARY_SHA256
 os.environ.update(SU2_CFD=str(exe),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 folder=E/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=False)
 w,cd=FrozenSurrogate().predict([spec['a'],spec['b']])
 frozen={'specification':spec,'identity':identity(),'frozen_before_solver_utc':datetime.now(timezone.utc).isoformat(),'retained_checkpoint_prediction_at_training_mach_1p8':{'waveform':w[0].tolist(),'cd':float(cd[0])},'prediction_scope':'This separate retained checkpoint did not generate the historical optimizer proposal; its prediction applies at Mach 1.8 only. The candidate file retains historical optimizer predictions.'}
 (folder/'design_frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
 cfd.run(spec['name'],a=spec['a'],b=spec['b'],level=spec['level'],mach=spec['mach'],iterations=4000)
 analyze(folder);checks=verify(folder)
 evidence={'identity':identity(),'specification':spec,'solver_version':'8.0.1','official_asset_url':URL,'official_asset_sha256':ASSET_SHA256,'checks':checks,'raw_sha256':{n:sha(folder/n) for n in RAW}}
 assert frozen['identity']==evidence['identity'];(folder/'design_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'specification':spec,'checks':checks},indent=2))
 if not checks['passed']:raise RuntimeError('Physical or convergence design check failed')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-index',type=int,choices=range(10),required=True);a=p.parse_args();run(a.case_index)
