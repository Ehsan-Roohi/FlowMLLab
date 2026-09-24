"""Controlled SU2 8.0.1 rerun of unchanged meshes/configurations and model weights."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
from datetime import datetime,timezone
import numpy as np
from analyze import analyze,load_csv
from freeze_model import FrozenSurrogate
from recompute_neural_cfd import case_data,verify_raw
from install_su2_801 import BINARY_SHA256,ASSET_SHA256,URL
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'results/week16_lowboom'
SOURCES=['weakwall_cfd.py','install_su2_801.py','analyze.py','freeze_model.py','independent_audit_report.py','recompute_neural_cfd.py','seeb_reference.py','cfd.py']
RAW=['mesh.su2','flow.cfg','metadata.json','history.csv','solver.log','restart_flow.csv','surface_flow.csv','metrics.json','extracted.npz','prediction.json','reference_extracted.npz','reference_run_evidence.json','reference_metrics.json']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity():
 return {'dataset_sha256':sha(E/'dataset.npz'),'checkpoint_sha256':sha(E/'reference/model_checkpoint.npz'),'source_sha256':{n:sha(ROOT/'qa/week16'/n) for n in SOURCES},'solver_binary_sha256':BINARY_SHA256}
def physical_checks(folder):
 settings={s.split('=',1)[0].strip():s.split('=',1)[1].strip() for s in (folder/'flow.cfg').read_text().splitlines() if '=' in s}
 g=1.4;R=287.058;T=float(settings['FREESTREAM_TEMPERATURE']);pinf=float(settings['FREESTREAM_PRESSURE']);M=float(settings['MACH_NUMBER'])
 assert T==288.15 and pinf==101325 and M==1.8
 factor=1+(g-1)*M*M/2;h_inf=g*R*T/(g-1)*factor;rho0=pinf/(R*T)*factor**(1/(g-1));fields=[]
 for fn in ['restart_flow.csv','surface_flow.csv']:
  u=load_csv(folder/fn);rho=u['Density'];v2=(u['Momentum_x']**2+u['Momentum_y']**2)/rho**2;p=(g-1)*(u['Energy']-.5*rho*v2);h=g/(g-1)*p/rho+.5*v2;dev=abs(h/h_inf-1);j=int(dev.argmax())
  checks={'finite_positive_fields':bool(np.isfinite(rho).all() and np.isfinite(p).all() and (rho>0).all() and (p>0).all()),'h0_max_deviation_at_most_10pct':bool(np.isfinite(h).all() and dev.max()<=.1),'density_below_110pct_stagnation_bound':bool(rho.max()<=1.1*rho0)}
  fields.append({'file':fn,'points':len(u),'max_h0_relative_deviation':float(dev.max()),'min_h0_over_freestream':float(h.min()/h_inf),'max_h0_over_freestream':float(h.max()/h_inf),'max_density_over_stagnation_bound':float(rho.max()/rho0),'max_h0_location':[float(u['x'][j]),float(u['y'][j])],'nodes_above_10pct':int((dev>.1).sum()),'checks':checks,'passed':all(checks.values())})
 return {'gamma':g,'gas_constant':R,'h0_freestream':h_inf,'isentropic_stagnation_density':rho0,'fields':fields,'passed':all(f['passed'] for f in fields)}
def verify(folder):
 numerical=verify_raw(folder)
 assert len(load_csv(folder/'restart_flow.csv'))==numerical['mesh_integrity']['nodes'], 'Restart node count differs from exported fluid mesh'
 physical=physical_checks(folder)
 return {'numerical':numerical,'physical':physical,'passed':numerical['passed'] and physical['passed']}
def run(index,reference_folder):
 ref=Path(reference_folder).resolve();name,par,x=case_data(index)
 old=json.loads((ref/'run_evidence.json').read_text());assert old['case_index']==index and old['test_name']==name
 for n,s in old['raw_sha256'].items():assert sha(ref/n)==s, f'Changed original reference {n}'
 assert old['identity']['checkpoint_sha256']==identity()['checkpoint_sha256'] and old['identity']['dataset_sha256']==identity()['dataset_sha256']
 exe=Path(os.environ.get('SU2_CFD',str(ROOT/'.tools/week16_su2_801/bin/SU2_CFD'))).resolve();assert sha(exe)==BINARY_SHA256
 folder=E/'runs'/f'checkpoint_v801_{index:03d}';folder.mkdir(parents=True,exist_ok=False)
 for n in ['mesh.su2','flow.cfg']:shutil.copy2(ref/n,folder/n)
 for n in ['extracted.npz','run_evidence.json','metrics.json']:shutil.copy2(ref/n,folder/('reference_'+n))
 msh={}
 for p in ref.glob('*.msh'):shutil.copy2(p,folder/p.name);msh[p.name]=sha(p)
 assert sha(folder/'mesh.su2')==old['raw_sha256']['mesh.su2'] and sha(folder/'flow.cfg')==old['raw_sha256']['flow.cfg']
 wave,cd=FrozenSurrogate().predict(par)
 prediction={'case_index':index,'test_name':name,'parameters':par.tolist(),'x':x.tolist(),'waveform':wave[0].tolist(),'cd':float(cd[0]),'recorded_before_solver_utc':datetime.now(timezone.utc).isoformat(),'identity':identity()}
 (folder/'prediction.json').write_text(json.dumps(prediction,indent=2)+'\n')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');start=time.time()
 with (folder/'solver.log').open('w') as log:result=subprocess.run([str(exe),'flow.cfg'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
 meta=json.loads((ref/'metadata.json').read_text());meta.update(name=folder.name,returncode=result.returncode,wall_seconds=time.time()-start,solver_version='8.0.1',solver_binary_sha256=BINARY_SHA256,reference_version='8.5.0',reference_test_name=name)
 (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 if result.returncode:raise RuntimeError((folder/'solver.log').read_text()[-4000:])
 analyze(folder);checks=verify(folder)
 evidence={'identity':identity(),'case_index':index,'test_name':name,'solver_version':'8.0.1','official_asset_url':URL,'official_asset_sha256':ASSET_SHA256,'checks':checks,'unchanged_reference_hashes':{n:old['raw_sha256'][n] for n in ['mesh.su2','flow.cfg']},'reference_workflow_run':35936740632,'copied_msh_sha256':msh,'raw_sha256':{n:sha(folder/n) for n in RAW}}
 assert prediction['identity']==evidence['identity']
 (folder/'run_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps({'case':name,'checks':checks},indent=2))
 if not checks['passed']:raise RuntimeError('Unchanged physical/convergence acceptance criteria failed')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-index',type=int,choices=range(8),required=True);p.add_argument('--reference-folder',type=Path,required=True);a=p.parse_args();run(a.case_index,a.reference_folder)
