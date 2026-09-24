"""Read-only physical plausibility audit of eight downloaded CFD artifacts."""
from pathlib import Path
import argparse,hashlib,json,zipfile,tarfile,io
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-dir',type=Path,required=True,help='Directory containing case-0.zip through case-7.zip from workflow 35936740632')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
BASE=args.artifact_dir
EXPECTED=['bb14263110616dc89a8eeff7b076172db79ac867f83462716199015a7c09a20d','b1bebf7f6b2c4c591e4470400a839176d3b074cb9191d090808191df9df8d2bc','f798560411617f3b25265b8910a9fbf9ff6def4664546f9f77864ca97c0fe96e','3807d3a7f568f07ed3eb467507bc82e3efbddc6a3a23b0053c6c4cc1b0aaf460','0304629543f5822c4767945837a7197218262a9f0ce42e33f77b7e07fd1b980b','c22002fe05c95ae6dcd153e6512b067de0f809bbcf240f7f1492bc18c50938e9','28eda8cfaaec83561b6fd9be685b835a64574cc6cc7460e99c9589f2f870791d','e600184ff1a7926078cfa7bf053088b62efb163463444feefe3225179fb42344']
IDS=[10782909532,10782959473,10782904522,10783118397,10783168194,10783481542,10782854569,10783531491]
rows=[]
for i in range(8):
 payload=(BASE/f'case-{i}.zip').read_bytes();sha=hashlib.sha256(payload).hexdigest();assert sha==EXPECTED[i]
 with zipfile.ZipFile(io.BytesIO(payload)) as z:
  assert z.testzip() is None
  with tarfile.open(fileobj=io.BytesIO(z.read(f'neural-case-{i}.tar.gz')),mode='r:gz') as t:
   prefix=f'results/week16_lowboom/runs/checkpoint_test_{i:03d}/'
   read=lambda name:t.extractfile(prefix+name).read()
   ev=json.loads(read('run_evidence.json'));cfg=read('flow.cfg');assert hashlib.sha256(cfg).hexdigest()==ev['raw_sha256']['flow.cfg']
   settings={line.split('=',1)[0].strip():line.split('=',1)[1].strip() for line in cfg.decode().splitlines() if '=' in line}
   gamma=1.4;gas_R=287.058
   p_inf=float(settings['FREESTREAM_PRESSURE']);T_inf=float(settings['FREESTREAM_TEMPERATURE']);M=float(settings['MACH_NUMBER'])
   assert p_inf==101325 and T_inf==288.15 and M==1.8
   rho_inf=p_inf/(gas_R*T_inf);factor=1+(gamma-1)*M*M/2
   h_inf=gamma*gas_R*T_inf/(gamma-1)*factor
   rho_bound=rho_inf*factor**(1/(gamma-1))
   fields=[]
   for fn in ['restart_flow.csv','surface_flow.csv']:
    raw=read(fn);rawsha=hashlib.sha256(raw).hexdigest();assert rawsha==ev['raw_sha256'][fn]
    a=np.genfromtxt(io.BytesIO(raw),delimiter=',',names=True,deletechars='')
    rho=a['Density'];v2=(a['Momentum_x']**2+a['Momentum_y']**2)/rho**2
    p=(gamma-1)*(a['Energy']-.5*rho*v2)
    h=gamma/(gamma-1)*p/rho+.5*v2
    dev=abs(h/h_inf-1); j=int(np.argmax(dev));k=int(np.argmax(rho))
    checks={'finite_positive_density_pressure':bool(np.isfinite(rho).all() and np.isfinite(p).all() and (rho>0).all() and (p>0).all()),'total_enthalpy_deviation_at_most_10pct':bool(np.isfinite(h).all() and dev.max()<=.1),'density_below_110pct_isentropic_stagnation_bound':bool(rho.max()<=1.1*rho_bound)}
    bad=dev>.1
    details={'points_exceeding_10pct':int(bad.sum()), 'fraction_exceeding_10pct':float(bad.mean()), 'affected_x_range':[float(a['x'][bad].min()),float(a['x'][bad].max())] if bad.any() else None, 'affected_r_range':[float(a['y'][bad].min()),float(a['y'][bad].max())] if bad.any() else None, 'h0_absolute_relative_deviation_percentiles':{str(q):float(np.percentile(dev,q)) for q in [50,95,99,99.9]}}
    fields.append({'diagnostics':details,'file':fn,'sha256':rawsha,'points':len(a),'min_density':float(rho.min()),'max_density':float(rho.max()),'max_density_over_stagnation_bound':float(rho.max()/rho_bound),'min_pressure':float(p.min()),'max_pressure':float(p.max()),'min_h0_over_freestream':float(h.min()/h_inf),'max_h0_over_freestream':float(h.max()/h_inf),'max_h0_relative_deviation':float(dev.max()),'worst_h0_location':{'x':float(a['x'][j]),'r':float(a['y'][j])},'max_density_location':{'x':float(a['x'][k]),'r':float(a['y'][k])},'checks':checks,'passed':all(checks.values())})
   rows.append({'case':f'checkpoint_test_{i:03d}','artifact_id':IDS[i],'artifact_sha256':sha,'config_sha256':hashlib.sha256(cfg).hexdigest(),'freestream':{'gamma':gamma,'gas_constant_J_kg_K':gas_R,'temperature_K':T_inf,'pressure_Pa':p_inf,'mach':M,'density_kg_m3':rho_inf,'total_enthalpy_J_kg':h_inf,'isentropic_stagnation_density_kg_m3':rho_bound},'fields':fields,'passed':all(f['passed'] for f in fields)})
report={'workflow_run':35936740632,'artifact_commit':'e593da167777805bacc2d56f421059cc02b27fa4','scope':'Independent read-only physical plausibility audit of actual volume and surface fields in eight retained CFD runs. No new CFD or fitting.','criteria':{'max_absolute_h0_relative_deviation':.1,'max_density_over_isentropic_stagnation_density':1.1},'equations':{'pressure':'(gamma-1)*(rhoE-0.5*(rho*u^2+rho*v^2))','total_enthalpy':'gamma/(gamma-1)*p/rho+0.5*(u^2+v^2)','freestream_total_enthalpy':'gamma*R*T_inf/(gamma-1)*(1+(gamma-1)*M_inf^2/2)','density_bound':'rho_inf*(1+(gamma-1)*M_inf^2/2)^(1/(gamma-1))'},'interpretation':'Uniform-inflow steady inviscid adiabatic flow should preserve stagnation enthalpy; shocks conserve it. Isentropic stagnation density supplies an upper plausibility bound for nonnegative entropy production. Finite-volume error allowances are broad rejection checks, not accuracy estimates.','limitations':['No nose stagnation pressure check is imposed on these pointed teaching bodies.','Positive fields and these allowances do not establish experimental validity, formal conservation order, or grid independence.','Gamma and gas constant are SU2 standard-air defaults; Mach, pressure and temperature are read from every retained configuration.'],'runs':rows,'passed':all(r['passed'] for r in rows)}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
for row in rows:
 print(row['case'],row['passed'],[(f['file'],100*f['max_h0_relative_deviation'],f['max_density_over_stagnation_bound']) for f in row['fields']])
print('ALL PASSED',report['passed'])
