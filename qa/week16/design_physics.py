"""Generic ideal-gas physical and convergence checks for retained design CFD."""
from pathlib import Path
import json
import numpy as np
from analyze import load_csv
from seeb_reference import check_su2_mesh

def physical_checks(folder):
 folder=Path(folder)
 settings={s.split('=',1)[0].strip():s.split('=',1)[1].strip() for s in (folder/'flow.cfg').read_text().splitlines() if '=' in s}
 g=1.4;R=287.058;T=float(settings['FREESTREAM_TEMPERATURE']);pinf=float(settings['FREESTREAM_PRESSURE']);M=float(settings['MACH_NUMBER'])
 assert T==288.15 and pinf==101325 and M in (1.7,1.8,1.9)
 factor=1+(g-1)*M*M/2;h_inf=g*R*T/(g-1)*factor;rho0=pinf/(R*T)*factor**(1/(g-1));fields=[]
 for fn in ['restart_flow.csv','surface_flow.csv']:
  u=load_csv(folder/fn);rho=u['Density'];v2=(u['Momentum_x']**2+u['Momentum_y']**2)/rho**2;p=(g-1)*(u['Energy']-.5*rho*v2);h=g/(g-1)*p/rho+.5*v2;dev=abs(h/h_inf-1);j=int(dev.argmax())
  checks={'finite_positive_fields':bool(np.isfinite(rho).all() and np.isfinite(p).all() and (rho>0).all() and (p>0).all()),'h0_max_deviation_at_most_10pct':bool(np.isfinite(h).all() and dev.max()<=.1),'density_below_110pct_stagnation_bound':bool(rho.max()<=1.1*rho0)}
  fields.append({'file':fn,'points':len(u),'max_h0_relative_deviation':float(dev.max()),'min_h0_over_freestream':float(h.min()/h_inf),'max_h0_over_freestream':float(h.max()/h_inf),'max_density_over_stagnation_bound':float(rho.max()/rho0),'max_h0_location':[float(u['x'][j]),float(u['y'][j])],'nodes_above_10pct':int((dev>.1).sum()),'checks':checks,'passed':all(checks.values())})
 return {'mach':M,'gamma':g,'gas_constant':R,'h0_freestream':h_inf,'isentropic_stagnation_density':rho0,'fields':fields,'passed':all(f['passed'] for f in fields)}

def verify(folder):
 folder=Path(folder);m=json.loads((folder/'metrics.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
 assert m['returncode']==meta['returncode']==0 and meta['axisymmetric']
 assert meta['level'] in (1.5,2) and meta['mach'] in (1.7,1.8,1.9)
 mesh=check_su2_mesh(folder/'mesh.su2',expected_cells=meta['cells'])
 physical=physical_checks(folder);assert physical['fields'][0]['points']==mesh['nodes']
 hist=load_csv(folder/'history.csv');res=float(hist['rms[Rho]'][-1]);drop=float(hist['rms[Rho]'][0]-res);tail=hist['CD'][-100:]
 assert len(tail)==100 and np.isfinite(tail).all() and abs(tail.mean())>0
 stable=float(np.ptp(tail)/abs(tail.mean()))
 numerical={'residual_at_most_minus9':res<=-9,'residual_drop_at_least5':drop>=5,'drag_stable_below_1e_minus4':stable<1e-4,'positive_pressure_drag':bool(np.isfinite(m['cd_pressure']) and m['cd_pressure']>0)}
 for key,value in [('density_residual_log10',res),('residual_drop',drop),('drag_tail_relative_range',stable)]:assert np.isclose(m[key],value,atol=1e-12,rtol=1e-12)
 return {'mesh_integrity':mesh,'iterations':len(hist),'density_residual_log10':res,'residual_drop':drop,'drag_tail_relative_range':stable,'numerical_checks':numerical,'physical':physical,'passed':all(numerical.values()) and physical['passed']}
