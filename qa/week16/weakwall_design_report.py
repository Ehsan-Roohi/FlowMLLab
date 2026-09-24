"""Validate retained design improvements with separate SU2 8.0.1 evidence."""
import argparse,json
from pathlib import Path
import numpy as np
from weakwall_designs import E,RAW,specifications,identity,sha,FrozenSurrogate
from design_physics import verify
R=E/'reference'
def report(write=True):
 rows=[];waves=[];surfaces=[];model=FrozenSurrogate()
 for spec in specifications():
  folder=E/'runs'/spec['name'];ev=json.loads((folder/'design_evidence.json').read_text());assert ev['identity']==identity() and ev['specification']==spec
  assert set(ev['raw_sha256'])==set(RAW)
  for n,s in ev['raw_sha256'].items():assert sha(folder/n)==s,f'Changed design evidence {folder/n}'
  frozen=json.loads((folder/'design_frozen.json').read_text());assert frozen['specification']==spec and frozen['identity']==identity()
  w,cd=model.predict([spec['a'],spec['b']]);pred=frozen['retained_checkpoint_prediction_at_training_mach_1p8']
  np.testing.assert_allclose(w[0],pred['waveform'],rtol=1e-12,atol=1e-14);np.testing.assert_allclose(cd[0],pred['cd'],rtol=1e-12,atol=1e-14)
  checks=verify(folder);assert checks==ev['checks'] and checks['passed']
  m=json.loads((folder/'metrics.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
  for key in ['a','b','level','mach']:assert m[key]==meta[key]==spec[key]
  for n in ['flow.cfg','mesh.su2']:assert sha(folder/n)==meta[n+'_sha256']
  with np.load(folder/'extracted.npz') as a:
   x=a['x'].copy();wave=a['cp'][1].copy();assert a['radii'][1]==.5
  assert np.isfinite(wave).all() and np.isclose(wave.max(),m['peak_cp'])
  if waves:assert np.array_equal(x,common_x)
  common_x=x;waves.append(wave)
  rows.append({'specification':spec,'cells':m['cells'],'peak_cp':m['peak_cp'],'cd_pressure':m['cd_pressure'],'checks':checks,'raw_sha256':ev['raw_sha256'],'design_evidence_sha256':sha(folder/'design_evidence.json')})
 def pair(i,j):
  return {'baseline_index':i,'optimized_index':j,'mach':rows[i]['specification']['mach'],'level':rows[i]['specification']['level'],'peak_reduction':1-rows[j]['peak_cp']/rows[i]['peak_cp'],'drag_ratio':rows[j]['cd_pressure']/rows[i]['cd_pressure'],'drag_change':rows[j]['cd_pressure']/rows[i]['cd_pressure']-1}
 design_pairs=[pair(0,1),pair(2,3)];offdesign=[pair(6,7),pair(8,9)]
 mesh=[]
 for i,j in [(0,2),(1,3)]:
  mesh.append({'design':rows[i]['specification']['design'],'coarser_index':i,'finer_index':j,'peak_relative_change':abs(rows[i]['peak_cp']/rows[j]['peak_cp']-1),'drag_relative_change':abs(rows[i]['cd_pressure']/rows[j]['cd_pressure']-1),'wave_relative_l2_change':float(np.linalg.norm(waves[i]-waves[j])/np.linalg.norm(waves[j]))})
 checks={'ten_runs_converged_and_physically_plausible':all(r['checks']['passed'] for r in rows),'peak_reduction_above_5pct_at_both_design_meshes':all(p['peak_reduction']>.05 for p in design_pairs),'drag_ratio_at_most_1p02_at_both_design_meshes':all(p['drag_ratio']<=1.02 for p in design_pairs),'both_peak_mesh_changes_below_5pct':all(m['peak_relative_change']<.05 for m in mesh),'both_drag_mesh_changes_below_3pct':all(m['drag_relative_change']<.03 for m in mesh)}
 arrays={'x':common_x,'names':[r['specification']['name'] for r in rows],'waveforms':waves,'cd_pressure':[r['cd_pressure'] for r in rows]}
 target=R/'weakwall_design_test.npz'
 if write:np.savez_compressed(target,**arrays)
 else:
  with np.load(target) as saved:
   for k,v in arrays.items():np.testing.assert_array_equal(saved[k],v)
 result={'solver_version':'8.0.1','identity':identity(),'runs':rows,'design_pairs':design_pairs,'mesh_sensitivity':mesh,'offdesign_pairs':offdesign,'thresholds':{'peak_reduction_min_exclusive':.05,'drag_ratio_max_inclusive':1.02,'peak_mesh_relative_change_max_exclusive':.05,'drag_mesh_relative_change_max_exclusive':.03,'max_h0_relative_deviation':.1,'max_density_over_stagnation_density':1.1},'checks':checks,'passed':all(checks.values()),'compact_arrays_sha256':sha(target),'scope':'Fresh physically checked CFD assessment of the existing historical candidate geometries; no optimization or retraining.','limitations':['The historical optimizer and retained portable checkpoint are different fitted models; these runs assess the retained geometries directly.','New SU2 8.0.1 results do not overwrite or retroactively validate 8.5.0 labels or reports.','Two-mesh differences are observed discretization sensitivity, not formal GCI or a continuum error estimate.','Off-design and alternative runs must pass physical and convergence checks; their performance is reported, not assumed to meet the design-point optimization gates.','Full-field thermodynamic allowances are plausibility rejection checks, not accuracy estimates.','Near-field pressure reduction at r/L=0.5 is not ground-level PLdB or full-aircraft performance.']}
 if write:
  (R/'weakwall_design_audit.json').write_text(json.dumps(result,indent=2)+'\n')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
  for ax,inds,title in zip(axs.flat,[(0,1,4,5),(2,3),(6,7),(8,9)],['Design point, mesh level 1.5','Design point, mesh level 2','Mach 1.7, mesh level 1.5','Mach 1.9, mesh level 1.5']):
   for i in inds:ax.plot(common_x,waves[i],label=rows[i]['specification']['design'])
   ax.set(title=title,xlabel='x/L at r/L=0.5',ylabel='Cp');ax.grid(alpha=.2);ax.legend(fontsize=8)
  fig.tight_layout();fig.savefig(R/'weakwall_design_validation.png',dpi=170);plt.close(fig)
 print(json.dumps({'design_pairs':design_pairs,'mesh_sensitivity':mesh,'checks':checks,'passed':result['passed']},indent=2))
 if not result['passed']:raise RuntimeError('Declared design verification gate failed')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');a=p.parse_args();report(write=not a.check_only)
