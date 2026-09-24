"""Audit eight unchanged-model SU2 8.0.1 comparisons without altering old reports."""
import argparse,json
from pathlib import Path
import numpy as np
from weakwall_cfd import E,RAW,identity,sha,verify,case_data,FrozenSurrogate
from independent_audit_report import metrics
R=E/'reference'
def report(write=True):
 names=[];pars=[];actual=[];predicted=[];cd=[];predcd=[];oldwaves=[];oldcd=[];rows=[]
 model=FrozenSurrogate()
 for i in range(8):
  name,par,x=case_data(i);folder=E/'runs'/f'checkpoint_v801_{i:03d}'
  evidence=json.loads((folder/'run_evidence.json').read_text());assert evidence['identity']==identity();assert evidence['case_index']==i and evidence['test_name']==name
  assert set(evidence['raw_sha256'])==set(RAW)
  for n,s in evidence['raw_sha256'].items():assert sha(folder/n)==s,f'Changed raw evidence {folder/n}'
  for n,s in evidence['copied_msh_sha256'].items():assert sha(folder/n)==s
  checks=verify(folder);assert checks==evidence['checks'] and checks['passed']
  old=json.loads((folder/'reference_run_evidence.json').read_text());assert old['case_index']==i and old['test_name']==name
  assert old['identity']['checkpoint_sha256']==identity()['checkpoint_sha256'] and old['identity']['dataset_sha256']==identity()['dataset_sha256']
  for n in ['mesh.su2','flow.cfg']:assert sha(folder/n)==old['raw_sha256'][n]==evidence['unchanged_reference_hashes'][n]
  for n in ['extracted.npz','metrics.json']:assert sha(folder/('reference_'+n))==old['raw_sha256'][n]
  p=json.loads((folder/'prediction.json').read_text());assert p['identity']==identity() and p['case_index']==i and p['test_name']==name
  assert np.array_equal(par,p['parameters']) and np.array_equal(x,p['x'])
  w,d=model.predict(par);np.testing.assert_allclose(w[0],p['waveform'],rtol=1e-12,atol=1e-14);np.testing.assert_allclose(d[0],p['cd'],rtol=1e-12,atol=1e-14)
  with np.load(folder/'extracted.npz') as a:assert np.array_equal(x,a['x']) and a['radii'][1]==.5;newwave=a['cp'][1].copy()
  with np.load(folder/'reference_extracted.npz') as a:assert np.array_equal(x,a['x']) and a['radii'][1]==.5;oldwave=a['cp'][1].copy()
  m=json.loads((folder/'metrics.json').read_text());om=json.loads((folder/'reference_metrics.json').read_text());assert m['a']==float(par[0]) and m['b']==float(par[1])
  names.append(name);pars.append(par);actual.append(newwave);predicted.append(p['waveform']);cd.append(m['cd_pressure']);predcd.append(p['cd']);oldwaves.append(oldwave);oldcd.append(om['cd_pressure'])
  rows.append({'name':name,'run':folder.name,'checks':checks,'raw_sha256':evidence['raw_sha256'],'run_evidence_sha256':sha(folder/'run_evidence.json'),'wave_relative_difference_v850':float(np.linalg.norm(newwave-oldwave)/np.linalg.norm(oldwave)),'drag_relative_difference_v850':float(abs(m['cd_pressure']/om['cd_pressure']-1))})
 scores=metrics(np.array(predicted),np.array(actual),np.array(predcd),np.array(cd));criteria={k:scores[k]<.1 for k in ['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error']}
 arrays=dict(names=names,parameters=pars,x=x,predicted=predicted,actual=actual,predicted_cd=predcd,actual_cd=cd,reference_v850=oldwaves,reference_v850_cd=oldcd)
 target=R/'weakwall_checkpoint_test.npz'
 if write:np.savez_compressed(target,**arrays)
 else:
  with np.load(target) as saved:
   for k,v in arrays.items():np.testing.assert_array_equal(saved[k],v)
 result={'solver_version':'8.0.1','identity':identity(),'names':names,'cases':8,'metrics':scores,'thresholds':{k:.1 for k in criteria},'checks':criteria,'physical_and_convergence_pass':True,'passed':all(criteria.values()),'run_evidence':rows,'compact_arrays_sha256':sha(target),'aggregate_wave_relative_difference_v850':float(np.linalg.norm(np.array(actual)-oldwaves)/np.linalg.norm(oldwaves)),'mean_drag_relative_difference_v850':float(np.mean(abs(np.array(cd)/oldcd-1))),'scope':'Unchanged retained neural checkpoint compared with eight actual SU2 8.0.1 reruns using exact archived 8.5.0 meshes and configurations, with full-field physical and convergence checks.','limitations':['Retrospective comparison at eight previously known geometries; not a new blind generalization set.','Training labels, model weights and old 8.5.0 audit results remain unchanged.','Official solver versions differ in multiple implementations; changes cannot be attributed exclusively to wall handling.','Old 8.5.0 full-field enthalpy checks failed and remain failed.','10% enthalpy and density allowances are plausibility rejection criteria, not estimates of solution accuracy.','Passing aggregate surrogate criteria does not require every individual waveform error below 10%.','No experimental neural validation, complete Chinese aircraft reproduction, or ground loudness claim.']}
 if write:
  (R/'weakwall_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axs=plt.subplots(4,2,figsize=(11,11),sharex=True)
  for i,ax in enumerate(axs.flat):
   ax.plot(x,actual[i],'k',label='SU2 8.0.1, unchanged mesh/config');ax.plot(x,predicted[i],'--',label='Unchanged retained checkpoint');ax.plot(x,oldwaves[i],color='gray',alpha=.6,lw=.8,label='8.5.0 numerical reference')
   ax.set(title=f'{names[i]}: error {100*scores["per_case_wave_relative_l2"][i]:.2f}%',ylabel='Cp');ax.grid(alpha=.2)
  for ax in axs[-1]:ax.set_xlabel('x/L at r/L=0.5')
  axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(R/'weakwall_checkpoint_validation.png',dpi=160);plt.close(fig)
 print(json.dumps({'metrics':scores,'checks':criteria,'passed':result['passed']},indent=2))
 if not result['passed']:raise RuntimeError('Aggregate neural accuracy gate failed')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');a=p.parse_args();report(write=not a.check_only)
