"""Plot retained geometries using checked SU2 8.0.1 raw fields (never runs CFD).

Example: python qa/week16/clean_figures.py --raw-root /tmp/clean-field-raw \
 --artifact-zips /tmp/design-clean-0.zip /tmp/design-clean-1.zip
Learning panels are generated only when the separate clean fit is available.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'results/week16_lowboom/reference'
ARTIFACTS=[(10785828508,'4f950536109a1f367a4dbd6877b5a67367b2aac4e6659ba1dff67c88463912c5'),(10786076437,'f1e9620bf11de8be7d83e0b689a26132b4f4f8db63d515a7886a957cbe3c0f70')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh(path):
 lines=iter(Path(path).read_text().splitlines());tri=[];xy=None
 for line in lines:
  if line.startswith('NELEM='):
   for _ in range(int(line.split('=')[1])):
    row=list(map(int,next(lines).split()))
    if row[0]==9:tri.extend([[row[1],row[2],row[3]],[row[1],row[3],row[4]]])
    elif row[0]==5:tri.append(row[1:4])
    else:raise ValueError('Unsupported fluid cell type')
  elif line.startswith('NPOIN='):
   xy=np.zeros((int(line.split('=')[1]),2))
   for _ in range(len(xy)):
    row=next(lines).split();xy[int(row[2])]=list(map(float,row[:2]))
   break
 return xy,np.array(tri)
def fields(raw_root,artifact_zips):
 provenance={'workflow_run':35943668623,'scope':'Actual SU2 8.0.1 axisymmetric Euler fields of retained historical geometries; no new optimization.','artifacts':[],'runs':[]}
 for i,(identifier,digest) in enumerate(ARTIFACTS):
  if artifact_zips:assert sha(artifact_zips[i])==digest,'Artifact archive digest mismatch'
  provenance['artifacts'].append({'id':identifier,'zip_sha256':digest,'archive_verified_locally':bool(artifact_zips)})
 fig,axes=plt.subplots(2,1,figsize=(12,7.5),sharex=True,sharey=True,layout='constrained')
 for ax,name,label in zip(axes,['design_v801_000_baseline','design_v801_001_optimized'],['Baseline geometry','Historical optimizer candidate (retained geometry)']):
  folder=next(Path(raw_root).rglob(name));e=json.loads((folder/'design_evidence.json').read_text());assert e['checks']['passed'] and e['solver_version']=='8.0.1'
  for filename,digest in e['raw_sha256'].items():assert sha(folder/filename)==digest,filename
  xy,tri=mesh(folder/'mesh.su2');d=np.genfromtxt(folder/'restart_flow.csv',delimiter=',',names=True,deletechars='');order=np.argsort(d['PointID']);d=d[order]
  np.testing.assert_array_equal(d['PointID'],np.arange(len(xy)));np.testing.assert_allclose(np.c_[d['x'],d['y']],xy,atol=1e-12)
  p=.4*(d['Energy']-.5*(d['Momentum_x']**2+d['Momentum_y']**2)/d['Density']);cp=(p-101325)/(.5*1.4*101325*1.8**2)
  triang=mtri.Triangulation(xy[:,0],xy[:,1],tri)
  im=ax.tricontourf(triang,cp,levels=np.linspace(-.12,.12,49),cmap='RdBu_r',extend='both')
  with np.load(folder/'extracted.npz') as z:
   order=np.argsort(z['surface_x']);ax.fill_between(z['surface_x'][order],0,z['surface_r'][order],color='#30343b',zorder=5)
  ax.axhline(.5,color='k',lw=.8,ls='--',alpha=.6);ax.text(1.65,.52,'sampling line r/L = 0.5',fontsize=9)
  ax.set(title=label,ylabel='r/L',xlim=(-.1,2.5),ylim=(0,1));ax.set_aspect('equal')
  provenance['runs'].append({'name':name,'evidence_sha256':sha(folder/'design_evidence.json'),'raw_sha256':e['raw_sha256'],'cp_min':float(cp.min()),'cp_max':float(cp.max())})
 axes[-1].set_xlabel('x/L');fig.colorbar(im,ax=axes,label=r'$C_p=(p-p_\infty)/(\frac{1}{2}\gamma p_\infty M_\infty^2)$',shrink=.82)
 fig.suptitle('Checked CFD fields • Mach 1.8 • axisymmetric Euler • SU2 8.0.1\nShared color scale and equal geometric aspect; saturated extrema extend beyond color limits',fontsize=12)
 target=REF/'clean_cfd_fields.png';fig.savefig(target,dpi=180);plt.close(fig)
 provenance['figure_sha256']=sha(target);provenance['script_sha256']=sha(__file__)
 (REF/'clean_cfd_fields_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
def learning():
 from freeze_model import FrozenSurrogate
 paths=[REF/n for n in ['clean_dataset_v801.npz','clean_model_v801.npz','clean_model_audit_v801.json','clean_model_training_v801.json']]
 if not all(p.exists() for p in paths):return False
 data=np.load(paths[0]);report=json.loads(paths[2].read_text());log=json.loads(paths[3].read_text());assert report['passed'] and sha(paths[0])==report['dataset_sha256'] and sha(paths[1])==report['checkpoint_sha256']
 fig,grid=plt.subplots(2,2,figsize=(10,7.5),layout='constrained');axes=grid.ravel();colors=['#2166ac','#1b9e77','#d95f02','#984ea3']
 splits=['train','validation','test','extrapolation']
 for split,color in zip(splits,colors):
  m=data['splits']==split;axes[0].scatter(*data['parameters'][m].T,label=f'{split} (n={m.sum()})',color=color,s=35)
 axes[0].set(xlabel='shape parameter a',ylabel='shape parameter b',title='Preserved geometry split');axes[0].legend(fontsize=8)
 vals=[100*report['same_mesh'][s]['wave_relative_l2'] for s in splits];axes[1].bar(splits,vals,color=colors);axes[1].tick_params(axis='x',rotation=25);axes[1].set(ylabel='Waveform relative L2 error (%)',title='Saved model: split-wise error')
 idx=np.flatnonzero(data['splits']=='test')[0];pred,_=FrozenSurrogate(paths[1]).predict(data['parameters'][idx:idx+1]);axes[2].plot(data['x'],data['waveforms'][idx],label='SU2 8.0.1');axes[2].plot(data['x'],pred[0],'--',label='Saved neural model');axes[2].set(xlabel='x/L',ylabel='Cp at r/L = 0.5',title=f'First held-out geometry: {data["names"][idx]}');axes[2].legend()
 axes[3].bar(np.arange(8),100*np.array(report['finer_mesh']['per_case_wave_relative_l2']),color='#2166ac');axes[3].set(xlabel='Held-out geometry index',ylabel='Waveform relative L2 error (%)',title='Each case against finer CFD',xticks=np.arange(8))
 axes[3].axhline(100*report['finer_mesh']['wave_relative_l2'],color='#d95f02',ls='--',label='Aggregate error');axes[3].legend(fontsize=9)
 fig.suptitle('Clean-label neural model: one fixed fit on 24 training geometries',fontsize=13)
 target=REF/'clean_model_learning.png';fig.savefig(target,dpi=180);plt.close(fig)
 (REF/'clean_model_learning_provenance.json').write_text(json.dumps({'input_sha256':{p.name:sha(p) for p in paths},'figure_sha256':sha(target),'script_sha256':sha(__file__),'final_training_objective':log['final_training_objective']},indent=2)+'\n');return True
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-root',type=Path);parser.add_argument('--artifact-zips',type=Path,nargs=2);args=parser.parse_args()
 REF.mkdir(exist_ok=True,parents=True)
 if args.raw_root:fields(args.raw_root,args.artifact_zips)
 print('Clean-model panels generated:',learning())
