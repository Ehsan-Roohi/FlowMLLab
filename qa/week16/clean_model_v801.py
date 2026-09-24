"""Fit once and separately audit a fixed neural model on physically checked 8.0.1 labels.

--train performs one prespecified fit on 24 training cases and saves new weights.
--check-only loads those weights and recomputes all metrics without fitting.
Both retain the old training dataset, model and historical reports unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,os,platform,time,warnings
import numpy as np
from freeze_model import FrozenSurrogate
from independent_audit_report import metrics
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'results/week16_lowboom';R=E/'reference'
DATA=R/'clean_dataset_v801.npz';CHECKPOINT=R/'clean_model_v801.npz';LOG=R/'clean_model_training_v801.json';REPORT=R/'clean_model_audit_v801.json'
FINE=R/'weakwall_checkpoint_test.npz';FINE_REPORT=R/'weakwall_checkpoint_audit.json';CAMPAIGN=R/'clean_campaign_audit.json'
SOURCES=['clean_model_v801.py','freeze_model.py','independent_audit_report.py']
SETTINGS={'hidden_layer_sizes':[32,32],'activation':'tanh','solver':'lbfgs','alpha':.01,'max_iter':3000,'max_fun':15000,'random_state':16,'tol':1e-7,'pod_modes':12,'pca_svd_solver':'full'}

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():return {n:sha(ROOT/'qa/week16'/n) for n in SOURCES}
def inputs():
 campaign=json.loads(CAMPAIGN.read_text());fine_report=json.loads(FINE_REPORT.read_text())
 assert campaign['passed'] and campaign['cases']==44 and campaign['dataset_sha256']==sha(DATA)
 assert campaign['split_counts']=={'train':24,'validation':6,'test':8,'extrapolation':6}
 assert len(campaign['runs'])==44 and all(row['checks']['passed'] for row in campaign['runs'])
 assert fine_report['passed'] and fine_report['physical_and_convergence_pass'] and fine_report['compact_arrays_sha256']==sha(FINE)
 with np.load(DATA,allow_pickle=False) as source:data={k:source[k].copy() for k in source.files}
 with np.load(FINE,allow_pickle=False) as source:fine={k:source[k].copy() for k in source.files}
 assert data['parameters'].shape==(44,2) and data['waveforms'].shape==(44,561) and data['cd'].shape==(44,)
 assert len(set(data['names']))==44 and np.isfinite(data['waveforms']).all() and np.isfinite(data['cd']).all() and (data['cd']>0).all()
 counts={str(s):int((data['splits']==s).sum()) for s in np.unique(data['splits'])};assert counts==campaign['split_counts']
 with np.load(E/'dataset.npz',allow_pickle=False) as historical:
  for k in ['parameters','names','splits','x']:np.testing.assert_array_equal(data[k],historical[k])
 test=data['splits']=='test';assert test.sum()==8
 for k,expected in [('names',data['names'][test]),('parameters',data['parameters'][test]),('x',data['x'])]:np.testing.assert_array_equal(fine[k],expected)
 assert fine['actual'].shape==(8,561) and fine['actual_cd'].shape==(8,) and np.isfinite(fine['actual']).all() and (fine['actual_cd']>0).all()
 return data,fine

def train():
 if any(p.exists() for p in [CHECKPOINT,LOG,REPORT]):raise FileExistsError('Refusing to replace a retained clean-model fit or report; declare a new model version explicitly.')
 data,_=inputs();train=data['splits']=='train'
 from sklearn.decomposition import PCA
 from sklearn.preprocessing import StandardScaler
 from sklearn.neural_network import MLPRegressor
 from sklearn.exceptions import ConvergenceWarning
 import sklearn,scipy
 source=source_hashes();dataset_hash=sha(DATA)
 sx=StandardScaler().fit(data['parameters'][train]);pca=PCA(n_components=12,svd_solver='full').fit(data['waveforms'][train]);z=np.c_[pca.transform(data['waveforms'][train]),np.log(data['cd'][train])];sy=StandardScaler().fit(z)
 model=MLPRegressor(hidden_layer_sizes=(32,32),activation='tanh',solver='lbfgs',alpha=.01,max_iter=3000,max_fun=15000,random_state=16,tol=1e-7)
 start=time.perf_counter()
 with warnings.catch_warnings(record=True) as recorded:
  warnings.simplefilter('always');model.fit(sx.transform(data['parameters'][train]),sy.transform(z))
 seconds=time.perf_counter()-start
 arrays=dict(format_version=np.array(1),layers=np.array(3),input_mean=sx.mean_,input_scale=sx.scale_,output_mean=sy.mean_,output_scale=sy.scale_,pca_mean=pca.mean_,pca_components=pca.components_,x=data['x'],training_names=data['names'][train])
 for i,(w,b) in enumerate(zip(model.coefs_,model.intercepts_)):arrays[f'weight_{i}']=w;arrays[f'bias_{i}']=b
 CHECKPOINT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(CHECKPOINT,**arrays)
 portable=FrozenSurrogate(CHECKPOINT);w,cd=portable.predict(data['parameters'][train]);reference=sy.inverse_transform(model.predict(sx.transform(data['parameters'][train])))
 np.testing.assert_allclose(w,pca.inverse_transform(reference[:,:12]),rtol=1e-12,atol=1e-14);np.testing.assert_allclose(cd,np.exp(reference[:,12]),rtol=1e-12,atol=1e-14)
 assert source==source_hashes() and dataset_hash==sha(DATA)
 log={'model_identity':'Clean-label fixed-architecture SU2 8.0.1 refit; separate from both historical models.','dataset_sha256':dataset_hash,'checkpoint_sha256':sha(CHECKPOINT),'campaign_audit_sha256':sha(CAMPAIGN),'source_sha256':source,'settings':SETTINGS,'training_names':data['names'][train].tolist(),'training_cases':24,'fit_calls':1,'training_seconds':seconds,'iterations':int(model.n_iter_),'final_training_objective':float(model.loss_),'warnings':[{'category':type(a.message).__name__,'message':str(a.message)} for a in recorded],'optimizer_converged_without_warning':not any(isinstance(a.message,ConvergenceWarning) for a in recorded),'pod_retained_training_variance':float(pca.explained_variance_ratio_.sum()),'portable_vs_sklearn_training_inference_agrees':True,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')},'scope':'One fixed fit uses only the 24 training geometries; no hyperparameter search, test tuning or optimization is performed. Complete fitting may vary slightly with libraries/BLAS; saved arrays establish model identity.'}
 LOG.write_text(json.dumps(log,indent=2)+'\n')
 return audit(write=True)

def audit(write=False):
 data,fine=inputs();log=json.loads(LOG.read_text());assert log['settings']==SETTINGS and log['fit_calls']==1 and log['training_cases']==24
 assert log['dataset_sha256']==sha(DATA) and log['checkpoint_sha256']==sha(CHECKPOINT) and log['campaign_audit_sha256']==sha(CAMPAIGN) and log['source_sha256']==source_hashes()
 model=FrozenSurrogate(CHECKPOINT);a=model.arrays;tr=data['splits']=='train'
 assert a['training_names'].tolist()==log['training_names']==data['names'][tr].tolist()
 assert not set(a['training_names'])&set(fine['names'])
 np.testing.assert_array_equal(a['x'],data['x'])
 for actual,expected in [(a['input_mean'],data['parameters'][tr].mean(axis=0)),(a['input_scale'],data['parameters'][tr].std(axis=0)),(a['pca_mean'],data['waveforms'][tr].mean(axis=0))]:np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
 basis=a['pca_components'];np.testing.assert_allclose(basis@basis.T,np.eye(12),atol=1e-12)
 centered=data['waveforms'][tr]-a['pca_mean'];_,_,vt=np.linalg.svd(centered,full_matrices=False);subspace_error=float(np.linalg.norm(basis.T@basis-vt[:12].T@vt[:12]));assert subspace_error<1e-9
 z=np.c_[centered@basis.T,np.log(data['cd'][tr])]
 for actual,expected in [(a['output_mean'],z.mean(axis=0)),(a['output_scale'],z.std(axis=0))]:np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
 split_metrics={}
 for split in ['train','validation','test','extrapolation']:
  mask=data['splits']==split;w,cd=model.predict(data['parameters'][mask]);split_metrics[split]={'cases':int(mask.sum()),**metrics(w,data['waveforms'][mask],cd,data['cd'][mask])}
 w,cd=model.predict(fine['parameters']);fine_metrics=metrics(w,fine['actual'],cd,fine['actual_cd'])
 criteria={k:fine_metrics[k]<.1 for k in ['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error']}
 criteria['fixed_fit_optimizer_converged_without_warning']=log['optimizer_converged_without_warning']
 result={'model_identity':log['model_identity'],'solver_version':'8.0.1','settings':SETTINGS,'dataset_sha256':sha(DATA),'checkpoint_sha256':sha(CHECKPOINT),'training_log_sha256':sha(LOG),'campaign_audit_sha256':sha(CAMPAIGN),'finer_reference_sha256':sha(FINE),'finer_audit_sha256':sha(FINE_REPORT),'source_sha256':source_hashes(),'training_names':a['training_names'].tolist(),'training_log':log,'identity_checks':{'all_44_campaign_physical_checks_pass':True,'finer_eight_physical_checks_pass':True,'geometry_split_and_coordinates_preserved':True,'training_only_statistics_and_pod_verified':True,'saved_weights_match_training_log':True},'pod_training_subspace_error':subspace_error,'same_mesh':split_metrics,'finer_mesh':fine_metrics,'thresholds':{'wave_relative_l2_max':.1,'peak_mean_relative_error_max':.1,'drag_mean_relative_error_max':.1},'checks':criteria,'passed':all(criteria.values()),'scope':'New model fitted once on 24 physically checked SU2 8.0.1 training labels, tested on the preserved same-family geometries and existing finer 8.0.1 CFD labels.','limitations':['Retrospective known-geometry study; finer reference solutions already existed before this fit, so no new prospective blind-generalization claim.','Physical checks are broad rejection criteria, not proof of CFD accuracy or continuum mesh independence.','Every original geometry is retained; extrapolation and worst-case errors are reported even when aggregate tests pass.','A new fitted model does not inherit the historical optimizer candidate or its predictions. No clean-model optimization is performed here.','No experimental neural validation, full Chinese-aircraft reproduction, atmospheric propagation or ground-noise calculation.']}
 if write:REPORT.write_text(json.dumps(result,indent=2)+'\n')
 else:
  saved=json.loads(REPORT.read_text())
  def same(a,b):
   if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
   if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
   if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
   return a==b
  assert same(saved,result),'Saved clean-model audit is stale'
 print(json.dumps({'same_mesh':split_metrics,'finer_mesh':fine_metrics,'checks':criteria,'passed':result['passed']},indent=2))
 if not result['passed']:raise RuntimeError('Prespecified clean-model fitting/accuracy gates failed; no automatic retuning is permitted.')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--train',action='store_true');g.add_argument('--check-only',action='store_true');args=p.parse_args()
 if args.train:train()
 else:audit(write=False)
