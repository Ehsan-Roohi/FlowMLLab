"""One-shot evaluation using the already-selected phase decoder, no Re115 tuning."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.cylinder_phase import fit_case, interpolate, align_and_predict
from run_cylinder_phase_stable import metrics, omega

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output; protocol=json.loads((out/'protocol.json').read_text())
    if (out/'metrics.json').exists(): raise ValueError('Evaluation already recorded; use a separately labeled reproduction')
    if digest(ROOT/'flowmllab/cylinder_phase.py')!=protocol['model_code_sha256']:raise ValueError('Model code changed after protocol freeze')
    manifest=json.loads((ROOT/'tmp/cylinder-release-manifest.json').read_text())
    expected={r['file']:r['sha256'] for r in manifest['cases']}
    cases={};models=[];source_hashes={}
    for re in [90,110,120,140,100]:
        p=args.data/f'cylinder_cfd_re{re:03d}.npz'
        if digest(p)!=expected[p.name]:raise ValueError('Release hash mismatch '+p.name)
        source_hashes[str(re)]=digest(p);cases[re]=load(p)
        if re!=100:models.append(fit_case(cases[re],reynolds=re,harmonics=6))
    # Freeze actual coefficient bytes before opening the held-out target.
    coefficients={str(m.reynolds):hashlib.sha256(m.coefficients.tobytes()).hexdigest() for m in models}
    frozen={'protocol_sha256':digest(out/'protocol.json'),'development_hashes':source_hashes,'coefficient_hashes':coefficients,'strouhal':{str(m.reynolds):m.strouhal for m in models},'harmonics':6}
    (out/'frozen_model.json').write_text(json.dumps(frozen,indent=2)+'\n')
    val=cases[100];initial=np.stack([val[k][:4] for k in ('u','v','p')],-1)
    vm=interpolate(models,100);vp,_=align_and_predict(vm,initial,len(val['u']),delta_t_star=25*.05/12)
    vp[:,val['solid'].astype(bool)]=0
    validation=metrics(val,vp,vm.strouhal)
    del vp
    p=ROOT/'data/cylinder_reserved_re115/cylinder_cfd_re115.npz'
    if digest(p)!=protocol['test_archive_sha256']:raise ValueError('Test checksum mismatch')
    (out/'first_use.json').write_text(json.dumps({'test_sha256':digest(p),'frozen_model_sha256':digest(out/'frozen_model.json'),'purpose':'first model scoring; no subsequent selection permitted'},indent=2)+'\n')
    target=load(p);truth=np.stack([target[k] for k in ('u','v','p')],-1);fluid=~target['solid'].astype(bool)
    dt=np.diff(target['snapshot_time'])*.05/12
    if not np.allclose(dt,dt[0]):raise ValueError('Nonuniform snapshot times')
    model=interpolate(models,115)
    pred,phase=align_and_predict(model,truth[:4],len(truth),delta_t_star=float(dt[0]));pred[:,~fluid]=0
    nearest=min(models,key=lambda m:(abs(m.reynolds-115),m.reynolds))
    base,_=align_and_predict(nearest,truth[:4],len(truth),delta_t_star=float(dt[0]));base[:,~fluid]=0
    times=target['snapshot_time'];idx=np.searchsorted(cases[110]['snapshot_time'],times)
    if not np.array_equal(cases[110]['snapshot_time'][idx],times):raise ValueError('Time matching failed')
    linear=np.stack([.5*(cases[110][k][idx]+cases[120][k][idx]) for k in ('u','v','p')],-1)
    report={'validation_reproduction':validation,'test':metrics(target,pred,model.strouhal),'nearest_development_baseline':metrics(target,base,nearest.strouhal),'time_matched_linear_baseline':metrics(target,linear,model.strouhal),'initial_true_frames':4,'initial_span_tU_D':float(3*dt[0]),'future_cfd_inputs':0,'phase':phase,'source':'reproduced public release decoder; Unity-specific modal run is separately audited','force_scope':'No force surrogate; CFD-only late-window statistics'}
    for key in ('predicted_strouhal','strouhal_relative_error','passes'):
        report['time_matched_linear_baseline'].pop(key, None)
    report['time_matched_linear_baseline']['frequency_scope']='No single frequency inferred from the superposed fields; only field errors compared.'
    for name,estimate in [('decoder',pred),('nearest',base),('linear',linear)]:
        report[name+'_field_errors']={k:float(np.linalg.norm((estimate[4:,...,i]-truth[4:,...,i])[:,fluid])/np.linalg.norm(truth[4:,...,i][:,fluid])) for i,k in enumerate(('u','v','p'))}
    keep=target['time']*.05/12>=75
    report['cfd_force_statistics_tstar_ge75']={'mean_Cd':float(target['drag_coefficient'][keep].mean()),'rms_Cl':float(np.sqrt(np.mean(target['lift_coefficient'][keep]**2)))}
    report['sampling_note']='Compact test has 53 future fields; initial four fields span 1.5625 convective units, unlike 0.3125 in the dense validation protocol.'
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    a,b=omega(truth),omega(pred);err=np.linalg.norm((b[4:]-a[4:])[:,fluid],axis=1)/np.linalg.norm(a[4:][:,fluid],axis=1)
    np.savetxt(out/'frame_errors.csv',np.column_stack((times[4:]*.05/12,err)),delimiter=',',header='tU_D,vorticity_relative_l2',comments='')
    fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for ax,field,title in [(axes[0,0],a[-1],'LBM reference'),(axes[0,1],b[-1],'Frozen phase decoder')]:
        im=ax.imshow(np.ma.masked_where(~fluid,field),origin='lower',extent=(0,20,0,8),cmap='RdBu_r',vmin=-2,vmax=2)
        ax.add_patch(Circle((5,47.5/12),.5,color='#263345'));ax.set(xlim=(3,20),ylim=(1,7),xlabel='x/D',ylabel='y/D',title=title,aspect='equal');fig.colorbar(im,ax=ax,label='vorticity D/U',extend='both')
    axes[1,0].plot(times[4:]*.05/12,100*err);axes[1,0].axhline(15,color='gray',ls='--');axes[1,0].set(xlabel='tU/D',ylabel='Frame vorticity error (%)',title='Future frames only')
    names=['Phase decoder','Nearest development','Time-matched linear'];keys=['test','nearest_development_baseline','time_matched_linear_baseline']
    axes[1,1].bar(names,[100*report[k]['vorticity_global_relative_l2'] for k in keys],color=['#2764a5','#7b8d9e','#cc8b38']);axes[1,1].set(ylabel='Global vorticity error (%)',title='Predeclared baseline comparison')
    fig.suptitle('FlowMLLab | First reserved Re=115 evaluation | 6 fixed harmonics',fontsize=16)
    fig.savefig(out/'re115_evaluation.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
