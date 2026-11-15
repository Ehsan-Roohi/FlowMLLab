"""Recompute metrics and update the g011 comparison from saved predictions only."""
from __future__ import annotations
import csv
import io
import json
import tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from train_week15_vanilla_deeponet import case_metrics, sha256

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/step_transolver'
ARCHIVE=ROOT/'results/step_operator_audit/wake_predictions.tgz'
LABELS={'geom':'Geo-DeepONet','fno':'FNO','ufno':'U-FNO'}

def load_fields(re):
    result={}
    with tarfile.open(ARCHIVE) as tar:
        for key in LABELS:
            path=f'wake_focused/{key}/seed_17/g011_Re{re}_medium_prediction.npz'
            with np.load(io.BytesIO(tar.extractfile(path).read()),allow_pickle=False) as z:
                result[key]={k:z[k].copy() for k in z.files}
    for seed in (17,29,43):
        with np.load(OUT/f'seed_{seed}/g011_Re{re}_medium_prediction.npz',allow_pickle=False) as z:
            result[f'transolver_{seed}']={k:z[k].copy() for k in z.files}
    ref=result['geom']
    for f in result.values():
        for key in ('truth','mask','shape'):
            np.testing.assert_array_equal(f[key],ref[key])
        np.testing.assert_allclose(f['coordinates'],ref['coordinates'],rtol=0,atol=1e-6)
        assert np.isfinite(f['prediction']).all()
    return result

def curl(a,x,y):
    return np.gradient(a[:,:,1],x,axis=1)-np.gradient(a[:,:,0],y,axis=0)

def build_fields(re, save=True):
    fields=load_fields(re);ref=fields['geom'];ny,nx=map(int,ref['shape'])
    xy=ref['coordinates'].reshape(ny,nx,2);x,y=xy[0,:,0],xy[:,0,1]
    mask=ref['mask'].reshape(ny,nx)
    interior=binary_erosion(mask,structure=np.ones((3,3)),border_value=0)
    keys=['geom','fno','ufno','transolver_17']
    arrays=[ref['truth']]+[fields[k]['prediction'] for k in keys]
    names=['CFD reference']+[LABELS[k] for k in keys[:3]]+['Transolver (original)']
    labels=[names[0]]
    for name,pred in zip(names[1:],arrays[1:]):
        m=case_metrics(ref['truth'],pred,ref['mask'])
        labels.append(f'{name}\nEv {m["velocity_percent"]:.2f}% | Ep {m["pressure_percent"]:.2f}%\nReverse IoU {m["reverse_iou"]:.3f}')
    arrays=[a.reshape(ny,nx,3).astype(float) for a in arrays]
    derived=[]
    for a in arrays:
        derived.append([np.ma.masked_where(~mask,np.hypot(a[:,:,0],a[:,:,1])),
            np.ma.masked_where(~mask,a[:,:,2]-a[:,:,2][mask].mean()),
            np.ma.masked_where(~interior,curl(a,x,y))])
    limits=[max(float(np.max(np.abs(v[j]))) for v in derived) for j in range(3)]
    fig=plt.figure(figsize=(17,9))
    grid=fig.add_gridspec(5,6,left=.17,right=.97,top=.87,bottom=.13,
                         width_ratios=[20,1,20,1,20,1],wspace=.30,hspace=.64)
    axs=np.empty((5,3),object);ims=[]
    for i,(a,vals,label) in enumerate(zip(arrays,derived,labels)):
        for j in range(3):
            ax=fig.add_subplot(grid[i,j*2]);axs[i,j]=ax
            cm=plt.get_cmap('viridis' if j==0 else 'RdBu_r').copy();cm.set_bad('#dadde2')
            im=ax.pcolormesh(x,y,vals[j],shading='nearest',cmap=cm,
                             vmin=0 if j==0 else -limits[j],vmax=limits[j])
            if i==0:ims.append(im)
            if j==0:
                ax.streamplot(np.linspace(x[0],x[-1],len(x)),np.linspace(y[0],y[-1],len(y)),
                    np.ma.masked_where(~mask,a[:,:,0]),np.ma.masked_where(~mask,a[:,:,1]),
                    color='white',density=.85,linewidth=.4,arrowsize=.5)
                ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=14,fontsize=9)
            ax.set_aspect('equal');ax.set_facecolor('#dadde2');ax.tick_params(labelsize=8)
            if j:ax.tick_params(labelleft=False)
            if i<4:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('x/H')
            if i==0:ax.set_title(['Speed + streamlines','Mean-centered pressure','Interior vorticity'][j],fontsize=11)
    for j,im in enumerate(ims):fig.colorbar(im,cax=fig.add_subplot(grid[:,2*j+1]))
    fig.suptitle(f'Step-flow geometry g011 | Re = {re}\nCFD and saved seed-17 predictions; shared scales, physical 5:1 aspect',fontsize=15,y=.98)
    fig.text(.5,.055,'Same historical geometry split and reference. Training losses/budgets differ; this is not a controlled architecture ranking.',ha='center',fontsize=10)
    fig.text(.5,.028,'Reverse IoU: all fluid cells with u < 0. Pressure is stored p* = Re p; each field is independently mean-centered.',ha='center',fontsize=9)
    if save:
        fig.savefig(OUT/f'fields_Re{re}.png',dpi=150)
    return fig

def main():
    records=[]
    for re in (25,50,100):
        fields=load_fields(re)
        for key,f in fields.items():
            records.append(dict(Re=re,model=LABELS.get(key,'Transolver'),
                seed=int(key.rsplit('_',1)[1]) if key.startswith('transolver') else 17,
                **case_metrics(f['truth'],f['prediction'],f['mask'])))
        plt.close(build_fields(re))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summaries=[]
    for re in (25,50,100):
        rows=[r for r in records if r['model']=='Transolver' and r['Re']==re]
        summaries.append(dict(Re=re,**{k+suffix:float(fn([r[k] for r in rows]))
             for k in ('velocity_percent','pressure_percent','reverse_iou')
             for suffix,fn in [('_mean',np.mean),('_std',lambda a:np.std(a,ddof=1))]}))
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    inputs=[ARCHIVE,*sorted(OUT.glob('seed_*/*prediction.npz'))]
    provenance={'builder_sha256':sha256(Path(__file__)),'inputs':{str(p.relative_to(ROOT)):sha256(p) for p in inputs},
                'plot_seed':17,'metrics':'recomputed from saved fields; identical truth and fluid masks asserted',
                'no_best_seed_selection':True}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
