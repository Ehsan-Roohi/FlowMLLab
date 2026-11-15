"""Restore every retained model in compact rainbow-color Week 15 comparisons."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from plot_step_transolver import load_fields
from plot_family_transolver import filtered_reverse
from train_week15_vanilla_deeponet import case_metrics,sha256
ROOT=Path(__file__).resolve().parents[1]

def draw(path,title,ref,models,filtered=False):
    shape=tuple(ref['shape']);mask=ref['mask'].astype(bool).reshape(shape)
    xy=ref['coordinates'].reshape(*shape,2);x,y=xy[0,:,0],xy[:,0,1]
    records=[('CFD reference',ref['truth'])]+models
    rows=(len(records)+1)//2
    fig=plt.figure(figsize=(12,rows*1.72+2.05),facecolor='white')
    height=rows*1.72+2.05
    values=[np.hypot(a[:,0],a[:,1]) for _,a in records]
    vmax=max(float(v[mask.ravel()].max()) for v in values)
    levels=np.linspace(0,vmax,15);cm=plt.get_cmap('jet').copy();cm.set_bad('#595959');norm=BoundaryNorm(levels,cm.N)
    metrics=[]
    for k,((name,arr),v) in enumerate(zip(records,values)):
        # Read rows left-to-right; no panel omitted or hidden.
        row,col=divmod(k,2)
        bottom=(height-.85-(row+1)*1.72)/height
        ax=fig.add_axes([.065+col*.49,bottom,.425,1.02/height])
        a=arr.reshape(*shape,3)
        im=ax.pcolormesh(x,y,np.ma.masked_where(~mask,v.reshape(shape)),shading='nearest',cmap=cm,norm=norm)
        ax.streamplot(np.linspace(x[0],x[-1],len(x)),np.linspace(y[0],y[-1],len(y)),np.ma.masked_where(~mask,a[:,:,0]),np.ma.masked_where(~mask,a[:,:,1]),color='white',density=.8,linewidth=.45,arrowsize=.5)
        ax.set(xlim=(0,5),ylim=(0,1),aspect='equal',ylabel='y/H',xlabel='x/H')
        ax.set_yticks([0,.5,1])
        if k+2<len(records):
            ax.set_xlabel('');ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=9);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10)
        ax.set_title(name,loc='left',fontsize=12,weight='bold',pad=18)
        if k:
            m=case_metrics(ref['truth'],arr,mask.ravel())
            if filtered:
                a1,a2=[filtered_reverse(a,mask.ravel(),shape) for a in (ref['truth'],arr)]
                iou=np.count_nonzero(a1&a2)/max(1,np.count_nonzero(a1|a2))
            else:iou=m['reverse_iou']
            ax.text(0,1.05,f'Velocity error {m["velocity_percent"]:.1f}%  |  reverse IoU {iou:.3f}',transform=ax.transAxes,fontsize=10)
            metrics.append(dict(model=name,velocity_percent=float(m['velocity_percent']),reverse_iou=float(iou)))
    fig.suptitle(title,fontsize=16,y=1-.11/height)
    cax=fig.add_axes([.3,.55/height,.4,.11/height]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=levels[::2]);cb.ax.tick_params(labelsize=9)
    fig.text(.5,.08/height,'Speed: shared banded RGB/rainbow scale. Each model uses its own streamlines. Seed 17; training protocols differ.',ha='center',fontsize=9)
    fig.savefig(path,dpi=150);plt.close(fig)
    return metrics

def main():
    fields=load_fields(100);ref=fields['geom']
    models=[(n,fields[k]['prediction']) for n,k in [('Geo-DeepONet','geom'),('FNO','fno'),('U-FNO','ufno'),('Original Transolver','transolver_17')]]
    p=ROOT/'results/step_transolver/all_models_g011_Re100.png'
    m1=draw(p,'Week 15 | central-obstacle channel g011 | Re = 100',ref,models)
    refs=ROOT/'results/family_transolver/reference'
    labels=[('Historical ordinary DeepONet','deeponet_seed17'),('Historical Geom-DeepONet','geom_seed17'),('Historical Geo-FNO','geofno_seed17'),('LR-tuned Geom-DeepONet','geom_author_fresh_lr1e-3_seed17_best_predictions'),('LR-tuned SMART','smart_author_lr1e-3_seed17_best_predictions'),('LR-tuned DoMINO','domino_author_lr1e-3_seed17_best_predictions'),('Matched GeoTransolver','geotransolver_author_seed17'),('PhysicsX-inspired proxy','px_context_proxy_seed17'),('LIFT wall-feature proxy','lift_wall_proxy_seed17')]
    paths=[refs/(key+'_g049_Re100.npz') for _,key in labels]
    paths.append(ROOT/'results/family_transolver/seed_17/g049_Re100_medium_prediction.npz')
    with np.load(paths[-1]) as z:ref={k:z[k].copy() for k in z.files}
    models=[];fluid=ref['mask'].astype(bool)
    for (name,_),path in zip(labels+[('Original Transolver','')],paths):
        with np.load(path) as z:
            np.testing.assert_array_equal(z['mask'].ravel(),ref['mask'].ravel())
            t=z['truth'].reshape(-1,3);t0=ref['truth']
            np.testing.assert_allclose(t[:,:2],t0[:,:2],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(t[fluid,2]-t[fluid,2].mean(),t0[fluid,2]-t0[fluid,2].mean(),rtol=1e-5,atol=1e-6)
            models.append((name,z['prediction'].reshape(-1,3).copy()))
    m2=draw(ROOT/'results/family_transolver/all_models_g049_Re100.png','Week 15 | double-step g049 | Re = 100 | all retained models',ref,models,True)
    provenance={'builder_sha256':sha256(Path(__file__)),'family_inputs':{str(p.relative_to(ROOT)):sha256(p) for p in paths},'central_obstacle_metrics':m1,'double_step_metrics':m2,'colormap':'jet, 14 discrete speed bands; common full-range scale per figure','original_archived_figures':'retained unchanged in results/week15_postaudit','no_retraining':True}
    (ROOT/'results/family_transolver/all_models_figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
if __name__=='__main__':main()
