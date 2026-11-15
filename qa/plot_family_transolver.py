"""Compact Week 15 figures and metrics from saved double-step predictions."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label
from train_family_transolver import OUT,load_data
from train_week15_vanilla_deeponet import case_metrics,sha256

def filtered_reverse(a,mask,shape):
    v=((a[:,0]<-.01)&mask).reshape(tuple(shape))
    labels,n=label(v) # scipy default: four-connected in 2D
    sizes=np.bincount(labels.ravel());keep=sizes>=8;keep[0]=False
    return keep[labels]

def main():
    d,gids,groups,_=load_data();rows=[]
    for seed in (17,29,43):
        for path in sorted((OUT/f'seed_{seed}').glob('*prediction.npz')):
            with np.load(path) as z:
                a,b=z['truth'],z['prediction'];mask=z['mask'].astype(bool)
                r1,r2=[filtered_reverse(v,mask,z['shape']) for v in (a,b)]
                union=np.count_nonzero(r1|r2)
                rows.append(dict(case=path.stem,seed=seed,Re=int(z['Re']),
                    **case_metrics(a,b,mask),reverse_iou_u_lt_minus_001=float(np.count_nonzero(r1&r2)/union) if union else 1.0))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={}
    for re in (25,50,100):
        rr=[r for r in rows if r['Re']==re]
        summaries[str(re)]={k:dict(mean=float(np.mean([np.mean([r[k] for r in rr if r['seed']==s]) for s in (17,29,43)])),std=float(np.std([np.mean([r[k] for r in rr if r['seed']==s]) for s in (17,29,43)],ddof=1))) for k in ('velocity_percent','pressure_percent','reverse_iou_u_lt_minus_001')}
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    provenance=[]
    for re in (25,50,100):
        paths=[OUT/f'seed_{s}/g049_Re{re}_medium_prediction.npz' for s in (17,29,43)]
        fs=[]
        for p in paths:
            with np.load(p) as z:fs.append({k:z[k].copy() for k in z.files})
        ref=fs[0];shape=tuple(ref['shape']);mask=ref['mask'].reshape(shape).astype(bool)
        xy=ref['coordinates'].reshape(*shape,2);x,y=xy[0,:,0],xy[:,0,1]
        arrays=[ref['truth']]+[f['prediction'] for f in fs]
        fig,axes=plt.subplots(4,1,figsize=(9,7.3));fig.subplots_adjust(left=.085,right=.97,top=.88,bottom=.14,hspace=.63)
        lim=max(float(np.max(np.hypot(a[:,0],a[:,1])[ref['mask'].astype(bool)])) for a in arrays)
        for i,(ax,arr) in enumerate(zip(axes,arrays)):
            a=arr.reshape(*shape,3);v=np.ma.masked_where(~mask,np.hypot(a[:,:,0],a[:,:,1]))
            cm=plt.get_cmap('viridis').copy();cm.set_bad('#dce1e7')
            im=ax.pcolormesh(x,y,v,shading='nearest',cmap=cm,vmin=0,vmax=lim)
            ax.streamplot(np.linspace(x[0],x[-1],len(x)),np.linspace(y[0],y[-1],len(y)),np.ma.masked_where(~mask,a[:,:,0]),np.ma.masked_where(~mask,a[:,:,1]),color='white',density=.9,linewidth=.55,arrowsize=.6)
            ax.set(xlim=(0,5),ylim=(0,1),aspect='equal',ylabel='y/H');ax.tick_params(labelsize=9)
            if i==0:title='CFD reference'
            else:
                rr=next(r for r in rows if r['case']==paths[i-1].stem and r['seed']==(17,29,43)[i-1])
                title=f'Transolver | seed {(17,29,43)[i-1]} | velocity error {rr["velocity_percent"]:.1f}% | IoU {rr["reverse_iou_u_lt_minus_001"]:.3f}'
            ax.set_title(title,loc='left',fontsize=11,weight='bold')
            if i<3:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('x/H',fontsize=10)
        fig.suptitle(f'Week 15: double-step test g049 | Re = {re}',fontsize=15,y=.97)
        cax=fig.add_axes([.25,.065,.55,.016]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=8);cb.set_label('Speed (stored units); shared scale',fontsize=9)
        fig.text(.5,.008,'Original Transolver • 100/8/19 split • g005 quarantined • 19,200 updates/seed • retrospective test',ha='center',fontsize=8)
        fig.savefig(OUT/f'g049_Re{re}.png',dpi=150);plt.close(fig)
        provenance.append(dict(figure=f'g049_Re{re}.png',inputs={str(p.relative_to(OUT)):sha256(p) for p in paths}))
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))

def comparison():
    """Use hash-verified archived tuned fields alongside the new seed-17 model."""
    import hashlib
    for re in (50,100):
        refs=OUT/'reference'
        paths=[next(refs.glob(f'{m}*g049_Re{re}.npz')) for m in ('geom_author','smart_author','domino_author')]
        paths.append(OUT/f'seed_17/g049_Re{re}_medium_prediction.npz')
        fields=[]
        for p in paths:
            with np.load(p) as z:fields.append({k:z[k].copy() for k in z.files})
        ref=fields[-1];shape=tuple(ref['shape']);mask=ref['mask'].reshape(shape).astype(bool)
        arrays=[ref['truth']]+[f['prediction'].reshape(-1,3) for f in fields]
        for f in fields:
            np.testing.assert_array_equal(f['mask'].reshape(-1),ref['mask'].reshape(-1))
            t=f['truth'].reshape(-1,3);t0=ref['truth'];fluid=ref['mask'].astype(bool)
            np.testing.assert_allclose(t[:,:2],t0[:,:2],rtol=1e-6,atol=1e-6)
            # Archived pressure uses a different constant gauge; compare centered fluid pressure.
            np.testing.assert_allclose(t[fluid,2]-t[fluid,2].mean(),t0[fluid,2]-t0[fluid,2].mean(),rtol=1e-5,atol=1e-6)
        xy=ref['coordinates'].reshape(*shape,2);x,y=xy[0,:,0],xy[:,0,1]
        lim=max(float(np.max(np.hypot(a[:,0],a[:,1])[ref['mask'].astype(bool)])) for a in arrays)
        fig=plt.figure(figsize=(9,11.4));axes=[fig.add_axes([.085,.77-i*.16,.885,.14]) for i in range(5)]
        names=['CFD reference','Tuned Geom-DeepONet','Tuned SMART','Tuned DoMINO','Original Transolver (new)']
        for i,(ax,arr,name) in enumerate(zip(axes,arrays,names)):
            a=arr.reshape(*shape,3);cm=plt.get_cmap('viridis').copy();cm.set_bad('#dce1e7')
            im=ax.pcolormesh(x,y,np.ma.masked_where(~mask,np.hypot(a[:,:,0],a[:,:,1])),shading='nearest',cmap=cm,vmin=0,vmax=lim)
            ax.streamplot(np.linspace(x[0],x[-1],len(x)),np.linspace(y[0],y[-1],len(y)),np.ma.masked_where(~mask,a[:,:,0]),np.ma.masked_where(~mask,a[:,:,1]),color='white',density=.85,linewidth=.5,arrowsize=.6)
            ax.set(xlim=(0,5),ylim=(0,1),aspect='equal',ylabel='y/H');ax.tick_params(labelsize=9)
            if i:
                m=case_metrics(ref['truth'],arr,ref['mask']);r1,r2=[filtered_reverse(v,ref['mask'].astype(bool),shape) for v in (ref['truth'],arr)]
                iou=np.count_nonzero(r1&r2)/max(1,np.count_nonzero(r1|r2))
                name+=f' | error {m["velocity_percent"]:.1f}% | IoU {iou:.3f}'
            ax.set_title(name,loc='left',fontsize=12,weight='bold',pad=5)
            if i<4:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('x/H')
        fig.suptitle(f'Week 15 | double-step g049 | Re = {re} | seed 17',fontsize=14,y=.97)
        cax=fig.add_axes([.25,.055,.55,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=8);cb.set_label('Speed; shared scale',fontsize=9)
        fig.text(.5,.004,'Same 100/8/19 split; validation-selected checkpoints. Training losses/tuning differ. Retrospective test.',ha='center',fontsize=8)
        fig.savefig(OUT/f'comparison_g049_Re{re}.png',dpi=150);plt.close(fig)
        (OUT/f'comparison_g049_Re{re}_provenance.json').write_text(json.dumps({str(p.relative_to(OUT)):sha256(p) for p in paths},indent=2)+'\n')

if __name__=='__main__':
    main()
    comparison()
