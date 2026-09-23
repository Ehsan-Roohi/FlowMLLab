"""Rebuild the retained tables, figures, and release checks from actual CFD."""
from pathlib import Path
import json,hashlib
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cfd import radius, VOLUME
from learning import ROOT,E,Surrogate
from analyze import load_csv
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def metrics(name):return json.loads((E/'runs'/name/'metrics.json').read_text())
def extracted(name):return np.load(E/'runs'/name/'extracted.npz')
def save(fig,name):
    buf=BytesIO();fig.savefig(buf,format='png',dpi=180,bbox_inches='tight');(E/(name+'.png')).write_bytes(buf.getvalue());plt.close(fig)
def main():
    data=np.load(E/'dataset.npz');design=json.loads((E/'design_candidates.json').read_text());learning=json.loads((E/'learning_metrics.json').read_text())
    base=metrics('baseline_fine_stable');fine=metrics('baseline_finer');large=metrics('baseline_large_domain');opt=metrics('optimized_fine');optfine=metrics('optimized_finer')
    benchmark=json.loads((E/'runs/cone_benchmark/benchmark.json').read_text())
    b=extracted('baseline_fine_stable');f=extracted('baseline_finer');o=extracted('optimized_fine');of=extracted('optimized_finer')
    validation={'baseline_peak_mesh_difference':abs(base['peak_cp']/fine['peak_cp']-1),'baseline_drag_mesh_difference':abs(base['cd_pressure']/fine['cd_pressure']-1),'baseline_wave_mesh_difference':float(np.linalg.norm(b['cp'][1]-f['cp'][1])/np.linalg.norm(f['cp'][1])),'baseline_domain_peak_difference':abs(large['peak_cp']/base['peak_cp']-1),'optimized_peak_mesh_difference':abs(opt['peak_cp']/optfine['peak_cp']-1),'optimized_drag_mesh_difference':abs(opt['cd_pressure']/optfine['cd_pressure']-1),'peak_reduction_fine':1-opt['peak_cp']/base['peak_cp'],'peak_reduction_finer':1-optfine['peak_cp']/fine['peak_cp'],'drag_change_fine':opt['cd_pressure']/base['cd_pressure']-1,'drag_change_finer':optfine['cd_pressure']/fine['cd_pressure']-1}
    rows=[]
    for c in design['candidates']:
        actual=metrics(c['name']+'_fine');rows.append(dict(name=c['name'],a=c['a'],b=c['b'],predicted_peak_cp=c['predicted_peak_cp'],actual_peak_cp=actual['peak_cp'],predicted_cd=c['predicted_cd'],actual_cd=actual['cd_pressure'],peak_reduction=1-actual['peak_cp']/base['peak_cp'],constraint_pass=actual['cd_pressure']<=design['drag_limit']))
    pd.DataFrame(rows).to_csv(E/'design_comparison.csv',index=False)
    pd.DataFrame(learning['comparison']).drop(columns=['warnings']).to_csv(E/'model_comparison.csv',index=False)
    dataset_rows=[metrics(str(n)) for n in data['names']]
    pd.DataFrame(dataset_rows).to_csv(E/'case_metrics.csv',index=False)
    shock={}
    for mach in [1.7,1.8,1.9]:
        if mach==1.8:bm,om=base,opt
        else:bm,om=metrics(f'baseline_M{mach}'),metrics(f'optimized_M{mach}')
        shock[str(mach)]={'peak_reduction':1-om['peak_cp']/bm['peak_cp'],'drag_change':om['cd_pressure']/bm['cd_pressure']-1}
    # Independent frustum volume calculation on the actual body mesh.
    meshvol={}
    for name in ['baseline_fine_stable','optimized_fine']:
        d=extracted(name);r=d['surface_r'];x=d['surface_x'];v=np.pi/3*np.sum(np.diff(x)*(r[:-1]**2+r[:-1]*r[1:]+r[1:]**2));meshvol[name]=float(abs(v/VOLUME-1))
    selected=[r for r in learning['comparison'] if r['model']==learning['selected_model'] and r['split']=='test'][0]
    gates={
      'cone_pressure_within_3pct':benchmark['passed'],
      'all_44_dataset_cases_converged':len(dataset_rows)==44 and all(r['converged'] for r in dataset_rows),
      'mesh_volume_within_0p2pct':max(meshvol.values())<.002,
      'baseline_peak_mesh_within_5pct':validation['baseline_peak_mesh_difference']<.05,
      'baseline_drag_mesh_within_3pct':validation['baseline_drag_mesh_difference']<.03,
      'domain_peak_within_1pct':validation['baseline_domain_peak_difference']<.01,
      'optimized_peak_mesh_within_5pct':validation['optimized_peak_mesh_difference']<.05,
      'optimized_drag_mesh_within_3pct':validation['optimized_drag_mesh_difference']<.03,
      'design_recomputations_converged':opt['converged'] and optfine['converged'],
      'optimized_drag_constraint_fine':validation['drag_change_fine']<=.02,
      'optimized_drag_constraint_finer':validation['drag_change_finer']<=.02,
      'peak_reduction_exceeds_5pct_both_meshes':min(validation['peak_reduction_fine'],validation['peak_reduction_finer'])>.05,
      'selected_model_test_peak_error_below_10pct':selected['peak_mean_relative_error']<.10,
    }
    summary={'scope':'Axisymmetric Euler near-field pressure optimization; no atmospheric propagation or PLdB claim','validation':validation,'benchmark':benchmark,'mach_stress_tests':shock,'volume_relative_error':meshvol,'models':learning,'designs':rows,'dataset_sha256':hashlib.sha256((E/'dataset.npz').read_bytes()).hexdigest(),'dataset_count':len(dataset_rows),'cfd_wall_seconds_sum':float(sum(r['wall_seconds'] for r in dataset_rows))}
    (E/'summary.json').write_text(json.dumps(summary,indent=2));(E/'release_check.json').write_text(json.dumps({'scientific_checks':gates,'scientific_pass':all(gates.values()),'notebook_executed':False,'lecture_visual_review':False,'release_ready':False},indent=2))
    x=np.linspace(0,1,501)
    fig,ax=plt.subplots(2,1,figsize=(9,5),gridspec_kw={'height_ratios':[1,1.6]})
    for i,row in enumerate(rows[:3]):
        r=radius(x,row['a'],row['b']);ax[0].plot(x,r,label=row['name']);ax[0].plot(x,-r,color=ax[0].lines[-1].get_color());d=extracted(row['name']+'_fine');ax[1].plot(d['x'],d['cp'][1],label=row['name'])
    rb=radius(x);ax[0].plot(x,rb,'k--',label='baseline');ax[0].plot(x,-rb,'k--');ax[0].set_aspect('equal');ax[0].set(xlabel='x/L',ylabel='r/L',title='Fixed length and volume: actual shape proportions');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,1.6),ncol=4,fontsize=8)
    ax[1].plot(b['x'],b['cp'][1],'k--',label='baseline');ax[1].set(xlabel='x/L',ylabel='Cp',title='Recomputed SU2 pressure signatures at r/L=0.5');ax[1].legend(ncol=4,fontsize=8);fig.tight_layout();save(fig,'design_shapes_signatures')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for name,label in [('baseline_medium','20k cells'),('baseline_fine_stable','45k cells'),('baseline_finer','81k cells')]:
        d=extracted(name);axs[0].plot(d['x'],d['cp'][1],label=label)
        h=load_csv(E/'runs'/name/'history.csv');axs[1].plot(h['Inner_Iter'],h['rms[Rho]'],label=label)
    axs[0].set(xlabel='x/L',ylabel='Cp',title='Mesh sensitivity');axs[1].set(xlabel='Iteration',ylabel='log10 density residual',title='Discrete convergence');[a.legend(fontsize=8) for a in axs];fig.tight_layout();save(fig,'numerical_verification')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for sp,mark in [('train','o'),('validation','s'),('test','^'),('extrapolation','x')]:
        mask=data['splits']==sp;axs[0].scatter(*data['parameters'][mask].T,label=sp,marker=mark,s=30)
    axs[0].set(xlabel='a: fore-aft redistribution',ylabel='b: shape redistribution',title='Frozen geometry split');axs[0].legend(fontsize=8)
    mask=data['splits']=='train';model=Surrogate(learning['selected_model']).fit(data['parameters'][mask],data['waveforms'][mask],data['cd'][mask])
    for sp,mark in [('test','o'),('extrapolation','x')]:
        mask=data['splits']==sp;pred,cd=model.predict(data['parameters'][mask]);axs[1].scatter(data['waveforms'][mask].max(axis=1),pred.max(axis=1),label=sp,marker=mark)
    lim=[.015,.075];axs[1].plot(lim,lim,'k--');axs[1].set(xlabel='CFD peak Cp',ylabel='Predicted peak Cp',title=f"{learning['selected_model'].upper()}: peak prediction");axs[1].legend();fig.tight_layout();save(fig,'dataset_learning')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for name,label in [('baseline_fine_stable','baseline'),('optimized_fine','optimized')]:
        d=extracted(name)
        for j,r in enumerate(d['radii']):axs[0].plot(d['x'],d['cp'][j],label=f'{label}, r/L={r}',ls='--' if label=='baseline' else '-')
    axs[0].set(xlabel='x/L',ylabel='Cp',title='Different extraction radii');axs[0].legend(fontsize=7,ncol=2)
    ms=[1.7,1.8,1.9];axs[1].plot(ms,[100*shock[str(m)]['peak_reduction'] for m in ms],'o-',label='Peak reduction');axs[1].plot(ms,[100*shock[str(m)]['drag_change'] for m in ms],'s-',label='Drag change');axs[1].axhline(0,color='gray',lw=.6);axs[1].set(xlabel='Mach number (fresh CFD)',ylabel='Change [%]',title='Off-design checks');axs[1].legend(fontsize=8);fig.tight_layout();save(fig,'condition_checks')
    # Actual computed field and actual Gmsh quads, no illustrative CFD.
    import meshio
    fig,axs=plt.subplots(2,1,figsize=(10,6.3),constrained_layout=True)
    for ax,name,label in zip(axs,['baseline_fine_stable','optimized_fine'],['Baseline','Optimized']):
        u=load_csv(E/'runs'/name/'restart_flow.csv');q=.5*1.4*101325*1.8**2;p=.4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/u['Density']);cp=(p-101325)/q
        import matplotlib.tri as tri
        mesh=meshio.read(E/'runs'/name/'mesh.msh');cells=mesh.cells_dict['quad'];pts=mesh.points
        # Coordinate lookup avoids any dependency on mesh file point order.
        from scipy.spatial import cKDTree
        dist,ix=cKDTree(np.c_[u['x'],u['y']]).query(pts[:,:2]);assert max(dist)<1e-8
        triangles=np.vstack([cells[:,[0,1,2]],cells[:,[0,2,3]]]);t=tri.Triangulation(pts[:,0],pts[:,1],triangles)
        c=ax.tripcolor(t,cp[ix],shading='gouraud',cmap='RdBu_r',vmin=-.1,vmax=.1,rasterized=True)
        mm=json.loads((E/'runs'/name/'metadata.json').read_text());rr=radius(x,mm['a'],mm['b']);ax.fill_between(x,0,rr,color='#222222');ax.axhline(.5,color='gray',lw=.7,ls='--');ax.set(xlim=(-.15,2.4),ylim=(0,1),aspect='equal',ylabel='r/L',xlabel='x/L',title=f'{label}: computed meridional Cp field')
    fig.colorbar(c,ax=axs,label='Cp',fraction=.025,pad=.02);save(fig,'cfd_fields')
    print(json.dumps({'validation':validation,'gates':gates},indent=2))
if __name__=='__main__':main()
