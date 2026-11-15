"""Import author-authorized frozen fields, verify hashes and reproduce metrics.

The private bundle path is an explicit CLI input; it is not a runtime notebook
dependency. Only selected sampled fields and compact metadata are imported.
Original checkpoints, sources, optimizer/RNG state and archives stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from flowmllab.deep_cavity_audit import load_bundle, evaluate, horizontal_profile

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
    src=args.source;dest=ROOT/'data/week13_d5';dest.mkdir(parents=True,exist_ok=True)
    out=ROOT/'results/week13_d5';out.mkdir(parents=True,exist_ok=True)
    original=src/'data/reynolds_common_fields.npz'
    expected='5e86d355262cda07cd341381de0697b958b04ffe7e6b096ae4a6c2d878e27ed3'
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    if digest!=expected: raise ValueError('Original author field hash differs')
    with np.load(original,allow_pickle=False) as z:
        keep={k:z[k] for k in z.files if not k.endswith('_speed')}
    np.savez_compressed(dest/original.name,**keep)
    metadata=json.loads((src/'historical_analysis/reynolds_analysis.json').read_text())
    manifest={'data_sha256':hashlib.sha256((dest/original.name).read_bytes()).hexdigest(),
              'original_sha256':digest,'original_bytes':original.stat().st_size,
              'article':'https://arxiv.org/abs/2610.38814','authors':['Ehsan Roohi','Christopher J. McDevitt'],
              'grid_shape_h_x':[605,121],'aspect_ratio_D_W':5,
              'lid':'Gaussian regularized; width sqrt(1e-3); not the discontinuous Ghia lid',
              'coordinates':'x/W and h/W above floor, with 0.002 offset from each wall',
              'pressure':'frozen common whole-interior gauge, preserved without recentering',
              'vorticity':'retained common-grid sampled finite differences, not native AD/spectral derivatives',
              'CFD_times':{str(re):metadata['cases'][str(re)]['CFD_time'] for re in (100,250,500,1000)},
              'workflow':'one retained trajectory per Re; Re1000 is control, not refined; unequal prior histories',
              'transform':'removed redundant speed arrays only; all retained arrays byte-identical',
              'claim':'frozen-field reproduction, not a complete training or CFD replay archive',
              'array_sha256':{k:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k,a in keep.items()}}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    data,_=load_bundle(dest)
    for k in keep: np.testing.assert_array_equal(data[k],keep[k])
    rows=evaluate(data)
    metrics_path=src/'reference_metrics/matched_region_metrics_20261030.json'
    original_metrics=json.loads(metrics_path.read_text())
    frozen=original_metrics['common_series']
    # The separate 161x801 workflow assessment is retained evidence. It is not
    # recomputed from the 121x605 common bundle or mislabeled as a new trial.
    workflow_rows=[{'workflow':name,'region':region,**values}
        for name,regions in original_metrics['detailed_workflows'].items()
        for region,values in regions.items()]
    pd.DataFrame(workflow_rows).to_csv(out/'workflow_metrics.csv',index=False)
    (out/'workflow_provenance.json').write_text(json.dumps({
        'source_sha256':hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        'source':'authors assessment_bundle_r1/reference_metrics/matched_region_metrics_20261030.json',
        'source_section':'detailed_workflows','sampling_grid_x_h':[161,801],
        'article':'https://arxiv.org/abs/2610.38814v2',
        'scope':'Exact retained metric values; not recomputed from the different common-grid bundle'},indent=2)+'\n')
    checks=0;roundoff=[]
    for row in rows:
        for key in ('count','velocity_RMS_U','CFD_velocity_RMS_U','velocity_NRMSE_percent',
                    'pressure_RMS_rhoU2','pressure_signed_mean_offset_rhoU2','pressure_local_shape_RMS_rhoU2','vorticity_RMS_UoverW'):
            # A globally gauged pressure mean can be ~1e-17. Reduction-order
            # roundoff then dominates relative error; scale its absolute bound
            # by the regional pressure discrepancy, not by an arbitrary floor.
            atol=(8*np.finfo(float).eps*row['pressure_RMS_rhoU2']
                  if key=='pressure_signed_mean_offset_rhoU2' else 1e-24)
            target=frozen[str(row['Re'])][row['region']][key]
            np.testing.assert_allclose(row[key],target,rtol=5e-13,atol=atol)
            if key=='pressure_signed_mean_offset_rhoU2':
                roundoff.append({'Re':row['Re'],'region':row['region'],
                    'absolute_difference':abs(row[key]-target),'absolute_bound':atol})
            checks+=1
    pd.DataFrame(rows).to_csv(out/'regional_metrics.csv',index=False)
    (out/'verification.json').write_text(json.dumps({'numeric_values_reproduced':checks,'original_sha256':digest,
        'data_sha256':manifest['data_sha256'],'all_retained_arrays_exact':True,
        'pressure_mean_roundoff':roundoff,
        'original_unchanged':hashlib.sha256(original.read_bytes()).hexdigest()==digest},indent=2)+'\n')
    for re in (100,250,500,1000):
        fig,axs=plt.subplots(3,3,figsize=(11,9))
        for col,height in enumerate((.1,.5,4.8)):
            for row,component in enumerate(('u','p','omega')):
                for method,color,style in [('cfd','black','-'),('pinn','#006db0','--')]:
                    x,vals,actual=horizontal_profile(data,re,method,component,height)
                    axs[row,col].plot(x,vals,style,color=color,label='Nektar++ CFD' if method=='cfd' else 'Retained PINN')
                if row==0:axs[row,col].set_title(f'h/W = {actual:.4f}')
                axs[row,col].set_xlabel('x/W');axs[row,col].grid(alpha=.18)
                axs[row,col].ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
                axs[row,col].set_ylabel({'u':'u/U','p':'p / (rho U²), global gauge','omega':'omega W/U'}[component])
        fig.suptitle(f'D/W = 5, Re = {re}: retained physical profiles',y=.995)
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
        fig.subplots_adjust(top=.86,bottom=.07,left=.08,right=.98,wspace=.42,hspace=.55)
        fig.savefig(out/f're{re}_profiles.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(7.4,8.8))
        speeds=[np.hypot(data[f're{re}_{m}_u'],data[f're{re}_{m}_v']) for m in ('cfd','pinn')]
        vmax=max(float(a.max()) for a in speeds)
        # Same full extrema and nonlinear display transform for the two fields.
        for ax,method,sp in zip(axs[:2],('cfd','pinn'),speeds):
            im=ax.pcolormesh(data['x'],data['h'],sp,shading='nearest',cmap='turbo',norm=SymLogNorm(1e-8,vmin=0,vmax=vmax))
            ax.streamplot(data['x'],data['h'],data[f're{re}_{method}_u'],data[f're{re}_{method}_v'],
                          color='black',density=.85,linewidth=.45,arrowsize=.5)
            ax.set_title('CFD' if method=='cfd' else 'PINN')
        du=data[f're{re}_pinn_u']-data[f're{re}_cfd_u'];bound=float(np.max(np.abs(du)))
        er=axs[2].pcolormesh(data['x'],data['h'],du,shading='nearest',cmap='RdBu_r',vmin=-bound,vmax=bound)
        axs[2].set_title('Signed u discrepancy')
        for ax in axs:
            ax.set_aspect('equal');ax.set_xlabel('x/W');ax.set_ylabel('h/W')
            ax.set_xlim(0,1);ax.set_ylim(0,5)
        fig.subplots_adjust(top=.92,bottom=.17,wspace=.75)
        cbax=fig.add_axes([.10,.07,.47,.018]);fig.colorbar(im,cax=cbax,orientation='horizontal').set_label('Speed / U (symmetric-log scale)')
        cbax=fig.add_axes([.69,.07,.23,.018]);fig.colorbar(er,cax=cbax,orientation='horizontal').set_label('u difference / U (linear)')
        fig.suptitle(f'D/W = 5, Re = {re}',y=.985)
        fig.savefig(out/f're{re}_fields.png',dpi=160);plt.close(fig)
    print('Verified',checks,'retained regional numbers and all arrays',flush=True)


if __name__=='__main__':main()
