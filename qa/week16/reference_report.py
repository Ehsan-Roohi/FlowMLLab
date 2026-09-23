"""Compare actual SU2 SEEB-ALR runs with unchanged NASA reference records."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib.pyplot as plt
from nasa_reference_data import load_reference
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/week16_lowboom/reference'

def build_report(write=True):
    experiments,lava=load_reference()
    runs=[]
    for p in R.glob('seeb_level_*/signature.npz'):
        m=json.loads((p.parent/'metadata.json').read_text())
        if 'converged' not in m:continue
        with np.load(p) as a: x=a['x_inches'].copy();y=a['dp_pinf'].copy()
        assert np.all(np.diff(x)>0) and np.isfinite(y).all()
        rows={}
        for label,e in experiments.items():
            mask=(e['x']>=25)&(e['x']<=46);xx=e['x'][mask];yy=e['pressure'][mask];unc=e['uncertainty'][mask]
            assert xx.min()>=x.min() and xx.max()<=x.max()
            pred=np.interp(xx,x,y)
            rows[label]={'wave_relative_l2':float(np.linalg.norm(pred-yy)/np.linalg.norm(yy)), 'peak_relative_error':float(abs(pred.max()/yy.max()-1)), 'fraction_in_supplied_uncertainty_band':float(np.mean(abs(pred-yy)<=unc)), 'points':len(xx)}
        runs.append({'level':m['level'],'cells':m['cells'],'converged':m['converged'],'metadata':m,'experimental_metrics':rows,'signature_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'x':x,'y':y})
    runs.sort(key=lambda z:z['level'])
    if not runs:raise RuntimeError('No completed SEEB-ALR runs. Execute seeb_reference.py first.')
    last=runs[-1];change=None
    if len(runs)>1:
        prev=runs[-2];mask=(last['x']>=25)&(last['x']<=46);yy=last['y'][mask]
        change=float(np.linalg.norm(np.interp(last['x'][mask],prev['x'],prev['y'])-yy)/np.linalg.norm(yy))
    checks={'required_three_mesh_levels': [z['level'] for z in runs]==[1,2,2.5],'all_runs_converged':all(z['converged'] for z in runs),'both_experimental_wave_errors_below_20pct':all(v['wave_relative_l2']<.20 for v in last['experimental_metrics'].values()),'both_experimental_peak_errors_below_10pct':all(v['peak_relative_error']<.10 for v in last['experimental_metrics'].values()),'last_two_mesh_change_below_5pct':change is not None and change<.05}
    result={'case':'NASA SEEB-ALR','mach':1.6,'height_inches':21.2,'window_inches':[25,46],'normalization':'delta_p/p_infinity','alignment':'Original NASA macros only; no fitted position/amplitude/offset','runs':[{k:v for k,v in row.items() if k not in ['x','y']} for row in runs],'last_two_mesh_wave_relative_l2':change,'thresholds':{'experimental_wave_relative_l2_max':.20,'experimental_peak_relative_error_max':.10,'last_two_mesh_wave_relative_l2_max':.05},'checks':checks,'passed':all(checks.values()),'scope':'Axisymmetric Euler off-body CFD validation, not neural experimental validation or full-aircraft reproduction. Mesh change is observed sensitivity, not formal GCI. Nose cap has two cells at each level.','recovery_note':'These are new runs using the reconstructed script. They supersede the unavailable earlier raw NASA runs; no earlier numbers are substituted.'}
    if write:
        # Geometry figure can be rebuilt from the retained CAD sampling alone.
        with np.load(R/f"seeb_level_{runs[0]['level']:g}"/'cad_meridian.npz') as cad:
            gx=cad['x'];gr=cad['r']
        gf,ga=plt.subplots(2,1,figsize=(9,5),gridspec_kw={'height_ratios':[2,1]})
        ga[0].plot(gx,gr,color='black');ga[0].plot(gx,-gr,color='black')
        ga[0].axhline(21.2/17.667,color='tab:red',ls='--',label='Pressure line H/L=21.2/17.667')
        ga[0].axhline(0,color='gray',lw=.6);ga[0].set(xlabel='x/L',ylabel='r/L',title='Original NASA meridian; equal physical aspect ratio');ga[0].set_aspect('equal',adjustable='datalim');ga[0].legend(fontsize=8)
        mask=gx<.015
        ga[1].plot(gx[mask],gr[mask],color='black');ga[1].plot([gx[0],gx[0]],[0,gr[0]],color='tab:orange',lw=2,label='Finite nose cap')
        ga[1].set(xlabel='x/L',ylabel='r/L',title='Nose inset (independent axis scaling)');ga[1].legend(fontsize=8)
        gf.tight_layout();gf.savefig(R/'seeb_geometry.png',dpi=180);plt.close(gf)
        (R/'seeb_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        fig,axs=plt.subplots(3,1,figsize=(9,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1]})
        for label,e in experiments.items():
            line,=axs[0].plot(e['x'],e['pressure'],label=label,lw=1.2)
            axs[0].fill_between(e['x'],e['pressure']-e['uncertainty'],e['pressure']+e['uncertainty'],color=line.get_color(),alpha=.12)
            mask=(e['x']>=25)&(e['x']<=46);x=e['x'][mask]
            axs[1].plot(x,np.interp(x,last['x'],last['y'])-e['pressure'][mask],label='SU2 minus '+label)
        axs[0].plot(lava['x'],lava['pressure'],'k--',label='NASA-hosted LAVA, 330k',lw=1.2)
        axs[0].plot(last['x'],last['y'],color='#a52775',label=f"SU2 Euler, {last['cells']:,} cells",lw=1.6)
        for row in runs:axs[2].plot(row['x'],row['y'],label=f"SU2 {row['cells']:,}",lw=1.2)
        axs[0].set(ylabel=r'$\Delta p/p_\infty$',title='SEEB-ALR: independent experimental and numerical comparisons')
        axs[1].set(ylabel='Signed pressure difference');axs[1].axhline(0,color='gray',lw=.7)
        axs[2].set(xlabel='Source-aligned x [inches]',ylabel=r'$\Delta p/p_\infty$')
        for ax in axs:ax.set_xlim(25,46);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(R/'seeb_validation.png',dpi=180);plt.close(fig)
    return result

if __name__=='__main__':
    z=build_report();print(json.dumps({'checks':z['checks'],'last_two_mesh_wave_relative_l2':z['last_two_mesh_wave_relative_l2'],'finest':z['runs'][-1]['experimental_metrics'],'passed':z['passed']},indent=2))
