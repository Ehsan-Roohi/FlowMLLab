"""Verify retained CPU studies and regenerate physical profiles and tables."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from flowmllab.pressure_velocity import divergence,pressure_matrix,centerlines
from flowmllab.pressure_velocity_studies import velocity_distance
from qa.refresh_week01_2_refinement import source_bytes_match
DEST=ROOT/'results/week01_2_literature'
ASSETS=ROOT/'lectures/source/week01_2_assets'


def load(directory=DEST):
    directory=Path(directory)
    data=json.loads((directory/'study.json').read_text())
    assert hashlib.sha256((directory/'fields.npz').read_bytes()).hexdigest()==data['provenance']['fields_sha256']
    with np.load(directory/'fields.npz',allow_pickle=False) as f:fields={k:f[k] for k in f.files}
    assert all(np.isfinite(v).all() for v in fields.values())
    return data,fields


def time_field(fields,case):
    return {key:fields[f'time_{case}_{key}'] for key in ('u_faces','v_faces','p','u','v','x','y','history')}


def tables(data):
    pressure=pd.DataFrame([{'cells':r['cells'],'linear solver':r['method'],
        'batch milliseconds':1000*r['total_seconds_median'],'setup milliseconds':1000*r['setup_seconds_median'],
        'max divergence':r['max_divergence_linf'],'max p discrepancy vs LU':r['max_pressure_relative_l2_vs_lu'],
        'converged':r['linear_solver_converged'],'GMRES iterations (all RHS)':r['gmres_iterations_all_rhs']}
        for r in data['pressure_rows']])
    temporal=pd.DataFrame([{'case':r['case'],'method':r['config']['method'].upper(),'dt':r['config']['dt'],
        'inner':r['config']['pressure_correctors'],'outer':r['config']['outer_correctors'],
        'velocity discrepancy %':100*r['velocity_discrepancy_vs_fine'],
        'median seconds':r['runtime_seconds_median'],'pressure solves':r['pressure_solves'],
        'last nonlinear step defect':r['last_nonlinear_step_defect'],'divergence':r['continuity_linf']}
        for r in data['time_rows']])
    return pressure,temporal


def audit(data,fields):
    sources={p:source_bytes_match((ROOT/p).read_bytes(),d) for p,d in data['provenance']['source_sha256'].items()}
    fine=time_field(fields,'reference_fine')
    for r in data['time_rows']:
        f=time_field(fields,r['case']);n=r['config']['cells']
        assert abs(f['history'][-1,1]-.2)<1e-10 and abs(r['physical_time']-.2)<1e-10
        mass=float(np.max(np.abs(divergence(f['u_faces'],f['v_faces'],1/n))))
        assert mass<1e-9 and np.isclose(mass,r['continuity_linf'],atol=1e-12)
        assert np.isclose(velocity_distance(f,fine),r['velocity_discrepancy_vs_fine'],rtol=1e-10,atol=1e-14)
        assert len(r['runtime_samples'])==3
    for name,case in [('reference_dt_change','reference_coarse'),('reference_loop_change','reference_loops')]:
        assert np.isclose(velocity_distance(time_field(fields,case),fine),data['reference_checks'][name],rtol=1e-10)
    for r in data['pressure_rows']:
        n=r['cells'];matrix=pressure_matrix(np.full((n,n-1),.05),np.full((n-1,n),.05),1/n)
        for j,diag in enumerate(r['rhs_diagnostics']):
            key=f'pressure_n{n}_{r["method"].replace(" ","_")}_rhs{j}_'
            mass=divergence(fields[key+'u'],fields[key+'v'],1/n)
            b=-divergence(fields[key+'u_star'],fields[key+'v_star'],1/n).ravel()
            residual=b-matrix@fields[key+'p'].ravel()
            np.testing.assert_allclose(mass.ravel(),-residual,atol=2e-11,rtol=1e-8)
            assert np.isclose(np.max(np.abs(mass)),diag['divergence_linf'],atol=1e-12)
    return {'source_checks':sources,'time_cases':len(data['time_rows']),'pressure_rows':len(data['pressure_rows']),
            'field_sha256':data['provenance']['fields_sha256'],'full_pressure_residual_and_divergence_identity':True,
            'same_physical_end_time':.2,'reference_sensitivity_recomputed':True}


def pressure_figure(rows,fields,cells=None,rhs=0):
    """Return pressure and conservation profiles for retained or fresh fields."""
    from qa.refresh_week01_2_figures import plt
    n=max(r['cells'] for r in rows) if cells is None else cells
    fig,ax=plt.subplots(2,1,figsize=(6.6,5.2),layout='constrained')
    selected=[r for r in rows if r['cells']==n and r['method']!='LU each RHS']
    for i,r in enumerate(selected):
        method=r['method'];color=['black','#cc2222','#0055cc','#008844'][i%4];style=['-','--',':','-.'][i%4]
        key=f'pressure_n{n}_'+method.replace(' ','_')+f'_rhs{rhs}_';p=fields[key+'p'];x=(np.arange(n)+.5)/n
        p_center=np.array([np.interp(.5,x,p[:,j]) for j in range(n)])
        ax[0].plot(x,p_center,color=color,ls=style,label=method,lw=1.7)
        ax[1].plot(x,fields[key+'div'][0,:],color=color,ls=style,lw=1.5)
    ax[0].set(xlabel='x/L',ylabel='p (zero mean)')
    ax[1].set(xlabel='x/L',ylabel='Cell divergence',yscale='symlog')
    ax[1].set_yscale('symlog',linthresh=1e-10)
    from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
    locator=SymmetricalLogLocator(base=10,linthresh=1e-10)
    locator.set_params(numticks=7)
    ticks=locator.tick_values(*ax[1].get_ylim())
    # Suppress labels at the tiny linear band; otherwise they crowd zero.
    ticks=[t for t in ticks if t==0 or abs(t)>1e-9]
    ax[1].yaxis.set_major_locator(FixedLocator(ticks))
    for a in ax:a.grid(alpha=.16)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='outside lower center',ncols=1,frameon=False,fontsize=10)
    return fig


def startup_figure(rows,fields):
    """Return velocity profiles and their signed differences on one grid."""
    from qa.refresh_week01_2_figures import plt
    from matplotlib.lines import Line2D
    fig,ax=plt.subplots(2,2,figsize=(6.6,5.8),layout='constrained')
    ref=time_field(fields,'reference_fine');yr,ur,xr,vr=centerlines(ref)
    steps=sorted({r['config']['dt'] for r in rows if not r['case'].startswith('reference')},reverse=True)
    styles={dt:['-','--',':','-.'][i%4] for i,dt in enumerate(steps)}
    for r in rows:
        if r['case'].startswith('reference'):continue
        f=time_field(fields,r['case']);y,u,x,v=centerlines(f)
        color='#cc2222' if r['config']['method']=='piso' else '#0055cc'
        style=styles[r['config']['dt']]
        ax[0,0].plot(u,y,color=color,ls=style,lw=1.4);ax[0,1].plot(x,v,color=color,ls=style,lw=1.4)
        ax[1,0].plot(y,u-ur,color=color,ls=style,lw=1.4);ax[1,1].plot(x,v-vr,color=color,ls=style,lw=1.4)
    ax[0,0].plot(ur,yr,c='black',lw=1.1);ax[0,1].plot(xr,vr,c='black',lw=1.1)
    ax[0,0].set(xlabel='u/U at x/L=0.5',ylabel='y/L');ax[0,1].set(xlabel='x/L',ylabel='v/U at y/L=0.5')
    ax[1,0].set(xlabel='y/L',ylabel='(u - comparison)/U');ax[1,1].set(xlabel='x/L',ylabel='(v - comparison)/U')
    handles=[Line2D([],[],c=c,label=m) for c,m in [('#cc2222','PISO'),('#0055cc','PIMPLE'),('black','Fine PIMPLE comparison')]]
    handles += [Line2D([],[],c='.4',ls=s,label=f'dt = {d:g}') for d,s in styles.items()]
    for a in ax.flat:a.grid(alpha=.16)
    fig.legend(handles=handles,loc='outside lower center',ncols=3,frameon=False,fontsize=10)
    return fig


def figures(data,fields):
    from qa.refresh_week01_2_figures import save
    save(pressure_figure(data['pressure_rows'],fields),'linear_pressure_profiles')
    save(startup_figure(data['time_rows'],fields),'startup_time_profiles')


def main():
    data,fields=load();verification=audit(data,fields);figures(data,fields)
    pressure,temporal=tables(data)
    pressure.to_csv(DEST/'pressure_table.csv',index=False);temporal.to_csv(DEST/'startup_table.csv',index=False)
    lines=[r'\begin{center}\small\begin{tabular}{rlrrr}\toprule',r'$N$ & Linear solve & Batch (ms) & Setup (ms) & $\|Dq\|_\infty$\\\midrule']
    for r in data['pressure_rows']:
        name=r['method'].replace('GMRES Jacobi rtol=0.0001',r'GMRES $10^{-4}$').replace('GMRES Jacobi rtol=1e-08',r'GMRES $10^{-8}$')
        lines.append(f'{r["cells"]} & {name} & {1e3*r["total_seconds_median"]:.2f} & {1e3*r["setup_seconds_median"]:.2f} & {r["max_divergence_linf"]:.2e}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    (ROOT/'lectures/source/week01_2_linear_work_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{center}\small\begin{tabular}{lrrrr}\toprule',r'Method & $\Delta t$ & Velocity discrepancy (\%) & Seconds & Pressure solves\\\midrule']
    for r in data['time_rows']:
        if r['case'].startswith('reference'):continue
        lines.append(f'{r["config"]["method"].upper()} & {r["config"]["dt"]:g} & {100*r["velocity_discrepancy_vs_fine"]:.3f} & {r["runtime_seconds_median"]:.3f} & {r["pressure_solves"]}'+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        f'The measured comparison-field change after halving its time step is {100*data["reference_checks"]["reference_dt_change"]:.4f}'+r'\%; doubling its outer/inner correction counts changes it by '+f'{100*data["reference_checks"]["reference_loop_change"]:.4g}'+r'\%. These are sensitivity checks, not rigorous error bounds.']
    (ROOT/'lectures/source/week01_2_startup_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (DEST/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
