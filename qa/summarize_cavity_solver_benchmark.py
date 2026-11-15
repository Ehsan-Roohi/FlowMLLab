"""Audit and summarize the matched cavity CPU benchmark; no reference data edits."""
from pathlib import Path
import argparse,csv,json,statistics,collections,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def verify_pairs(entries):
    groups=collections.defaultdict(list)
    for e in entries:
        d=e['result']
        if e['path'].startswith('production/'):
            groups[(d['cells'],d['scenario'],d['repeat'])].append(e)
    checked=[]
    for key,pair in sorted(groups.items()):
        assert len(pair)==2,('Incomplete pair',key)
        a,b=(next(e['result'] for e in pair if e['result']['method']==m) for m in ['direct','gmres_mg'])
        required=['seed_sha256','reference_sha256','packed_initial_state_sha256','host','slurm_job_id','environment','thread_environment','schedule','nonlinear_tolerance','source_sha256_lf','candidate_config']
        missing=[k for k in required if k not in a or k not in b]
        if missing:
            assert not (a['success'] and b['success']),('Successful pair lacks provenance',key)
            checked.append({'cells':key[0],'scenario':key[1],'repeat':key[2],'both_pass':False,'metadata_incomplete':missing})
            continue
        for k in required:
            assert a[k]==b[k],('Pair mismatch',key,k)
        assert set(a['thread_environment'].values())=={'1'}
        for d in [a,b]:
            if d['success']:
                assert d['final_residual_linf']<1e-8
                assert d['reference_discrepancies']['velocity_relative_l2']<1e-8
                assert d['reference_discrepancies']['omega']['relative_l2']<1e-7
                assert all(s['success'] and s['original_relative_defect']<=1e-5 for s in d['linear_solves'])
        checked.append({'cells':key[0],'scenario':key[1],'repeat':key[2],'both_pass':a['success'] and b['success'],'host':a['host']})
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    root=args.directory;evidence=json.loads((root/'evidence.json').read_text(encoding='utf-8-sig'));entries=evidence['trials']
    checked=verify_pairs(entries)
    for k,v in evidence['metadata'].items():
        if k.endswith('/status.json'):assert v['state']=='complete',(k,'still active')
    prod=[e for e in entries if e['path'].startswith('production/')]
    selected=evidence['metadata'].get('selection.json',{})
    expected_sources=evidence['metadata']['staging_manifest.json']['sources']
    for e in prod:
        d=e['result']
        if d.get('source_sha256_lf'):
            assert all(v==expected_sources[k] for k,v in d['source_sha256_lf'].items()),'Frozen numerical source changed'
        if d.get('candidate_config'):
            assert d['candidate_config']['ilu_fill_factor']==selected['fill']
            assert d['candidate_config']['ilu_drop_tol']==selected['drop']
    groups=collections.defaultdict(list)
    for e in prod:
        d=e['result'];groups[d['cells'],d['scenario'],d['method']].append(d)
    rows=[]
    for (n,scenario,method),data in sorted(groups.items()):
        valid=[d for d in data if d['success']]
        row=dict(cells=n,nodes=n+1,scenario=scenario,method=method,attempts=len(data),passed=len(valid))
        if valid:
            times=[d['solve_wall_seconds'] for d in valid]
            row.update(cpu_seconds_median=statistics.median(d['solve_cpu_seconds'] for d in valid),wall_seconds_median=statistics.median(times),wall_seconds_min=min(times),wall_seconds_max=max(times),peak_RSS_MiB_max=max(d['peak_rss_KiB'] for d in valid)/1024,
                residual_linf_max=max(d['final_residual_linf'] for d in valid),velocity_relative_l2_max=max(d['reference_discrepancies']['velocity_relative_l2'] for d in valid),omega_relative_l2_max=max(d['reference_discrepancies']['omega']['relative_l2'] for d in valid),jacobian_seconds_median=statistics.median(d['profiling']['jacobian_seconds'] for d in valid),preconditioner_setup_seconds_median=statistics.median(d['profiling']['preconditioner_setup_seconds'] for d in valid),linear_seconds_median=statistics.median(d['profiling']['linear_total_seconds'] for d in valid),krylov_iterations_median=statistics.median(d['profiling']['krylov_iterations'] for d in valid))
        rows.append(row)
    fieldnames=list(dict.fromkeys(k for r in rows for k in r))
    if rows:
        with (root/'timings.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    summary={'retrieved_utc':evidence['retrieved_utc'],'paired_checks':checked,'summary':rows,'pilot':[]}
    for e in entries:
        if e['path'].startswith(('pilot/','pilot_strong/')):
            d=e['result'];summary['pilot'].append({'tag':e['path'],'success':d['success'],'method':d['method'],'scenario':d['scenario'],'config':d.get('candidate_config'),'seconds':d.get('solve_wall_seconds'),'error':d.get('error'),'profiling':d.get('profiling')})
    (root/'verified_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'mathtext.fontset':'stix','font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':11,'axes.labelsize':12,'legend.fontsize':10,'savefig.dpi':220})
    available=[n for n in [512,256,128] if all(any(e['result'].get('cells')==n and e['result'].get('scenario')==s and e['result'].get('method')==m and e['result'].get('success') and 'profiles' in e for e in entries) for s in ['warm7500','warm10000'] for m in ['direct','gmres_mg'])]
    if available:
        n=available[0];fig,axes=plt.subplots(2,3,figsize=(11,6.6),layout='constrained')
        for i,scenario in enumerate(['warm7500','warm10000']):
            for method,color,style in [('direct','black','-'),('gmres_mg','#b22222','--')]:
                e=next(e for e in entries if e['result'].get('cells')==n and e['result'].get('scenario')==scenario and e['result'].get('method')==method and e['result'].get('success') and 'profiles' in e)
                f=e['profiles'];label='Newton + direct' if method=='direct' else 'Newton + GMRES / multigrid'
                axes[i,0].plot(f['u_center'],f['y'],style,color=color,lw=1.4,label=label)
                axes[i,1].plot(f['x'],f['v_center'],style,color=color,lw=1.4,label=label)
                axes[i,2].plot(f['x'],f['omega_horizontal_center'],style,color=color,lw=1.4,label=label)
            axes[i,0].set(xlabel=r'$u(1/2,y)/U$',ylabel=r'$y/L$')
            axes[i,1].set(xlabel=r'$x/L$',ylabel=r'$v(x,1/2)/U$')
            axes[i,2].set(xlabel=r'$x/L$',ylabel=r'$\omega(x,1/2)L/U$')
            for ax in axes[i]:ax.grid(alpha=.2);ax.text(.02,1.04,fr'$Re={7500 if i==0 else 10000}$',transform=ax.transAxes,fontsize=11)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
        fig.savefig(root/'physical_profiles.png');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
        for ax,scenario in zip(axes,['warm7500','warm10000']):
            for method,color,style in [('direct','black','o-'),('gmres_mg','#b22222','s--')]:
                d=next(e['result'] for e in entries if e['result'].get('cells')==n and e['result'].get('scenario')==scenario and e['result'].get('method')==method and e['result'].get('success') and 'profiles' in e)
                h=d['history'][0]['iterations'];ax.semilogy([v['iteration'] for v in h],[max(v['poisson_linf'],v['steady_vorticity_linf']) for v in h],style,color=color,lw=1.4,ms=4,label='Newton + direct' if method=='direct' else 'Newton + GMRES / multigrid')
            ax.axhline(1e-8,color='.5',lw=.8,ls=':');ax.set(xlabel='Newton iteration',ylabel='Original equation residual (max norm)');ax.grid(alpha=.2);ax.text(.02,1.04,fr'$Re={7500 if scenario=="warm7500" else 10000}$',transform=ax.transAxes)
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=2,frameon=False);fig.savefig(root/'nonlinear_convergence.png');plt.close(fig)
    hardware=next((e['result'] for e in prod if e['result'].get('success')),entries[0]['result'])
    hardware_note=f"Hardware: {hardware.get('cpu_model','unrecorded')}, node {hardware.get('host','unrecorded')}; Python {hardware.get('environment',{}).get('python','?')}, NumPy {hardware.get('environment',{}).get('numpy','?')}, SciPy {hardware.get('environment',{}).get('scipy','?')}. The two grid studies ran in separate one-core Slurm steps on the same node; paired methods within each study ran sequentially."
    warm_ratios=[]
    for n in [256,512]:
        for scenario in ['warm7500','warm10000']:
            paired_rows={v['method']:v for v in rows if v['cells']==n and v['scenario']==scenario}
            if len(paired_rows)==2 and all(v['passed']==v['attempts']==3 for v in paired_rows.values()):
                warm_ratios.append(paired_rows['gmres_mg']['wall_seconds_median']/paired_rows['direct']['wall_seconds_median'])
    if len(warm_ratios)==4 and min(warm_ratios)>1:
        verdict=f"The original sparse-direct solver was faster in all four matched warm-case/grid comparisons. The tested GMRES/multigrid implementation took {min(warm_ratios):.2f} to {max(warm_ratios):.2f} times as long, with the same accepted numerical solution. The original production solver is retained."
    elif len(warm_ratios)==4:
        verdict='All four warm-case/grid comparisons completed; the timing table records the measured wins and losses without claiming a universal best solver.'
    else:
        verdict='At least one matched warm-case/grid comparison lacks three successful repetitions; no complete four-case speed claim is made.'
    lines=['# Matched cavity solver benchmark','',verdict,'',hardware_note,'',
        'The existing damped Newton solver with a sparse direct linear solve is compared with the same outer Newton iteration using GMRES and a coupled geometric Galerkin multigrid preconditioner. No PDE, grid, boundary condition, line search, or nonlinear acceptance tolerance was changed.','',
        'Warm cases use the same retained Re=6500 field for Re=7500 and the same retained Re=9000 field for Re=10000. Continuation starts from the same retained 128-cell Re=1000 field and follows the same 20-root path. These are not cold-start timings.','',
        'Solve wall time includes Jacobian assembly and preconditioner construction; it excludes queueing, imports, initial input I/O, seed generation, verification, and checkpoint writing. One CPU thread per trial; methods run sequentially on the same allocation with fresh processes. Warm comparisons use AB, BA, AB order.','',
        'Each successful solution must have original nonlinear residual <1e-8, combined velocity relative L2 difference <1e-8, and vorticity relative L2 difference <1e-7 against the retained same-grid CFD field. These are agreement checks, not error against Ghia or a continuum exact solution.','',
        '## Pilot','', '| Trial | ILUT fill / drop | Passed | Solve seconds | Failure |','|---|---|---|---:|---|']
    for d in summary['pilot']:
        c=d['config'] or {};cfg=f"{c.get('ilu_fill_factor','—')} / {c.get('ilu_drop_tol','—')}" if d['method']=='gmres_mg' else '—'
        lines.append(f"| {d['tag'].removesuffix('/result.json')} | {cfg} | {d['success']} | {d['seconds'] if d['seconds'] is not None else '—'} | {d['error'] or ''} |")
    if rows:
        lines += ['', '## Matched results','','| Grid nodes | Case | Method | Passed | Median seconds [range] | Peak RSS MiB |','|---|---|---|---|---:|---:|']
        for r in rows:
            t=f"{r['wall_seconds_median']:.3f} [{r['wall_seconds_min']:.3f}, {r['wall_seconds_max']:.3f}]" if r['passed'] else 'failed'
            mem=f"{r['peak_RSS_MiB_max']:.1f}" if r['passed'] else '—'
            lines.append(f"| {r['nodes']} × {r['nodes']} | {r['scenario']} | {r['method']} | {r['passed']}/{r['attempts']} | {t} | {mem} |")
        lines += ['', 'The continuation comparison has one repetition per method; the warm cases have three. A ratio from one continuation pair should not be treated as a hardware-independent speedup.','']
        for n in [256,512]:
            for scenario in ['warm7500','warm10000','continuation']:
                pair={r['method']:r for r in rows if r['cells']==n and r['scenario']==scenario}
                if len(pair)==2 and all(x['passed']==x['attempts'] for x in pair.values()):
                    ratio=pair['gmres_mg']['wall_seconds_median']/pair['direct']['wall_seconds_median']
                    lines.append(f'- {n+1} nodes, {scenario}: GMRES-multigrid / direct median wall-time ratio = {ratio:.4f} (above 1 means multigrid is slower).')
    candidate_warm=[v for v in rows if v['method']=='gmres_mg' and v['scenario'].startswith('warm') and v['passed']==v['attempts']==3]
    if len(candidate_warm)==4:
        fractions=[v['preconditioner_setup_seconds_median']/v['wall_seconds_median'] for v in candidate_warm]
        newton_counts=sorted({h['newton_iterations'] for e in prod if e['result']['success'] and e['result']['scenario'].startswith('warm') for h in e['result']['history']})
        lines += ['',f"Profiling explains the observed slowdown: construction of the candidate preconditioner accounted for approximately {100*min(fractions):.1f}% to {100*max(fractions):.1f}% of warm-case solve time (ratios of median timings). Both methods used {newton_counts} Newton steps in these cases. This implementation rebuilds the strong ILUT-based hierarchy at every Newton step; that setup cost dominates, even though GMRES converges in few iterations.",'']
    paired=evidence['metadata'].get('paired_fields_audit.json',{}).get('pairs',[])
    if paired:
        lines += ['', '## Direct comparison of complete fields','',f"Across the {len(paired)} representative repeat-0 field pairs, the largest relative L2 difference is {max(p['paired_velocity_relative_l2'] for p in paired):.5e} for combined velocity and {max(p['paired_omega_relative_l2'] for p in paired):.5e} for vorticity. These compare the two methods directly over the entire grid; they are not differences from Ghia data.",'', 'Velocity difference is sqrt(sum((u_MG-u_direct)^2 + (v_MG-v_direct)^2)) / sqrt(sum(u_direct^2 + v_direct^2)). Vorticity uses the analogous relative L2 norm.','']
    lines += ['', '## Limits','',
        'This benchmark assesses one experimental preconditioner, not all multigrid or Newton–Krylov methods. It is not a reimplementation of Ghia et al.’s coupled strongly implicit multigrid algorithm. Both methods compute the same two-dimensional steady branch. Converging that branch does not establish temporal stability of a high-Reynolds-number physical flow. Memory includes the complete Python child process, not just the factorization.','',
        'Historical Ghia timings are not a matched benchmark: the hardware, solver, starting field, discretization and stopping conditions differ. The present paired measurements support algorithm comparisons only on the recorded hardware and inputs.','',
        'The three original lightweight 128-cell configurations failed. A bounded follow-up increased ILUT strength while keeping the equations and all acceptance gates unchanged. The successful setting was frozen before any 256/512-cell candidate evaluation.','',
        'Exact source hashes, input array hashes, NPZ container hashes, host/CPU model, per-linear-solve histories, failures, and numerical gates are in `evidence.json` and `verified_summary.json`. Complete recovery states and representative full fields remain in the recorded Unity Workspace.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'verified_pairs':len(checked),'summary_rows':len(rows),'representative_grid_cells':available[0] if available else None,'report':str(root/'REPORT.md')}))


if __name__=='__main__':main()
