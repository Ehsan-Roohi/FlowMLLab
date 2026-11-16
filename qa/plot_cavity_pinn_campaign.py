"""Plot frozen Ghia PINN evidence only; no training, selection, or score rewriting.

Every scored input is SHA256 checked. Native omega scores remain untouched.
The additional centered-curl diagnostic is derived from both stored velocity
fields with one operator on a common 509 x 509 grid.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'qa'))
from lecture_typography import configure_matplotlib
REYNOLDS = (100, 400, 1000, 3200, 5000, 7500, 10000)
SEEDS = (7, 23, 61)
COLLARS = (.02, .01)
COLORS = {7:'#0072B2', 23:'#D55E00', 61:'#009E73'}
STYLES = {.02:'-', .01:'--'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def entry(path):
    return {'path':Path(path).resolve().relative_to(ROOT.resolve()).as_posix(), 'sha256':sha(path)}


def verified(item):
    p = (ROOT/item['path']).resolve()
    if not p.is_relative_to(ROOT.resolve()) or sha(p) != item['sha256']:
        raise ValueError('Evidence path/hash rejected: '+item['path'])
    return p


def stamp(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None: raise ValueError('Evidence timestamp lacks timezone')
    return t


def load_manifest(path, allow_partial=False):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data['schema_version']!=1 or data['campaign_kind']!='ghia_constant_lid':
        raise ValueError('Unsupported campaign schema')
    ids=set(); identities=set()
    for r in data['records']:
        if r['run_id'] in ids: raise ValueError('Duplicate scientific run ID')
        ids.add(r['run_id'])
        if r['outcome']!='scored': continue
        ident=(r['Re'],r['seed'],r['collar'])
        if ident in identities: raise ValueError('Repeated Re/seed/collar trajectory')
        identities.add(ident)
        if r['Re'] not in REYNOLDS or r['seed'] not in SEEDS or r['collar'] not in COLLARS:
            raise ValueError('Unexpected campaign identity')
        if r['prediction']['sha256']!=r['freeze']['prediction_sha256'] or r['prediction']['sha256']!=r['score']['prediction_sha256']:
            raise ValueError('Prediction differs from the scored freeze')
        if stamp(r['freeze']['at_utc']) > stamp(r['score']['reference_accessed_at_utc']):
            raise ValueError('Reference access precedes the evaluation freeze')
        for key in ('prediction','reference','history','ghia_profiles'): verified(r[key])
    expected={(re,seed,collar) for re in REYNOLDS for seed in SEEDS for collar in COLLARS}
    if identities!=expected and not allow_partial:
        raise ValueError(f'Expected 42 scored endpoints, found {len(identities)}; --allow-partial is diagnostic only')
    return data


def arrays(row):
    fields=[]
    for key in ('prediction','reference'):
        with np.load(verified(row[key]), allow_pickle=False) as f:
            fields.append({k:f[k].copy() for k in f.files})
    pred,ref=fields
    for c in ('x','y'):
        if pred[c].ndim!=1 or not np.array_equal(pred[c],ref[c]): raise ValueError('Unmatched native coordinates')
        if not np.all(np.diff(pred[c])>0): raise ValueError('Non-increasing native coordinates')
    shape=(len(pred['y']),len(pred['x']))
    for d in fields:
        for k in ('u','v','psi','omega'):
            if d[k].shape!=shape or not np.isfinite(d[k]).all(): raise ValueError('Invalid native field '+k)
    return pred,ref


def centered_curl(field):
    x,y=field['x'],field['y'];u,v=field['u'],field['v']
    if not np.allclose(np.diff(x),np.diff(x)[0],rtol=1e-12,atol=0) or not np.allclose(np.diff(y),np.diff(y)[0],rtol=1e-12,atol=0):
        raise ValueError('This centered-curl diagnostic requires a uniform native grid')
    return ((v[1:-1,2:]-v[1:-1,:-2])/(x[2:]-x[:-2])[None,:]
            -(u[2:,1:-1]-u[:-2,1:-1])/(y[2:]-y[:-2])[:,None])


def save_rgb(fig,path,pdf=None):
    fig.savefig(path,dpi=200,facecolor='white')
    with Image.open(path) as im: rgb=im.convert('RGB')
    rgb.save(path)
    if pdf is not None: pdf.savefig(fig,facecolor='white')
    plt.close(fig)
    return entry(path)


def legend_handles():
    return [Line2D([],[],color='black',lw=1.5,label='Retained CFD'),
            Line2D([],[],color='black',marker='o',mfc='white',ls='none',label='Ghia printed samples'),
            *[Line2D([],[],color=COLORS[s],lw=1.6,label=f'PINN seed {s}') for s in SEEDS],
            *[Line2D([],[],color='.3',lw=1.6,ls=STYLES[e],label=fr'$\epsilon={e:g}$') for e in COLLARS],
            Line2D([],[],color='#B2182B',marker='x',ls='none',label='Flagged printed entry')]


def profile_figure(re,rows):
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.7))
    fig.subplots_adjust(left=.09,right=.98,top=.9,bottom=.27,wspace=.3,hspace=.3)
    fig.suptitle(f'Fixed-budget PINN baseline, Re = {re}\nSigned centerlines and errors at published Ghia positions',fontsize=13,y=.985)
    first=True
    for r in sorted(rows,key=lambda x:(x['seed'],-x['collar'])):
        pred,ref=arrays(r)
        ix=int(np.argmin(abs(ref['x']-.5)));iy=int(np.argmin(abs(ref['y']-.5)))
        if ref['x'][ix]!=.5 or ref['y'][iy]!=.5: raise ValueError('Native centerline missing')
        if first:
            axes[0,0].plot(ref['u'][:,ix],ref['y'],color='black',lw=1.8,zorder=2)
            axes[0,1].plot(ref['x'],ref['v'][iy],color='black',lw=1.8,zorder=2)
        axes[0,0].plot(pred['u'][:,ix],pred['y'],color=COLORS[r['seed']],ls=STYLES[r['collar']],lw=1.15)
        axes[0,1].plot(pred['x'],pred['v'][iy],color=COLORS[r['seed']],ls=STYLES[r['collar']],lw=1.15)
        with verified(r['ghia_profiles']).open(encoding='utf-8',newline='') as f: samples=list(csv.DictReader(f))
        for j,component in enumerate(('u','v')):
            points=[x for x in samples if x['component']==component]
            pos=np.array([float(x['position']) for x in points]);truth=np.array([float(x['published']) for x in points])
            estimate=np.array([float(x['prediction']) for x in points]);error=np.array([float(x['signed_error_U']) for x in points])
            included=np.array([x['screened_included'].lower() in ('true','1') for x in points])
            if not np.allclose(estimate-truth,error,rtol=1e-12,atol=1e-14): raise ValueError('Signed Ghia error changed')
            if first:
                if component=='u': axes[0,j].scatter(truth[included],pos[included],s=30,facecolors='white',edgecolors='black',zorder=6)
                else: axes[0,j].scatter(pos[included],truth[included],s=30,facecolors='white',edgecolors='black',zorder=6)
                if (~included).any():
                    if component=='u': axes[0,j].scatter(truth[~included],pos[~included],s=45,c='#B2182B',marker='x',zorder=7)
                    else: axes[0,j].scatter(pos[~included],truth[~included],s=45,c='#B2182B',marker='x',zorder=7)
            order=np.argsort(pos)
            # Lines connect only actual measured positions, with markers preserving sample coverage.
            axes[1,j].plot(pos[order],error[order],color=COLORS[r['seed']],ls=STYLES[r['collar']],marker='o',ms=2.5,lw=1.05)
            axes[1,j].scatter(pos[~included],error[~included],s=28,c='#B2182B',marker='x',zorder=8)
        first=False
    if first:
        for ax in axes.flat: ax.text(.5,.5,'No scored endpoint',ha='center',transform=ax.transAxes)
    axes[0,0].set(xlabel=r'$u/U$ at $x/W=0.5$',ylabel=r'$y/W$',ylim=(0,1))
    axes[0,1].set(xlabel=r'$x/W$',ylabel=r'$v/U$ at $y/W=0.5$',xlim=(0,1))
    for j,c in enumerate(('u','v')):
        axes[1,j].axhline(0,color='.35',lw=.8,zorder=1)
        axes[1,j].set(xlabel=r'$y/W$' if c=='u' else r'$x/W$',ylabel=f'Signed {c} error / U',xlim=(0,1))
    for ax in axes.flat: ax.grid(alpha=.18);ax.tick_params(labelsize=10)
    fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.027,'Dense curves: exact retained interior nodes. Error markers: direct frozen predictions at printed positions.\nFlagged entries remain visible; no reference was used to choose a seed or endpoint.',ha='center',fontsize=9)
    return fig


def streamline_levels(ref,pred):
    values=np.r_[ref['psi'].ravel(),pred['psi'].ravel()]
    neg=max(0.,-float(values.min()));pos=max(0.,float(values.max()))
    # Identical levels for the two methods; logarithmic weak-circulation levels retain secondary eddies.
    levels=np.r_[-np.geomspace(neg*1e-5,neg*.97,15)[::-1] if neg>1e-12 else [],
                 np.geomspace(pos*1e-5,pos*.97,12) if pos>1e-12 else []]
    return np.unique(levels)


def overlay(ax,pred,ref,levels):
    ax.contour(ref['x'],ref['y'],ref['psi'],levels=levels,colors='black',linestyles='solid',linewidths=.9)
    ax.contour(pred['x'],pred['y'],pred['psi'],levels=levels,colors='#D55E00',linestyles='--',linewidths=.9)
    ax.set_aspect('equal');ax.set(xlabel=r'$x/W$',ylabel=r'$y/W$')


def streamline_figure(re,pred,ref):
    fig,axes=plt.subplots(1,3,figsize=(12.1,4.6),gridspec_kw={'width_ratios':[1.5,1,1]})
    levels=streamline_levels(ref,pred)
    for ax in axes: overlay(ax,pred,ref,levels)
    axes[0].set(xlim=(0,1),ylim=(0,1));axes[1].set(xlim=(0,.22),ylim=(0,.22));axes[2].set(xlim=(.78,1),ylim=(0,.22))
    axes[0].set_title('Whole cavity',fontsize=12);axes[1].set_title('Lower left',fontsize=12);axes[2].set_title('Lower right',fontsize=12)
    fig.suptitle(fr'Re = {re}; seed 7; $\epsilon=0.02$: matched streamfunction contours',fontsize=14,y=.98)
    fig.legend(handles=[Line2D([],[],color='black',lw=1.3,label='Retained CFD'),Line2D([],[],color='#D55E00',ls='--',lw=1.3,label='Frozen PINN')],
               loc='lower center',bbox_to_anchor=(.5,.08),ncol=2,frameon=False)
    fig.text(.5,.025,'The same contour values are used for both methods. Log-spaced weak-circulation levels expose secondary structures.\nContours use retained interior samples; no boundary rows or arrow fields have been invented.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.06,right=.98,top=.82,bottom=.24,wspace=.32)
    return fig,levels


def field_figure(re,pred,ref,norms):
    fig,axes=plt.subplots(3,3,figsize=(11.5,11.2))
    fig.subplots_adjust(left=.065,right=.925,bottom=.11,top=.93,wspace=.38,hspace=.48)
    x,y=ref['x'],ref['y'];wp=centered_curl(pred);wr=centered_curl(ref)
    speed=[np.hypot(d['u'],d['v']) for d in (ref,pred)]
    error=np.hypot(pred['u']-ref['u'],pred['v']-ref['v'])
    content=[(0,0,speed[0],x,y,'Retained CFD speed','turbo',norms['speed'],r'$|\mathbf{u}|/U$'),
             (0,1,speed[1],x,y,'Frozen PINN speed','turbo',norms['speed'],r'$|\mathbf{u}|/U$'),
             (0,2,error,x,y,'Vector-error magnitude','turbo',norms['vector_error'],r'$|\hat{\mathbf{u}}-\mathbf{u}|/U$'),
             (1,0,wr,x[1:-1],y[1:-1],'CFD centered curl','RdBu_r',norms['omega'],r'$\omega W/U$'),
             (1,1,wp,x[1:-1],y[1:-1],'PINN centered curl','RdBu_r',norms['omega'],r'$\omega W/U$'),
             (1,2,wp-wr,x[1:-1],y[1:-1],'Signed curl error','RdBu_r',norms['omega_error'],r'$(\hat\omega-\omega)W/U$'),
             (2,0,pred['u']-ref['u'],x,y,'Signed horizontal-velocity error','RdBu_r',norms['signed_velocity'],r'$(\hat u-u)/U$'),
             (2,1,pred['v']-ref['v'],x,y,'Signed vertical-velocity error','RdBu_r',norms['signed_velocity'],r'$(\hat v-v)/U$')]
    for i,j,z,cx,cy,title,cmap,norm,label in content:
        ax=axes[i,j]
        im=ax.pcolormesh(cx,cy,z,cmap=cmap,norm=norm,shading='nearest',rasterized=True)
        ax.set(title=title,xlabel=r'$x/W$',ylabel=r'$y/W$',xlim=(0,1),ylim=(0,1),aspect='equal')
        ax.title.set_fontsize(11)
        cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.035);cb.set_label(label,fontsize=10);cb.ax.tick_params(labelsize=8)
    overlay(axes[2,2],pred,ref,streamline_levels(ref,pred));axes[2,2].set(title='Matched streamfunction contours',xlim=(0,1),ylim=(0,1));axes[2,2].title.set_fontsize(11)
    fig.suptitle(fr'Re = {re}; seed 7; $\epsilon=0.02$: frozen fields on matched coordinates',fontsize=15,y=.978)
    fig.legend(handles=[Line2D([],[],color='black',label='CFD contours'),Line2D([],[],color='#D55E00',ls='--',label='PINN contours')],loc='lower center',bbox_to_anchor=(.5,.047),ncol=2,frameon=False)
    fig.text(.5,.012,'Shared scales across Re1000 and Re7500. Velocity: 511 x 511; common centered curl: 509 x 509.\nCurl scales are symmetric-log (linear threshold 0.25 U/W); velocity and its errors are linear.\nFrozen native-AD/transported-omega scores are a separate comparison. No CFD pressure field is available.',ha='center',fontsize=9)
    return fig



def paired_field_figure(variable,selected,norms):
    fig,axes=plt.subplots(2,3,figsize=(8.7,7.6))
    fig.subplots_adjust(left=.075,right=.98,bottom=.26,top=.88,wspace=.3,hspace=.3)
    image_handles=[]
    for i,re in enumerate((1000,7500)):
        _,pred,ref=selected[re]
        if variable=='speed':
            x,y=ref['x'],ref['y']
            values=(np.hypot(ref['u'],ref['v']),np.hypot(pred['u'],pred['v']),np.hypot(pred['u']-ref['u'],pred['v']-ref['v']))
            color='turbo';normal=(norms['speed'],norms['speed'],norms['vector_error'])
        else:
            x,y=ref['x'][1:-1],ref['y'][1:-1]
            wr,wp=centered_curl(ref),centered_curl(pred)
            values=(wr,wp,wp-wr);color='RdBu_r';normal=(norms['omega'],norms['omega'],norms['omega_error'])
        for j,(z,norm) in enumerate(zip(values,normal)):
            ax=axes[i,j];im=ax.pcolormesh(x,y,z,cmap=color,norm=norm,shading='nearest',rasterized=True)
            ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',xlabel=r'$x/W$')
            ax.set_ylabel(f'Re = {re}\n'+r'$y/W$' if j==0 else r'$y/W$')
            ax.tick_params(labelsize=10)
            if i==0: image_handles.append(im)
    for j,title in enumerate(('Retained CFD','Frozen PINN','Vector-error magnitude' if variable=='speed' else 'Signed curl error')):
        axes[0,j].set_title(title,fontsize=12,pad=10)
    cax=fig.add_axes([.10,.135,.5,.025]);bar=fig.colorbar(image_handles[0],cax=cax,orientation='horizontal')
    eaxis=fig.add_axes([.73,.135,.22,.025]);ebar=fig.colorbar(image_handles[2],cax=eaxis,orientation='horizontal')
    if variable=='speed':
        title='Speed and velocity error';bar.set_label(r'$|\mathbf{u}|/U$');ebar.set_label(r'$|\hat{\mathbf{u}}-\mathbf{u}|/U$')
        caption='Exact retained 511 x 511 interior nodes; linear shared scales. No pressure truth is available.'
    else:
        title='Vorticity from one common centered-curl operator';bar.set_label(r'$\omega W/U$');ebar.set_label(r'$(\hat\omega-\omega)W/U$')
        caption='Common 509 x 509 inner grid; symmetric-log scales, linear threshold 0.25 U/W.\nThese derived curls do not replace the original native-vorticity scores.'
        magnitude=10.**np.floor(np.log10(norms['omega_error'].vmax))
        ebar.set_ticks([-magnitude,0,magnitude]);ebar.ax.tick_params(labelsize=10)
    fig.suptitle(title+'\n'+r'Predeclared seed 7, $\epsilon=0.02$; fixed-budget baseline',fontsize=13,y=.975)
    fig.text(.5,.025,caption,ha='center',fontsize=10)
    return fig


def criterion_summary(manifest):
    scored=[r for r in manifest['records'] if r['outcome']=='scored']
    checks=sorted({k for r in scored for k in r['scores']['numeric_checks']})
    counts={k:sum(bool(r['scores']['numeric_checks'].get(k,False)) for r in scored) for k in checks}
    pairs=manifest.get('collar_comparisons',[])
    return {'scored_endpoints':len(scored),'non_scored_endpoints':len(manifest['records'])-len(scored),
            'numeric_pass_counts':counts,'all_four_numeric_targets':sum(all(r['scores']['numeric_checks'].values()) for r in scored),
            'collar_pairs':len(pairs),'collar_pairs_within_2_percent':sum(bool(r['numeric_target_le_0_02']) for r in pairs),
            'physical_acceptance':'Not implied by the numeric counts; inspect all signed profiles and representative fields.'}


def write_narrative(path,manifest,summary):
    scored=[r for r in manifest['records'] if r['outcome']=='scored']
    lines=['# Fresh label-free Ghia PINN baseline','',
      'These are case-specific PDE-trained networks. The existing supervised POD-DeepONet is a different experiment.', '',
      f"{summary['scored_endpoints']} frozen endpoints were scored. Of those, {summary['all_four_numeric_targets']} meet all four recorded numeric targets; {summary['collar_pairs_within_2_percent']} of {summary['collar_pairs']} matched collar pairs meet the 2% sensitivity target. A completed optimizer budget is not evidence of accurate flow.",'',
      '| Re | Scored | Ghia u/v target | Interior vector target | All four targets | Interior vector error range |',
      '|---:|---:|---:|---:|---:|---:|']
    for re in REYNOLDS:
        rows=[r for r in scored if r['Re']==re]
        vals=[r['scores']['regions']['interior']['velocity_relative_l2'] for r in rows]
        count=lambda key:sum(bool(r['scores']['numeric_checks'].get(key,False)) for r in rows)
        rng=f'{100*min(vals):.2f}% to {100*max(vals):.2f}%' if vals else 'No scored result'
        lines.append(f"| {re} | {len(rows)}/6 | {count('both_screened_ghia_relative_l2_le_0_02')}/{len(rows)} | {count('interior_velocity_relative_l2_le_0_05')}/{len(rows)} | {sum(all(r['scores']['numeric_checks'].values()) for r in rows)}/{len(rows)} | {rng} |")
    lines += ['', 'All three predeclared seeds and both collars appear in the centerline figures. Markers on error curves are the original Ghia positions, evaluated directly by the frozen network. Flagged printed entries remain visible. The field figures use predeclared seed 7 and collar 0.02 at Re1000 and Re7500; they are not a best-seed selection.', '',
      'The native scores compare PINN automatic-differentiation vorticity with retained CFD transported vorticity at 511 x 511 interior nodes. The additional visualization applies one centered curl operator to both retained velocity fields at 509 x 509 common inner nodes; it never changes those original scores.', '',
      'A zero collar difference is not sufficient sensitivity evidence: sparse fixed sampling can place no points in the removed annulus, leaving the training sets and trajectory identical. Inspect sample identities and annulus counts. The retained high-Re CFD fields have unresolved grid sensitivity. Collar sensitivity does not establish full-domain strong-residual convergence. No CFD pressure truth is available. Do not infer unseen-Re generalization or an optimizer ranking from this fixed-baseline campaign.', '',
      '## Figures','', '[All seven Reynolds numbers: signed centerlines and Ghia errors](all_reynolds_centerlines.pdf)', '',
      '[Re1000 physical fields](re1000_fields.png) · [Re1000 streamline overlay](re1000_streamlines.png)', '',
      '[Re7500 physical fields](re7500_fields.png) · [Re7500 streamline overlay](re7500_streamlines.png)', '',
      'Figure provenance and fixed scales are recorded in `figure_manifest.json`. The numeric ledger remains `../evaluation_manifest.json`.']
    if 'sampling_coverage' in summary:
        c=summary['sampling_coverage']
        lines[4:4]=[f"The checkpoint-bound sampling audit finds {c['identical_objective_pairs']}/{c['pairs']} collar pairs with identical collocation tensors and discrete objectives. Seeds 7 and 61 add no annulus points; seed 23 adds only one. Of those identical-objective pairs, {c['identical_model_pairs_among_identical_objectives']} also have identical frozen models. The remaining Re1000/seed61 weight difference is an execution-sensitivity observation, not a collar response; its cause is not established.",'']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'results/cavity_pinn_campaign/evaluation_manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/cavity_pinn_campaign/figures')
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT.resolve()): raise ValueError('Output must stay inside this checkout')
    manifest=load_manifest(args.manifest,args.allow_partial)
    configure_matplotlib();plt.rcParams.update({'font.size':11,'axes.titlesize':12})
    out=args.output;out.mkdir(parents=True,exist_ok=True);figures=[]
    scored=[r for r in manifest['records'] if r['outcome']=='scored']
    with PdfPages(out/'all_reynolds_centerlines.pdf') as pdf:
        for re in REYNOLDS:
            fig=profile_figure(re,[r for r in scored if r['Re']==re])
            item=save_rgb(fig,out/f're{re}_centerlines.png',pdf);item.update(kind='signed_ghia_centerlines',Re=re);figures.append(item)
    figures.append(entry(out/'all_reynolds_centerlines.pdf'))
    selected={}
    for re in (1000,7500):
        rows=[r for r in scored if (r['Re'],r['seed'],r['collar'])==(re,7,.02)]
        if len(rows)==1:selected[re]=(rows[0],*arrays(rows[0]))
    scales={}
    if selected:
        speed=max(float(np.hypot(d['u'],d['v']).max()) for _,p,r in selected.values() for d in (p,r))
        ve=max(float(np.hypot(p['u']-r['u'],p['v']-r['v']).max()) for _,p,r in selected.values())
        se=max(float(abs(p[k]-r[k]).max()) for _,p,r in selected.values() for k in ('u','v'))
        om=max(float(abs(centered_curl(d)).max()) for _,p,r in selected.values() for d in (p,r))
        oe=max(float(abs(centered_curl(p)-centered_curl(r)).max()) for _,p,r in selected.values())
        scales=dict(speed_vmax=speed,vector_error_vmax=ve,signed_velocity_absmax=se,curl_absmax=om,curl_error_absmax=oe,curl_symlog_linthresh=.25)
        norms={'speed':Normalize(0,max(speed,1e-12)),'vector_error':Normalize(0,max(ve,1e-12)),
               'signed_velocity':TwoSlopeNorm(0,-max(se,1e-12),max(se,1e-12)),
               'omega':SymLogNorm(.25,vmin=-max(om,.25),vmax=max(om,.25)),
               'omega_error':SymLogNorm(.25,vmin=-max(oe,.25),vmax=max(oe,.25))}
        with PdfPages(out/'representative_physical_fields.pdf') as pdf:
            for re,(row,pred,ref) in selected.items():
                item=save_rgb(field_figure(re,pred,ref,norms),out/f're{re}_fields.png',pdf)
                item.update(kind='native_fields_and_derived_centered_curl',Re=re,run_id=row['run_id'],prediction=row['prediction'],reference=row['reference']);figures.append(item)
                fig,levels=streamline_figure(re,pred,ref)
                item=save_rgb(fig,out/f're{re}_streamlines.png',pdf);item.update(kind='matched_streamfunction_contours',Re=re,run_id=row['run_id'],levels=levels.tolist());figures.append(item)
        figures.append(entry(out/'representative_physical_fields.pdf'))
        if set(selected)=={1000,7500}:
            with PdfPages(out/'paired_velocity_vorticity.pdf') as pdf:
                for variable in ('speed','curl'):
                    item=save_rgb(paired_field_figure(variable,selected,norms),out/f'paired_{variable}.png',pdf)
                    item.update(kind='paired_speed_fields' if variable=='speed' else 'paired_centered_curl_fields',Re=[1000,7500],seed=7,collar=.02)
                    figures.append(item)
            figures.append(entry(out/'paired_velocity_vorticity.pdf'))
    summary=criterion_summary(manifest)
    coverage_path=args.manifest.parent/'sampling_coverage.json'
    coverage=None
    if coverage_path.exists():
        coverage=json.loads(coverage_path.read_text(encoding='utf-8'))
        if coverage['schema_version']!=1 or coverage['summary']['runs']!=42:raise ValueError('Unexpected sampling audit')
        summary['sampling_coverage']=coverage['summary']
    write_narrative(out/'README.md',manifest,summary)
    record={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'source_evaluation':entry(args.manifest),
            'generator':entry(__file__),'partial_diagnostic':args.allow_partial,'figures':figures,'summary':summary,'shared_scales':scales,
            'selection':'Predeclared Re1000/Re7500, seed7, collar0.02; no reference-based selection',
            'derived_vorticity':'(v[i,j+1]-v[i,j-1])/(x[j+1]-x[j-1]) - (u[i+1,j]-u[i-1,j])/(y[i+1]-y[i-1]); common 509 x 509 interior',
            'native_vorticity':'Original native PINN AD and retained CFD transported omega scores unchanged',
            'visual_review':'pending; generated files require readable individual-page inspection'}
    if coverage is not None:record['sampling_coverage']=entry(coverage_path)
    (out/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
