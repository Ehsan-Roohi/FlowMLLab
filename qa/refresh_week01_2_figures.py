"""Publication-size Week 1.2 figures, rendered from geometry and archived fields.

No CFD trajectory is rerun. Missing sparse-checkout inputs are read from Git
objects into memory. Each PDF is 6.6 inches wide, matching the lecture text.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'lectures/source/week01_2_assets'
BLUE, GREEN, RED = '#1859a9', '#177548', '#b32b2b'
PURPLE = '#783d99'
COLORS = [BLUE, RED, GREEN, PURPLE]
METHODS = ['simple', 'piso', 'pimple', 'streamfunction-vorticity']
NAMES = ['SIMPLE', 'PISO', 'PIMPLE', r'$\psi$-$\omega$ (FD)']
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 13,
                     'axes.titlesize': 14, 'mathtext.fontset': 'stix',
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': 'white',
                     'figure.constrained_layout.h_pad': .12,
                     'figure.constrained_layout.w_pad': .08})

def figure(height):
    fig, ax = plt.subplots(figsize=(6.6, height))
    fig.subplots_adjust(left=.035, right=.965, bottom=.035, top=.965)
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
    return fig, ax

def text(ax, x, y, value, color='black', size=15, **kw):
    return ax.text(x, y, value, fontsize=size, color=color,
                   bbox=dict(facecolor='white', edgecolor='none', pad=1.8),
                   ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), **kw)

def arrow(ax, start, end, color='black', style='->', lw=1.6):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=13))

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'png']:
        target = OUT / f'{name}.{ext}'
        tmp = OUT / f'.{name}.tmp.{ext}'
        opts = {'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {'dpi': 190}
        fig.savefig(tmp, facecolor='white', **opts)
        tmp.replace(target)
    plt.close(fig)

def grids():
    fig, ax = figure(4.3)
    ax.set_aspect('equal')
    # Pressure centers separated by h; staggered velocities on cell faces.
    for a in [1.7, 3.9, 6.1, 8.3]:
        ax.plot([a, a], [1.3, 8.7], color='.8', lw=.8)
    for b in [1.7, 3.9, 6.1, 8.3]:
        ax.plot([1.3, 8.7], [b, b], color='.8', lw=.8)
    ax.add_patch(Rectangle((3.9, 3.9), 2.2, 2.2, fc='#edf3fa', ec='black', lw=1.6))
    for x, y, s in [(5,5,'P'), (7.2,5,'E'), (2.8,5,'W'), (5,7.2,'N'), (5,2.8,'S')]:
        ax.plot(x,y,'o',mfc='white',mec='black',mew=1.5,ms=9,zorder=8)
        ax.annotate(rf'$p_{s}$',xy=(x,y),xytext=(12,12),
                    textcoords='offset points',ha='left',va='bottom',fontsize=16,
                    bbox=dict(fc='white',ec='none',pad=1),
                    arrowprops=dict(arrowstyle='-',color='.4',lw=.7,shrinkB=5),zorder=9)
    for x,s in [(3.9,'w'),(6.1,'e')]:
        ax.plot(x,5,'o',c=BLUE,ms=6); arrow(ax,(x,5),(x+.7,5),BLUE)
        text(ax,x,4.25,rf'$u_{s}$',BLUE,17)
    for y,s in [(3.9,'s'),(6.1,'n')]:
        ax.plot(5,y,'^',c=GREEN,ms=7);arrow(ax,(5,y),(5,y+.7),GREEN)
        text(ax,4.35,y,rf'$v_{s}$',GREEN,17)
    arrow(ax,(3.9,.8),(6.1,.8),style='<->');text(ax,5,.8,r'$h$',size=17)
    arrow(ax,(8,1),(9,1));text(ax,9.3,1,r'$x$')
    arrow(ax,(8,1),(8,2.3));text(ax,8,2.7,r'$y$')
    save(fig,'mac_grid')

    fig, ax = figure(4.05)
    ax.add_patch(Rectangle((3.4,2.6),3.2,4.8,fc='#edf3fa',ec=BLUE,lw=2))
    ax.plot([3.4,6.6],[5,5],ls=':',color='.6')
    for x,s in [(3.4,r'$p_P$'),(5,r'$u_e$'),(6.6,r'$p_E$')]:
        ax.plot(x,5,'o',c='black',ms=6);text(ax,x,4.25,s,size=18)
    text(ax,5,6.45,r'$V_e$',size=20)
    for start,end,pos,s in [((5,7.4),(5,9),(5,9.55),r'$\boldsymbol{n}=(0,1)$'),
        ((6.6,5),(8,5),(8.4,6),r'$\boldsymbol{n}=(1,0)$'),
        ((3.4,5),(2,5),(1.65,6),r'$\boldsymbol{n}=(-1,0)$'),
        ((5,2.6),(5,1.2),(5,.6),r'$\boldsymbol{n}=(0,-1)$')]:
        arrow(ax,start,end);text(ax,*pos,s,size=15)
    arrow(ax,(3.4,3),(6.6,3),style='<->');text(ax,5,3,r'$x_E-x_P=h$',size=16)
    save(fig,'momentum_volume')

def locations():
    # Each location drawing occupies the full text width, with a separate key.
    for interpolated in [False, True]:
        fig,ax=plt.subplots(figsize=(6.6,5.2))
        fig.subplots_adjust(left=.13,right=.96,bottom=.13,top=.96)
        ax.set(xlim=(-.85,1.85),ylim=(-1.3,1.3),aspect='equal',xlabel=r'$x/h$',ylabel=r'$y/h$')
        ax.set_xticks([-.5,0,.5,1,1.5]);ax.set_yticks([-1,-.5,0,.5,1])
        for x in [-.5,.5,1.5]:ax.axvline(x,c='.86',lw=.7,zorder=0)
        for y in [-.5,.5]:ax.axhline(y,c='.86',lw=.7,zorder=0)
        ax.add_patch(Rectangle((0,-.5),1,1,fc='#edf3fa',ec=BLUE,lw=1.8))
        if not interpolated:
            points=[(-.5,0,r'$u_{\rm W}$',-.10,.20),(.5,0,r'$u_e$',.14,.18),
                    (1.5,0,r'$u_{\rm E}$',.12,.20),(.5,1,r'$u_{\rm N}$',.17,.12),
                    (.5,-1,r'$u_{\rm S}$',.17,-.06)]
            for x,y,s,dx,dy in points:
                ax.plot(x,y,'o',color=BLUE,ms=8);text(ax,x+dx,y+dy,s,BLUE,18)
            for x,y,s in [(0,.5,r'$v_{n,P}$'),(1,.5,r'$v_{n,E}$'),(0,-.5,r'$v_{s,P}$'),(1,-.5,r'$v_{s,E}$')]:
                ax.plot(x,y,'^',c=GREEN,ms=8);text(ax,x+.16,y+(.18 if y>0 else -.22),s,GREEN,16)
            for x,s in [(0,r'$p_P$'),(1,r'$p_E$')]:
                ax.plot(x,0,'s',mfc='white',mec='black',ms=8);text(ax,x,-.23,s,size=18)
        else:
            for x,y,s,tx,ty in [(-.5,0,r'$u_{\rm W}$',-.55,.24),(1.5,0,r'$u_{\rm E}$',1.55,.24),
                                (.5,1,r'$u_{\rm N}$',.79,1.02),(.5,-1,r'$u_{\rm S}$',.79,-1.02)]:
                ax.plot([.5,x],[0,y],'--',c=BLUE,alpha=.4,lw=1)
                ax.plot(x,y,'o',mfc='white',mec=BLUE,ms=7);text(ax,tx,ty,s,BLUE,17)
            ax.plot(.5,0,'o',c=BLUE,ms=8);text(ax,.71,.18,r'$u_e$',BLUE,18)
            for x,y,s,tx,ty in [(0,0,r'$\widehat u_{\rm W}$',-.13,-.24),
                (1,0,r'$\widehat u_{\rm E}$',1.15,-.24),(.5,.5,r'$\widehat u_{\rm N}$',.86,.68),
                (.5,-.5,r'$\widehat u_{\rm S}$',.86,-.67)]:
                ax.plot(x,y,'x',c=RED,ms=11,mew=2);text(ax,tx,ty,s,RED,19)
        save(fig,'hat_interpolated_locations' if interpolated else 'hat_stored_locations')

def transport():
    fig,ax=figure(4.7)
    # Geometry above, arithmetic below; long prose belongs in the caption.
    ax.add_patch(Rectangle((1.4,4.2),4.3,2.8,fc='#edf3fa',ec=BLUE,lw=1.7))
    ax.plot([1.4,5.7],[7,7],c=RED,lw=3)
    ax.plot(3.55,7,'ko',ms=5)
    arrow(ax,(3.55,7),(3.55,9.1),GREEN,lw=2.4)
    # Horizontal component is drawn horizontally, offset from the face and
    # linked by a light construction line, to avoid covering the red face.
    ax.plot([3.55,3.55],[7,8.05],ls=':',c='.55',lw=1)
    arrow(ax,(3.55,8.05),(5.6,8.05),BLUE,lw=2.4)
    text(ax,2.1,9.25,r'$c_{\rm N}=0.02$',GREEN,18)
    text(ax,7.2,8.05,r'$\widehat u_{\rm N}=0.06$',BLUE,18)
    text(ax,7.25,6.55,'North face',RED,15)
    ax.plot([5.7,6.15],[7,6.55],c='.5',lw=1)
    text(ax,3.55,5.3,r'$V_e$',size=19)
    arrow(ax,(1.4,3.65),(5.7,3.65),style='<->');text(ax,3.55,3.65,r'$h=0.25$',size=17)
    text(ax,7.4,4.5,'Arrow lengths\nare schematic.',size=13)
    ax.axhline(2.85,xmin=.05,xmax=.95,c='.8',lw=.8)
    text(ax,5,2.25,r'$Q_{\rm N}=h c_{\rm N}=0.25(0.02)=0.005$',GREEN,18)
    text(ax,5,1.05,r'$F^u_{\rm N}=Q_{\rm N}\widehat u_{\rm N}=0.005(0.06)=0.00030$',BLUE,18)
    save(fig,'north_face_transport')

    fig,ax=figure(4.1)
    for x in [1,5]:ax.add_patch(Rectangle((x,4),4,4.4,fc='#f3f6fa',ec='.5',lw=1.1))
    ax.plot([5,5],[4,8.4],c=BLUE,lw=2)
    for x,s in [(3,r'$p_P^\prime$'),(7,r'$p_E^\prime$')]:
        ax.plot(x,6,'ko',ms=6);text(ax,x,5.25,s,size=20)
    ax.plot(5,6,'o',c=BLUE,ms=7)
    arrow(ax,(4.7,7.35),(6.5,7.35),BLUE,lw=2)
    text(ax,5.7,8.9,r'$u_e^*=0.12$',BLUE,18)
    arrow(ax,(5.6,4.5),(4.4,4.5),RED,lw=2)
    text(ax,5,3.3,r'$u_e^\prime=-0.02$',RED,18)
    text(ax,5,1.95,r'$u_e^\prime=-d_e\,(p_E^\prime-p_P^\prime)/h$',size=18)
    text(ax,5,.7,r'$h=0.25,\quad d_e=0.5,\quad p_E^\prime-p_P^\prime=0.01$',size=16)
    save(fig,'shared_face_correction')

def checkerboard():
    x=np.arange(7)*.25;p=(-1.)**np.arange(7)
    fig,axes=plt.subplots(2,1,figsize=(6.6,4.8),layout='constrained')
    axes[0].plot(x,p,'ko--',ms=6)
    axes[0].set(ylabel=r'$p_i$',title='(a) Alternating pressure samples',yticks=[-1,0,1])
    axes[1].plot(x[1:-1],(p[2:]-p[:-2])/.5,'o',c=BLUE,label='Centered difference at a cell center',ms=7)
    axes[1].plot((x[1:]+x[:-1])/2,(p[1:]-p[:-1])/.25,'x',c=RED,label='Adjacent-center difference at the midpoint',ms=9,mew=2)
    axes[1].set(ylabel='Pressure gradient',title='(b) Response of the two discrete gradients')
    axes[1].legend(loc='upper center',ncols=1,fontsize=12)
    axes[1].set_ylim(-11,15)
    for ax in axes:ax.set_xlabel(r'$x$');ax.grid(alpha=.18)
    save(fig,'checkerboard_pressure')

def loops():
    rows={
        'simple': ['Guess pressure and face velocities',
            'Freeze convection; assemble and relax momentum',
            'Solve momentum for predictor velocities',
            'Solve pressure correction from cell mass imbalance',
            'Correct face velocities; relax pressure',
            'Check momentum and continuity defects',
            'Retain converged steady fields'],
        'piso': ['Save old-time velocity at time n',
            'Assemble transient momentum; solve predictor',
            'Solve pressure; correct face velocities',
            'Update correction terms using latest velocities',
            'Solve pressure again; correct velocities',
            'Complete configured pressure corrections',
            'Accept step; advance to time n+1'],
        'pimple': ['Save old-time velocity at time n',
            'Rebuild momentum from latest new-time estimate',
            'Solve momentum for predictor velocities',
            'Solve pressure; correct face velocities',
            'Update correction terms; repeat inner corrections',
            'Complete configured outer passes',
            'Accept step; test steady convergence']}
    for name,labels in rows.items():
        fig,ax=figure(6.4)
        left,right=1.3,8.25;ys=np.linspace(9.15,1.1,7)
        for i,(y,label) in enumerate(zip(ys,labels)):
            ax.add_patch(FancyBboxPatch((left,y-.43),right-left,.86,boxstyle='round,pad=.04',fc='#edf3fa',ec=BLUE,lw=1.2))
            # Deliberate wrapping at semantic breaks, never compressed text.
            import textwrap
            text(ax,(left+right)/2,y,textwrap.fill(label,47),size=14)
            if i<6:arrow(ax,((left+right)/2,y-.48),((left+right)/2,ys[i+1]+.48))
        source,target=(5,1) if name=='simple' else ((5,3) if name=='piso' else (5,1))
        ax.plot([right,9,9],[ys[source],ys[source],ys[target]],c=RED,lw=1.6)
        arrow(ax,(9,ys[target]),(right,ys[target]),RED)
        label={'simple':'Not converged','piso':'More correctors','pimple':'More outer passes'}[name]
        text(ax,9.55,(ys[source]+ys[target])/2,label,RED,14,rotation=90)
        if name=='pimple':
            ax.plot([left,.55,.55],[ys[4],ys[4],ys[3]],c=GREEN,lw=1.6)
            arrow(ax,(.55,ys[3]),(left,ys[3]),GREEN)
            text(ax,.12,(ys[4]+ys[3])/2,'Inner loop',GREEN,13,rotation=90)
        footer={'simple':'Steady iterations',
                'piso':'Old-time data stay fixed during corrections.',
                'pimple':'Old-time data stay fixed inside both loops.'}[name]
        text(ax,4.8,.18,footer,size=13)
        save(fig,name+'_visual')
    fig,ax=figure(4.1)
    labels=[('SIMPLE','Steady iteration','Rebuild momentum;\ncorrect pressure and velocity'),
            ('PISO','One physical step','Freeze momentum coefficients;\nrepeat pressure-velocity corrections'),
            ('PIMPLE','One physical step','Outer: rebuild momentum\nInner: pressure-velocity corrections')]
    for y,(name,clock,work) in zip([8.3,5.1,1.9],labels):
        text(ax,1.1,y,name,BLUE,17)
        ax.add_patch(Rectangle((2.5,y-1.15),7.2,2.3,fc='#edf3fa',ec=BLUE,lw=1.2))
        text(ax,6.1,y+.65,clock,size=14,weight='bold')
        text(ax,6.1,y-.35,work,size=14)
        arrow(ax,(2.7,y-1),(2.7,y+.9),RED)
    save(fig,'loop_map')

def blob(relative):
    p=ROOT/relative
    return p.read_bytes() if p.is_file() else subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT)

def archived_rows():
    manifest=json.loads(blob('results/week01_2_pressure_velocity/manifest.json'))
    rows=[]
    for rec in manifest['runs']:
        data=blob('results/week01_2_pressure_velocity/'+rec['name']+'.npz')
        assert hashlib.sha256(data).hexdigest()==rec['field_sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as f:rows.append({**rec,**{k:f[k] for k in f.files}})
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','results/week01_2_formulations'],cwd=ROOT,text=True).splitlines()
    for name in paths:
        if not re.search(r'/[^/]+-re(?:400|1000)-n\d+\.json$',name):continue
        rec=json.loads(blob(name))
        if rec.get('status')=='failed':continue
        data=blob(name[:-5]+'.npz');assert hashlib.sha256(data).hexdigest()==rec['field_sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as f:rows.append({**rec,**{k:f[k] for k in f.files}})
    return rows

def profiles(rows,re,name=None):
    from flowmllab.pressure_velocity import centerlines,GHIA_X,GHIA_Y,GHIA_U,GHIA_V
    from flowmllab.pressure_velocity_experiments import REFERENCE,benchmark_valid_mask
    fig,axes=plt.subplots(2,1,figsize=(6.6,7.1),layout='constrained')
    for method,color,label in zip(METHODS,COLORS,NAMES):
        for r in sorted(rows,key=lambda r:r['config']['cells']):
            c=r['config']
            if c['reynolds']!=re or c['method']!=method:continue
            if re==100 and c['cells']!=64:continue
            y,u,x,v=centerlines(r);ls={32:'-',64:'--',128:':'}[c['cells']]
            axes[0].plot(u,y,c=color,ls=ls,lw=1.6,label=f'{label}, h=1/{c["cells"]}')
            axes[1].plot(x,v,c=color,ls=ls,lw=1.6)
    ref={'u':GHIA_U,'v':GHIA_V} if re==100 else REFERENCE[re]
    axes[0].plot(ref['u'],GHIA_Y,'ko',ms=4,label='Ghia et al. (1982)')
    x=np.asarray(GHIA_X);v=np.asarray(ref['v']);mask=benchmark_valid_mask(re,'v',x)
    axes[1].plot(x[mask],v[mask],'ko',ms=4)
    if (~mask).any():axes[1].plot(x[~mask],v[~mask],'rx',ms=9,mew=2)
    axes[0].set(xlabel=r'$u/U$ at $x/L=0.5$',ylabel=r'$y/L$')
    axes[1].set(xlabel=r'$x/L$',ylabel=r'$v/U$ at $y/L=0.5$')
    for ax in axes:ax.grid(alpha=.17)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=2,fontsize=12,frameon=False)
    save(fig,name or f'profiles_re{re}')

def fields(rows,key,re=100,cells=64):
    from flowmllab.cavity_visuals import RGB,flow_structure,field_normalization
    rr=[next(r for r in rows if r['config']['method']==m and r['config']['reynolds']==re and r['config']['cells']==cells) for m in METHODS]
    vals=[np.hypot(r['u'],r['v']) if key=='speed' else r[key] for r in rr]
    norm=field_normalization(key,vals)
    fig,axes=plt.subplots(2,2,figsize=(6.6,6.6),layout='constrained')
    for ax,r,z,label in zip(axes.flat,rr,vals,NAMES):
        im=ax.pcolormesh(r['x'],r['y'],z,cmap=RGB,norm=norm,shading='nearest',rasterized=True)
        flow_structure(ax,r['x'],r['y'],r['u'],r['v'])
        ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xlabel=r'$x/L$',ylabel=r'$y/L$',title=label)
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    label={'speed':r'$|\boldsymbol{u}|/U$','p':r'$p/(\rho U^2)$','omega':r'$\omega L/U$'}[key]
    cb=fig.colorbar(im,ax=list(axes.flat),location='bottom',shrink=.85,aspect=30,pad=.05,label=label)
    if key=='speed':cb.set_ticks([0,.25,.5,.75,1])
    save(fig,'field_'+key+f'_re{re}')

def convergence(rows):
    fig,axes=plt.subplots(2,1,figsize=(6.6,5.9),layout='constrained')
    for m,color,label in zip(METHODS,COLORS,NAMES):
        r=next(r for r in rows if r['config']['method']==m and r['config']['reynolds']==100 and r['config']['cells']==64)
        if 'history' not in r:continue
        z=r['history']
        axes[0].semilogy(z[:,2],z[:,3],c=color,label=label)
        axes[1].semilogy(z[:,2],np.maximum(z[:,4],1e-16),c=color)
    axes[0].set_ylabel('FV momentum defect, Linf')
    axes[1].set_ylabel('Cell divergence, Linf')
    for ax in axes:ax.set_xlabel('Recorded solver wall time (s)');ax.grid(alpha=.17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,frameon=False)
    save(fig,'convergence_re100')

def refinement(rows,re):
    from flowmllab.cavity_visuals import RGB,flow_structure
    fig,axes=plt.subplots(2,2,figsize=(6.6,6.4),layout='constrained')
    for ax,(m,n) in zip(axes.flat,[('simple',64),('streamfunction-vorticity',64),('streamfunction-vorticity',128)]):
        r=next(r for r in rows if r['config']['method']==m and r['config']['reynolds']==re and r['config']['cells']==n)
        im=ax.pcolormesh(r['x'],r['y'],np.hypot(r['u'],r['v']),cmap=RGB,vmin=0,vmax=1,shading='nearest',rasterized=True)
        flow_structure(ax,r['x'],r['y'],r['u'],r['v'])
        title=('SIMPLE' if m=='simple' else r'$\psi$-$\omega$ (FD)')+f', h=1/{n}'
        ax.set(aspect='equal',xlabel=r'$x/L$',ylabel=r'$y/L$',title=title,xlim=(0,1),ylim=(0,1))
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    axes[1,1].axis('off')
    axes[1,1].text(.05,.7,'Color: speed / U\n\nBlack curves: velocity\nstreamlines\n\nSame lid speed and\ncolor limits in all panels.',transform=axes[1,1].transAxes,fontsize=14,va='center')
    fig.colorbar(im,ax=list(axes.flat),location='bottom',label=r'$|\boldsymbol{u}|/U$',shrink=.85,aspect=30,pad=.05)
    save(fig,f'refinement_re{re}')

SUPPLEMENT_CAPTIONS = {}

def supplementary(rows,include_historical=True):
    """Notebook companions use the same final-size fonts as the lecture."""
    for re in ([400,1000] if include_historical else []):
        for key,title in [('speed','speed'),('p','zero-mean pressure'),('omega','signed vorticity')]:
            fields(rows,key,re=re,cells=32)
            SUPPLEMENT_CAPTIONS[f'field_{key}_re{re}']=(f'Retained Re={re} {title} at h=1/32, with computed velocity streamlines. '
                'All four formulations are shown on the same physical color scale. Signed fields use symmetric logarithmic normalization. '
                'These coarse-grid solutions are distinct from the selected fine-grid comparisons.')
        fig,axes=plt.subplots(2,1,figsize=(6.6,5.9),layout='constrained')
        for m,color,label in zip(METHODS,COLORS,NAMES):
            r=next(r for r in rows if r['config']['method']==m and r['config']['reynolds']==re and r['config']['cells']==32)
            if m=='streamfunction-vorticity':axes[1].semilogy(r['residual_steps'],r['history_rate'],c=color,label=label)
            else:axes[0].semilogy(r['history'][:,0],r['history'][:,3],c=color,label=label)
        axes[0].set(xlabel='FV outer iterations',ylabel='FV momentum defect, Linf')
        axes[1].set(xlabel='FD explicit time steps',ylabel='Relative vorticity-update rate')
        for ax in axes:ax.legend(fontsize=12);ax.grid(alpha=.17)
        name=f'convergence_re{re}';save(fig,name)
        SUPPLEMENT_CAPTIONS[name]=(f'Retained Re={re}, h=1/32 convergence. Upper: FV steady momentum defect. '
            'Lower: FD relative vorticity-update rate, divided by its time step. The equations, iteration units and stopping norms differ; '
            'their ordinate magnitudes cannot be used to rank formulation accuracy. Recorded costs are in the tables.')
    # Independent follow-up suite: keep its provenance separate from old runs.
    from qa.review_student_cavity import load
    fresh=load()
    from flowmllab.pressure_velocity import centerlines,GHIA_Y
    from flowmllab.pressure_velocity_experiments import REFERENCE,benchmark_valid_mask
    for re in [400,1000]:
        central=[r for r in fresh if r['config']['convection']=='central']
        name=f'review_profiles_re{re}';profiles(central,re,name)
        SUPPLEMENT_CAPTIONS[name]=(f'Independent follow-up Re={re} velocity profiles for central convection. '
            'Solid lines use 32 FV cells per side; dashed lines use 64 where available. Black markers are Ghia samples. '
            'At Re=400 the red cross retains the isolated published anomaly without including it in the qualified norm. '
            'These fields and their timings belong to the separately identified follow-up suite.')
        fig,axes=plt.subplots(2,1,figsize=(6.6,5.9),layout='constrained')
        for r in central:
            c=r['config']
            if c['reynolds']!=re or c['cells']!=32:continue
            j=METHODS.index(c['method']);h=r['history']
            count=1 if c['method']=='simple' else c['pressure_correctors']*(c['outer_correctors'] if c['method']=='pimple' else 1)
            axes[0].semilogy(count*h[:,0],h[:,3],c=COLORS[j],label=NAMES[j])
            axes[1].semilogy(count*h[:,0],np.maximum(h[:,4],1e-17),c=COLORS[j])
        axes[0].axhline(1e-6,c='black',ls=':');axes[0].set_ylabel('Momentum defect\n(Linf)')
        axes[1].set_ylabel('Cell divergence, Linf')
        for ax in axes:ax.set_xlabel('Cumulative pressure solves');ax.grid(alpha=.17)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=3,frameon=False)
        name=f'review_coupling_re{re}';save(fig,name)
        SUPPLEMENT_CAPTIONS[name]=(f'Independent follow-up Re={re}, 32-cell central-convection histories against cumulative pressure solves. '
            'The dotted momentum line is the 1e-6 target. Divergence fluctuations near machine precision are not spatial error. '
            'Solve count is a work indicator, not an assertion that every linear solve has equal cost.')
    fig,axes=plt.subplots(2,1,figsize=(6.6,6.3),layout='constrained')
    for r in fresh:
        c=r['config']
        if c['method']!='simple' or c['reynolds']!=400:continue
        y,u,x,v=centerlines(r);label=f'N={c["cells"]}, {c["convection"]}'
        color=RED if c['convection']=='upwind' else (GREEN if c['cells']==64 else BLUE)
        axes[0].plot(u,y,c=color,label=label);axes[1].plot(x,v,c=color)
    axes[0].plot(REFERENCE[400]['u'],GHIA_Y,'ko',ms=4,label='Ghia reference')
    from flowmllab.pressure_velocity import GHIA_X
    x=np.asarray(GHIA_X);mask=benchmark_valid_mask(400,'v',x)
    axes[1].plot(x[mask],np.asarray(REFERENCE[400]['v'])[mask],'ko',ms=4)
    axes[0].set(xlabel=r'$u/U$',ylabel=r'$y/L$');axes[1].set(xlabel=r'$x/L$',ylabel=r'$v/U$')
    for ax in axes:ax.grid(alpha=.17)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2,frameon=False,fontsize=12)
    save(fig,'review_convection_re400')
    SUPPLEMENT_CAPTIONS['review_convection_re400']=('Independent follow-up Re=400 SIMPLE profiles: first-order upwind and central convection on the indicated meshes. '
        'Changing convection changes the spatial equations; this is separate from changing the pressure-velocity coupling algorithm. '
        'The isolated Ghia horizontal-profile anomaly is omitted from the qualified markers.')

CAPTIONS = {
 'mac_grid': 'Staggered (MAC) storage. Open black circles mark cell centers where pressure is stored; blue filled circles mark horizontal velocities and green triangles mark vertical velocities. The shaded square is the pressure control volume around P, of side length h.',
 'checkerboard_pressure': 'A synthetic pressure diagnostic with h=0.25. (a) Pressure samples at equally spaced cell centers alternate between +1 and -1. (b) Blue circles use the two samples on either side of a cell center; red crosses use the two adjacent samples around their midpoint. The blue result is zero while the red results are +8 or -8. The legend identifies these two difference formulas and their evaluation positions; it does not show two flow solutions.',
 'momentum_volume': 'Horizontal-momentum control volume V_e. The unknown u_e is at its center, while p_P and p_E are at the west and east face midpoints. Arrows show outward unit normals. The two pressure samples are separated by h.',
 'hat_stored_locations': 'Stored samples around the horizontal-momentum volume (blue square). Blue circles: horizontal velocity; green triangles: vertical velocity; open black squares: pressure. Coordinates are relative to pressure center P and scaled by h.',
 'hat_interpolated_locations': 'Evaluation points for transported horizontal velocity. Blue circles are stored u samples and red crosses are interpolated values on momentum-volume faces. Dashed segments connect each pair used in an average. The west and east crosses coincide geometrically with pressure centers; pressure symbols are omitted here for clarity.',
 'north_face_transport': 'Horizontal momentum transported through the north face. The normal speed c_N carries fluid upward through the red face; the transported component is the horizontal velocity u-hat_N, offset above the face for clarity. Q_N is outward volume flux per unit depth. F_N^u is horizontal-momentum flux per unit density and unit depth. The values are nondimensional; arrow lengths are schematic.',
 'shared_face_correction': 'A shared-face pressure correction. When p-prime_E exceeds p-prime_P by 0.01, the velocity increment is westward: u-prime_e=-0.02. The predictor 0.12 becomes 0.10. The same face velocity enters the two neighboring outward flux balances with opposite signs.',
 'simple_visual': 'SIMPLE for a steady problem. Follow the blue boxes downward. The red return path rebuilds momentum after an unconverged iteration. Correct face velocities fully, relax pressure as specified, and test both momentum and cell continuity.',
 'piso_visual': 'PISO within one physical time step. The red return path updates correction terms before another pressure solve, using fixed momentum coefficients. The saved old-time velocity remains unchanged until the new time level is accepted.',
 'pimple_visual': 'PIMPLE within one physical time step. Green: inner pressure-velocity corrections with fixed momentum coefficients. Red: outer passes rebuild those coefficients from the latest new-time estimate. The retained code completes the configured passes, accepts the step, and tests steady convergence between steps. Neither correction loop advances the saved old-time data.',
 'loop_map': 'Comparison of repeated work. A SIMPLE cycle is a steady nonlinear iteration. PISO repeats pressure-velocity corrections within one step. PIMPLE nests those corrections inside momentum-coefficient updates. The red arrows denote repetition; they do not denote physical time advancement.',
 'profiles_re100': 'Retained Re=100 centerline velocities at matched spacing h=1/64. Curves show all four formulations; black markers are Ghia et al. (1982) samples. FV uses 64 pressure cells per side and FD uses 65 boundary-inclusive nodes. Line styles and colors identify the computed series.',
 'field_speed_re100': 'Retained Re=100 speed and velocity streamlines at h=1/64. All panels use the same physical color limits. Streamline arrows show circulation direction; their density does not measure vorticity.',
 'field_omega_re100': 'Retained Re=100 signed vorticity with velocity streamlines at h=1/64. One symmetric logarithmic color scale is shared by all panels, linear for absolute vorticity up to 1. The displayed colorbar values are nondimensional physical vorticity, including wall peaks.',
 'field_p_re100': 'Retained Re=100 zero-mean pressure and velocity streamlines at h=1/64. One symmetric logarithmic color scale is shared, linear for absolute pressure up to 0.01. FD pressure is reconstructed after the vorticity solve. No independent pressure benchmark is claimed.',
 'convergence_re100': 'Retained Re=100 finite-volume convergence histories at h=1/64. The horizontal coordinate is recorded solver-call wall time from one local run per method. Momentum and continuity are separate defects. The streamfunction-vorticity solver uses a different native stopping norm and is included in the adjacent timing tables, not on these FV residual axes.',
 'profiles_re400': 'Retained Re=400 centerline velocities. Color identifies formulation and line style identifies spacing. Black markers are Ghia samples; the red cross retains the isolated published horizontal-profile anomaly at x/L=0.9063, excluded only from the qualified error norm. Selected fine cases do not constitute a complete refinement matrix.',
 'profiles_re1000': 'Retained Re=1000 centerline velocities. Color identifies formulation and line style identifies spacing; black markers are Ghia et al. (1982) samples. Finer FD solutions show the effect of spatial resolution separately from the coupling algorithm.',
 'refinement_re400': 'Retained Re=400 speed and computed velocity streamlines: SIMPLE at h=1/64, and independent streamfunction-vorticity solutions at h=1/64 and 1/128. The color scale and lid speed are common. Small recirculation structures must be supported by the resolved velocity field.',
 'refinement_re1000': 'Retained Re=1000 speed and computed velocity streamlines for the same selected refinement cases. The finer grid changes resolved boundary layers and corner circulation. Additional pressure corrections alone cannot recover spatial structures absent from the mesh.'
}

def caption_tex(name):
    # Mathematical labels remain properly typeset in captions as well as figures.
    s=CAPTIONS[name]
    replacements={'u-hat_N':r'$\widehat u_{\rm N}$','F_N^u':r'$F^u_{\rm N}$','Q_N':r'$Q_{\rm N}$','c_N':r'$c_{\rm N}$',
        'p-prime_E':r'$p_E^\prime$','p-prime_P':r'$p_P^\prime$','u-prime_e=-0.02':r'$u_e^\prime=-0.02$',
        'V_e':r'$V_e$','u_e':r'$u_e$','p_P':r'$p_P$','p_E':r'$p_E$'}
    return re.sub('|'.join(re.escape(k) for k in replacements),
                  lambda m: replacements[m.group()],s)

def integrate():
    folder=ROOT/'lectures/source'
    main=folder/'week01_2_pressure_velocity.tex'
    src=main.read_text(encoding='utf-8')
    if r'\usepackage{caption}' not in src:
        src=src.replace(r'\usepackage{fontspec}',r'''\usepackage{caption}
\captionsetup{font=normalsize,labelfont=bf,justification=raggedright,singlelinecheck=false,skip=6pt}
\newcommand{\lessonfigure}[3]{%
\begin{center}\includegraphics[width=\linewidth]{week01_2_assets/#1.pdf}%
\captionof{figure}{#2}\label{fig:#3}\end{center}}
\usepackage{fontspec}''')
    # Each field figure gets its own page rather than two reduced figures.
    src=src.replace(r'\topic{18}{Pressure and native convergence histories}',r'\topic{18}{Pressure and native convergence histories}')
    main.write_text(src,encoding='utf-8',newline='\n')
    aliases={'centerlines.png':'profiles_re100','speed.png':'field_speed_re100','omega.png':'field_omega_re100',
             'p.png':'field_p_re100','convergence.png':'convergence_re100'}
    pattern=r'\\begin\{center\}\\includegraphics\[[^\]]*\]\{([^}]+)\}\\end\{center\}'
    for path in folder.glob('week01_2*.tex'):
        s=path.read_text(encoding='utf-8')
        def repl(m):
            target=m.group(1);name=Path(target).stem
            if '/week01_2_pressure_velocity/figures/' in target:name=aliases[Path(target).name]
            elif '/week01_2_formulations/' in target:
                if target.endswith('centerlines.png'):name='profiles_re400'
                elif target.endswith('refinement.png'):name='refinement_re400'
            if name=='hat_locations':name='hat_stored_locations'
            if name not in CAPTIONS:raise ValueError(target)
            return '\\lessonfigure{'+name+'}{'+caption_tex(name)+'}{'+name.replace('_','-')+'}'
        revised=[]
        for line in re.sub(pattern,repl,s).splitlines():
            m=re.match(r'\\lessonfigure\{([^}]+)\}',line)
            if m:
                name=m.group(1)
                line='\\lessonfigure{'+name+'}{'+caption_tex(name)+'}{'+name.replace('_','-')+'}'
            revised.append(line)
        updated='\n'.join(revised)+'\n'
        if updated.splitlines()!=s.splitlines():
            path.write_text(updated,encoding='utf-8',newline='\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrate',action='store_true')
    args=parser.parse_args()
    grids();locations();transport();checkerboard();loops()
    rows=archived_rows()
    for re in [100,400,1000]:profiles(rows,re)
    for key in ['speed','omega','p']:fields(rows,key)
    convergence(rows)
    for re in [400,1000]:refinement(rows,re)
    supplementary(rows)
    (OUT/'supplementary_captions.json').write_text(json.dumps(SUPPLEMENT_CAPTIONS,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.integrate:integrate()
    print(json.dumps({'figures':len(CAPTIONS),'cfd_trajectories_recomputed':False,'width_inches':6.6}))

if __name__=='__main__':main()
